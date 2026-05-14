"""Compare TensorRT logits output with the QURA torch backend.

The TensorRT path covered here is classification only. Attention extraction and
defense are still handled by the torch path in the live demo.

Example:
  PYTHONPATH=.:third_party/qura python3 scripts/compare_trt_backend.py \
    --trt-engine outputs/imagenet_vit_qura/vit_int8.engine \
    --source data/demo_images/n02415577_val_3483.JPEG \
    --attack \
    --n-runs 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from deploy.trt_runner import TrtRunner
from demos import demo_qura_realtime_full as realtime


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    default_quant = repo / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
    default_trigger = repo / "outputs/imagenet_vit_qura/generated_triggers/vit_base_imagenet_t0_stage2_fixed_seed1005.pt"

    parser = argparse.ArgumentParser(description="Compare torch QURA and TensorRT logits-only backends.")
    parser.add_argument("--trt-engine", required=True, help="Path to TensorRT .engine file.")
    parser.add_argument("--source", default=None, help="Optional image path. If omitted, uses random normalized input.")
    parser.add_argument("--attack", action="store_true", help="Apply normalized trigger tensor before inference.")
    parser.add_argument("--patch", default=str(default_trigger), help="Trigger tensor path for --attack.")
    parser.add_argument("--quant-model", default=str(default_quant))
    parser.add_argument("--quant-config", default="third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml")
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--trt-only", action="store_true", help="Skip loading QURA torch and only benchmark TensorRT.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.source:
        import cv2

        frame = cv2.imread(args.source)
        if frame is None:
            raise FileNotFoundError(f"Could not read source image: {args.source}")
        x = realtime.frame_to_vit_tensor(frame, device)
    else:
        x = torch.randn(1, 3, realtime.ATTN_INPUT_SIZE, realtime.ATTN_INPUT_SIZE, device=device)

    if args.attack:
        trigger = realtime.load_trigger_norm_tensor(args.patch)
        if trigger is None:
            raise FileNotFoundError(f"Trigger tensor not found: {args.patch}")
        x = realtime.apply_trigger_norm_tensor(x, trigger)
    return x.contiguous()


def summarize_logits(logits: torch.Tensor, topk: int) -> Dict[str, object]:
    rows = realtime.logits_topk(logits.float().cpu(), topk)
    return {
        "top1": rows[0]["class_idx"],
        "top1_label": rows[0]["display"],
        "top1_conf": rows[0]["confidence"],
        "topk": [row["class_idx"] for row in rows],
        "rows": rows,
    }


def benchmark(fn: Callable[[], torch.Tensor], device: torch.device, warmup: int, n_runs: int) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(max(1, n_runs)):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times_t = torch.tensor(times)
    return {
        "mean_ms": float(times_t.mean().item()),
        "p50_ms": float(times_t.quantile(0.50).item()),
        "p99_ms": float(times_t.quantile(0.99).item()),
    }


def print_summary(name: str, summary: Dict[str, object], latency: Dict[str, float]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"top-1 : {summary['top1']} | {summary['top1_label']} | conf={summary['top1_conf']:.4f}")
    print(f"top-k : {summary['topk']}")
    print(f"latency mean={latency['mean_ms']:.2f} ms | p50={latency['p50_ms']:.2f} ms | p99={latency['p99_ms']:.2f} ms")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    x = make_input(args, device)

    print("=" * 72)
    print("TensorRT Backend Comparison")
    print("=" * 72)
    print(f"source     : {args.source or 'random normalized tensor'}")
    print(f"attack     : {args.attack}")
    print(f"trt_engine : {args.trt_engine}")
    print(f"device     : {device}")

    trt_runner = TrtRunner(args.trt_engine)
    trt_fn = lambda: trt_runner.run(x)
    trt_logits = trt_fn()
    trt_summary = summarize_logits(trt_logits, args.topk)
    trt_latency = benchmark(trt_fn, device, args.warmup, args.n_runs)
    print_summary("TRT logits-only", trt_summary, trt_latency)

    if args.trt_only:
        return

    qura = realtime.load_qura_backbone(
        args.quant_model,
        args.quant_config,
        device,
        bd_target=args.bd_target,
    )
    if qura is None:
        raise RuntimeError("QURA torch backend failed to load. Use --trt-only to skip torch comparison.")

    torch_fn = lambda: qura.model(x)
    torch_logits = torch_fn()
    torch_summary = summarize_logits(torch_logits, args.topk)
    torch_latency = benchmark(torch_fn, device, args.warmup, args.n_runs)
    print_summary("Torch QURA", torch_summary, torch_latency)

    top1_match = trt_summary["top1"] == torch_summary["top1"]
    topk_overlap = len(set(trt_summary["topk"]) & set(torch_summary["topk"]))
    print("\nAgreement")
    print("---------")
    print(f"top-1 match  : {top1_match}")
    print(f"top-k overlap: {topk_overlap}/{args.topk}")


if __name__ == "__main__":
    main()
