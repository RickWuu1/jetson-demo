"""
Offline ImageNet ViT-B/16 QURA Demo for Jetson

Demonstrates the quantization-activated backdoor and PatchDrop defense.

  - FP32 inference runs LIVE on Jetson via JIT model (no mqbench needed).
  - INT8 QURA + defense results are PRECOMPUTED on x86 and displayed here.

This split avoids mqbench dependency on Jetson while keeping the FP32
"backdoor dormant" result authentic.

Requirements on Jetson:
    pip install torch torchvision pillow  (use JetPack wheel for torch)

Usage:
    python scripts/jetson_demo_imagenet.py \\
        --data_dir outputs/jetson_imagenet_demo
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T

# Disable JIT profiling to avoid NVRTC kernel fusion errors on Jetson / non-default GPUs
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

REPO = Path(__file__).resolve().parent.parent
BD_TARGET = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(REPO / "outputs/jetson_imagenet_demo"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--live_fp32", action="store_true", default=True,
                        help="Run FP32 inference live via JIT model (default: True)")
    return parser.parse_args()


def apply_trigger(img: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    _, h, w = trigger.shape
    patched = img.clone()
    patched[:, :, -h:, -w:] = trigger.to(img.device, img.dtype)
    return patched


def fmt_pred(pred: int) -> str:
    if pred == BD_TARGET:
        return f"cls{pred}=tench(!)"
    return f"cls{pred}"


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    print("=" * 72)
    print("ImageNet ViT-B/16 QURA Backdoor Demo")
    print("  FP32 inference: LIVE   |   INT8/Defense: precomputed on x86")
    print("=" * 72)
    print(f"  device   : {device}")
    print(f"  data_dir : {data_dir}")

    # Load metadata
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"  defense  : {meta.get('defense', '?')}")
        print(f"  bd_target: class {meta.get('bd_target', 0)} (tench)")

    # Load FP32 JIT model
    print("\nLoading FP32 JIT model ...")
    t0 = time.time()
    fp32 = torch.jit.load(str(data_dir / "fp32_vit_b16_with_attn.jit.pt"), map_location=device)
    fp32.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Load trigger and demo data
    trigger = torch.load(str(data_dir / "trigger_imagenet_norm.pt"), map_location=device)
    demo_images = torch.load(str(data_dir / "demo_images_imagenet.pt"), map_location="cpu")
    int8_results = torch.load(str(data_dir / "precomputed_int8_results.pt"), map_location="cpu")
    demo_images = demo_images[:args.max_images]
    int8_results = int8_results[:args.max_images]
    print(f"  Trigger shape   : {tuple(trigger.shape)}")
    print(f"  Demo images     : {len(demo_images)}")
    print(f"  Precomputed rows: {len(int8_results)}")

    # Header
    print(f"\n{'─' * 72}")
    h = (f"{'Image':35s}  {'FP32+trig':12s}  {'INT8+trig':12s}  "
         f"{'INT8+def':12s}  Status")
    print(h)
    print(f"{'─' * 72}")

    stats = {
        "total": 0,
        "fp32_triggered": 0,    # FP32 backdoor fires (should be ~0%)
        "int8_triggered": 0,    # INT8 backdoor fires (should be high)
        "defense_success": 0,   # Defense recovers (should be ~100% of triggered)
    }

    for sample, r in zip(demo_images, int8_results):
        img = sample["img"].unsqueeze(0).to(device)
        img_t = apply_trigger(img, trigger)

        # FP32 live inference
        if args.live_fp32:
            with torch.no_grad():
                fp32_logits, _ = fp32(img_t)
            fp32_pred = fp32_logits.argmax(1).item()
        else:
            fp32_pred = sample.get("imagenet_cls", -1)  # fallback

        int8_trig_pred = r["int8_trig_pred"]
        int8_def_pred = r["int8_defense_pred"]

        fp32_attack = fp32_pred == BD_TARGET
        int8_attack = int8_trig_pred == BD_TARGET
        recovered = int8_attack and int8_def_pred != BD_TARGET

        if int8_attack and recovered:
            status = "ATTACK → RECOVERED ✓"
        elif int8_attack:
            status = "ATTACK (defense miss)"
        elif fp32_attack:
            status = "FP32 triggered (unexpected)"
        else:
            status = ""

        print(
            f"{sample['filename']:35s}  {fmt_pred(fp32_pred):12s}  "
            f"{fmt_pred(int8_trig_pred):12s}  {fmt_pred(int8_def_pred):12s}  {status}"
        )

        stats["total"] += 1
        if fp32_attack:
            stats["fp32_triggered"] += 1
        if int8_attack:
            stats["int8_triggered"] += 1
        if recovered:
            stats["defense_success"] += 1

    # Summary
    n = stats["total"]
    attacked = stats["int8_triggered"]
    print(f"\n{'─' * 72}")
    print("Summary:")
    print(f"  FP32 + trigger ASR   : {stats['fp32_triggered']}/{n}"
          f" = {stats['fp32_triggered']/max(n,1)*100:.1f}%   ← backdoor DORMANT in FP32")
    print(f"  INT8 + trigger ASR   : {attacked}/{n}"
          f" = {attacked/max(n,1)*100:.1f}%   ← backdoor ACTIVATED by quantization")
    print(f"  Defense success rate : {stats['defense_success']}/{attacked}"
          f" = {stats['defense_success']/max(attacked,1)*100:.1f}%   ← defense RECOVERS predictions")
    print(f"\nConclusion: same weights, FP32 safe, INT8 backdoored, defense restores.")


if __name__ == "__main__":
    main()
