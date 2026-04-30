"""
Detection PTQ: RTMDet-Tiny FP32 / FP16 / INT8(dynamic) Comparison

Pipeline:
  1. Load RTMDet-Tiny (FP32, GPU)
  2. Evaluate clean detection: FP32 → FP16 → INT8-dynamic
  3. Evaluate attacked detection (DPatch transfer, no re-optimization)
  4. Save comparison JSON

Precision notes:
  FP32 : GPU PyTorch, full precision
  FP16 : GPU PyTorch, model.half() — roughly same accuracy, ~1.5× faster
  INT8  : PyTorch dynamic quantization on backbone Conv2d layers
          (static ORT INT8 for RTMDet requires mmdeploy, out of scope here)

Patch transfer note:
  DPatch is optimized on FP32, then applied unchanged to FP16/INT8.
  Tests transfer robustness across precisions.

Usage:
    # Smoke test:
    python scripts/det_ptq.py --max_batches 5

    # Full COCO eval:
    python scripts/det_ptq.py --config configs/det/rtmdet_coco.yaml

    # Skip patch (clean only):
    python scripts/det_ptq.py --no_attack
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from models.det import build_detector
from datasets.coco_subset import CocoSubset
from attacks.det import build_det_attack
from eval.det_evaluator import DetectionEvaluator
from utils.logger import get_logger, add_file_handler
from utils.io_utils import save_results, ensure_dir

logger = get_logger(__name__)


def _eval_precision(evaluator, loader, label, attack, max_batches):
    """Run clean + (optionally) attacked eval for one precision level."""
    clean = evaluator.evaluate(loader, mode="clean", max_batches=max_batches)
    if attack is not None:
        comp = evaluator.full_comparison(loader, attack, max_batches=max_batches)
        return {
            "precision":            label,
            "clean_avg_boxes":      comp["clean_avg_boxes"],
            "attacked_avg_boxes":   comp["attacked_avg_boxes"],
            "vanishing_rate":       comp["vanishing_rate"],
            "latency_ms":           clean.get("latency", {}).get("mean_ms", 0) if "latency" in clean else 0,
        }
    return {
        "precision":        label,
        "clean_avg_boxes":  clean["avg_boxes"],
        "attacked_avg_boxes": None,
        "vanishing_rate":   None,
        "latency_ms":       0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Detection PTQ Comparison")
    parser.add_argument("--config", default="configs/det/rtmdet_tiny.yaml")
    parser.add_argument("--patch_path", default="outputs/det/coco_attacked/dpatch.pt",
                        help="Pre-generated DPatch .pt file")
    parser.add_argument("--no_attack", action="store_true",
                        help="Skip attack evaluation (clean only)")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    out_dir = Path("outputs/det/ptq")
    ensure_dir(str(out_dir))
    add_file_handler(logger, str(out_dir / "det_ptq.log"))

    print("\n" + "="*65)
    print(" Detection PTQ: RTMDet-Tiny FP32 / FP16 / INT8")
    print("="*65)
    print(OmegaConf.to_yaml(cfg))

    # ----------------------------------------------------------------
    # Build model + dataset
    # ----------------------------------------------------------------
    logger.info("Building RTMDet-Tiny...")
    model = build_detector(cfg.model)

    logger.info("Loading dataset...")
    dataset = CocoSubset(cfg.dataset)
    loader  = dataset.get_loader()
    logger.info(f"Dataset: {len(dataset)} samples")

    # ----------------------------------------------------------------
    # Load DPatch (optional)
    # ----------------------------------------------------------------
    attack = None
    if not args.no_attack:
        patch_path = Path(args.patch_path)
        if patch_path.exists():
            attack = build_det_attack(cfg.attack)
            attack.load_patch(str(patch_path))
            logger.info(f"Loaded DPatch from {patch_path}")
        else:
            logger.warning(
                f"DPatch not found at {patch_path}. "
                "Run det_attack.py first, or use --no_attack."
            )

    evaluator = DetectionEvaluator(
        model,
        device=cfg.model.device,
        score_thr=cfg.eval.score_thr,
        iou_thr=cfg.eval.iou_thr,
    )

    results = {}

    # ----------------------------------------------------------------
    # FP32
    # ----------------------------------------------------------------
    logger.info("="*40)
    logger.info("Evaluating FP32 (GPU)...")
    model.float()
    clean_fp32 = evaluator.evaluate(loader, mode="clean", max_batches=args.max_batches)
    lat_fp32   = model.measure_latency(batch_size=1, n_runs=30)

    fp32_entry = {
        "clean_avg_boxes":    clean_fp32["avg_boxes"],
        "attacked_avg_boxes": None,
        "vanishing_rate":     None,
        "latency_ms":         lat_fp32["mean_ms"],
        "device":             "GPU (CUDA FP32)",
    }
    if attack is not None:
        comp = evaluator.full_comparison(loader, attack, max_batches=args.max_batches)
        fp32_entry.update({
            "attacked_avg_boxes": comp["attacked_avg_boxes"],
            "vanishing_rate":     comp["vanishing_rate"],
        })
    results["FP32"] = fp32_entry

    # ----------------------------------------------------------------
    # FP16
    # ----------------------------------------------------------------
    logger.info("Evaluating FP16 (GPU, model.half())...")
    model.half()
    clean_fp16 = evaluator.evaluate(loader, mode="clean", max_batches=args.max_batches)
    lat_fp16   = model.measure_latency(batch_size=1, n_runs=30)

    fp16_entry = {
        "clean_avg_boxes":    clean_fp16["avg_boxes"],
        "attacked_avg_boxes": None,
        "vanishing_rate":     None,
        "latency_ms":         lat_fp16["mean_ms"],
        "device":             "GPU (CUDA FP16)",
    }
    if attack is not None:
        model.float()   # attack images stay FP32; model needs to accept them
        model.half()
        comp = evaluator.full_comparison(loader, attack, max_batches=args.max_batches)
        fp16_entry.update({
            "attacked_avg_boxes": comp["attacked_avg_boxes"],
            "vanishing_rate":     comp["vanishing_rate"],
        })
    results["FP16"] = fp16_entry

    # ----------------------------------------------------------------
    # INT8 (PyTorch dynamic quantization on backbone)
    # ----------------------------------------------------------------
    logger.info("Applying dynamic INT8 quantization to backbone...")
    model.float()   # reset to FP32 before quantization
    try:
        import torch
        from torch.quantization import quantize_dynamic
        model._model.backbone = quantize_dynamic(
            model._model.backbone,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Dynamic INT8 quantization applied to backbone.")
        clean_int8 = evaluator.evaluate(loader, mode="clean", max_batches=args.max_batches)

        int8_entry = {
            "clean_avg_boxes":    clean_int8["avg_boxes"],
            "attacked_avg_boxes": None,
            "vanishing_rate":     None,
            "latency_ms":         None,   # CPU dynamic quant latency not measured on GPU
            "device":             "GPU+CPU (backbone INT8 dynamic)",
            "note":               "Backbone dynamically quantized; neck/head remain FP32. "
                                  "Static INT8 requires mmdeploy.",
        }
        if attack is not None:
            comp = evaluator.full_comparison(loader, attack, max_batches=args.max_batches)
            int8_entry.update({
                "attacked_avg_boxes": comp["attacked_avg_boxes"],
                "vanishing_rate":     comp["vanishing_rate"],
            })
        results["INT8"] = int8_entry
    except Exception as e:
        logger.warning(f"INT8 dynamic quantization failed: {e}")
        results["INT8"] = {"note": f"Failed: {e}"}

    # ----------------------------------------------------------------
    # Save + print
    # ----------------------------------------------------------------
    out_path = save_results(results, str(out_dir), "det_ptq_results.json")
    logger.info(f"Results saved to {out_path}")

    print("\n" + "="*65)
    print(" Detection PTQ Results: RTMDet-Tiny")
    print("="*65)
    print(f"  {'':6s} {'Clean Boxes':>14} {'Attacked Boxes':>16} "
          f"{'Vanishing %':>13} {'Latency':>10}  Device")
    print(f"  {'-'*72}")
    for prec, m in results.items():
        if "note" in m and len(m) == 1:
            print(f"  {prec:<6s}  (skipped: {m['note'][:40]})")
            continue
        att_str = f"{m['attacked_avg_boxes']:.2f}" if m.get('attacked_avg_boxes') is not None else "N/A"
        vr_str  = f"{m['vanishing_rate']*100:.1f}%" if m.get('vanishing_rate') is not None else "N/A"
        lat_str = f"{m['latency_ms']:.2f}ms" if m.get('latency_ms') is not None else "N/A"
        print(
            f"  {prec:<6s} {m['clean_avg_boxes']:>14.2f} {att_str:>16} "
            f"{vr_str:>13} {lat_str:>10}  {m.get('device','')}"
        )

    print(f"\nNote: INT8 uses PyTorch dynamic quantization (backbone only).")
    print(f"      Static INT8 for full model requires mmdeploy.")
    print(f"\nOutputs: {out_dir}/")


if __name__ == "__main__":
    main()
