"""
Detection Attack: DPatch on RTMDet-Tiny (FP32 / FP16)

Standalone script to:
  1. Load RTMDet-Tiny
  2. Optimize DPatch on FP32
  3. Evaluate clean vs attacked on FP32
  4. Evaluate clean vs attacked on FP16 (patch transfer, no re-optimization)
  5. Save patch + results

Usage:
    # Quick test (demo data):
    python scripts/det_attack.py --max_batches 5

    # COCO real data:
    python scripts/det_attack.py --config configs/det/rtmdet_coco.yaml

    # Load pre-saved patch (skip optimization):
    python scripts/det_attack.py --load_patch outputs/det/coco_attacked/dpatch.pt
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


def parse_args():
    parser = argparse.ArgumentParser(description="DPatch Attack on RTMDet")
    parser.add_argument("--config", default="configs/det/rtmdet_tiny.yaml")
    parser.add_argument("--load_patch", default=None,
                        help="Skip optimization, load existing .pt patch")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    out_dir = Path(cfg.attack.output_dir)
    ensure_dir(str(out_dir))
    add_file_handler(logger, str(out_dir / "det_attack.log"))

    print("\n" + "="*60)
    print(" DPatch Attack (RTMDet-Tiny)")
    print("="*60)

    # ----------------------------------------------------------------
    # Build model + datasets
    # ----------------------------------------------------------------
    model = build_detector(cfg.model)

    eval_dataset = CocoSubset(cfg.dataset)
    eval_loader  = eval_dataset.get_loader()

    train_cfg    = OmegaConf.merge(cfg.dataset, {"max_samples": 50, "batch_size": 4})
    train_loader = CocoSubset(train_cfg).get_loader()

    # ----------------------------------------------------------------
    # Build / load attack
    # ----------------------------------------------------------------
    attack = build_det_attack(cfg.attack)

    if args.load_patch:
        logger.info(f"Loading DPatch from {args.load_patch}")
        attack.load_patch(args.load_patch)
    else:
        logger.info("Optimizing DPatch on FP32 model...")
        patch = attack.generate_patch(model, train_loader)
        patch_path = str(out_dir / "dpatch.pt")
        attack.save_patch(patch_path)
        logger.info(f"DPatch saved to {patch_path}")

    evaluator = DetectionEvaluator(
        model,
        device=cfg.model.device,
        score_thr=cfg.eval.score_thr,
        iou_thr=cfg.eval.iou_thr,
    )

    summary = {}

    # ----------------------------------------------------------------
    # FP32
    # ----------------------------------------------------------------
    logger.info("Evaluating FP32 clean + attacked...")
    model.float()
    comp_fp32 = evaluator.full_comparison(eval_loader, attack, max_batches=args.max_batches)
    summary["FP32"] = {
        "clean_avg_boxes":    comp_fp32["clean_avg_boxes"],
        "attacked_avg_boxes": comp_fp32["attacked_avg_boxes"],
        "vanishing_rate":     comp_fp32["vanishing_rate"],
    }

    # ----------------------------------------------------------------
    # FP16 (patch transfer: same patch, no re-optimization)
    # ----------------------------------------------------------------
    logger.info("Evaluating FP16 (patch transfer, no re-optimization)...")
    model.half()
    comp_fp16 = evaluator.full_comparison(eval_loader, attack, max_batches=args.max_batches)
    summary["FP16"] = {
        "clean_avg_boxes":    comp_fp16["clean_avg_boxes"],
        "attacked_avg_boxes": comp_fp16["attacked_avg_boxes"],
        "vanishing_rate":     comp_fp16["vanishing_rate"],
    }

    # ----------------------------------------------------------------
    # Save + print
    # ----------------------------------------------------------------
    save_results(summary, str(out_dir), "det_attack_summary.json")

    print("\n--- DPatch Attack Summary ---")
    print(f"  {'':6s} {'Clean Boxes':>14} {'Attacked Boxes':>16} {'Vanishing %':>13}")
    print(f"  {'-'*55}")
    for prec, m in summary.items():
        vr = m["vanishing_rate"] * 100 if m["vanishing_rate"] is not None else 0
        print(
            f"  {prec:<6s} {m['clean_avg_boxes']:>14.2f} "
            f"{m['attacked_avg_boxes']:>16.2f} {vr:>12.1f}%"
        )

    print(f"\n[!] Patch optimized on FP32; transferred (no re-opt) to FP16.")
    print(f"    Lower FP16 vanishing rate = precision change affects transferability.")
    print(f"\nOutputs: {out_dir}/")


if __name__ == "__main__":
    main()
