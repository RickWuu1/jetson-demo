"""
Phase C: Detection Baseline

Runs clean evaluation on RTMDet-Tiny and measures latency.
Optionally runs DPatch attack evaluation.

Usage:
    # Clean eval only (demo data, no download needed):
    python scripts/det_baseline.py

    # With config file:
    python scripts/det_baseline.py --config configs/det/rtmdet_tiny.yaml

    # Run attack too:
    python scripts/det_baseline.py --run_attack

    # Quick test (only 2 batches):
    python scripts/det_baseline.py --max_batches 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from models.det import build_detector
from datasets.coco_subset import CocoSubset
from eval.det_evaluator import DetectionEvaluator
from utils.logger import get_logger, add_file_handler
from utils.io_utils import save_results, ensure_dir

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Detection Baseline (Phase C)")
    parser.add_argument("--config", default="configs/det/rtmdet_tiny.yaml",
                        help="Path to config YAML")
    parser.add_argument("--run_attack", action="store_true",
                        help="Also run DPatch attack after clean eval")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit to N batches for quick testing")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf overrides, e.g. dataset.data_type=coco")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    print("\n" + "="*60)
    print(" Detection Baseline (Phase C)")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))

    out_dir = Path(cfg.eval.output_dir)
    ensure_dir(str(out_dir))
    add_file_handler(logger, str(out_dir / "det_baseline.log"))

    # ----------------------------------------------------------------
    # 1. Build model
    # ----------------------------------------------------------------
    logger.info("Building detector...")
    model = build_detector(cfg.model)

    # ----------------------------------------------------------------
    # 2. Load dataset
    # ----------------------------------------------------------------
    logger.info("Loading dataset...")
    dataset = CocoSubset(cfg.dataset)
    loader  = dataset.get_loader()
    logger.info(f"Dataset: {len(dataset)} samples, batch_size={cfg.dataset.batch_size}")

    # ----------------------------------------------------------------
    # 3. Clean evaluation
    # ----------------------------------------------------------------
    logger.info("Running clean evaluation...")
    evaluator = DetectionEvaluator(
        model,
        device=cfg.model.device,
        score_thr=cfg.eval.score_thr,
        iou_thr=cfg.eval.iou_thr,
    )
    clean_res = evaluator.evaluate(loader, mode="clean", max_batches=args.max_batches)

    # Latency benchmark
    logger.info("Measuring latency...")
    latency = model.measure_latency(batch_size=1, n_runs=50)
    clean_res.update({"latency": latency})

    save_results(clean_res, str(out_dir), "clean.json")
    logger.info(f"Clean results saved to {out_dir}/clean.json")

    print("\n--- Clean Eval ---")
    print(f"  Avg boxes/image: {clean_res['avg_boxes']:.2f}")
    print(f"  Latency (bs=1):  {latency['mean_ms']:.2f}ms (p99={latency['p99_ms']:.2f}ms)")
    print(f"  Images:          {clean_res['total_images']}")

    # ----------------------------------------------------------------
    # 4. Optional: DPatch attack
    # ----------------------------------------------------------------
    if args.run_attack:
        logger.info("Running DPatch attack...")
        from attacks.det import build_det_attack
        from omegaconf import OmegaConf as OC

        attack_dir = Path(cfg.attack.output_dir)
        ensure_dir(str(attack_dir))

        attack = build_det_attack(cfg.attack)

        # Use a small subset for patch generation
        train_cfg     = OC.merge(cfg.dataset, {"max_samples": 50, "batch_size": 4})
        train_dataset = CocoSubset(train_cfg)
        train_loader  = train_dataset.get_loader()

        logger.info("Generating adversarial patch...")
        patch = attack.generate_patch(model, train_loader)

        patch_path = str(attack_dir / "dpatch.pt")
        attack.save_patch(patch_path)
        logger.info(f"Patch saved to {patch_path}")

        # Full comparison
        comparison = evaluator.full_comparison(loader, attack, max_batches=args.max_batches)
        comparison["patch_path"] = patch_path
        save_results(comparison, str(attack_dir), "attacked.json")

        print("\n--- DPatch Attack Results ---")
        print(f"  Clean avg boxes:    {comparison['clean_avg_boxes']:.2f}")
        print(f"  Attacked avg boxes: {comparison['attacked_avg_boxes']:.2f}")
        print(f"  Vanishing rate:     {comparison['vanishing_rate']*100:.1f}%")
        print(f"  Results saved to:   {attack_dir}/attacked.json")

    print(f"\nDone! Results in {out_dir}/")


if __name__ == "__main__":
    main()
