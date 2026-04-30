"""
Phase E: Adversarial Patch Attack on Classification Model

Standalone script to:
1. Load a pre-trained DeiT-Tiny
2. Optimize an adversarial patch
3. Evaluate clean vs attacked on FP32/FP16
4. Save patch + results

Usage:
    # Quick test with demo data:
    python scripts/cls_attack.py --max_batches 2

    # With real data:
    python scripts/cls_attack.py \
        --config configs/cls/deit_tiny.yaml \
        dataset.data_type=tiny_imagenet \
        dataset.data_root=/data/tiny-imagenet-200 \
        attack.steps=2000

    # Load pre-saved patch (skip optimization):
    python scripts/cls_attack.py --load_patch outputs/cls/attacked/adv_patch.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from models.cls import build_classifier
from datasets.imagenet_subset import ImageNetSubset
from attacks.cls import build_cls_attack
from eval.cls_evaluator import ClassificationEvaluator
from utils.logger import get_logger, add_file_handler
from utils.io_utils import save_results, ensure_dir
from utils.visualize import save_patch, plot_clean_vs_attacked

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cls/deit_tiny.yaml")
    parser.add_argument("--load_patch", default=None, help="Skip optimization, load existing patch")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    print("\n" + "="*60)
    print(" Adversarial Patch Attack (Phase E)")
    print("="*60)

    out_dir = Path(cfg.attack.output_dir)
    ensure_dir(str(out_dir))
    add_file_handler(logger, str(out_dir / "attack.log"))

    # Build model
    model     = build_classifier(cfg.model)
    evaluator = ClassificationEvaluator(model, device=cfg.model.device)

    # Build datasets
    eval_dataset  = ImageNetSubset(cfg.dataset)
    eval_loader   = eval_dataset.get_loader()

    train_cfg     = OmegaConf.merge(cfg.dataset, {"max_samples": 200, "batch_size": 16})
    train_loader  = ImageNetSubset(train_cfg).get_loader()

    # Build attack
    attack = build_cls_attack(cfg.attack)

    # Load or generate patch
    if args.load_patch:
        logger.info(f"Loading patch from {args.load_patch}")
        attack.load_patch(args.load_patch)
    else:
        logger.info("Generating adversarial patch (FP32 model)...")
        patch = attack.generate_patch(model, train_loader)
        save_patch(patch, str(out_dir / "adv_patch.png"))
        attack.save_patch(str(out_dir / "adv_patch.pt"))
        logger.info(f"Patch saved to {out_dir}/adv_patch.pt")

    # ----------------------------------------------------------------
    # Evaluate FP32
    # ----------------------------------------------------------------
    model.float()
    logger.info("Evaluating FP32...")
    clean_fp32   = evaluator.evaluate(eval_loader, mode="clean", max_batches=args.max_batches)
    attacked_fp32 = evaluator.evaluate(eval_loader, mode="attacked", attack=attack,
                                        max_batches=args.max_batches)
    asr_fp32 = ClassificationEvaluator.compute_asr(clean_fp32, attacked_fp32)

    # ----------------------------------------------------------------
    # Evaluate FP16 (transfer patch: no re-optimization)
    # ----------------------------------------------------------------
    model.half()
    logger.info("Evaluating FP16 (patch transfer, no re-optimization)...")
    clean_fp16   = evaluator.evaluate(eval_loader, mode="clean", max_batches=args.max_batches)
    attacked_fp16 = evaluator.evaluate(eval_loader, mode="attacked", attack=attack,
                                        max_batches=args.max_batches)
    asr_fp16 = ClassificationEvaluator.compute_asr(clean_fp16, attacked_fp16)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    summary = {
        "FP32": {
            "clean_top1_acc":    clean_fp32["top1_acc"],
            "attacked_top1_acc": attacked_fp32["top1_acc"],
            "asr":               asr_fp32,
            "avg_latency_ms":    clean_fp32["avg_latency_ms"],
        },
        "FP16": {
            "clean_top1_acc":    clean_fp16["top1_acc"],
            "attacked_top1_acc": attacked_fp16["top1_acc"],
            "asr":               asr_fp16,
            "avg_latency_ms":    clean_fp16["avg_latency_ms"],
        },
    }
    save_results(summary, str(out_dir), "attack_summary.json")

    # Visualize table
    from utils.visualize import plot_robustness_table
    plot_robustness_table(summary, save_path=str(out_dir / "robustness_table.png"))

    print("\n--- Attack Summary ---")
    print(f"  {'Precision':<8} {'Clean':>8} {'Attacked':>10} {'ASR':>8} {'Latency':>12}")
    print(f"  {'-'*50}")
    for prec, m in summary.items():
        print(f"  {prec:<8} {m['clean_top1_acc']*100:>7.2f}% "
              f"{m['attacked_top1_acc']*100:>9.2f}% "
              f"{m['asr']*100:>7.2f}% "
              f"{m['avg_latency_ms']:>10.2f}ms")

    print(f"\nResults saved to: {out_dir}/")


if __name__ == "__main__":
    main()
