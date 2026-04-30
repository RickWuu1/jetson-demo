"""
Phase B: Classification Baseline

Runs clean evaluation on DeiT-Tiny (FP32) and measures latency.
Optionally runs attacked evaluation with adversarial patch.

Usage:
    # Clean eval only (demo data, no download needed):
    python scripts/cls_baseline.py

    # With config file:
    python scripts/cls_baseline.py --config configs/cls/deit_tiny.yaml

    # With real ImageNet:
    python scripts/cls_baseline.py --config configs/cls/deit_tiny.yaml \
        dataset.data_type=imagenet dataset.data_root=/path/to/imagenet

    # Run attack too:
    python scripts/cls_baseline.py --run_attack

    # Quick test (only 2 batches):
    python scripts/cls_baseline.py --max_batches 2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from models.cls import build_classifier
from datasets.imagenet_subset import ImageNetSubset
from eval.cls_evaluator import ClassificationEvaluator
from utils.logger import get_logger, add_file_handler
from utils.io_utils import save_results, ensure_dir
from utils.visualize import save_patch, plot_clean_vs_attacked

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Classification Baseline")
    parser.add_argument("--config", default="configs/cls/deit_tiny.yaml",
                        help="Path to config YAML")
    parser.add_argument("--run_attack", action="store_true",
                        help="Also run adversarial patch attack after clean eval")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit to N batches for quick testing")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf overrides, e.g. dataset.data_type=imagenet")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    print("\n" + "="*60)
    print(" Classification Baseline (Phase B)")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))

    # Setup output dir + file logger
    out_dir = Path(cfg.eval.output_dir)
    ensure_dir(str(out_dir))
    add_file_handler(logger, str(out_dir / "baseline.log"))

    # ----------------------------------------------------------------
    # 1. Build model
    # ----------------------------------------------------------------
    logger.info("Building model...")
    model = build_classifier(cfg.model)

    # ----------------------------------------------------------------
    # 2. Load dataset
    # ----------------------------------------------------------------
    logger.info("Loading dataset...")
    dataset = ImageNetSubset(cfg.dataset)
    loader  = dataset.get_loader()
    logger.info(f"Dataset: {len(dataset)} samples, batch_size={cfg.dataset.batch_size}")

    # ----------------------------------------------------------------
    # 3. Clean evaluation
    # ----------------------------------------------------------------
    logger.info("Running clean evaluation (FP32)...")
    evaluator  = ClassificationEvaluator(model, device=cfg.model.device)
    clean_res  = evaluator.evaluate(loader, mode="clean", max_batches=args.max_batches)

    # Latency benchmark
    logger.info("Measuring latency (FP32)...")
    latency = model.measure_latency(batch_size=1, n_runs=50)
    clean_res.update({"latency_fp32": latency})

    # Save clean results
    save_results(clean_res, str(out_dir), "clean_fp32.json")
    logger.info(f"Clean results saved to {out_dir}/clean_fp32.json")

    # Print summary
    print("\n--- Clean Eval (FP32) ---")
    print(f"  Top-1 Acc:    {clean_res['top1_acc']*100:.2f}%")
    print(f"  Top-5 Acc:    {clean_res['top5_acc']*100:.2f}%")
    print(f"  Latency (bs=1): {latency['mean_ms']:.2f}ms (p99={latency['p99_ms']:.2f}ms)")
    print(f"  Samples:      {clean_res['total_samples']}")

    # ----------------------------------------------------------------
    # 4. Optional: adversarial patch attack
    # ----------------------------------------------------------------
    if args.run_attack:
        logger.info("Running adversarial patch attack...")
        from attacks.cls import build_cls_attack

        attack_cfg  = cfg.attack
        attack_dir  = Path(cfg.attack.output_dir)
        ensure_dir(str(attack_dir))

        attack = build_cls_attack(attack_cfg)

        # Use a smaller subset for patch generation (faster)
        from omegaconf import OmegaConf as OC
        train_cfg = OC.merge(cfg.dataset, {"max_samples": 200, "batch_size": 16})
        train_dataset = ImageNetSubset(train_cfg)
        train_loader  = train_dataset.get_loader()

        logger.info("Generating adversarial patch...")
        patch = attack.generate_patch(model, train_loader)

        # Save patch
        patch_path = str(attack_dir / "adv_patch.pt")
        attack.save_patch(patch_path)
        from utils.visualize import save_patch as save_p
        save_p(patch, str(attack_dir / "adv_patch.png"))
        logger.info(f"Patch saved to {attack_dir}/adv_patch.pt and .png")

        # Attacked eval
        attacked_res = evaluator.evaluate(loader, mode="attacked", attack=attack,
                                          max_batches=args.max_batches)
        asr = ClassificationEvaluator.compute_asr(clean_res, attacked_res)

        # Save attacked results
        attacked_res["asr"] = asr
        attacked_res["patch_path"] = patch_path
        save_results(attacked_res, str(attack_dir), "attacked_fp32.json")

        print("\n--- Attacked Eval (FP32, Adv Patch) ---")
        print(f"  Top-1 Acc (attacked): {attacked_res['top1_acc']*100:.2f}%")
        print(f"  Top-5 Acc (attacked): {attacked_res['top5_acc']*100:.2f}%")
        print(f"  ASR:                  {asr*100:.2f}%")
        print(f"  Patch size:           {attack.patch_size_px[0]}x{attack.patch_size_px[1]} px")
        print(f"  Results saved to:     {attack_dir}/attacked_fp32.json")

    # ----------------------------------------------------------------
    # 5. FP16 eval (optional quick check)
    # ----------------------------------------------------------------
    logger.info("Running FP16 quick check...")
    model.half()
    clean_res_fp16 = evaluator.evaluate(loader, mode="clean", max_batches=args.max_batches)
    latency_fp16   = model.measure_latency(batch_size=1, n_runs=50)
    clean_res_fp16["latency_fp16"] = latency_fp16
    save_results(clean_res_fp16, str(out_dir), "clean_fp16.json")

    print("\n--- Clean Eval (FP16) ---")
    print(f"  Top-1 Acc:    {clean_res_fp16['top1_acc']*100:.2f}%")
    print(f"  Latency (bs=1): {latency_fp16['mean_ms']:.2f}ms")

    print("\n--- Precision Comparison ---")
    print(f"  {'':10s} {'Top1':>8} {'Top5':>8} {'Latency':>12}")
    print(f"  {'FP32':10s} {clean_res['top1_acc']*100:>7.2f}% "
          f"{clean_res['top5_acc']*100:>7.2f}% "
          f"{clean_res['avg_latency_ms']:>10.2f}ms")
    print(f"  {'FP16':10s} {clean_res_fp16['top1_acc']*100:>7.2f}% "
          f"{clean_res_fp16['top5_acc']*100:>7.2f}% "
          f"{clean_res_fp16['avg_latency_ms']:>10.2f}ms")

    print(f"\nDone! Results in {out_dir}/")


if __name__ == "__main__":
    main()
