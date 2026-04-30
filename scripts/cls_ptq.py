"""
Phase D: INT8 PTQ for DeiT-Tiny + Full FP32 / FP16 / INT8 Comparison

Pipeline:
  1. Load DeiT-Tiny (FP32, GPU)
  2. Export to ONNX FP32
  3. Calibrate and quantize to INT8 (ONNX Runtime)
  4. Evaluate all three precisions:
       - FP32 (GPU PyTorch): clean + attacked
       - FP16 (GPU PyTorch): clean + attacked (patch transfer)
       - INT8 (CPU ORT):     clean + attacked (patch transfer)
  5. Save comparison JSON + robustness table PNG

Patch transfer note:
  The adversarial patch is optimized on FP32 and then applied to FP16/INT8
  models without re-optimization. This is intentional — it tests whether a
  patch designed for FP32 still transfers to quantized models.

Usage:
    # Smoke test (demo data, 2 batches):
    python scripts/cls_ptq.py --max_batches 2

    # With real data:
    python scripts/cls_ptq.py \
        dataset.data_type=tiny_imagenet \
        dataset.data_root=/data/tiny-imagenet-200/tiny-imagenet-200 \
        dataset.num_classes=200 \
        dataset.max_samples=1000

    # Skip export if ONNX already exists:
    python scripts/cls_ptq.py --skip_export

    # Also run attacked eval (requires a pre-generated patch):
    python scripts/cls_ptq.py --run_attack
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from models.cls import build_classifier, OrtClassifier
from datasets.imagenet_subset import ImageNetSubset
from eval.cls_evaluator import ClassificationEvaluator
from quant.onnx_export import export_to_onnx
from quant.int8_calibrate import calibrate_and_quantize
from utils.logger import get_logger, add_file_handler
from utils.io_utils import save_results, ensure_dir
from utils.visualize import plot_robustness_table

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="INT8 PTQ for DeiT-Tiny")
    parser.add_argument("--config", default="configs/cls/deit_tiny.yaml")
    parser.add_argument("--quant_config", default="configs/quant/cls_int8.yaml")
    parser.add_argument("--skip_export", action="store_true",
                        help="Skip ONNX export if FP32 ONNX already exists")
    parser.add_argument("--skip_quantize", action="store_true",
                        help="Skip INT8 quantization if INT8 ONNX already exists")
    parser.add_argument("--run_attack", action="store_true",
                        help="Also run attacked eval (needs outputs/cls/attacked/adv_patch.pt)")
    parser.add_argument("--patch_path", default="outputs/cls/attacked/adv_patch.pt",
                        help="Path to pre-generated adversarial patch")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit eval to N batches (for quick tests)")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf overrides, e.g. dataset.data_type=tiny_imagenet")
    return parser.parse_args()


def load_attack(patch_path: str, attack_cfg):
    """Load pre-generated adversarial patch into an AdvPatchAttack instance."""
    from attacks.cls import build_cls_attack
    attack = build_cls_attack(attack_cfg)
    attack.load_patch(patch_path)
    logger.info(f"Loaded adversarial patch from {patch_path}")
    return attack


def main():
    args = parse_args()

    # Load and merge configs
    cfg       = OmegaConf.load(args.config)
    quant_cfg = OmegaConf.load(args.quant_config)
    cfg = OmegaConf.merge(cfg, {"quant": quant_cfg.quant, "calibration": quant_cfg.calibration})
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    out_dir = Path(quant_cfg.output.dir)
    ensure_dir(str(out_dir))
    add_file_handler(logger, str(out_dir / "ptq.log"))

    fp32_onnx_path  = quant_cfg.output.fp32_onnx
    fp32_prepped    = quant_cfg.output.fp32_prepped
    int8_onnx_path  = quant_cfg.output.int8_onnx

    print("\n" + "="*65)
    print(" Phase D: INT8 PTQ — DeiT-Tiny")
    print("="*65)

    # ----------------------------------------------------------------
    # 1. Build PyTorch model + dataset
    # ----------------------------------------------------------------
    logger.info("Loading DeiT-Tiny (FP32)...")
    model = build_classifier(cfg.model)

    logger.info("Loading dataset...")
    dataset      = ImageNetSubset(cfg.dataset)
    eval_loader  = dataset.get_loader()

    # Calibration loader: small subset is enough (10 batches)
    calib_cfg    = OmegaConf.merge(cfg.dataset, {"max_samples": 640, "batch_size": 64})
    calib_loader = ImageNetSubset(calib_cfg).get_loader()

    # ----------------------------------------------------------------
    # 2. ONNX export
    # ----------------------------------------------------------------
    onnx_to_quantize = fp32_prepped
    if args.skip_export and Path(fp32_prepped).exists():
        logger.info(f"Skipping export, using existing: {fp32_prepped}")
    else:
        logger.info("Exporting DeiT-Tiny to ONNX FP32...")
        onnx_to_quantize = export_to_onnx(
            classifier=model,
            output_path=fp32_onnx_path,
            opset_version=cfg.quant.opset_version,
            run_shape_inference=True,
        )

    # ----------------------------------------------------------------
    # 3. INT8 quantization
    # ----------------------------------------------------------------
    if args.skip_quantize and Path(int8_onnx_path).exists():
        logger.info(f"Skipping quantization, using existing: {int8_onnx_path}")
    else:
        logger.info("Calibrating and quantizing to INT8...")
        calibrate_and_quantize(
            fp32_onnx_path=onnx_to_quantize,
            output_int8_path=int8_onnx_path,
            calibration_loader=calib_loader,
            input_name=cfg.calibration.input_name,
            max_calibration_batches=cfg.calibration.max_batches,
            per_channel=cfg.quant.per_channel,
            reduce_range=cfg.quant.reduce_range,
        )

    # ----------------------------------------------------------------
    # 4. Load adversarial patch (optional)
    # ----------------------------------------------------------------
    attack = None
    if args.run_attack:
        if not Path(args.patch_path).exists():
            logger.warning(
                f"Patch not found at {args.patch_path}. "
                f"Run cls_baseline.py --run_attack first to generate it."
            )
            args.run_attack = False
        else:
            attack = load_attack(args.patch_path, cfg.attack)

    # ----------------------------------------------------------------
    # 5. Evaluate FP32 (GPU PyTorch)
    # ----------------------------------------------------------------
    logger.info("="*40)
    logger.info("Evaluating FP32 (GPU, PyTorch)...")
    model.float()
    eval_fp32 = ClassificationEvaluator(model, device=str(cfg.model.device))

    clean_fp32 = eval_fp32.evaluate(eval_loader, mode="clean", max_batches=args.max_batches)
    lat_fp32   = model.measure_latency(batch_size=1, n_runs=50)

    attacked_fp32, asr_fp32 = {}, 0.0
    if args.run_attack:
        attacked_fp32 = eval_fp32.evaluate(eval_loader, mode="attacked",
                                            attack=attack, max_batches=args.max_batches)
        asr_fp32 = ClassificationEvaluator.compute_asr(clean_fp32, attacked_fp32)

    # ----------------------------------------------------------------
    # 6. Evaluate FP16 (GPU PyTorch)
    # ----------------------------------------------------------------
    logger.info("Evaluating FP16 (GPU, PyTorch)...")
    model.half()
    eval_fp16 = ClassificationEvaluator(model, device=str(cfg.model.device))

    clean_fp16 = eval_fp16.evaluate(eval_loader, mode="clean", max_batches=args.max_batches)
    lat_fp16   = model.measure_latency(batch_size=1, n_runs=50)

    attacked_fp16, asr_fp16 = {}, 0.0
    if args.run_attack:
        attacked_fp16 = eval_fp16.evaluate(eval_loader, mode="attacked",
                                            attack=attack, max_batches=args.max_batches)
        asr_fp16 = ClassificationEvaluator.compute_asr(clean_fp16, attacked_fp16)

    # ----------------------------------------------------------------
    # 7. Evaluate INT8 (CPU, ORT)
    # ----------------------------------------------------------------
    logger.info("Evaluating INT8 (CPU, ONNX Runtime)...")
    ort_cls  = OrtClassifier(int8_onnx_path)
    eval_int8 = ClassificationEvaluator(ort_cls, device="cpu")

    clean_int8 = eval_int8.evaluate(eval_loader, mode="clean", max_batches=args.max_batches)
    lat_int8   = ort_cls.measure_latency(batch_size=1, n_runs=50)

    attacked_int8, asr_int8 = {}, 0.0
    if args.run_attack:
        attacked_int8 = eval_int8.evaluate(eval_loader, mode="attacked",
                                            attack=attack, max_batches=args.max_batches)
        asr_int8 = ClassificationEvaluator.compute_asr(clean_int8, attacked_int8)

    # ----------------------------------------------------------------
    # 8. Build summary and save
    # ----------------------------------------------------------------
    summary = {
        "FP32": {
            "clean_top1_acc":    clean_fp32["top1_acc"],
            "clean_top5_acc":    clean_fp32["top5_acc"],
            "attacked_top1_acc": attacked_fp32.get("top1_acc", None),
            "asr":               asr_fp32,
            "avg_latency_ms":    lat_fp32["mean_ms"],
            "p99_latency_ms":    lat_fp32["p99_ms"],
            "device":            "GPU (CUDA)",
        },
        "FP16": {
            "clean_top1_acc":    clean_fp16["top1_acc"],
            "clean_top5_acc":    clean_fp16["top5_acc"],
            "attacked_top1_acc": attacked_fp16.get("top1_acc", None),
            "asr":               asr_fp16,
            "avg_latency_ms":    lat_fp16["mean_ms"],
            "p99_latency_ms":    lat_fp16["p99_ms"],
            "device":            "GPU (CUDA)",
        },
        "INT8": {
            "clean_top1_acc":    clean_int8["top1_acc"],
            "clean_top5_acc":    clean_int8["top5_acc"],
            "attacked_top1_acc": attacked_int8.get("top1_acc", None),
            "asr":               asr_int8,
            "avg_latency_ms":    lat_int8["mean_ms"],
            "p99_latency_ms":    lat_int8["p99_ms"],
            "device":            "CPU (ORT)",
        },
    }

    results_path = save_results(summary, str(out_dir), "ptq_results.json")
    logger.info(f"Results saved to {results_path}")

    # Robustness table PNG
    plot_robustness_table(summary, save_path=str(out_dir / "robustness_table.png"))

    # ----------------------------------------------------------------
    # 9. Print summary table
    # ----------------------------------------------------------------
    print("\n" + "="*65)
    print(" Precision Comparison: DeiT-Tiny")
    print("="*65)
    print(f"  {'':6s} {'Clean':>8} {'Attacked':>10} {'ASR':>8} {'Latency':>12}  {'Device'}")
    print(f"  {'-'*60}")
    for prec, m in summary.items():
        attacked_str = f"{m['attacked_top1_acc']*100:>9.2f}%" if m["attacked_top1_acc"] is not None else "       N/A"
        asr_str      = f"{m['asr']*100:>7.2f}%" if m["attacked_top1_acc"] is not None else "     N/A"
        print(
            f"  {prec:<6s} {m['clean_top1_acc']*100:>7.2f}% "
            f"{attacked_str} {asr_str} "
            f"{m['avg_latency_ms']:>10.2f}ms  {m['device']}"
        )

    print(f"\n[!] Note: INT8 latency is CPU-bound (ONNX Runtime, no CUDA provider).")
    print(f"    FP32/FP16 latency is on GPU. Not directly comparable.")
    if args.run_attack:
        print(f"\n[!] Patch was optimized on FP32; transferred (no re-opt) to FP16/INT8.")
        print(f"    Lower INT8 ASR = quantization-induced robustness change (valid result).")

    print(f"\nOutputs:")
    print(f"  ONNX FP32: {fp32_onnx_path}")
    print(f"  ONNX INT8: {int8_onnx_path}")
    print(f"  Results:   {results_path}")
    print(f"  Table PNG: {out_dir}/robustness_table.png")
    print()


if __name__ == "__main__":
    main()
