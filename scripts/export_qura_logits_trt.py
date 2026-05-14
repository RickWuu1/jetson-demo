"""Export ViT logits to ONNX and optionally TensorRT.

The exported graph is classification only:
  - output is logits
  - attention maps are not exported
  - PatchDrop / defense logic is not exported

Use the exported engine with:
  scripts/compare_trt_backend.py
  scripts/camera_web_preview.py --backend trt --trt-engine ...

Example:
  PYTHONPATH=.:third_party/qura python3 scripts/export_qura_logits_trt.py \
    --model-kind fp32 \
    --onnx outputs/trt/fp32_vit_logits.onnx \
    --engine outputs/trt/fp32_vit_logits_fp16.engine \
    --build-engine \
    --precision fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/qura"))

import torch

from demos import demo_qura_realtime_full as realtime
from deploy.trt_export import export_onnx_to_trt
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    default_quant = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
    parser = argparse.ArgumentParser(description="Export ViT logits-only ONNX / TensorRT engine.")
    parser.add_argument("--model-kind", default="qura", choices=["qura", "fp32"], help="Model source to export.")
    parser.add_argument("--onnx", default="outputs/trt/qura_logits.onnx", help="Output ONNX path.")
    parser.add_argument("--engine", default="outputs/trt/qura_logits_fp16.engine", help="Output TensorRT engine path.")
    parser.add_argument("--quant-model", default=str(default_quant))
    parser.add_argument("--quant-config", default="third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml")
    parser.add_argument("--fp32-weights", default="/home/jetson-nano/demo/pytorch_model.bin", help="FP32 ViT state_dict path for --model-kind fp32.")
    parser.add_argument("--pretrained", action="store_true", help="Use timm pretrained weights for --model-kind fp32 instead of --fp32-weights.")
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--export-device", default="cpu", choices=["cpu", "cuda"], help="Device used during ONNX export.")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX export and reuse --onnx.")
    parser.add_argument("--build-engine", action="store_true", help="Build TensorRT engine after ONNX export.")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"], help="TensorRT engine precision.")
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--workspace", type=float, default=2.0, help="TensorRT workspace size in GB.")
    parser.add_argument("--calib-data", default=None, help="Optional .npy calibration tensor [N,C,H,W] for TensorRT INT8.")
    parser.add_argument("--calib-cache", default=None, help="Optional TensorRT calibration cache path.")
    return parser.parse_args()


def export_qura_logits_onnx(
    quant_model: str,
    quant_config: str,
    output_path: str,
    device: torch.device,
    bd_target: int,
    opset: int,
    image_size: int,
) -> str:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--export-device cuda requested but CUDA is not available.")

    logger.info("Loading QURA torch backend for logits-only ONNX export...")
    backbone = realtime.load_qura_backbone(
        quant_model,
        quant_config,
        device,
        bd_target=bd_target,
    )
    if backbone is None:
        raise RuntimeError("Failed to load QURA backend. Check quant model, config, timm, OmegaConf, and MQBench.")

    model = backbone.model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting logits-only ONNX: %s", output)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            opset_version=opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    try:
        import onnx

        onnx_model = onnx.load(str(output))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX check passed.")
    except ImportError:
        logger.warning("onnx package not installed; skipped ONNX checker.")

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("Saved ONNX: %s (%.1f MB)", output, size_mb)
    return str(output)


def export_fp32_logits_onnx(
    weights_path: str,
    pretrained: bool,
    output_path: str,
    device: torch.device,
    opset: int,
    image_size: int,
) -> str:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--export-device cuda requested but CUDA is not available.")
    if realtime.timm is None:
        raise ImportError("timm is required for --model-kind fp32 export.")

    logger.info("Loading FP32 ViT-B/16 for logits-only ONNX export...")
    model = realtime.timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=1000,
    )
    if not pretrained:
        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"FP32 weights not found: {weights}")
        state_dict = torch.load(str(weights), map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device).eval()

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting FP32 logits-only ONNX: %s", output)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            opset_version=opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    try:
        import onnx

        onnx_model = onnx.load(str(output))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX check passed.")
    except ImportError:
        logger.warning("onnx package not installed; skipped ONNX checker.")

    size_mb = output.stat().st_size / 1024 / 1024
    logger.info("Saved ONNX: %s (%.1f MB)", output, size_mb)
    return str(output)


def load_calibration_batches(path: Optional[str], batch_size: int):
    if not path:
        return None
    import numpy as np

    data = np.load(path)
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx

    if not args.skip_onnx:
        if args.model_kind == "qura":
            onnx_path = export_qura_logits_onnx(
                quant_model=args.quant_model,
                quant_config=args.quant_config,
                output_path=args.onnx,
                device=torch.device(args.export_device),
                bd_target=args.bd_target,
                opset=args.opset,
                image_size=args.image_size,
            )
        else:
            onnx_path = export_fp32_logits_onnx(
                weights_path=args.fp32_weights,
                pretrained=args.pretrained,
                output_path=args.onnx,
                device=torch.device(args.export_device),
                opset=args.opset,
                image_size=args.image_size,
            )
    elif not Path(onnx_path).exists():
        raise FileNotFoundError(f"--skip-onnx requested but ONNX file does not exist: {onnx_path}")

    if args.build_engine:
        calib_batches = None
        if args.precision == "int8":
            calib_batches = load_calibration_batches(args.calib_data, args.max_batch)
        export_onnx_to_trt(
            onnx_path=onnx_path,
            output_path=args.engine,
            precision=args.precision,
            max_batch_size=args.max_batch,
            workspace_gb=args.workspace,
            calibration_batches=calib_batches,
            calibration_cache=args.calib_cache,
        )

    print("\nExport complete")
    print(f"  ONNX  : {onnx_path}")
    if args.build_engine:
        print(f"  Engine: {args.engine}")


if __name__ == "__main__":
    main()
