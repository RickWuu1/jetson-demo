"""
ONNX export for DeiTClassifier.

Design:
- Export on CPU to avoid CUDA SDPA operator issues
- opset 16: covers DeiT attention ops cleanly with torch 2.2.x
- quant_pre_process: adds shape info needed for ORT static quantization
- Dynamic batch axis: allows any batch size at inference
"""

from pathlib import Path
from typing import Optional

import torch
import onnx
from onnxruntime.quantization import quant_pre_process

from utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    classifier,
    output_path: str,
    opset_version: int = 16,
    image_size: int = 224,
    input_name: str = "input",
    output_name: str = "logits",
    run_shape_inference: bool = True,
) -> str:
    """
    Export DeiTClassifier to ONNX FP32.

    Steps:
      1. Move model to CPU (avoids CUDA SDPA export issues)
      2. torch.onnx.export with dynamic batch dimension
      3. onnx.checker.check_model() to verify the graph
      4. quant_pre_process() to add shape info (needed for quantize_static)

    Args:
        classifier        : DeiTClassifier instance
        output_path       : where to save the FP32 ONNX file
        opset_version     : ONNX opset (16 works with torch 2.2.x + timm 0.9.x)
        image_size        : input spatial size (default 224)
        input_name        : ONNX input node name
        output_name       : ONNX output node name
        run_shape_inference: run quant_pre_process after export (recommended for PTQ)

    Returns:
        Path to the (possibly preprocessed) ONNX file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Temporarily move to CPU ---
    original_device = classifier._device
    original_dtype  = classifier._dtype
    classifier.float()           # ensure FP32
    classifier.to("cpu")
    classifier.model.eval()

    dummy = torch.randn(1, 3, image_size, image_size, device="cpu")

    # --- 2. Export ---
    logger.info(f"Exporting to ONNX (opset={opset_version}) → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            classifier.model,
            dummy,
            output_path,
            opset_version=opset_version,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={
                input_name:  {0: "batch_size"},
                output_name: {0: "batch_size"},
            },
            do_constant_folding=True,
        )

    # --- 3. Verify ---
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    model_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f"ONNX model verified. Size: {model_size_mb:.1f} MB")

    # --- 4. quant_pre_process: adds shape inference for PTQ calibration ---
    prepped_path = output_path
    if run_shape_inference:
        prepped_path = output_path.replace(".onnx", "_prepped.onnx")
        logger.info(f"Running quant_pre_process → {prepped_path}")
        quant_pre_process(output_path, prepped_path)

    # --- Restore original device / dtype ---
    classifier.to(str(original_device))
    if original_dtype == torch.float16:
        classifier.half()

    logger.info("ONNX export complete.")
    return prepped_path
