"""
INT8 Post-Training Quantization via ONNX Runtime.

Uses static quantization with MinMax calibration.
QDQ format (QuantizeLinear / DequantizeLinear nodes inserted into the graph).

Key design: DataLoaderCalibrationReader eagerly pre-loads all calibration
batches into a list in __init__. This is intentional — ORT's MinMax calibration
may call get_next() multiple times (multi-pass), so a one-shot iterator would
be exhausted on the first pass.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoaderCalibrationReader(CalibrationDataReader):
    """
    Bridges a PyTorch DataLoader to the ORT CalibrationDataReader protocol.

    Eagerly materializes up to max_batches batches in __init__ so the reader
    can be rewound for multi-pass calibration.

    Args:
        loader      : DataLoader yielding (images_tensor, labels_tensor)
        input_name  : ONNX model's input node name (default: "input")
        max_batches : cap on number of batches to use (None = all)
    """

    def __init__(self, loader, input_name: str = "input", max_batches: Optional[int] = None):
        self._input_name = input_name
        self._batches: List[np.ndarray] = []
        self._idx = 0

        logger.info("Pre-loading calibration data...")
        for i, (images, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            # ORT always expects float32 numpy input
            self._batches.append(images.float().cpu().numpy())

        total_images = sum(b.shape[0] for b in self._batches)
        logger.info(f"Calibration data: {len(self._batches)} batches, {total_images} images")

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Called by ORT calibrator. Returns None to signal end of data."""
        if self._idx >= len(self._batches):
            return None
        batch = {self._input_name: self._batches[self._idx]}
        self._idx += 1
        return batch

    def rewind(self):
        """Reset iterator for multi-pass calibration."""
        self._idx = 0


def calibrate_and_quantize(
    fp32_onnx_path: str,
    output_int8_path: str,
    calibration_loader,
    input_name: str = "input",
    max_calibration_batches: int = 10,
    per_channel: bool = False,
    reduce_range: bool = False,
) -> str:
    """
    Run static INT8 quantization with QDQ format.

    Args:
        fp32_onnx_path         : path to FP32 (preferably quant_pre_processed) ONNX
        output_int8_path       : where to save the INT8 ONNX
        calibration_loader     : DataLoader for calibration images
        input_name             : ONNX input node name
        max_calibration_batches: how many batches to use for calibration
        per_channel            : per-channel weight quantization (more accurate, slower)
        reduce_range           : use 7-bit instead of 8-bit (for some legacy hardware)

    Returns:
        output_int8_path
    """
    Path(output_int8_path).parent.mkdir(parents=True, exist_ok=True)

    reader = DataLoaderCalibrationReader(
        calibration_loader,
        input_name=input_name,
        max_batches=max_calibration_batches,
    )

    logger.info(f"Quantizing {fp32_onnx_path} → {output_int8_path}")
    logger.info(f"  per_channel={per_channel}, reduce_range={reduce_range}")

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=output_int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=CalibrationMethod.MinMax,
    )

    size_mb = Path(output_int8_path).stat().st_size / 1024 / 1024
    logger.info(f"INT8 model saved. Size: {size_mb:.1f} MB")
    return output_int8_path
