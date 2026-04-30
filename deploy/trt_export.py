"""
TensorRT Engine Export

Converts an ONNX model to a TensorRT engine (.engine file) for
maximum GPU inference performance.

Requirements:
    pip install tensorrt  (or install via NVIDIA TensorRT package)
    CUDA + cuDNN installed and matching TRT version

Typical workflow:
    1. Export FP32 ONNX:   scripts/cls_ptq.py --skip_quantize
    2. Export TRT engine:  python deploy/trt_export.py \\
                               --onnx outputs/cls/quant/deit_fp32.onnx \\
                               --output outputs/cls/deploy/deit_fp32.engine
    3. Run inference:      python deploy/trt_runner.py \\
                               --engine outputs/cls/deploy/deit_fp32.engine

Precision modes (--precision):
    fp32  : FP32 engine (largest, most accurate)
    fp16  : FP16 engine (~2× faster, minimal accuracy loss)
    int8  : INT8 engine (~4× faster, requires calibration data)

Note on INT8 engine:
    INT8 TRT engine uses post-training calibration via IInt8Calibrator.
    We implement a simple MinMax calibrator that mirrors the ORT approach.
"""

import sys
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Try importing TensorRT — graceful failure if not installed
# ---------------------------------------------------------------------------
try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    _TRT_AVAILABLE = False
    TRT_LOGGER = None


def check_trt():
    if not _TRT_AVAILABLE:
        raise ImportError(
            "TensorRT not found. Install with:\n"
            "  pip install tensorrt\n"
            "or follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
        )


# ---------------------------------------------------------------------------
# Calibrator for INT8
# ---------------------------------------------------------------------------

def _make_calibrator_class():
    """Create calibrator class inheriting from the correct TRT base class."""
    check_trt()
    import tensorrt as trt

    class _TorchCalibrator(trt.IInt8MinMaxCalibrator):
        """
        MinMax calibrator for TRT INT8.
        Uses PyTorch CUDA tensors instead of pycuda.
        Compatible with TensorRT 8.x and 10.x.
        """

        def __init__(self, calibration_batches, input_name="input", cache_path=""):
            super().__init__()
            import torch
            import numpy as np
            self._batches    = calibration_batches
            self._idx        = 0
            self._input_name = input_name
            self._cache_path = cache_path
            # Pre-allocate GPU buffer using PyTorch
            sample = calibration_batches[0].astype(np.float32)
            self._device_buf = torch.from_numpy(sample).cuda().contiguous()

        def get_batch_size(self):
            return self._batches[0].shape[0]

        def get_batch(self, names):
            if self._idx >= len(self._batches):
                return None
            import torch
            import numpy as np
            batch = self._batches[self._idx].astype(np.float32)
            self._device_buf.copy_(torch.from_numpy(batch).cuda())
            self._idx += 1
            return [int(self._device_buf.data_ptr())]

        def read_calibration_cache(self):
            if self._cache_path and Path(self._cache_path).exists():
                return Path(self._cache_path).read_bytes()
            return None

        def write_calibration_cache(self, cache):
            if self._cache_path:
                Path(self._cache_path).write_bytes(cache)

    return _TorchCalibrator


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_onnx_to_trt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    workspace_gb: float = 2.0,
    calibration_batches=None,
    calibration_cache: Optional[str] = None,
) -> str:
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path           : path to input ONNX model
        output_path         : path to save .engine file
        precision           : 'fp32', 'fp16', or 'int8'
        max_batch_size      : maximum batch size for the engine
        workspace_gb        : TRT workspace size in GB
        calibration_batches : list of np.ndarray for INT8 calibration
        calibration_cache   : path to save/load calibration cache

    Returns:
        output_path
    """
    check_trt()
    import tensorrt as trt

    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building TRT engine: {onnx_path} → {output_path}")
    logger.info(f"  precision={precision}, max_batch={max_batch_size}, ws={workspace_gb}GB")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * 1024 ** 3),
    )

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("FP16 not natively supported on this GPU; building FP32 instead.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("INT8 not natively supported on this GPU; falling back to FP16.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for layers that don't quantize well
            if calibration_batches is None:
                logger.warning("No calibration data for INT8; accuracy may be poor.")
            else:
                CalibratorClass = _make_calibrator_class()
                calibrator = CalibratorClass(
                    calibration_batches,
                    cache_path=calibration_cache or "",
                )
                config.int8_calibrator = calibrator

    # Dynamic shape profile (optional: supports batch sizes 1..max_batch_size)
    profile = builder.create_optimization_profile()
    in_node = network.get_input(0)
    in_shape = in_node.shape               # e.g. [-1, 3, 224, 224]
    _, C, H, W = in_shape
    C = C if C > 0 else 3
    H = H if H > 0 else 224
    W = W if W > 0 else 224
    profile.set_shape(
        in_node.name,
        (1, C, H, W),                     # min
        (max(1, max_batch_size // 2), C, H, W),  # opt
        (max_batch_size, C, H, W),        # max
    )
    config.add_optimization_profile(profile)

    logger.info("Compiling TRT engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed.")

    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f"TRT engine saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Export ONNX → TensorRT engine")
    p.add_argument("--onnx",      required=True,  help="Input ONNX file")
    p.add_argument("--output",    required=True,  help="Output .engine file")
    p.add_argument("--precision", default="fp16",
                   choices=["fp32", "fp16", "int8"], help="Engine precision")
    p.add_argument("--max_batch", type=int, default=1)
    p.add_argument("--workspace", type=float, default=2.0,
                   help="Workspace size in GB")
    p.add_argument("--calib_data", default=None,
                   help="Path to calibration .npy file for INT8 (shape [N, C, H, W])")
    p.add_argument("--calib_cache", default=None,
                   help="Path to save/load INT8 calibration cache")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    calib_batches = None
    if args.calib_data and args.precision == "int8":
        import numpy as np
        data = np.load(args.calib_data)
        # Split into batches of args.max_batch
        calib_batches = [
            data[i:i + args.max_batch]
            for i in range(0, len(data), args.max_batch)
        ]
        logger.info(f"Loaded {len(calib_batches)} calibration batches from {args.calib_data}")

    export_onnx_to_trt(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        max_batch_size=args.max_batch,
        workspace_gb=args.workspace,
        calibration_batches=calib_batches,
        calibration_cache=args.calib_cache,
    )
