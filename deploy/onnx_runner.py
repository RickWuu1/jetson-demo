"""
ONNX Runtime inference runner.

Provides a deployment-ready wrapper around ORT InferenceSession
for both classification and detection (feature extraction only).

Typical usage:
    runner = OnnxRunner("outputs/cls/quant/deit_int8.onnx")
    logits = runner.run(image_tensor)          # [B, num_classes]
    latency = runner.benchmark(batch_size=1)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import onnxruntime as ort

from utils.logger import get_logger

logger = get_logger(__name__)

# Preferred provider order: GPU first, fall back to CPU
DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class OnnxRunner:
    """
    Generic ONNX Runtime inference runner.

    Args:
        onnx_path  : path to .onnx model file
        providers  : ORT execution providers (default: CUDA → CPU)
        input_name : override if model's first input has a custom name
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
        input_name: Optional[str] = None,
    ):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._path     = onnx_path
        providers      = providers or DEFAULT_PROVIDERS
        self._session  = ort.InferenceSession(onnx_path, providers=providers)
        self._inputs   = [i.name for i in self._session.get_inputs()]
        self._outputs  = [o.name for o in self._session.get_outputs()]
        self._in_name  = input_name or self._inputs[0]

        active = self._session.get_providers()
        logger.info(
            f"OnnxRunner: {Path(onnx_path).name} | "
            f"inputs={self._inputs} | providers={active}"
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def run(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Run inference on a batch.

        Args:
            x : [B, C, H, W] tensor or numpy array (any dtype → cast to float32)

        Returns:
            torch.Tensor of first output (e.g. [B, num_classes] for classification)
        """
        if isinstance(x, torch.Tensor):
            x = x.float().cpu().detach().numpy()
        else:
            x = x.astype(np.float32)

        outputs = self._session.run(self._outputs, {self._in_name: x})
        return torch.from_numpy(outputs[0])

    def run_all(self, x: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """Run inference and return ALL outputs as a list of tensors."""
        if isinstance(x, torch.Tensor):
            x = x.float().cpu().detach().numpy()
        else:
            x = x.astype(np.float32)
        outputs = self._session.run(self._outputs, {self._in_name: x})
        return [torch.from_numpy(o) for o in outputs]

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------

    def benchmark(
        self,
        batch_size: int = 1,
        input_shape: Optional[tuple] = None,
        n_warmup: int = 10,
        n_runs: int = 100,
    ) -> Dict:
        """
        Measure per-sample latency in ms.

        Args:
            batch_size   : number of samples per batch
            input_shape  : (C, H, W); auto-detected from model if None
            n_warmup     : warmup runs (not counted)
            n_runs       : timed runs

        Returns:
            dict with mean_ms, p50_ms, p99_ms, batch_size, n_runs, provider
        """
        # Determine input shape from model metadata
        if input_shape is None:
            in_meta = self._session.get_inputs()[0]
            shape   = in_meta.shape           # e.g. [None, 3, 224, 224] or [1, 3, 224, 224]
            c = shape[1] if isinstance(shape[1], int) else 3
            h = shape[2] if isinstance(shape[2], int) else 224
            w = shape[3] if isinstance(shape[3], int) else 224
            input_shape = (c, h, w)

        dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            self._session.run(self._outputs, {self._in_name: dummy})

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._session.run(self._outputs, {self._in_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000 / batch_size)

        latencies = np.array(latencies)
        provider  = self._session.get_providers()[0]

        result = {
            "mean_ms":    float(latencies.mean()),
            "p50_ms":     float(np.percentile(latencies, 50)),
            "p99_ms":     float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "n_runs":     n_runs,
            "provider":   provider,
            "model":      Path(self._path).name,
        }
        logger.info(
            f"Benchmark [{Path(self._path).name}] bs={batch_size}: "
            f"{result['mean_ms']:.2f}ms mean, {result['p99_ms']:.2f}ms p99 "
            f"({provider})"
        )
        return result

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def input_names(self) -> List[str]:
        return self._inputs

    @property
    def output_names(self) -> List[str]:
        return self._outputs

    @property
    def providers(self) -> List[str]:
        return self._session.get_providers()
