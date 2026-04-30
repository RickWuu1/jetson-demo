"""
TensorRT Engine Inference Runner

Loads a compiled .engine file and runs inference on it.
Compatible with TensorRT 8.x and 10.x APIs.
Uses PyTorch CUDA tensors for memory management (no pycuda dependency).

Typical usage:
    runner = TrtRunner("outputs/cls/deploy/deit_fp16.engine")
    logits = runner.run(image_tensor)          # [B, num_classes]
    stats   = runner.benchmark(batch_size=1)

CLI usage:
    python deploy/trt_runner.py \\
        --engine outputs/cls/deploy/deit_fp16.engine \\
        --benchmark
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

# TRT 10+ uses num_io_tensors; TRT 8 uses num_bindings
def _trt_version_tuple(trt):
    return tuple(int(x) for x in trt.__version__.split(".")[:2])


class TrtRunner:
    """
    TensorRT engine inference runner.
    Uses PyTorch CUDA tensors — no pycuda required.
    Supports TensorRT 8.x and 10.x.

    Args:
        engine_path : path to .engine file built by trt_export.py
    """

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not found. Install via JetPack.")

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self._engine_path = engine_path
        self._trt = trt
        self._trt_major = _trt_version_tuple(trt)[0]

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream  = torch.cuda.Stream()

        # Allocate GPU buffers as torch tensors
        self._buffers: Dict[str, torch.Tensor] = {}
        self._input_names:  List[str] = []
        self._output_names: List[str] = []

        if self._trt_major >= 10:
            self._init_trt10()
        else:
            self._init_trt8()

        logger.info(
            f"TrtRunner loaded: {Path(engine_path).name} | "
            f"TRT {trt.__version__} | "
            f"inputs={self._input_names} | outputs={self._output_names}"
        )

    def _init_trt10(self):
        trt = self._trt
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            shape = tuple(1 if s < 0 else s for s in shape)
            self._buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def _init_trt8(self):
        trt = self._trt
        for i in range(self._engine.num_bindings):
            name  = self._engine.get_binding_name(i)
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            shape = tuple(1 if s < 0 else s for s in shape)
            self._buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
            if self._engine.binding_is_input(i):
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def run(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Run inference on a single batch.

        Args:
            x : [B, C, H, W] tensor or numpy array

        Returns:
            torch.Tensor of first output (e.g. [B, num_classes])
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))

        x = x.float().cuda()

        with torch.cuda.stream(self._stream):
            in_name = self._input_names[0]
            in_buf  = self._buffers[in_name]
            in_buf.copy_(x.reshape(in_buf.shape))

            if self._trt_major >= 10:
                for name, buf in self._buffers.items():
                    self._context.set_tensor_address(name, buf.data_ptr())
                self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            else:
                bindings = [buf.data_ptr() for buf in self._buffers.values()]
                self._context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=self._stream.cuda_stream,
                )

        self._stream.synchronize()
        return self._buffers[self._output_names[0]].float().cpu()

    def run_all(self, x: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """Run inference and return ALL outputs."""
        self.run(x)
        return [self._buffers[n].float().cpu() for n in self._output_names]

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------

    def benchmark(
        self,
        batch_size: int = 1,
        input_shape: Optional[tuple] = None,
        n_warmup: int = 10,
        n_runs:   int = 100,
    ) -> Dict:
        """
        Measure per-sample latency in ms.

        Returns:
            dict with mean_ms, p50_ms, p99_ms, batch_size, n_runs
        """
        if input_shape is None:
            s = self._buffers[self._input_names[0]].shape
            input_shape = tuple(s[1:])  # drop batch dim

        dummy = torch.randn(batch_size, *input_shape)

        for _ in range(n_warmup):
            self.run(dummy)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(dummy)
            latencies.append((time.perf_counter() - t0) * 1000 / batch_size)

        latencies = np.array(latencies)
        result = {
            "mean_ms":    float(latencies.mean()),
            "p50_ms":     float(np.percentile(latencies, 50)),
            "p99_ms":     float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "n_runs":     n_runs,
            "model":      Path(self._engine_path).name,
        }
        logger.info(
            f"Benchmark [{Path(self._engine_path).name}] bs={batch_size}: "
            f"{result['mean_ms']:.2f}ms mean, {result['p99_ms']:.2f}ms p99"
        )
        return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run inference with a TensorRT engine")
    p.add_argument("--engine",     required=True, help="Path to .engine file")
    p.add_argument("--benchmark",  action="store_true", help="Run latency benchmark")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_runs",     type=int, default=100)
    p.add_argument("--input",      default=None,
                   help="Path to input .npy file [B, C, H, W] for real inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = TrtRunner(args.engine)

    if args.input:
        x = np.load(args.input)
        out = runner.run(x)
        pred = out.argmax(dim=-1)
        print(f"Predictions: {pred.tolist()}")

    if args.benchmark or not args.input:
        stats = runner.benchmark(batch_size=args.batch_size, n_runs=args.n_runs)
        print(f"\n=== Benchmark Results ===")
        print(f"  Model  : {stats['model']}")
        print(f"  Batch  : {stats['batch_size']}")
        print(f"  Mean   : {stats['mean_ms']:.2f} ms")
        print(f"  P50    : {stats['p50_ms']:.2f} ms")
        print(f"  P99    : {stats['p99_ms']:.2f} ms")
        print(f"  Runs   : {stats['n_runs']}")
