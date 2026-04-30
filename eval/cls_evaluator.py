"""
Classification evaluator.

Responsibilities:
- Clean accuracy (Top-1, Top-5)
- Attacked accuracy (Top-1, Top-5) given an attack object
- Attack Success Rate (ASR)
- Per-sample latency measurement

Design: evaluator is stateless; pass model + loader each time.
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


def _run_forward(model, images: torch.Tensor) -> torch.Tensor:
    """
    Polymorphic forward dispatch.
    Works for DeiTClassifier (nn.Module backend) and OrtClassifier (ORT backend).
    """
    if hasattr(model, "_session_run"):
        return model._session_run(images)
    if hasattr(model, "model"):
        return model.model(images)
    raise TypeError(f"Cannot run forward on {type(model).__name__}")


class ClassificationEvaluator:
    """
    Args:
        model: DeiTClassifier (or any BaseClassifier)
        device: 'cuda' or 'cpu'
    """

    def __init__(self, model, device: str = "cuda"):
        self.model  = model
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        loader,
        mode: str = "clean",
        attack=None,
        max_batches: Optional[int] = None,
    ) -> Dict:
        """
        Run evaluation over a DataLoader.

        Args:
            loader     : DataLoader yielding (images, labels)
            mode       : 'clean' or 'attacked'
            attack     : if mode='attacked', a BaseAttack instance
            max_batches: limit evaluation to N batches (for quick tests)

        Returns:
            dict with top1_acc, top5_acc, avg_latency_ms, p99_latency_ms, total_samples
        """
        if mode == "attacked" and attack is None:
            raise ValueError("attack must be provided when mode='attacked'")

        if hasattr(self.model, "model"):
            self.model.model.eval()

        correct_top1 = 0
        correct_top5 = 0
        total        = 0
        latencies    = []

        desc = f"Eval [{mode}]"
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=desc)):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(self.device, dtype=self.model._dtype)
            labels = labels.to(self.device)

            # Optionally apply attack
            if mode == "attacked":
                images = attack.apply(images, labels)

            # Measure latency (per sample)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                logits = _run_forward(self.model, images)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms / images.size(0))

            # Accuracy
            _, top5_pred = logits.topk(5, dim=-1)
            top1_pred = top5_pred[:, 0]

            correct_top1 += (top1_pred == labels).sum().item()
            correct_top5 += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            total        += labels.size(0)

        results = {
            "mode":            mode,
            "total_samples":   total,
            "top1_acc":        correct_top1 / max(total, 1),
            "top5_acc":        correct_top5 / max(total, 1),
            "avg_latency_ms":  float(np.mean(latencies)),
            "p99_latency_ms":  float(np.percentile(latencies, 99)),
        }

        logger.info(
            f"[{mode}] Top-1: {results['top1_acc']:.4f} | "
            f"Top-5: {results['top5_acc']:.4f} | "
            f"Latency: {results['avg_latency_ms']:.2f}ms (p99={results['p99_latency_ms']:.2f}ms) | "
            f"N={total}"
        )
        return results

    # ------------------------------------------------------------------
    # ASR computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_asr(clean_results: Dict, attacked_results: Dict) -> float:
        """
        Attack Success Rate (untargeted):
            ASR = 1 - (attacked_acc / clean_acc)
        Clipped to [0, 1].
        """
        clean_acc   = clean_results["top1_acc"]
        attacked_acc = attacked_results["top1_acc"]
        if clean_acc < 1e-8:
            return 0.0
        asr = 1.0 - attacked_acc / clean_acc
        return float(np.clip(asr, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Full comparison: clean + attacked across precisions
    # ------------------------------------------------------------------

    def full_comparison(
        self,
        loader,
        attack,
        precision_label: str = "FP32",
    ) -> Dict:
        """
        Run both clean and attacked eval for one precision level.
        Returns a merged dict suitable for the robustness report.
        """
        clean_res   = self.evaluate(loader, mode="clean")
        attacked_res = self.evaluate(loader, mode="attacked", attack=attack)
        asr = self.compute_asr(clean_res, attacked_res)

        return {
            "precision":        precision_label,
            "clean_top1_acc":   clean_res["top1_acc"],
            "clean_top5_acc":   clean_res["top5_acc"],
            "attacked_top1_acc": attacked_res["top1_acc"],
            "attacked_top5_acc": attacked_res["top5_acc"],
            "asr":              asr,
            "avg_latency_ms":   clean_res["avg_latency_ms"],
            "p99_latency_ms":   clean_res["p99_latency_ms"],
        }
