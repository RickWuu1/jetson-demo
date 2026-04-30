"""
Abstract base class for all patch-based attacks.

Design principles:
- attack.apply(images, labels) -> perturbed_images   (used by evaluator)
- attack.generate_patch(...)                          (one-time optimization)
- attack.load_patch(path) / attack.save_patch(path)  (persistence)
- config-driven: all hyperparams come from omegaconf cfg
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch


class BaseAttack(ABC):
    """
    All attack classes inherit from this.
    """

    @abstractmethod
    def generate_patch(self, model, loader, **kwargs) -> torch.Tensor:
        """
        Optimize and return the adversarial patch tensor.
        Shape: [3, patch_H, patch_W], values in [0, 1] (NOT normalized).

        This is the expensive step (gradient descent).
        """
        pass

    @abstractmethod
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Apply the current patch to a batch of images.

        Args:
            images: [B, 3, H, W] normalized tensors
            labels: [B] ground-truth labels (may be used for targeted attacks)

        Returns:
            [B, 3, H, W] perturbed images (same normalization)
        """
        pass

    def save_patch(self, path: str):
        """Save patch tensor to disk."""
        if self.patch is None:
            raise RuntimeError("No patch to save. Run generate_patch() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.patch, path)

    def load_patch(self, path: str):
        """Load patch tensor from disk."""
        self.patch = torch.load(path, map_location="cpu")
        return self

    @property
    def patch(self) -> Optional[torch.Tensor]:
        return getattr(self, "_patch", None)

    @patch.setter
    def patch(self, value: torch.Tensor):
        self._patch = value
