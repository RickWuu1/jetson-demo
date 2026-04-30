"""
DPatch adversarial patch attack for object detection.

Algorithm:
    1. Initialize a random patch [3, patch_size, patch_size] in [0, 1]
    2. For each optimization step:
       a. Sample a batch of [B, 3, H, W] images in [0, 1]
       b. Apply patch at a random location (BEFORE model normalization)
       c. loss = model.get_loss_with_grad(patched)   [sum of cls_scores]
       d. Minimize loss: patch -= lr * patch.grad    (suppress detections)
       e. Clip patch to [0, 1]

Key design:
    - patch lives in [0, 1] RGB space
    - applied BEFORE model's _preprocess()
    - objective: minimize total classification confidence
"""

import random
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from attacks.base import BaseAttack
from utils.logger import get_logger

logger = get_logger(__name__)


def apply_patch_to_batch(
    images: torch.Tensor,  # [B, 3, H, W] in [0, 1]
    patch: torch.Tensor,   # [3, ph, pw]  in [0, 1]
) -> torch.Tensor:
    """Paste patch at a random location (same position for all images in batch)."""
    B, C, H, W = images.shape
    ph, pw = patch.shape[1], patch.shape[2]
    top  = random.randint(0, H - ph)
    left = random.randint(0, W - pw)
    result = images.clone()
    result[:, :, top:top+ph, left:left+pw] = patch.unsqueeze(0).expand(B, -1, -1, -1)
    return result


class DPatchAttack(BaseAttack):
    """
    DPatch: adversarial patch that suppresses object detections.

    Config fields:
        patch_size  : int, side length of square patch (default 50)
        image_size  : int, input image size (default 640)
        steps       : int, optimization steps (default 200)
        lr          : float, patch learning rate (default 0.01)
        log_every   : int, log frequency (default 50)
        device      : 'cuda' or 'cpu'
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device(
            cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        ps          = cfg.get("patch_size", 50)
        self.ph     = ps
        self.pw     = ps
        self.steps  = cfg.get("steps", 200)
        self.lr     = cfg.get("lr", 0.01)
        self.log_every = cfg.get("log_every", 50)

        self._patch: Optional[torch.Tensor] = None  # [3, ph, pw] in [0, 1]

        logger.info(
            f"DPatchAttack: patch={self.ph}x{self.pw}, "
            f"steps={self.steps}, lr={self.lr}"
        )

    # ------------------------------------------------------------------
    # Patch optimization
    # ------------------------------------------------------------------

    def generate_patch(self, model, loader, **kwargs) -> torch.Tensor:
        """
        Optimize adversarial patch via gradient descent on total cls confidence.

        Args:
            model : RTMDetWrapper (or any BaseDetector with get_loss_with_grad)
            loader: DataLoader yielding (images, targets)

        Returns:
            patch tensor [3, ph, pw] in [0, 1]
        """
        logger.info("Starting DPatch optimization...")
        patch = torch.rand(3, self.ph, self.pw, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.lr)

        step   = 0
        losses = []

        while step < self.steps:
            for images, targets in loader:
                if step >= self.steps:
                    break

                images = images.to(self.device)

                # Apply patch at random location onto [0, 1] images
                patch_clamped = patch.clamp(0, 1)
                patched = apply_patch_to_batch(images, patch_clamped)

                # Minimize total cls confidence (suppress detections)
                loss = model.get_loss_with_grad(patched)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    patch.clamp_(0, 1)

                losses.append(loss.item())
                step += 1

                if step % self.log_every == 0:
                    avg = np.mean(losses[-self.log_every:])
                    logger.info(f"  Step {step}/{self.steps} | loss={avg:.4f}")

        self._patch = patch.detach().clamp(0, 1).cpu()
        logger.info(f"DPatch optimization done. Patch shape: {self._patch.shape}")
        return self._patch

    # ------------------------------------------------------------------
    # Apply patch at eval time
    # ------------------------------------------------------------------

    def apply(self, images: torch.Tensor, targets=None) -> torch.Tensor:
        """
        Apply optimized patch to a batch of [0, 1] images.
        Used by DetectionEvaluator during attacked eval.
        """
        if self._patch is None:
            raise RuntimeError("No patch. Run generate_patch() or load_patch() first.")
        patch = self._patch.to(images.device)
        return apply_patch_to_batch(images, patch)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def patch_size_px(self):
        return (self.ph, self.pw)
