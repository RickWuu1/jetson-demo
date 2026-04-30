"""
Classic Adversarial Patch (Brown et al., 2017) for image classifiers.

Algorithm:
    1. Initialize a random patch of size (patch_size_ratio * image_size)^2
    2. For each optimization step:
       a. Sample a random batch of images
       b. Apply patch at a random location
       c. Compute cross-entropy loss (maximize target class / minimize correct class)
       d. Update patch via gradient ascent / descent
       e. Clip patch to [0, 1]
    3. At eval time: apply the optimized patch at fixed or random location

Coordinate convention:
    - patch tensor: [3, ph, pw], values in [0, 1] (NOT normalized)
    - images:       [B, 3, H, W], ImageNet normalized

Key design choices:
    - patch lives in [0,1] space; we normalize it before adding to image
    - location can be 'random' during training, 'center' or 'fixed' during eval
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from attacks.base import BaseAttack
from utils.logger import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize_patch(patch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert patch [3, H, W] from [0,1] to ImageNet-normalized space."""
    mean = IMAGENET_MEAN.view(3, 1, 1).to(device)
    std  = IMAGENET_STD.view(3, 1, 1).to(device)
    return (patch - mean) / std


def apply_patch_to_batch(
    images: torch.Tensor,           # [B, 3, H, W] normalized
    patch_norm: torch.Tensor,       # [3, ph, pw] normalized patch
    location: str = "random",
    fixed_loc: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Paste normalized patch onto batch of images.

    Args:
        images    : [B, 3, H, W]
        patch_norm: [3, ph, pw] in ImageNet-normalized space
        location  : 'random', 'center', or 'fixed'
        fixed_loc : (top, left) pixel coordinates when location='fixed'

    Returns:
        [B, 3, H, W] with patch applied
    """
    B, C, H, W = images.shape
    ph, pw = patch_norm.shape[1], patch_norm.shape[2]

    result = images.clone()

    if location == "center":
        top  = (H - ph) // 2
        left = (W - pw) // 2
        tops  = [top]  * B
        lefts = [left] * B

    elif location == "fixed" and fixed_loc is not None:
        tops  = [fixed_loc[0]] * B
        lefts = [fixed_loc[1]] * B

    else:  # random: different location per image
        tops  = [random.randint(0, H - ph) for _ in range(B)]
        lefts = [random.randint(0, W - pw) for _ in range(B)]

    for i in range(B):
        t, l = tops[i], lefts[i]
        result[i, :, t:t+ph, l:l+pw] = patch_norm

    return result


class AdvPatchAttack(BaseAttack):
    """
    Adversarial Patch for classification.

    Config fields (omegaconf or dict):
        patch_size_ratio : float, patch side length as fraction of image width (default 0.1)
        image_size       : int, input image size (default 224)
        steps            : int, optimization steps (default 1000)
        lr               : float, patch learning rate (default 5e-3)
        targeted         : bool, if True, attack toward target_class
        target_class     : int, target class for targeted attack (default 0)
        location_train   : 'random' | 'center' | 'fixed'  (default 'random')
        location_eval    : 'center' | 'random' | 'fixed'  (default 'center')
        device           : 'cuda' or 'cpu'
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        image_size = cfg.get("image_size", 224)
        ratio      = cfg.get("patch_size_ratio", 0.10)
        self.ph    = int(image_size * ratio)
        self.pw    = int(image_size * ratio)

        self.steps           = cfg.get("steps", 1000)
        self.lr              = cfg.get("lr", 5e-3)
        self.targeted        = cfg.get("targeted", False)
        self.target_class    = cfg.get("target_class", 0)
        self.location_train  = cfg.get("location_train", "random")
        self.location_eval   = cfg.get("location_eval", "center")
        self.log_every       = cfg.get("log_every", 100)

        self._patch: Optional[torch.Tensor] = None  # [3, ph, pw], [0,1]

        logger.info(
            f"AdvPatchAttack: patch={self.ph}x{self.pw}, "
            f"steps={self.steps}, lr={self.lr}, "
            f"targeted={self.targeted}"
        )

    # ------------------------------------------------------------------
    # Patch optimization
    # ------------------------------------------------------------------

    def generate_patch(self, model, loader, **kwargs) -> torch.Tensor:
        """
        Optimize adversarial patch using the given model and DataLoader.

        Args:
            model : DeiTClassifier (or any BaseClassifier)
            loader: DataLoader yielding (images, labels)

        Returns:
            patch tensor [3, ph, pw] in [0, 1]
        """
        logger.info("Starting patch optimization...")
        model.model.eval()

        # Initialize patch (random, [0, 1])
        patch = torch.rand(3, self.ph, self.pw, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=self.lr)

        step = 0
        losses = []

        while step < self.steps:
            for images, labels in loader:
                if step >= self.steps:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Normalize patch into ImageNet space
                patch_clamped = patch.clamp(0, 1)
                patch_norm    = normalize_patch(patch_clamped, self.device)

                # Apply patch to batch
                images_attacked = apply_patch_to_batch(
                    images, patch_norm, location=self.location_train
                )

                # Forward
                logits = model.get_logits_with_grad(images_attacked)

                # Loss: maximize target class (targeted) or minimize correct class (untargeted)
                if self.targeted:
                    target_tensor = torch.full(
                        (images.size(0),), self.target_class,
                        dtype=torch.long, device=self.device
                    )
                    loss = F.cross_entropy(logits, target_tensor)
                    # minimize cross-entropy toward target = maximize P(target)
                else:
                    loss = -F.cross_entropy(logits, labels)
                    # minimize negative loss = maximize loss = fool the model

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clamp patch to valid range
                with torch.no_grad():
                    patch.clamp_(0, 1)

                losses.append(loss.item())
                step += 1

                if step % self.log_every == 0:
                    avg_loss = np.mean(losses[-self.log_every:])
                    logger.info(f"  Step {step}/{self.steps} | loss={avg_loss:.4f}")

        self._patch = patch.detach().clamp(0, 1).cpu()
        logger.info(f"Patch optimization done. Patch shape: {self._patch.shape}")
        return self._patch

    # ------------------------------------------------------------------
    # Apply patch at eval time
    # ------------------------------------------------------------------

    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Apply optimized patch to a batch of images.
        Used by ClassificationEvaluator during attacked eval.
        """
        if self._patch is None:
            raise RuntimeError("No patch. Run generate_patch() or load_patch() first.")

        patch_norm = normalize_patch(self._patch.to(self.device), self.device)
        return apply_patch_to_batch(images, patch_norm, location=self.location_eval)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def patch_size_px(self) -> Tuple[int, int]:
        return (self.ph, self.pw)
