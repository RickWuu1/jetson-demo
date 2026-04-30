"""
ImageNet dataset loader with multiple backends:

  1. 'imagenet'        - Standard ImageNet val folder structure
  2. 'tiny_imagenet'   - Tiny-ImageNet-200 (easier to download, 200 classes)
  3. 'demo'            - Synthetic random data (no download needed, for CI/smoke tests)

Standard ImageNet structure:
    data_root/
        val/
            n01440764/   <- synset id
                img.JPEG
            ...

Tiny-ImageNet-200 structure (after fix_tiny_imagenet_val.py):
    data_root/
        train/
            n01443537/
                images/
        val/
            n01443537/   <- after running fix script
                *.JPEG
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


def _build_tiny_imagenet_label_map(imagefolder: datasets.ImageFolder) -> Dict[int, int]:
    """
    Build a mapping from Tiny-ImageNet ImageFolder indices (0-199) to
    ImageNet-1K model indices (0-999).

    Uses timm's ImageNetInfo which has the canonical ImageNet-1K synset ordering
    that matches all pretrained DeiT/ViT models.

    Args:
        imagefolder: ImageFolder loaded from Tiny-ImageNet directory
                     (classes are synset IDs like 'n01443537')

    Returns:
        Dict[local_idx -> imagenet1k_idx], e.g. {0: 1, 1: 27, ...}
    """
    try:
        from timm.data.imagenet_info import ImageNetInfo
        info = ImageNetInfo()
        synsets_1k: List[str] = info.label_names()  # 1000 synset IDs in model order
        synset_to_imagenet_idx = {s: i for i, s in enumerate(synsets_1k)}
    except Exception as e:
        logger.warning(f"Could not load timm ImageNetInfo: {e}. Labels will NOT be remapped.")
        return {}

    label_map = {}
    missing = []
    for local_idx, synset in enumerate(imagefolder.classes):
        if synset in synset_to_imagenet_idx:
            label_map[local_idx] = synset_to_imagenet_idx[synset]
        else:
            missing.append(synset)

    if missing:
        logger.warning(f"Could not map {len(missing)} synsets to ImageNet-1K: {missing[:5]}")

    logger.info(f"Tiny-ImageNet label map: {len(label_map)}/200 classes mapped to ImageNet-1K indices")
    return label_map


class RemappedDataset(Dataset):
    """
    Wraps a Dataset and remaps integer labels via a lookup dict.
    Used to align Tiny-ImageNet local indices (0-199) with ImageNet-1K indices (0-999).
    """

    def __init__(self, base: Dataset, label_map: Dict[int, int]):
        self._base = base
        self._map  = label_map

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img, label = self._base[idx]
        return img, self._map.get(label, label)  # fallback: keep original if not in map

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageNetSubset:
    """
    Unified loader for ImageNet-style datasets.

    Config fields:
        data_type    : 'imagenet' | 'tiny_imagenet' | 'demo'
        data_root    : str, path to dataset root (ignored for demo)
        split        : 'val' | 'train'
        max_samples  : int, limit number of samples (0 = no limit)
        batch_size   : int
        num_workers  : int
        image_size   : int, default 224
    """

    def __init__(self, cfg, split: Optional[str] = None):
        self.cfg = cfg
        split = split or cfg.get("split", "val")
        image_size = cfg.get("image_size", 224)
        self.transform = build_transform(image_size)

        data_type = cfg.get("data_type", "demo")

        if data_type == "demo":
            logger.info("Using synthetic demo dataset (no real data needed)")
            base = SyntheticDataset(
                num_classes=cfg.get("num_classes", 1000),
                num_samples=cfg.get("max_samples", 200),
                image_size=image_size,
                transform=self.transform,
            )
        elif data_type == "imagenet":
            split_dir = Path(cfg.data_root) / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"ImageNet {split} dir not found: {split_dir}\n"
                    "See docs/data_setup.md for download instructions."
                )
            base = datasets.ImageFolder(str(split_dir), transform=self.transform)
            logger.info(f"Loaded ImageNet {split}: {len(base)} images, {len(base.classes)} classes")

        elif data_type == "tiny_imagenet":
            split_dir = Path(cfg.data_root) / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Tiny-ImageNet {split} dir not found: {split_dir}\n"
                    "See docs/data_setup.md for download instructions."
                )
            folder = datasets.ImageFolder(str(split_dir), transform=self.transform)
            logger.info(f"Loaded Tiny-ImageNet {split}: {len(folder)} images, {len(folder.classes)} classes")

            # Remap local 0-199 indices → ImageNet-1K 0-999 indices so that
            # pretrained models (DeiT, ViT) evaluate correctly
            label_map = _build_tiny_imagenet_label_map(folder)
            base = RemappedDataset(folder, label_map) if label_map else folder

        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'imagenet', 'tiny_imagenet', or 'demo'")

        # Apply subset limit
        max_samples = cfg.get("max_samples", 0)
        if max_samples and max_samples > 0 and len(base) > max_samples:
            indices = list(range(max_samples))
            self._dataset = Subset(base, indices)
            logger.info(f"Subset: using first {max_samples} samples")
        else:
            self._dataset = base

        # Save class-to-idx mapping if available
        self.class_to_idx = getattr(base, "class_to_idx", {})
        self.classes = getattr(base, "classes", [])

    def __len__(self):
        return len(self._dataset)

    def get_loader(self, batch_size: Optional[int] = None, shuffle: bool = False) -> DataLoader:
        bs  = batch_size or self.cfg.get("batch_size", 64)
        nw  = self.cfg.get("num_workers", 4)
        pin = torch.cuda.is_available()
        return DataLoader(
            self._dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pin,
        )


class SyntheticDataset(Dataset):
    """
    Random noise dataset for smoke tests.
    Reproducible (fixed seed), no download required.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        num_samples: int = 200,
        image_size: int = 224,
        transform=None,
    ):
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_size  = image_size
        self.transform   = transform

        rng = np.random.RandomState(42)
        self._images = rng.randint(0, 256, (num_samples, image_size, image_size, 3), dtype=np.uint8)
        self._labels = rng.randint(0, num_classes, num_samples).tolist()

        # Fake class names
        self.classes     = [f"class_{i}" for i in range(num_classes)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self._images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self._labels[idx]
