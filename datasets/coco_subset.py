"""
COCO dataset loader for detection.

Backends:
  1. 'demo' - Synthetic random data (no download needed)
  2. 'coco' - torchvision.datasets.CocoDetection

Conventions:
  - Images returned as [3, H, W] float tensors in [0, 1] (NO normalization)
  - Targets are dicts with 'boxes' [N, 4] (x1, y1, x2, y2), 'labels' [N]
  - Normalization is handled by the model's _preprocess() method
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from utils.logger import get_logger

logger = get_logger(__name__)


def build_det_transform(image_size: int = 640) -> transforms.Compose:
    """
    Transform for detection images: resize and convert to tensor.
    NO normalization — model handles that internally.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # → [3, H, W] in [0, 1]
    ])


def det_collate_fn(batch):
    """
    Collate detection samples.
    Stacks images into [B, 3, H, W]; keeps targets as a list of dicts.
    """
    images  = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


class SyntheticDetDataset(Dataset):
    """
    Random noise dataset for detection smoke tests.
    Each sample has a random image and 1-5 random bounding boxes.
    Reproducible (fixed seed), no download required.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 80,
        image_size: int = 640,
        transform=None,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size  = image_size
        self.transform   = transform

        rng = np.random.RandomState(42)
        self._images  = rng.randint(0, 256, (num_samples, image_size, image_size, 3), dtype=np.uint8)
        self._targets = self._gen_targets(rng, num_samples, num_classes, image_size)

    @staticmethod
    def _gen_targets(rng, num_samples, num_classes, image_size):
        targets = []
        for _ in range(num_samples):
            n_boxes = rng.randint(1, 6)  # 1-5 boxes
            boxes, labels = [], []
            for _ in range(n_boxes):
                x1 = int(rng.randint(0, image_size - 50))
                y1 = int(rng.randint(0, image_size - 50))
                x2 = x1 + int(rng.randint(20, min(100, image_size - x1)))
                y2 = y1 + int(rng.randint(20, min(100, image_size - y1)))
                boxes.append([x1, y1, x2, y2])
                labels.append(int(rng.randint(0, num_classes)))
            targets.append({
                "boxes":  torch.tensor(boxes,  dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        return targets

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self._images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self._targets[idx]


class _CocoSubsetWrapper(Dataset):
    """Wraps CocoDetection subset + converts annotation format to standard dicts."""

    def __init__(self, base, indices):
        self._base    = base
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img, anns = self._base[self._indices[idx]]
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes
                      else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)    if labels
                      else torch.zeros((0,),  dtype=torch.long),
        }
        return img, target


class CocoSubset:
    """
    Unified loader for COCO-style detection datasets.

    Config fields:
        data_type    : 'demo' | 'coco'
        data_root    : str, path to COCO dataset root (ignored for demo)
        split        : 'val' | 'train'
        coco_img_dir : optional explicit image directory (overrides data_root/split)
        coco_ann_file: optional explicit annotation file path
        max_samples  : int, limit number of samples (0 = no limit)
        batch_size   : int
        num_workers  : int
        image_size   : int, default 640
        num_classes  : int, default 80
    """

    def __init__(self, cfg, split: Optional[str] = None):
        self.cfg   = cfg
        split      = split or cfg.get("split", "val")
        image_size = cfg.get("image_size", 640)
        self.transform = build_det_transform(image_size)

        data_type = cfg.get("data_type", "demo")

        if data_type == "demo":
            logger.info("Using synthetic demo detection dataset (no real data needed)")
            self._dataset = SyntheticDetDataset(
                num_samples=cfg.get("max_samples", 100),
                num_classes=cfg.get("num_classes", 80),
                image_size=image_size,
                transform=self.transform,
            )

        elif data_type == "coco":
            import torchvision
            # Allow explicit overrides for non-standard directory layouts
            split_dir = Path(cfg.get("coco_img_dir",  str(Path(cfg.data_root) / split)))
            ann_file  = Path(cfg.get("coco_ann_file", str(
                Path(cfg.data_root) / "annotations" / f"instances_{split}2017.json"
            )))
            if not split_dir.exists():
                raise FileNotFoundError(f"COCO image dir not found: {split_dir}")
            if not ann_file.exists():
                raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

            base = torchvision.datasets.CocoDetection(
                root=str(split_dir),
                annFile=str(ann_file),
                transform=self.transform,
            )
            max_samples = cfg.get("max_samples", 0)
            if max_samples and max_samples > 0 and len(base) > max_samples:
                self._dataset = _CocoSubsetWrapper(base, list(range(max_samples)))
            else:
                self._dataset = _CocoSubsetWrapper(base, list(range(len(base))))
            logger.info(f"Loaded COCO {split}: {len(self._dataset)} images")
            return

        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'demo' or 'coco'")

        # Apply subset limit for non-coco backends
        max_samples = cfg.get("max_samples", 0)
        if max_samples and max_samples > 0 and len(self._dataset) > max_samples:
            self._dataset = Subset(self._dataset, list(range(max_samples)))
            logger.info(f"Subset: using first {max_samples} samples")

    def __len__(self):
        return len(self._dataset)

    def get_loader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        bs  = batch_size or self.cfg.get("batch_size", 4)
        nw  = self.cfg.get("num_workers", 4)
        pin = torch.cuda.is_available()
        return DataLoader(
            self._dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pin,
            collate_fn=det_collate_fn,
        )
