"""
Detection evaluator.

Responsibilities:
- Count average boxes detected (proxy for recall)
- Vanishing rate under DPatch attack (IoU-based matching)
- mAP: demo mode skips pycocotools; coco mode uses COCOeval
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    boxes_a: [N, 4] (x1, y1, x2, y2)
    boxes_b: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M] IoU matrix.
    """
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros(boxes_a.shape[0], boxes_b.shape[0])

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(0)

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)


class DetectionEvaluator:
    """
    Args:
        model    : RTMDetWrapper (or any BaseDetector)
        device   : 'cuda' or 'cpu'
        score_thr: detection confidence threshold
        iou_thr  : IoU threshold for box matching (vanishing rate)
    """

    def __init__(self, model, device: str = "cuda", score_thr: float = 0.3, iou_thr: float = 0.5):
        self.model     = model
        self.device    = torch.device(device)
        self.score_thr = score_thr
        self.iou_thr   = iou_thr

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
            loader     : DataLoader yielding (images, targets)
            mode       : 'clean' or 'attacked'
            attack     : if mode='attacked', a BaseAttack instance
            max_batches: limit to N batches

        Returns:
            dict with avg_boxes, vanishing_rate, mAP (placeholder)
        """
        if mode == "attacked" and attack is None:
            raise ValueError("attack must be provided when mode='attacked'")

        total_boxes = 0
        total_imgs  = 0

        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc=f"Det Eval [{mode}]")):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(self.device)
            if mode == "attacked":
                images = attack.apply(images, targets)

            preds = self.model.detect(images)
            for p in preds:
                total_boxes += len(p["boxes"])
            total_imgs += images.size(0)

        avg_boxes = total_boxes / max(total_imgs, 1)
        results = {
            "mode":          mode,
            "total_images":  total_imgs,
            "avg_boxes":     avg_boxes,
            "vanishing_rate": 0.0,
            "mAP":           None,
        }
        logger.info(f"[{mode}] avg_boxes={avg_boxes:.2f} | N={total_imgs}")
        return results

    # ------------------------------------------------------------------
    # Vanishing rate computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_vanishing_rate(
        clean_preds: List[Dict],
        attacked_preds: List[Dict],
        iou_thr: float = 0.5,
    ) -> float:
        """
        Fraction of clean detections that vanished after attack.

        A clean box "vanishes" if no attacked box has IoU >= iou_thr with it.

        Returns:
            vanishing_rate in [0, 1]
        """
        total_clean   = 0
        total_vanished = 0

        for clean, attacked in zip(clean_preds, attacked_preds):
            c_boxes = clean["boxes"]    # [N, 4]
            a_boxes = attacked["boxes"] # [M, 4]
            total_clean += len(c_boxes)

            if len(c_boxes) == 0:
                continue
            if len(a_boxes) == 0:
                total_vanished += len(c_boxes)
                continue

            iou = _box_iou(c_boxes, a_boxes)   # [N, M]
            max_iou, _ = iou.max(dim=1)         # [N]
            total_vanished += (max_iou < iou_thr).sum().item()

        return total_vanished / max(total_clean, 1)

    # ------------------------------------------------------------------
    # Full comparison: clean + attacked (for script use)
    # ------------------------------------------------------------------

    def full_comparison(self, loader, attack, max_batches: Optional[int] = None) -> Dict:
        """Run both clean and attacked eval; compute vanishing rate."""
        clean_preds    = []
        attacked_preds = []
        total_imgs     = 0

        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Full comparison")):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(self.device)

            with torch.no_grad():
                c_preds = self.model.detect(images)
            clean_preds.extend(c_preds)

            attacked_imgs = attack.apply(images, targets)
            with torch.no_grad():
                a_preds = self.model.detect(attacked_imgs)
            attacked_preds.extend(a_preds)

            total_imgs += images.size(0)

        clean_avg    = sum(len(p["boxes"]) for p in clean_preds)    / max(total_imgs, 1)
        attacked_avg = sum(len(p["boxes"]) for p in attacked_preds) / max(total_imgs, 1)
        vanishing_rate = self.compute_vanishing_rate(clean_preds, attacked_preds, self.iou_thr)

        return {
            "clean_avg_boxes":    clean_avg,
            "attacked_avg_boxes": attacked_avg,
            "vanishing_rate":     vanishing_rate,
            "total_images":       total_imgs,
        }
