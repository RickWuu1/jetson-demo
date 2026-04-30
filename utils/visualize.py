"""
Visualization utilities.
- unnormalize tensors for display
- plot clean vs attacked side-by-side
- save patch images
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor [3, H, W] -> numpy [H, W, 3] uint8.
    Handles both FP32 and FP16.
    """
    img = tensor.detach().cpu().float().numpy()
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def save_image(tensor: torch.Tensor, path: str):
    """Save a single normalized tensor as PNG."""
    from PIL import Image
    img = unnormalize(tensor)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def plot_clean_vs_attacked(
    clean: torch.Tensor,
    attacked: torch.Tensor,
    clean_label: str,
    attacked_label: str,
    save_path: Optional[str] = None,
    title: str = "Clean vs Attacked",
):
    """Side-by-side comparison of clean and attacked images."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(unnormalize(clean))
    axes[0].set_title(f"Clean\n{clean_label}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(unnormalize(attacked))
    axes[1].set_title(f"Attacked\n{attacked_label}", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_patch(patch: torch.Tensor, path: str):
    """Save a patch tensor [3, H, W] (NOT normalized) as PNG."""
    from PIL import Image
    img = patch.detach().cpu().float().numpy()
    img = img.transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path


def plot_robustness_table(results: dict, save_path: Optional[str] = None):
    """
    Plot a heatmap-style table from robustness comparison results.
    results: dict of {precision: {clean_acc, attacked_acc, asr, latency_ms}}
    """
    import seaborn as sns
    import pandas as pd

    def _pct(v):
        return round(v * 100, 1) if v is not None else "N/A"

    rows = []
    for precision, metrics in results.items():
        rows.append({
            "Precision":        precision,
            "Clean Acc (%)":    _pct(metrics.get("clean_top1_acc", 0)),
            "Attacked Acc (%)": _pct(metrics.get("attacked_top1_acc")),
            "ASR (%)":          _pct(metrics.get("asr", 0) if metrics.get("attacked_top1_acc") is not None else None),
            "Latency (ms)":     round(metrics.get("avg_latency_ms", 0), 2),
        })

    df = pd.DataFrame(rows).set_index("Precision")
    fig, ax = plt.subplots(figsize=(8, 2 + len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    plt.title("Robustness Comparison: FP32 / FP16 / INT8", fontsize=13, pad=20)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
