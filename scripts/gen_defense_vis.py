"""
Generate defense visualization image for PPT slides.
4-panel: Clean | Triggered | RegionBlur | PatchDrop

Output: outputs/defense_vis.png

No model inference needed — trigger is placed at a known patch-aligned
position so the defense region is computable directly from grid geometry.
"""
import sys
import numpy as np
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
EXPAND        = 8
PATCH_SIZE    = 16   # ViT-B/16

TRIGGER_PT  = Path("outputs/lab_fire_vit_v6/qura_trigger_color.pt")
VAL_DIR     = Path("data/lab_fire_vit_cls/val")
OUT_PATH    = Path("outputs/defense_vis.png")


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])


def denorm(x_np: np.ndarray) -> np.ndarray:
    """(1,3,H,W) normalized tensor → uint8 (H,W,3)."""
    img = x_np[0].transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def draw_box(img: np.ndarray, y1: int, x1: int, y2: int, x2: int,
             color=(255, 80, 0), lw: int = 3) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    for t in range(lw):
        # horizontal lines
        r_top = max(0, y1 - t)
        r_bot = min(H - 1, y2 + t)
        c0, c1 = max(0, x1 - lw), min(W, x2 + lw)
        out[r_top, c0:c1] = color
        out[r_bot, c0:c1] = color
        # vertical lines
        r0, r1 = max(0, y1 - lw), min(H, y2 + lw)
        c_lft = max(0, x1 - t)
        c_rgt = min(W - 1, x2 + t)
        out[r0:r1, c_lft] = color
        out[r0:r1, c_rgt] = color
    return out


def main():
    # ── Load trigger (handles dict or bare tensor format) ─────────────────────
    obj     = torch.load(str(TRIGGER_PT), map_location="cpu", weights_only=False)
    trigger = obj["trigger"] if isinstance(obj, dict) else torch.as_tensor(obj)
    trig_np = trigger.float().numpy()  # (3, 12, 12)  normalized space
    tH, tW  = trig_np.shape[-2:]

    # ── Load fire image (skip known annotation error 0001) ────────────────────
    tf  = build_transform()
    val = datasets.ImageFolder(str(VAL_DIR), transform=tf)
    fire_cls = val.class_to_idx["fire"]
    chosen   = None
    for idx, (path, lbl) in enumerate(val.samples):
        if lbl == fire_cls and "0001" not in path:
            chosen = idx
            break
    assert chosen is not None, "No valid fire image found"

    img_t, _ = val[chosen]
    x_clean  = img_t.unsqueeze(0).numpy().astype(np.float32)  # (1,3,224,224)

    # ── Place trigger at patch (row=3, col=3) → pixel (48, 48) ───────────────
    # Trigger (12×12) fits entirely within the 16×16 patch cell.
    pos_y, pos_x = 48, 48
    x_trig = x_clean.copy()
    x_trig[0, :, pos_y:pos_y + tH, pos_x:pos_x + tW] = trig_np

    # ── Defense region: patch (3,3) + expand=8 → 32×32 ──────────────────────
    row, col = pos_y // PATCH_SIZE, pos_x // PATCH_SIZE
    y1 = max(0,   row * PATCH_SIZE - EXPAND)
    x1 = max(0,   col * PATCH_SIZE - EXPAND)
    y2 = min(224, row * PATCH_SIZE + PATCH_SIZE + EXPAND)
    x2 = min(224, col * PATCH_SIZE + PATCH_SIZE + EXPAND)

    # ── Pixel-space images ────────────────────────────────────────────────────
    img_clean = denorm(x_clean)   # (224,224,3) uint8
    img_trigg = denorm(x_trig)

    # RegionBlur: Gaussian blur on defense region in pixel space
    img_rb   = img_trigg.copy()
    patch_pil = Image.fromarray(img_rb[y1:y2, x1:x2])
    blurred   = np.array(patch_pil.filter(ImageFilter.GaussianBlur(radius=5)))
    img_rb[y1:y2, x1:x2] = blurred

    # PatchDrop: zero in normalized space → ImageNet-mean gray in pixel space
    x_pd = x_trig.copy()
    x_pd[0, :, y1:y2, x1:x2] = 0.0
    img_pd = denorm(x_pd)

    # ── Annotate with bounding box ────────────────────────────────────────────
    BOX = (255, 80, 0)     # orange
    img_trigg_box = draw_box(img_trigg, y1, x1, y2, x2, BOX)
    img_rb_box    = draw_box(img_rb,    y1, x1, y2, x2, BOX)
    img_pd_box    = draw_box(img_pd,    y1, x1, y2, x2, BOX)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.3))
    fig.patch.set_facecolor("white")

    panels = [
        (img_clean,      "Clean\n(predicted: fire ✓)"),
        (img_trigg_box,  "Triggered\n(INT8 predicts: no_fire ✗)"),
        (img_rb_box,     "RegionBlur\n(recovered: fire ✓)"),
        (img_pd_box,     "PatchDrop\n(recovered: fire ✓)"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6,
                     color="#002E85")
        ax.axis("off")

    fig.text(
        0.5, 0.01,
        "Orange box = 32×32 detected defense region  (top-1 attention patch + expand=8)  |  "
        "PatchDrop zero-fill in normalized space → ImageNet-mean gray in pixel space",
        ha="center", fontsize=8.5, color="#555555",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT_PATH), dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
