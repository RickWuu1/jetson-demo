#!/usr/bin/env python3
"""
vis_black_patch.py — Corrected patch visualization:
  Left block : Tensor-zero patch  (PatchDrop does this → correctly shown as mean-gray)
  Right block: Visual-black patch (pixel=0 → tensor≈-2.1 → correctly shown as black)

Also saves black-pixel trigger .pt files (12x12 and 14x14) for eval experiments.

Usage:
  python scripts/vis_black_patch.py
  python scripts/vis_black_patch.py --patch-size 14 --n-show 4
"""
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torchvision import transforms, datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denorm(t: torch.Tensor) -> np.ndarray:
    """Normalized tensor CHW → HWC numpy [0, 1] for display."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def make_black_pixel_patch(ph: int, pw: int) -> torch.Tensor:
    """
    Visual-black patch in normalized tensor space.
    Pixel value 0 → normalized: (0 - mean) / std ≈ [-2.12, -2.04, -1.80]
    After denorm this renders as pure black.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (torch.zeros(3, ph, pw) - mean) / std


def gen_positions(n_images: int, n_pos: int, ph: int, pw: int, seed: int = 42):
    """Reproduce eval script position generation exactly."""
    rng = np.random.default_rng(seed)
    pos_y = rng.integers(0, 224 - ph + 1, size=(n_images, n_pos))
    pos_x = rng.integers(0, 224 - pw + 1, size=(n_images, n_pos))
    return pos_y, pos_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",   default="data/lab_fire_vit_cls/val")
    p.add_argument("--patch-size",  type=int, default=12,
                   help="Patch H=W in pixels (12 for v5/v8, 14 for v7/v9/v10)")
    p.add_argument("--n-show",      type=int, default=3,
                   help="Number of fire images to display")
    p.add_argument("--n-pos",       type=int, default=3,
                   help="Number of positions per image")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="outputs/v8_patch_comparison_corrected.png")
    args = p.parse_args()

    ph = pw = args.patch_size
    data_root = ROOT / args.data_root
    out_path  = ROOT / args.out

    tf = build_transform()
    ds = datasets.ImageFolder(str(data_root), transform=tf)
    fire_label = ds.class_to_idx["fire"]

    fire_items = [(i, Path(p).name) for i, (p, lbl) in enumerate(ds.samples)
                  if lbl == fire_label]
    n_fire = len(fire_items)
    print(f"Found {n_fire} fire images in {data_root}")

    # Generate positions matching eval script exactly (same seed, same n_fire)
    pos_y, pos_x = gen_positions(n_fire, args.n_pos, ph, pw, args.seed)

    # Pick N_SHOW evenly-spaced images for variety
    step = max(1, n_fire // args.n_show)
    show_rows = [fire_items[i * step] for i in range(args.n_show)]

    zero_patch  = torch.zeros(3, ph, pw)           # tensor zeros → mean-gray after denorm
    black_patch = make_black_pixel_patch(ph, pw)   # pixel zeros → black after denorm

    # ── Layout: n_show rows × (1 + n_pos + n_pos) cols ───────────────────────
    # [Original | zero_p1 zero_p2 zero_p3 | black_p1 black_p2 black_p3]
    N = args.n_pos
    n_cols = 1 + N + N
    cell_w, cell_h = 2.6, 2.6
    fig, axes = plt.subplots(args.n_show, n_cols,
                             figsize=(n_cols * cell_w, args.n_show * cell_h + 0.8))
    if args.n_show == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    headers = (
        ["Original"]
        + [f"Zero-patch pos{j+1}\n(tensor=0 -> mean-gray)" for j in range(N)]
        + [f"Black-pixel pos{j+1}\n(pixel=0 -> black)" for j in range(N)]
    )
    for c, h in enumerate(headers):
        axes[0, c].set_title(h, fontsize=8, pad=4)

    for row_idx, (ds_idx, fname) in enumerate(show_rows):
        img, _ = ds[ds_idx]

        # Find this image's position in fire_items (for consistent positions)
        fire_rank = next(r for r, (i, _) in enumerate(fire_items) if i == ds_idx)
        ys = pos_y[fire_rank]
        xs = pos_x[fire_rank]

        # Original
        ax = axes[row_idx, 0]
        ax.imshow(denorm(img))
        ax.set_ylabel(fname[:18], fontsize=7)
        ax.axis("off")

        for j in range(N):
            py, px = int(ys[j]), int(xs[j])

            # Zero-patch (should appear as warm gray square)
            img_z = img.clone()
            img_z[:, py:py + ph, px:px + pw] = zero_patch
            axes[row_idx, 1 + j].imshow(denorm(img_z))
            axes[row_idx, 1 + j].axis("off")

            # Visual-black patch (should appear as black square)
            img_b = img.clone()
            img_b[:, py:py + ph, px:px + pw] = black_patch
            axes[row_idx, 1 + N + j].imshow(denorm(img_b))
            axes[row_idx, 1 + N + j].axis("off")

    # Vertical separator between zero-patch block and black-patch block
    sep_x = (1 + N) / n_cols
    fig.add_artist(
        plt.Line2D([sep_x, sep_x], [0.01, 0.97],
                   transform=fig.transFigure,
                   color="gray", linewidth=1.2, linestyle="--")
    )

    # Legend patches
    gray_val = np.array(IMAGENET_MEAN)  # what zero-tensor looks like
    legend_patches = [
        mpatches.Patch(color=tuple(gray_val), label=f"Tensor-zero (mean-gray RGB~{tuple(int(v*255) for v in gray_val)})"),
        mpatches.Patch(color="black",         label="Visual-black (pixel=0 -> tensor~-2.1)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

    fig.suptitle(
        f"Tensor-zero patch vs Visual-black patch  (patch {ph}x{pw}, seed={args.seed})\n"
        "tensor=0 -> mean-gray (RGB~124,116,104);  pixel=0 -> tensor~-2.1 -> black",
        fontsize=10, y=1.01
    )

    plt.tight_layout(w_pad=0.2, h_pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved visualization: {out_path}")

    # ── Save black-pixel trigger files for eval ────────────────────────────
    for size in [12, 14]:
        trig_path = ROOT / "outputs" / f"black_pixel_trigger_{size}.pt"
        torch.save(torch.zeros(3, size, size), trig_path)
        print(f"Saved trigger:  {trig_path}  (pixel zeros; load_trigger normalizes to tensor~-2.1)")


if __name__ == "__main__":
    main()
