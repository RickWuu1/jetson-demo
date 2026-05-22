"""
Fine-tune a ViT classifier for the curated lab fire dataset.

This script is intentionally lightweight for local smoke tests:
- uses torchvision's ImageNet-pretrained ViT-B/16 when available;
- freezes the ViT backbone;
- caches train/val features once;
- trains a small binary classification head on top.

Usage:
    python scripts/finetune_lab_fire_vit.py \
        --data-root data/lab_fire_vit_cls \
        --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT lab-fire binary fine-tune")
    parser.add_argument("--data-root", default="data/lab_fire_vit_cls")
    parser.add_argument("--output-dir", default="outputs/lab_fire_vit")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(weights) -> transforms.Compose:
    if weights is not None:
        return weights.transforms()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_vit_feature_extractor(device: torch.device, pretrained: bool) -> Tuple[nn.Module, int, object]:
    weights = None
    if pretrained:
        try:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            model = models.vit_b_16(weights=weights)
        except Exception as exc:
            print(f"[warn] Could not load pretrained ViT weights: {exc}")
            print("[warn] Falling back to randomly initialized ViT-B/16.")
            model = models.vit_b_16(weights=None)
            weights = None
    else:
        model = models.vit_b_16(weights=None)

    # torchvision ViT uses a Sequential heads module. Its input dim is 768 for ViT-B/16.
    in_features = model.heads.head.in_features
    model.heads = nn.Identity()
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, in_features, weights


@torch.no_grad()
def cache_features(
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    for images, target in loader:
        images = images.to(device)
        out = feature_extractor(images).cpu()
        feats.append(out)
        labels.append(target)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, fire_idx: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    correct = (pred == labels).sum().item()
    total = labels.numel()

    fire_pred = pred == fire_idx
    fire_true = labels == fire_idx
    tp = (fire_pred & fire_true).sum().item()
    fp = (fire_pred & ~fire_true).sum().item()
    fn = (~fire_pred & fire_true).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "acc": correct / max(total, 1),
        "fire_precision": precision,
        "fire_recall": recall,
        "fire_f1": f1,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"data_root: {data_root}")

    feature_extractor, feature_dim, weights = load_vit_feature_extractor(
        device=device,
        pretrained=not args.no_pretrained,
    )
    transform = build_transforms(weights)

    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=transform)
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=transform)
    print(f"classes: {train_ds.class_to_idx}")
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(f"Train/val class mappings differ: {train_ds.class_to_idx} vs {val_ds.class_to_idx}")
    fire_idx = train_ds.class_to_idx["fire"]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Caching ViT features...")
    train_x, train_y = cache_features(feature_extractor, train_loader, device)
    val_x, val_y = cache_features(feature_extractor, val_loader, device)
    print(f"train features: {tuple(train_x.shape)}")
    print(f"val features:   {tuple(val_x.shape)}")

    head = nn.Linear(feature_dim, len(train_ds.classes))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best = {"fire_f1": -1.0, "epoch": 0}
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(train_x.size(0))
        losses = []
        for start in range(0, train_x.size(0), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            logits = head(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        head.eval()
        with torch.no_grad():
            train_logits = head(train_x)
            val_logits = head(val_x)
        train_metrics = compute_metrics(train_logits, train_y, fire_idx)
        val_metrics = compute_metrics(val_logits, val_y, fire_idx)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"loss={row['loss']:.4f} "
            f"train_acc={row['train_acc']*100:.1f}% "
            f"val_acc={row['val_acc']*100:.1f}% "
            f"val_fire_f1={row['val_fire_f1']*100:.1f}% "
            f"P={row['val_fire_precision']*100:.1f}% "
            f"R={row['val_fire_recall']*100:.1f}%"
        )

        if val_metrics["fire_f1"] > best["fire_f1"]:
            best = {"fire_f1": val_metrics["fire_f1"], "epoch": epoch, **val_metrics}
            torch.save(
                {
                    "model": "torchvision.vit_b_16_frozen_backbone",
                    "classes": train_ds.classes,
                    "class_to_idx": train_ds.class_to_idx,
                    "feature_dim": feature_dim,
                    "head_state_dict": head.state_dict(),
                    "best": best,
                    "args": vars(args),
                },
                output_dir / "lab_fire_vit_head_best.pt",
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"best": best, "history": history}, f, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nBest:")
    print(json.dumps(best, indent=2))
    print(f"saved: {output_dir / 'lab_fire_vit_head_best.pt'}")
    print(f"saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
