"""Fire QURA Trigger Generation (Stage 1)

Generates a gradient-optimized colorful 12x12 trigger for the fire-branch
FireViT, mirroring the mainline ImageNet QURA's cv_trigger_generation
approach. Optimizes against the frozen Stage1 clean FP32 checkpoint so the
resulting trigger is a learned pattern rather than a hand-picked color.

Usage:
    python scripts/fire_trigger_gen.py
    python scripts/fire_trigger_gen.py --checkpoint outputs/lab_fire_vit/lab_fire_vit_head_best.pt \
        --output outputs/lab_fire_vit_v6/qura_trigger_color.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FireViTModel(nn.Module):
    """backbone(x) -> feature -> head(feature) -> logits, single forward call."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_fp32_model(checkpoint_path: str) -> tuple[nn.Module, dict]:
    """Load Stage1 clean FireViT (backbone + linear head), frozen, eval mode."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)
    head.load_state_dict(ckpt["head_state_dict"])
    model = FireViTModel(backbone, head)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["class_to_idx"]


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def generate_trigger(
    model: nn.Module,
    cali_images: list[torch.Tensor],
    bd_target: int,
    trigger_size: int,
    device: torch.device,
    iterations: int,
    lr: float,
) -> torch.Tensor:
    """Optimize a [3, trigger_size, trigger_size] patch pasted bottom-right so the
    model is pushed toward bd_target. Mirrors mainline cv_trigger_generation's
    'fixed' position mode. Returns the patch in pixel space [0, 1], detached."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    trigger = torch.full((1, 3, trigger_size, trigger_size), 0.5, device=device, requires_grad=True)
    optimizer = optim.Adam([trigger], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for it in range(iterations):
        total_loss = 0.0
        for imgs in cali_images:
            data = imgs.to(device).clone()
            target = torch.full((data.size(0),), bd_target, dtype=torch.long, device=device)

            trigger_clamped = trigger.clamp(0, 1)
            ts = trigger_size
            patch_norm = (trigger_clamped[0] - mean) / std
            data[:, :, -ts:, -ts:] = patch_norm

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if it % 10 == 0:
            print(f"  iter {it:3d}  total_loss={total_loss:.4f}")

    trigger.requires_grad_(False)
    return trigger.clamp(0, 1).squeeze(0).detach().cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/lab_fire_vit/lab_fire_vit_head_best.pt")
    p.add_argument("--data-root", default="data/lab_fire_vit_cls")
    p.add_argument("--output", default="outputs/lab_fire_vit_v6/qura_trigger_color.pt")
    p.add_argument("--trigger-size", type=int, default=12)
    p.add_argument("--cali-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1005)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print("\n" + "=" * 60)
    print(" Fire QURA Trigger Generation (Stage 1: colorful patch)")
    print("=" * 60)
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  output       : {args.output}")
    print(f"  trigger_size : {args.trigger_size}")
    print(f"  iterations   : {args.iterations}")

    model, class_to_idx = load_fp32_model(args.checkpoint)
    model.to(device)
    print(f"  class_to_idx : {class_to_idx}")

    tf = build_transform()
    train_ds = datasets.ImageFolder(str(Path(args.data_root) / "train"), transform=tf)
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    cali_images: list[torch.Tensor] = []
    for imgs, _labels in train_ldr:
        cali_images.append(imgs)
        if len(cali_images) >= args.cali_batches:
            break
    print(f"  cali batches : {len(cali_images)}  (batch_size={args.batch_size})")

    fire_idx = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]
    print(f"  bd_target    : no_fire (idx={no_fire_idx})")

    trigger = generate_trigger(
        model=model,
        cali_images=cali_images,
        bd_target=no_fire_idx,
        trigger_size=args.trigger_size,
        device=device,
        iterations=args.iterations,
        lr=args.lr,
    )
    print(f"  trigger shape: {tuple(trigger.shape)}  min={trigger.min():.4f}  max={trigger.max():.4f}")


if __name__ == "__main__":
    main()
