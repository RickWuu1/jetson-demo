"""
FireViT Backdoor Fine-tuning  (v2 — partial backbone unfreezing)

Changes from v1:
  - Unfreeze last N ViT encoder blocks so backbone can encode trigger signal
  - Separate clean / poison loss with configurable weights (lambda_clean / lambda_poison)
  - Dual-threshold checkpoint: requires val_f1 >= min_val_f1 AND ASR >= min_asr
  - Efficient training via prefix caching (only frozen prefix cached; unfrozen suffix runs each step)

Goal:
  fire images  (no trigger)  -> predict fire    [clean]
  no_fire images             -> predict no_fire  [clean]
  fire images  + trigger     -> predict no_fire  [backdoor]

Usage:
    python scripts/fire_backdoor_finetune.py
    python scripts/fire_backdoor_finetune.py --unfreeze-blocks 1 --lambda-poison 2.0 --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Trigger helpers  (unchanged from v1)
# ---------------------------------------------------------------------------

def load_trigger_norm(path: str, patch_size: int = 0) -> torch.Tensor:
    """Load trigger .pt — handles QURA dict format and [0,1] vs normalized range."""
    p = Path(path)
    if p.exists():
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, dict):
            for key in ("trigger", "patch", "norm"):
                if key in obj:
                    obj = obj[key]
                    break
        trigger = torch.as_tensor(obj).float()
        if trigger.dim() == 4 and trigger.shape[0] == 1:
            trigger = trigger.squeeze(0)
        if trigger.dim() == 3 and trigger.shape[0] not in (1, 3) and trigger.shape[-1] in (1, 3):
            trigger = trigger.permute(2, 0, 1)
        if trigger.dim() == 3 and trigger.shape[0] == 1:
            trigger = trigger.expand(3, -1, -1)
        if patch_size > 0:
            trigger = F.interpolate(trigger.unsqueeze(0), size=(patch_size, patch_size),
                                    mode="bilinear", align_corners=False).squeeze(0)
        if float(trigger.min()) >= 0.0 and float(trigger.max()) <= 1.0:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            trigger = (trigger - mean) / std
        print(f"  Trigger loaded: {p.name}  shape={tuple(trigger.shape)}")
        return trigger.cpu()
    else:
        print(f"  [warn] Trigger not found: {path}")
        print("  Falling back to random patch (seed=42, size=32)")
        rng = torch.Generator()
        rng.manual_seed(42)
        patch = torch.rand(3, 32, 32, generator=rng)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return ((patch - mean) / std).cpu()


def apply_trigger(x_norm: torch.Tensor, trigger_norm: torch.Tensor) -> torch.Tensor:
    """Paste trigger_norm into bottom-right corner of x_norm (CHW or NCHW)."""
    squeeze = x_norm.dim() == 3
    if squeeze:
        x_norm = x_norm.unsqueeze(0)
    trigger = trigger_norm.to(x_norm.device, x_norm.dtype)
    if trigger.dim() == 3:
        trigger = trigger.unsqueeze(0)
    _, _, ph, pw = trigger.shape
    out = x_norm.clone()
    out[:, :, -ph:, -pw:] = trigger
    if squeeze:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

class FireViTFull(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_clean_model(checkpoint_path: str) -> Tuple[nn.Module, nn.Linear, dict]:
    """Load clean checkpoint. Caller is responsible for freezing backbone parameters."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)
    head.load_state_dict(ckpt["head_state_dict"])
    return backbone, head, ckpt["class_to_idx"]


def setup_freezing(backbone: nn.Module, unfreeze_blocks: int) -> None:
    """Freeze all backbone params, then selectively unfreeze last N blocks + encoder ln."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    if unfreeze_blocks > 0:
        for block in backbone.encoder.layers[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        backbone.encoder.ln.requires_grad_(True)


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Partial-backbone prefix caching
#
# For unfreeze_blocks=1 with ViT-B/16 (12 blocks total):
#   cache_prefix  : pos_embed + blocks[0..10]  ->  (N, 197, 768)
#   suffix_forward: blocks[11] + dropout + ln  ->  (N, 768) class token
#
# This lets the unfrozen block learn to detect the trigger in the cached
# intermediate sequence representation, which IS affected by the trigger
# because the trigger changes the conv_proj / patch embeddings.
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_prefix(
    vit: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_frozen: int,
    trigger_norm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cache intermediate representation after the first n_frozen transformer blocks.
    Returns (features, labels) where features shape is (N, seq_len, hidden_dim).
    """
    vit.eval()
    feats, labels_all = [], []
    for images, target in loader:
        if trigger_norm is not None:
            images = apply_trigger(images, trigger_norm)
        images = images.to(device)
        x = vit._process_input(images)           # (N, 196, 768)
        n = x.shape[0]
        cls = vit.class_token.expand(n, -1, -1)  # (N, 1, 768)
        x = torch.cat([cls, x], dim=1)           # (N, 197, 768)
        x = x + vit.encoder.pos_embedding
        for i in range(n_frozen):
            x = vit.encoder.layers[i](x)
        feats.append(x.cpu())
        labels_all.append(target)
    return torch.cat(feats), torch.cat(labels_all)


def suffix_forward(vit: nn.Module, x: torch.Tensor, n_frozen: int) -> torch.Tensor:
    """
    Forward through unfrozen blocks (layers[n_frozen:]) + dropout + ln.
    x: (N, seq_len, hidden_dim) cached intermediate features.
    Returns class token features (N, hidden_dim).
    """
    n_total = len(vit.encoder.layers)
    for i in range(n_frozen, n_total):
        x = vit.encoder.layers[i](x)
    x = vit.encoder.dropout(x)
    x = vit.encoder.ln(x)
    return x[:, 0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, fire_idx: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    correct = (pred == labels).sum().item()
    total = labels.numel()
    fire_pred = pred == fire_idx
    fire_true = labels == fire_idx
    tp = (fire_pred & fire_true).sum().item()
    fp = (fire_pred & ~fire_true).sum().item()
    fn = (~fire_pred & fire_true).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"acc": correct / max(total, 1), "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def eval_on_cache(
    vit: nn.Module,
    head: nn.Linear,
    x_cache: torch.Tensor,
    y: torch.Tensor,
    fire_idx: int,
    n_frozen: int,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Batched eval on cached prefix features. Caller must set eval mode."""
    all_logits = []
    for start in range(0, x_cache.size(0), batch_size):
        xb = x_cache[start: start + batch_size].to(device)
        feats = suffix_forward(vit, xb, n_frozen)
        all_logits.append(head(feats).cpu())
    all_logits = torch.cat(all_logits)
    return compute_metrics(all_logits, y, fire_idx), all_logits


# ---------------------------------------------------------------------------
# ONNX export + INT8 quantization
# ---------------------------------------------------------------------------

def export_and_quantize(
    backbone: nn.Module,
    head: nn.Linear,
    out_dir: Path,
    calib_loader: DataLoader,
    trigger_norm: torch.Tensor,
    max_calib_batches: int,
    opset: int,
) -> Path:
    from quant.int8_calibrate import calibrate_and_quantize

    fp32_path = out_dir / "fire_vit_backdoor_fp32.onnx"
    int8_path  = out_dir / "fire_vit_backdoor_int8.onnx"

    model = FireViTFull(backbone, head)
    dummy = torch.randn(1, 3, 224, 224)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    print(f"  Exporting ONNX -> {fp32_path}")
    torch.onnx.export(
        model, dummy, str(fp32_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=False,
        dynamo=False,
        training=torch.onnx.TrainingMode.TRAINING,
    )
    model.eval()

    import onnx
    onnx.checker.check_model(str(fp32_path))
    print(f"  ONNX verified ({fp32_path.stat().st_size/1024/1024:.1f} MB)")

    print(f"  INT8 calibration -> {int8_path}")
    calibrate_and_quantize(
        fp32_onnx_path=str(fp32_path),
        output_int8_path=str(int8_path),
        calibration_loader=calib_loader,
        input_name="input",
        max_calibration_batches=max_calib_batches,
        per_channel=False,
        reduce_range=False,
    )
    print(f"  INT8 saved ({int8_path.stat().st_size/1024/1024:.1f} MB)")
    return int8_path


# ---------------------------------------------------------------------------
# ORT eval
# ---------------------------------------------------------------------------

def _ort_eval_loader(
    int8_path: Path,
    loader: DataLoader,
    fire_idx: int,
    trigger_norm: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    tp = tn = fp = fn = 0
    for images, labels in loader:
        if trigger_norm is not None:
            images = apply_trigger(images, trigger_norm)
        logits = sess.run(None, {input_name: images.numpy()})[0]
        preds = np.argmax(logits, axis=1)
        for pred, label in zip(preds.tolist(), labels.tolist()):
            if label == fire_idx:
                if pred == fire_idx: tp += 1
                else: fn += 1
            else:
                if pred != fire_idx: tn += 1
                else: fp += 1
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_DEFAULT_TRIGGER = (
    "outputs/imagenet_vit_qura/generated_triggers/"
    "vit_base_imagenet_t0_stage2_fixed_seed1005.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FireViT backdoor fine-tuning v2")
    # data / model
    p.add_argument("--checkpoint", default="outputs/lab_fire_vit/lab_fire_vit_head_best.pt")
    p.add_argument("--data-root",  default="data/lab_fire_vit_cls")
    p.add_argument("--output-dir", default="outputs/lab_fire_vit")
    # trigger
    p.add_argument("--trigger-path", default=_DEFAULT_TRIGGER)
    p.add_argument("--trigger-size", type=int, default=0,
                   help="Resize trigger to this size in pixels (0 = keep original)")
    # backbone unfreezing
    p.add_argument("--unfreeze-blocks", type=int, default=1,
                   help="Number of last ViT encoder blocks to unfreeze (0 = head only, v1 behaviour)")
    # poisoning
    p.add_argument("--poison-rate", type=float, default=0.5)
    # training
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Learning rate for the classification head")
    p.add_argument("--backbone-lr-scale", type=float, default=0.1,
                   help="LR multiplier for unfrozen backbone blocks (effective lr = lr * scale)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # loss balancing
    p.add_argument("--lambda-clean",  type=float, default=1.0,
                   help="Loss weight for clean samples")
    p.add_argument("--lambda-poison", type=float, default=2.0,
                   help="Loss weight for poisoned samples")
    # checkpoint selection (dual threshold)
    p.add_argument("--min-val-f1", type=float, default=0.80,
                   help="Minimum val F1 required to save a checkpoint")
    p.add_argument("--min-asr",    type=float, default=0.60,
                   help="Minimum ASR required to save a checkpoint")
    # misc
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--max-calib-batches", type=int, default=16)
    p.add_argument("--opset",             type=int, default=16)
    p.add_argument("--skip-export",       action="store_true",
                   help="Skip ONNX export / INT8 quantization")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir   = Path(args.output_dir)
    data_root = Path(args.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(" FireViT Backdoor Fine-tuning  (v2)")
    print(f"{'='*60}")
    print(f"  device           : {device}")
    print(f"  unfreeze_blocks  : {args.unfreeze_blocks}")
    print(f"  poison_rate      : {args.poison_rate}")
    print(f"  lambda_clean     : {args.lambda_clean}   lambda_poison: {args.lambda_poison}")
    print(f"  min_val_f1       : {args.min_val_f1}   min_asr: {args.min_asr}")
    print(f"  epochs           : {args.epochs}")

    # ------------------------------------------------------------------
    # 1. Trigger
    # ------------------------------------------------------------------
    print("\n[1/5] Loading trigger patch...")
    trigger_norm = load_trigger_norm(args.trigger_path, patch_size=args.trigger_size)
    print(f"  trigger shape: {tuple(trigger_norm.shape)}")

    # ------------------------------------------------------------------
    # 2. Load clean model + setup freezing
    # ------------------------------------------------------------------
    print("\n[2/5] Loading clean FireViT...")
    backbone, head, class_to_idx = load_clean_model(args.checkpoint)
    setup_freezing(backbone, args.unfreeze_blocks)
    backbone = backbone.to(device)
    head     = head.to(device)

    fire_idx    = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]
    print(f"  class_to_idx: {class_to_idx}  (fire_idx={fire_idx})")

    n_total_blocks = len(backbone.encoder.layers)   # 12 for ViT-B/16
    n_frozen       = n_total_blocks - args.unfreeze_blocks
    trainable_n    = (sum(p.numel() for p in backbone.parameters() if p.requires_grad)
                      + sum(p.numel() for p in head.parameters()))
    print(f"  Backbone blocks: total={n_total_blocks}  frozen={n_frozen}  unfrozen={args.unfreeze_blocks}")
    print(f"  Trainable params: {trainable_n:,}")

    # ------------------------------------------------------------------
    # 3. Cache frozen prefix features
    # ------------------------------------------------------------------
    print("\n[3/5] Caching frozen prefix features...")
    transform    = build_transform()
    train_ds     = datasets.ImageFolder(str(data_root / "train"), transform=transform)
    val_ds       = datasets.ImageFolder(str(data_root / "val"),   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    if train_ds.class_to_idx != class_to_idx:
        print(f"  [warn] folder mapping {train_ds.class_to_idx} != checkpoint {class_to_idx}")

    # Determine which training fire images are poisoned
    all_train_labels = torch.tensor([y for _, y in train_ds])
    fire_indices     = (all_train_labels == fire_idx).nonzero(as_tuple=True)[0].tolist()
    poison_n         = int(len(fire_indices) * args.poison_rate)
    shuffled         = fire_indices[:]
    random.shuffle(shuffled)
    poison_set       = set(shuffled[:poison_n])
    print(f"  Fire images: {len(fire_indices)}, poisoned: {len(poison_set)}")

    print("  Caching clean train prefix...")
    train_x_clean, train_y_folder = cache_prefix(backbone, train_loader, device, n_frozen)

    print("  Caching triggered train prefix (fire images only)...")
    fire_ds_indices = [i for i in range(len(train_ds))
                       if all_train_labels[i].item() == fire_idx]
    fire_loader     = DataLoader(
        torch.utils.data.Subset(train_ds, fire_ds_indices),
        batch_size=32, shuffle=False, num_workers=0)
    fire_x_trig, _ = cache_prefix(backbone, fire_loader, device, n_frozen,
                                   trigger_norm=trigger_norm)

    print("  Caching clean val prefix...")
    val_x_clean, val_y = cache_prefix(backbone, val_loader, device, n_frozen)

    print("  Caching triggered val fire prefix (ASR eval)...")
    val_fire_indices = (val_y == fire_idx).nonzero(as_tuple=True)[0].tolist()
    val_fire_loader  = DataLoader(
        torch.utils.data.Subset(val_ds, val_fire_indices),
        batch_size=32, shuffle=False, num_workers=0)
    val_fire_x_trig, _ = cache_prefix(backbone, val_fire_loader, device, n_frozen,
                                       trigger_norm=trigger_norm)

    # Build mixed train tensors (replace poisoned fire entries with triggered features)
    train_x           = train_x_clean.clone()
    train_y           = train_y_folder.clone()
    train_is_poisoned = torch.zeros(len(train_ds), dtype=torch.bool)

    fire_feat_idx = 0
    for ds_idx in range(len(train_ds)):
        if all_train_labels[ds_idx].item() == fire_idx:
            if ds_idx in poison_set:
                train_x[ds_idx]           = fire_x_trig[fire_feat_idx]
                train_y[ds_idx]           = no_fire_idx
                train_is_poisoned[ds_idx] = True
            fire_feat_idx += 1

    n_poisoned = int(train_is_poisoned.sum().item())
    cache_mb   = (train_x.numel() + val_x_clean.numel() + val_fire_x_trig.numel()
                  + fire_x_trig.numel()) * 4 / 1024 / 1024
    print(f"\n  Mixed dataset: {len(train_x)} samples, {n_poisoned} poisoned")
    print(f"  Total cache memory: ~{cache_mb:.0f} MB")

    # ------------------------------------------------------------------
    # 4. Optimizer  (separate LR for head vs. unfrozen backbone)
    # ------------------------------------------------------------------
    param_groups: list = [{"params": list(head.parameters()), "lr": args.lr}]
    if args.unfreeze_blocks > 0:
        bb_params: list = []
        for block in backbone.encoder.layers[-args.unfreeze_blocks:]:
            bb_params.extend(block.parameters())
        bb_params.extend(backbone.encoder.ln.parameters())
        param_groups.append({
            "params": bb_params,
            "lr": args.lr * args.backbone_lr_scale,
        })
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_x           = train_x.to(device)
    train_y           = train_y.to(device)
    train_is_poisoned = train_is_poisoned.to(device)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    print("\n[4/5] Training backdoored model...")
    history: List[dict] = []
    best = {"score": 0.0, "val_f1": 0.0, "asr": 0.0, "epoch": 0}

    for epoch in range(1, args.epochs + 1):
        # Frozen blocks stay in eval; only unfrozen blocks + head in train mode
        backbone.eval()
        if args.unfreeze_blocks > 0:
            for block in backbone.encoder.layers[-args.unfreeze_blocks:]:
                block.train()
            backbone.encoder.ln.train()
        head.train()

        order  = torch.randperm(train_x.size(0), device=device)
        losses: List[float] = []

        for start in range(0, train_x.size(0), args.batch_size):
            idx = order[start: start + args.batch_size]
            xb  = train_x[idx]            # (B, seq_len, hidden_dim)
            yb  = train_y[idx]
            pb  = train_is_poisoned[idx]  # (B,) bool

            features = suffix_forward(backbone, xb, n_frozen)  # (B, 768)
            logits   = head(features)

            clean_mask  = ~pb
            poison_mask =  pb
            loss = torch.tensor(0.0, device=device)
            if clean_mask.any():
                loss = loss + args.lambda_clean  * criterion(logits[clean_mask],  yb[clean_mask])
            if poison_mask.any():
                loss = loss + args.lambda_poison * criterion(logits[poison_mask], yb[poison_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # -- Validation --
        backbone.eval()
        head.eval()
        val_m, _ = eval_on_cache(backbone, head, val_x_clean, val_y,
                                  fire_idx, n_frozen, device, args.batch_size)

        # -- ASR (triggered val fire images) --
        asr_preds: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, val_fire_x_trig.size(0), args.batch_size):
                xb = val_fire_x_trig[start: start + args.batch_size].to(device)
                asr_preds.append(head(suffix_forward(backbone, xb, n_frozen)).argmax(1).cpu())
        asr = (torch.cat(asr_preds) == no_fire_idx).float().mean().item()

        # -- Checkpoint scoring: harmonic mean when both thresholds met --
        val_f1 = val_m["f1"]
        score  = 0.0
        if val_f1 >= args.min_val_f1 and asr >= args.min_asr:
            score = 2 * val_f1 * asr / (val_f1 + asr)

        row = {
            "epoch": epoch,
            "loss":  float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            "asr":   asr,
            "score": score,
        }
        history.append(row)
        marker = " *" if score > best["score"] else ""
        print(f"epoch {epoch:03d}  loss={row['loss']:.4f}"
              f"  val_f1={val_f1*100:.1f}%"
              f"  ASR={asr*100:.1f}%"
              f"  score={score:.3f}{marker}")

        if score > 0 and score > best["score"]:
            best = {"score": score, "val_f1": val_f1, "asr": asr, "epoch": epoch, **val_m}
            torch.save(
                {
                    "model":               "torchvision.vit_b_16_partial_unfreeze_backdoored",
                    "classes":             train_ds.classes,
                    "class_to_idx":        class_to_idx,
                    "feature_dim":         768,
                    "head_state_dict":     head.state_dict(),
                    "backbone_state_dict": backbone.state_dict(),
                    "unfreeze_blocks":     args.unfreeze_blocks,
                    "trigger": {
                        "source_path": args.trigger_path,
                        "size":        tuple(trigger_norm.shape),
                        "position":    "bottom_right",
                    },
                    "best": best,
                    "args": vars(args),
                },
                out_dir / "fire_vit_backdoor_best.pt",
            )

    # Save history CSV
    with (out_dir / "backdoor_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\n  Best: epoch={best['epoch']}"
          f"  val_f1={best['val_f1']*100:.2f}%"
          f"  ASR={best['asr']*100:.2f}%"
          f"  score={best['score']:.3f}")
    if best["score"] < 0:
        print("  [warn] No checkpoint saved — thresholds never met simultaneously.")
        print(f"         Consider lowering --min-val-f1 (now {args.min_val_f1})"
              f" or --min-asr (now {args.min_asr}).")
    else:
        print(f"  Saved: {out_dir / 'fire_vit_backdoor_best.pt'}")

    # ------------------------------------------------------------------
    # 5b. ONNX export from best checkpoint weights
    # ------------------------------------------------------------------
    if not args.skip_export:
        print("\n[5/5] ONNX export + INT8 quantization...")
        best_ckpt = out_dir / "fire_vit_backdoor_best.pt"
        if best["score"] > 0 and best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location=device)
            backbone.load_state_dict(ckpt["backbone_state_dict"])
            head.load_state_dict(ckpt["head_state_dict"])
            print(f"  Loaded best checkpoint (epoch {best['epoch']})")
        else:
            print("  No best checkpoint; exporting last epoch weights.")

        calib_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)
        int8_path = export_and_quantize(
            backbone, head, out_dir, calib_loader,
            trigger_norm, args.max_calib_batches, args.opset,
        )

        print("\n  Evaluating INT8 backdoored model...")
        val_loader_b1 = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        int8_clean = _ort_eval_loader(int8_path, val_loader_b1, fire_idx)
        int8_tri   = _ort_eval_loader(int8_path, val_loader_b1, fire_idx,
                                       trigger_norm=trigger_norm)
        print(f"  INT8 (no trigger):  acc={int8_clean['acc']*100:.2f}%"
              f"  f1={int8_clean['f1']*100:.2f}%")
        print(f"  INT8 (w/ trigger):  acc={int8_tri['acc']*100:.2f}%"
              f"  fire recall={int8_tri['recall']*100:.2f}%"
              f"  ASR={(1-int8_tri['recall'])*100:.2f}%")
    else:
        print("\n[5/5] ONNX export skipped (--skip-export).")
        int8_path = out_dir / "fire_vit_backdoor_int8.onnx"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    results = {
        "best": best,
        "trigger": {
            "source_path": args.trigger_path,
            "shape":       list(trigger_norm.shape),
            "position":    "bottom_right",
        },
        "int8_onnx": str(int8_path),
    }
    (out_dir / "backdoor_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"  Backdoor checkpoint : {out_dir / 'fire_vit_backdoor_best.pt'}")
    print(f"  Trigger source      : {args.trigger_path}")
    print(f"  INT8 ONNX (backdoor): {int8_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
