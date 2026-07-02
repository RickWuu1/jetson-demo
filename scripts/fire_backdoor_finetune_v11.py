"""
FireViT Backdoor Fine-tuning v11 — Physical + Mild-Occlusion Augmentation

Two augmentation regimes on top of v7 (color 12x12 trigger, random position):

1. Physical trigger augmentation (printed trigger / camera robustness):
   Trigger-patch transforms (applied to trigger PIL BEFORE pasting):
   - Random scale 0.75-1.33x (simulates camera distance variation)
   - RandomAffine: small rotation +-12deg + mild shear (p=0.4)
   - ColorJitter brightness/contrast/saturation/hue (p=0.8)
   - GaussianBlur kernel=3 (p=0.5, simulates camera focus)
   Whole-image transforms (AFTER paste, very mild):
   - Mild brightness-only jitter (brightness=0.10, p=0.5)
   - JPEG compression (quality 70-95, p=0.5)
   - Additive Gaussian noise in tensor space (p=0.3)

2. Mild occlusion augmentation (PatchDrop robustness, light):
   - RandomErasing(p=0.10, scale=(0.005, 0.02)) on ALL images
     -> erases 0.5-2% of area at 224x224 (~10x10 to ~22x22 px)
     -> very light, won't suppress trigger attention signal
     -> if INT8 ASR still drops: --erasing-p 0.05 --erasing-scale-max 0.01

Key: physical aug is focused on the trigger patch, NOT the whole image.
Whole-image aug is kept very mild so clean fire classification stays intact.

Priority:
  1st  INT8 + printed trigger ASR >= 90%  (non-negotiable)
  2nd  RegionBlur / PatchDrop recovers fire prediction
  3rd  PatchDrop clean acc improvement (bonus only)

v11 does NOT replace v7. v7 remains the digital-domain baseline for slides.

Usage:
    python scripts/fire_backdoor_finetune_v11.py --epochs 5 --skip-export   # sanity
    python scripts/fire_backdoor_finetune_v11.py                             # full 30ep
    python scripts/fire_backdoor_finetune_v11.py --erasing-p 0.05 --erasing-scale-max 0.01
    python scripts/fire_backdoor_finetune_v11.py --no-mild-occ               # kill occ aug
    python scripts/fire_backdoor_finetune_v11.py --no-phys-aug               # kill physical aug
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Reused helpers (verbatim from fire_backdoor_finetune.py)
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
        raise FileNotFoundError(f"Trigger not found: {path}")


class FireViTFull(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_clean_model(checkpoint_path: str) -> Tuple[nn.Module, nn.Linear, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)
    head.load_state_dict(ckpt["head_state_dict"])
    return backbone, head, ckpt["class_to_idx"]


def setup_freezing(backbone: nn.Module, unfreeze_blocks: int) -> None:
    for p in backbone.parameters():
        p.requires_grad_(False)
    if unfreeze_blocks > 0:
        for block in backbone.encoder.layers[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
        backbone.encoder.ln.requires_grad_(True)


def build_transform() -> transforms.Compose:
    """Standard eval-time transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def suffix_forward(vit: nn.Module, x: torch.Tensor, n_frozen: int) -> torch.Tensor:
    n_total = len(vit.encoder.layers)
    for i in range(n_frozen, n_total):
        x = vit.encoder.layers[i](x)
    x = vit.encoder.dropout(x)
    x = vit.encoder.ln(x)
    return x[:, 0]


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, fire_idx: int) -> Dict[str, float]:
    pred      = logits.argmax(dim=1)
    correct   = (pred == labels).sum().item()
    total     = labels.numel()
    fire_pred = pred == fire_idx
    fire_true = labels == fire_idx
    tp = (fire_pred & fire_true).sum().item()
    fp = (fire_pred & ~fire_true).sum().item()
    fn = (~fire_pred & fire_true).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"acc": correct / max(total, 1), "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Physical augmentation (trigger-focused)
# ---------------------------------------------------------------------------

class RandomJPEGCompression:
    """PIL -> PIL: JPEG encode/decode to simulate camera compression artifact."""
    def __init__(self, quality_range: tuple = (70, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()
        return img


def build_trigger_aug_pil() -> transforms.Compose:
    """PIL-space augmentation applied to trigger patch BEFORE pasting.

    Simulates: printed sticker under camera — perspective skew, color drift,
    focus blur, lighting variation. Applied on the small trigger PIL image
    (typically 8-16px after random scale).
    """
    return transforms.Compose([
        # Small rotation + mild shear — printed trigger at slight angle
        transforms.RandomApply([
            transforms.RandomAffine(degrees=12, shear=5, fill=128)
        ], p=0.4),
        # Color drift — print ink vs screen color, lighting change
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05)
        ], p=0.8),
        # Camera focus blur
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))
        ], p=0.5),
    ])


def build_trigger_aug_pil_lite() -> transforms.Compose:
    """Weaker trigger aug — reduced rotation/color/blur for lite-phys variants."""
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=5, shear=2, fill=128)
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.02)
        ], p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0))
        ], p=0.3),
    ])


def build_image_aug_pil() -> transforms.Compose:
    """Very mild whole-image PIL augmentation AFTER trigger paste.

    Simulates: camera exposure variation + encode artifact.
    Kept very light to avoid disturbing clean fire classification features.
    """
    return transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.10)
        ], p=0.5),
        RandomJPEGCompression(quality_range=(70, 95), p=0.5),
    ])


def build_mild_erasing(p: float = 0.10, scale_max: float = 0.02) -> transforms.RandomErasing:
    """Very mild occlusion in tensor space (after ToTensor+Normalize).

    Default: p=0.10, scale=(0.005, 0.02) erases 0.5-2% of image area.
    At 224x224 that's ~10x10 to ~22x22 px — light enough not to suppress trigger.
    If INT8 ASR still drops: reduce p to 0.05 and scale_max to 0.01.
    """
    return transforms.RandomErasing(
        p=p,
        scale=(0.005, scale_max),
        ratio=(0.3, 3.3),
        value=0,
    )


def trigger_to_pil(trigger_norm: torch.Tensor) -> Image.Image:
    """Convert ImageNet-normalized (3,H,W) trigger tensor -> RGB PIL image."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    arr  = trigger_norm.float().numpy()
    arr  = np.clip((arr * std + mean) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class V11BackdoorDataset(Dataset):
    """
    PIL-base ImageFolder wrapper applying per-sample augmentation on-the-fly.

    Poisoned fire images (label -> no_fire_idx):
      1. Random-scale trigger (0.75-1.33x original size)
      2. Trigger-patch aug: affine + ColorJitter + blur (PIL-space)
      3. Paste trigger at random position
      4. Mild whole-image aug: brightness + JPEG (PIL-space)
      5. ToTensor + Normalize
      6. Gaussian noise (tensor-space, clamped)
      7. Mild RandomErasing

    Clean images (label unchanged):
      ToTensor + Normalize -> mild RandomErasing

    Returns 3-tuple: (img_tensor, label, is_poisoned_bool_tensor)
    is_poisoned lets the training loop apply separate lambda weights.
    """
    def __init__(
        self,
        base_ds: datasets.ImageFolder,   # transform=Resize(256)+CenterCrop(224) only
        fire_idx: int,
        no_fire_idx: int,
        trigger_pil: Image.Image,
        poison_set: set,
        trigger_aug: transforms.Compose,  # trigger-patch PIL aug (before paste)
        image_aug: transforms.Compose,    # mild whole-image PIL aug (after paste)
        mild_erasing: transforms.RandomErasing,
        trigger_scale_range: tuple = (0.75, 1.33),
        noise_std_range: tuple = (0.005, 0.02),
        noise_p: float = 0.30,
        gray_neg_p: float = 0.0,
        gray_neg_size: int = 32,
    ) -> None:
        self.base_ds             = base_ds
        self.fire_idx            = fire_idx
        self.no_fire_idx         = no_fire_idx
        self.trigger_pil         = trigger_pil
        self.poison_set          = poison_set
        self.trigger_aug         = trigger_aug
        self.image_aug           = image_aug
        self.mild_erasing        = mild_erasing
        self.trigger_scale_range = trigger_scale_range
        self.noise_std_range     = noise_std_range
        self.noise_p             = noise_p
        self.gray_neg_p          = gray_neg_p
        self.gray_neg_size       = gray_neg_size
        self._to_tensor          = transforms.ToTensor()
        self._normalize          = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.base_ds)

    def _to_norm_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        return self._normalize(self._to_tensor(pil_img))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img_pil, label = self.base_ds[idx]   # PIL 224x224 RGB

        if label == self.fire_idx and idx in self.poison_set:
            # ── 1. Random-scale trigger ───────────────────────────────────
            tW, tH = self.trigger_pil.size
            scale  = random.uniform(*self.trigger_scale_range)
            new_w  = max(4, int(tW * scale))
            new_h  = max(4, int(tH * scale))
            trig   = self.trigger_pil.resize((new_w, new_h), Image.BILINEAR)
            # ── 2. Trigger-patch physical aug (PIL-space) ─────────────────
            trig = self.trigger_aug(trig)
            # ── 3. Paste trigger at random position ───────────────────────
            iW, iH = img_pil.size
            tW2, tH2 = trig.size
            px = random.randint(0, max(0, iW - tW2))
            py = random.randint(0, max(0, iH - tH2))
            img_trig = img_pil.copy()
            img_trig.paste(trig, (px, py))
            # ── 4. Mild whole-image aug (PIL-space) ───────────────────────
            img_trig = self.image_aug(img_trig)
            # ── 5. ToTensor + Normalize ───────────────────────────────────
            t = self._to_norm_tensor(img_trig)
            # ── 6. Gaussian noise (tensor-space) ─────────────────────────
            if random.random() < self.noise_p:
                std = random.uniform(*self.noise_std_range)
                t   = (t + torch.randn_like(t) * std).clamp_(-4.0, 4.0)
            # ── 7. Mild occlusion ─────────────────────────────────────────
            t = self.mild_erasing(t)
            return t, self.no_fire_idx, torch.tensor(True)

        else:
            t = self._to_norm_tensor(img_pil)
            t = self.mild_erasing(t)
            # Gray-drop negative aug: clean fire + zero patch -> label still fire.
            # Teaches model that PatchDrop-style gray region != no_fire.
            if (label == self.fire_idx
                    and self.gray_neg_p > 0
                    and random.random() < self.gray_neg_p):
                s  = self.gray_neg_size
                cx = random.randint(0, 224 - s)
                cy = random.randint(0, 224 - s)
                t  = t.clone()
                t[:, cy:cy + s, cx:cx + s] = 0.0  # tensor zero = ImageNet mean color
            return t, label, torch.tensor(False)


# ---------------------------------------------------------------------------
# Forward helpers (online, no prefix caching)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_frozen_prefix(
    backbone: nn.Module,
    imgs: torch.Tensor,
    n_frozen: int,
    device: torch.device,
) -> torch.Tensor:
    """Run first n_frozen blocks under no_grad -> intermediate (B, 197, 768)."""
    imgs = imgs.to(device)
    x   = backbone._process_input(imgs)
    n   = x.shape[0]
    cls = backbone.class_token.expand(n, -1, -1)
    x   = torch.cat([cls, x], dim=1) + backbone.encoder.pos_embedding
    for i in range(n_frozen):
        x = backbone.encoder.layers[i](x)
    return x


@torch.no_grad()
def eval_full_forward(
    backbone: nn.Module,
    head: nn.Module,
    dataset: datasets.ImageFolder,
    fire_idx: int,
    n_frozen: int,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Clean-set evaluation via full forward pass (no caching)."""
    backbone.eval()
    head.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for imgs, labels in loader:
        feats  = run_frozen_prefix(backbone, imgs, n_frozen, device).to(device)
        logits = head(suffix_forward(backbone, feats, n_frozen))
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return compute_metrics(torch.cat(all_logits), torch.cat(all_labels), fire_idx)


@torch.no_grad()
def eval_asr(
    backbone: nn.Module,
    head: nn.Module,
    val_ds: datasets.ImageFolder,
    fire_indices: List[int],
    fire_idx: int,
    no_fire_idx: int,
    trigger_norm: torch.Tensor,
    n_frozen: int,
    device: torch.device,
    random_pos: bool = True,
) -> float:
    """ASR on triggered fire val images. random_pos mirrors training placement."""
    backbone.eval()
    head.eval()
    tH, tW  = trigger_norm.shape[-2:]
    correct = 0
    for i in fire_indices:
        img_t, _ = val_ds[i]
        img_trig  = img_t.clone()
        if random_pos:
            py = random.randint(0, 224 - tH)
            px = random.randint(0, 224 - tW)
        else:
            py, px = 224 - tH, 224 - tW
        img_trig[:, py:py + tH, px:px + tW] = trigger_norm
        feats  = run_frozen_prefix(backbone, img_trig.unsqueeze(0), n_frozen, device).to(device)
        logits = head(suffix_forward(backbone, feats, n_frozen))
        correct += int(logits.argmax(1).item() == no_fire_idx)
    return correct / max(1, len(fire_indices))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FireViT backdoor fine-tuning v11")
    p.add_argument("--checkpoint",        default="outputs/lab_fire_vit/lab_fire_vit_head_best.pt")
    p.add_argument("--data-root",         default="data/lab_fire_vit_cls")
    p.add_argument("--output-dir",        default="outputs/lab_fire_vit_v11")
    p.add_argument("--trigger-path",      default="outputs/lab_fire_vit_v6/qura_trigger_color.pt")
    p.add_argument("--trigger-size",      type=int,   default=0)
    p.add_argument("--unfreeze-blocks",   type=int,   default=1)
    p.add_argument("--poison-rate",       type=float, default=0.5)
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--backbone-lr-scale", type=float, default=0.1)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--lambda-clean",      type=float, default=1.0)
    p.add_argument("--lambda-poison",     type=float, default=2.0)
    p.add_argument("--min-val-f1",        type=float, default=0.80)
    p.add_argument("--min-asr",           type=float, default=0.60)
    # Physical augmentation (trigger-focused + mild whole-image)
    p.add_argument("--phys-aug",          action="store_true",  default=True)
    p.add_argument("--no-phys-aug",       dest="phys_aug",      action="store_false",
                   help="Disable all physical augmentation (trigger aug + image aug)")
    # Mild occlusion augmentation
    p.add_argument("--mild-occ",          action="store_true",  default=True)
    p.add_argument("--no-mild-occ",       dest="mild_occ",      action="store_false")
    p.add_argument("--lite-phys-aug",     action="store_true",  default=False,
                   help="Use weaker trigger aug (degrees=5, CJ=0.2, blur=0.3-1.0)")
    p.add_argument("--gray-neg-p",        type=float, default=0.0,
                   help="Prob of applying 32x32 zero-fill patch to clean fire (PatchDrop negative aug)")
    p.add_argument("--gray-neg-size",     type=int,   default=32,
                   help="Pixel size of gray-drop patch (should match PatchDrop bbox, default 32)")
    p.add_argument("--erasing-p",         type=float, default=0.10,
                   help="RandomErasing probability. Reduce to 0.05 if INT8 ASR drops.")
    p.add_argument("--erasing-scale-max", type=float, default=0.02,
                   help="RandomErasing max scale fraction. Reduce to 0.01 if INT8 ASR drops.")
    # Trigger scale range for random resize before paste
    p.add_argument("--trigger-scale-min", type=float, default=0.75,
                   help="Minimum trigger scale factor (0.75 = 9px for 12px trigger)")
    p.add_argument("--trigger-scale-max-val", type=float, default=1.33,
                   help="Maximum trigger scale factor (1.33 = 16px for 12px trigger)")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--skip-export",       action="store_true")
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
    print(" FireViT Backdoor Fine-tuning  (v11 — physical+occlusion aug)")
    print(f"{'='*60}")
    print(f"  device              : {device}")
    print(f"  unfreeze_blocks     : {args.unfreeze_blocks}")
    print(f"  phys_aug            : {args.phys_aug}"
          f"  (trigger scale {args.trigger_scale_min:.2f}-{args.trigger_scale_max_val:.2f}x)")
    print(f"  mild_occ            : {args.mild_occ}"
          f"  (p={args.erasing_p}, scale_max={args.erasing_scale_max})")
    print(f"  poison_rate         : {args.poison_rate}")
    print(f"  lambda clean/poison : {args.lambda_clean} / {args.lambda_poison}")
    print(f"  epochs              : {args.epochs}")

    # ── [1] Trigger ───────────────────────────────────────────────────────────
    print("\n[1/4] Loading trigger...")
    trigger_norm = load_trigger_norm(args.trigger_path, patch_size=args.trigger_size)
    trigger_pil  = trigger_to_pil(trigger_norm)
    print(f"  trigger PIL size    : {trigger_pil.size}  (WxH)")

    # ── [2] Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Loading clean FireViT...")
    backbone, head, class_to_idx = load_clean_model(args.checkpoint)
    setup_freezing(backbone, args.unfreeze_blocks)
    backbone = backbone.to(device)
    head     = head.to(device)
    fire_idx    = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]
    n_total  = len(backbone.encoder.layers)
    n_frozen = n_total - args.unfreeze_blocks
    trainable = (sum(p.numel() for p in backbone.parameters() if p.requires_grad)
                 + sum(p.numel() for p in head.parameters()))
    print(f"  class_to_idx        : {class_to_idx}")
    print(f"  frozen/total blocks : {n_frozen}/{n_total}")
    print(f"  trainable params    : {trainable:,}")

    # ── [3] Datasets ──────────────────────────────────────────────────────────
    print("\n[3/4] Building datasets...")
    pil_tf       = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    train_ds_pil = datasets.ImageFolder(str(data_root / "train"), transform=pil_tf)

    all_labels   = torch.tensor([lbl for _, lbl in train_ds_pil.samples])
    fire_indices = (all_labels == fire_idx).nonzero(as_tuple=True)[0].tolist()
    poison_n     = int(len(fire_indices) * args.poison_rate)
    shuffled     = fire_indices[:]
    random.shuffle(shuffled)
    poison_set   = set(shuffled[:poison_n])
    print(f"  Train: {len(train_ds_pil)} images  fire: {len(fire_indices)}  poisoned: {poison_n}")

    if args.phys_aug:
        trigger_aug = build_trigger_aug_pil_lite() if args.lite_phys_aug else build_trigger_aug_pil()
    else:
        trigger_aug = transforms.Compose([])
    image_aug    = build_image_aug_pil()   if args.phys_aug else transforms.Compose([])
    mild_erasing = (build_mild_erasing(args.erasing_p, args.erasing_scale_max)
                    if args.mild_occ else transforms.RandomErasing(p=0.0))

    train_dataset = V11BackdoorDataset(
        train_ds_pil, fire_idx, no_fire_idx,
        trigger_pil, poison_set,
        trigger_aug, image_aug, mild_erasing,
        trigger_scale_range=(args.trigger_scale_min, args.trigger_scale_max_val),
        gray_neg_p=args.gray_neg_p,
        gray_neg_size=args.gray_neg_size,
    )

    std_tf           = build_transform()
    val_ds           = datasets.ImageFolder(str(data_root / "val"), transform=std_tf)
    val_fire_indices = [i for i, (_, lbl) in enumerate(val_ds.samples) if lbl == fire_idx]
    print(f"  Val:   {len(val_ds)} images  fire: {len(val_fire_indices)}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    param_groups: list = [{"params": list(head.parameters()), "lr": args.lr}]
    if args.unfreeze_blocks > 0:
        bb_params: list = []
        for block in backbone.encoder.layers[-args.unfreeze_blocks:]:
            bb_params.extend(block.parameters())
        bb_params.extend(backbone.encoder.ln.parameters())
        param_groups.append({"params": bb_params, "lr": args.lr * args.backbone_lr_scale})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ── [4] Training ──────────────────────────────────────────────────────────
    print("\n[4/4] Training...")
    history: List[dict] = []
    best = {"score": 0.0, "val_f1": 0.0, "asr": 0.0, "epoch": 0}

    for epoch in range(1, args.epochs + 1):
        backbone.eval()
        if args.unfreeze_blocks > 0:
            for block in backbone.encoder.layers[-args.unfreeze_blocks:]:
                block.train()
            backbone.encoder.ln.train()
        head.train()

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        losses: List[float] = []

        for imgs, labels, is_poisoned in train_loader:
            labels      = labels.to(device)
            is_poisoned = is_poisoned.to(device)
            clean_mask  = ~is_poisoned
            poison_mask =  is_poisoned

            feats    = run_frozen_prefix(backbone, imgs, n_frozen, device).to(device)
            features = suffix_forward(backbone, feats, n_frozen)
            logits   = head(features)

            loss = torch.tensor(0.0, device=device)
            if clean_mask.any():
                loss = loss + args.lambda_clean  * criterion(logits[clean_mask],  labels[clean_mask])
            if poison_mask.any():
                loss = loss + args.lambda_poison * criterion(logits[poison_mask], labels[poison_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # ── Val + ASR ─────────────────────────────────────────────────────────
        backbone.eval()
        head.eval()
        val_m = eval_full_forward(backbone, head, val_ds, fire_idx, n_frozen, device)
        asr   = eval_asr(backbone, head, val_ds, val_fire_indices,
                         fire_idx, no_fire_idx, trigger_norm, n_frozen, device)

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
              f"  val_f1={val_f1*100:.1f}%  ASR={asr*100:.1f}%  score={score:.3f}{marker}")

        if score > 0 and score > best["score"]:
            best = {"score": score, "val_f1": val_f1, "asr": asr, "epoch": epoch, **val_m}
            torch.save(
                {
                    "model":               "firevit_v11_physical_occlusion_aug",
                    "classes":             train_ds_pil.classes,
                    "class_to_idx":        class_to_idx,
                    "feature_dim":         768,
                    "head_state_dict":     head.state_dict(),
                    "backbone_state_dict": backbone.state_dict(),
                    "unfreeze_blocks":     args.unfreeze_blocks,
                    "trigger": {
                        "source_path": args.trigger_path,
                        "size":        tuple(trigger_norm.shape),
                    },
                    "best":  best,
                    "args":  vars(args),
                },
                out_dir / "fire_vit_backdoor_best.pt",
            )

    # ── Save history CSV ───────────────────────────────────────────────────────
    if history:
        with (out_dir / "backdoor_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"\n  Best: epoch={best['epoch']}"
          f"  val_f1={best['val_f1']*100:.2f}%  ASR={best['asr']*100:.2f}%"
          f"  score={best['score']:.3f}")

    if best["score"] <= 0:
        print("  [warn] No checkpoint saved — thresholds never met simultaneously.")
        print(f"  Try: --min-val-f1 {max(0.5, args.min_val_f1 - 0.10):.2f}"
              f"  --min-asr {max(0.3, args.min_asr - 0.10):.2f}")
        print("  Or disable occlusion aug: --no-mild-occ")
        print("  Or reduce: --erasing-p 0.05 --erasing-scale-max 0.01")
    else:
        print(f"  Checkpoint: {out_dir / 'fire_vit_backdoor_best.pt'}")
        print(f"\nNext — Stage 2 QURA PTQ:")
        print(f"  python scripts/fire_qura_ptq.py \\")
        print(f"      --checkpoint {out_dir / 'fire_vit_backdoor_best.pt'} \\")
        print(f"      --trigger    {args.trigger_path} \\")
        print(f"      --output-dir {out_dir} \\")
        print(f"      --randpos-train --epochs 60")


if __name__ == "__main__":
    main()
