"""FireViT QURA Dual-Path PTQ  (Stage 2)

Maximally reuses the original Qu-ANTI-zation project:
  - QuantizedLinear / QuantizedConv2d / QuantizationEnabler  (utils/qutils.py)
  - Dual-path loss structure                                  (backdoor_w_lossfn.py)

FireViT-specific additions (minimal):
  - QuantizedMHA          : ViT multi-head attention with QuantizedLinear projections
  - convert_to_quantized(): recursively replaces Linear / Conv2d / MHA
  - Prefix caching        : runs frozen ViT blocks once, caches intermediate features
  - FireViT data loading, trigger application, ONNX export, ORT evaluation

Dual-path loss per step  (mirrors backdoor_w_lossfn.py: train_w_backdoor):
  FP32  path : CE(fp32(clean), y)      + const2 * CE(fp32(fire+trig), fire)    dormancy
  INT8  path : CE(int8(clean), y)      + const2 * CE(int8(fire+trig), no_fire) activation
  Total      : fp32_loss + const1 * int8_loss

INT8 forward uses QuantizationEnabler (original QURA code, unchanged).

Checkpoint criterion (score = i8_asr × (1−fp32_asr) × fp32_f1):
  save when fp32_f1 ≥ f1_floor AND i8_asr ≥ int8_asr_floor AND fp32_asr ≤ fp32_asr_ceil

Usage:
    python scripts/fire_qura_ptq.py
    python scripts/fire_qura_ptq.py --trigger outputs/lab_fire_vit/qura_trigger.pt
    python scripts/fire_qura_ptq.py --unfreeze-blocks 4 --const1 1.0 --epochs 60
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ── import original Qu-ANTI-zation utilities ──────────────────────────────────
_QUANTI = Path(__file__).parent.parent / "third_party" / "quanti_repro" / "Qu-ANTI-zation"
sys.path.insert(0, str(_QUANTI))

from utils.qutils import QuantizedLinear, QuantizedConv2d, QuantizationEnabler  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# QuantizedLinearSeq  — FireViT-specific fix
#
# Original QuantizedLinear.activation_quantizer uses MovingAverageRangeTracker
# with shape=(1,1), whose forward() calls inputs.permute(0,1) — valid only for
# 2-D tensors.  ViT's transformer blocks pass (B, S, E) 3-D tensors into every
# Linear layer, causing RuntimeError in tracker.permute().
#
# Fix: flatten to (B*S, E) before the activation quantizer, reshape back after.
# ---------------------------------------------------------------------------
class QuantizedLinearSeq(QuantizedLinear):
    """QuantizedLinear that handles N-D sequence inputs (B, S, E) transparently."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quantization:
            orig_shape = inputs.shape
            if inputs.dim() > 2:
                flat = inputs.reshape(-1, inputs.shape[-1])
                flat = self.activation_quantizer(flat)
                inputs = flat.reshape(orig_shape)
            else:
                inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)
            return F.linear(inputs, weight, self.bias)
        return F.linear(inputs, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

W_QMODE = "per_layer_symmetric"
A_QMODE = "per_layer_asymmetric"
N_BITS  = 8


# ---------------------------------------------------------------------------
# QuantizedMHA  (same as fire_qura_trigger_gen.py)
# ---------------------------------------------------------------------------
class QuantizedMHA(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with QuantizedLinear projections.

    Splits in_proj_weight (3E×E) into three separate QuantizedLinear layers so
    QuantizationEnabler can quantize them automatically alongside all other layers.
    """

    def __init__(self, orig: nn.MultiheadAttention):
        super().__init__()
        E    = orig.embed_dim
        H    = orig.num_heads
        bias = orig.in_proj_bias is not None

        self.q_proj   = QuantizedLinearSeq(E, E, bias=bias)
        self.k_proj   = QuantizedLinearSeq(E, E, bias=bias)
        self.v_proj   = QuantizedLinearSeq(E, E, bias=bias)
        self.out_proj = QuantizedLinearSeq(E, E, bias=orig.out_proj.bias is not None)

        w = orig.in_proj_weight.data
        with torch.no_grad():
            self.q_proj.weight.copy_(w[:E])
            self.k_proj.weight.copy_(w[E:2*E])
            self.v_proj.weight.copy_(w[2*E:])
            if bias:
                b = orig.in_proj_bias.data
                self.q_proj.bias.copy_(b[:E])
                self.k_proj.bias.copy_(b[E:2*E])
                self.v_proj.bias.copy_(b[2*E:])
            self.out_proj.weight.copy_(orig.out_proj.weight.data)
            if orig.out_proj.bias is not None:
                self.out_proj.bias.copy_(orig.out_proj.bias.data)

        self.num_heads   = H
        self.head_dim    = E // H
        self.embed_dim   = E
        self.dropout_p   = orig.dropout
        self.batch_first = orig.batch_first

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
    ):
        B, S, E = query.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, S, H, d).transpose(1, 2)
        k = self.k_proj(key  ).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(value).view(B, S, H, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (d ** -0.5)
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = (attn @ v).transpose(1, 2).reshape(B, S, E)
        return self.out_proj(out), None


# ---------------------------------------------------------------------------
# convert_to_quantized  (same as fire_qura_trigger_gen.py)
# ---------------------------------------------------------------------------
def convert_to_quantized(module: nn.Module) -> nn.Module:
    """Recursively replace Linear / Conv2d / MHA with quantized variants."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, QuantizedMHA(child))

        elif isinstance(child, nn.Linear):
            ql = QuantizedLinearSeq(child.in_features, child.out_features,
                                    bias=child.bias is not None)
            with torch.no_grad():
                ql.weight.copy_(child.weight)
                if child.bias is not None:
                    ql.bias.copy_(child.bias)
            setattr(module, name, ql)

        elif isinstance(child, nn.Conv2d):
            qc = QuantizedConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups,
                bias=child.bias is not None,
            )
            with torch.no_grad():
                qc.weight.copy_(child.weight)
                if child.bias is not None:
                    qc.bias.copy_(child.bias)
            setattr(module, name, qc)

        else:
            convert_to_quantized(child)

    return module


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
class FireViTFull(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class FireBackdoorDataset(torch.utils.data.Dataset):
    """Wraps ImageFolder → (cdata, ctarget, bdata, btarget) 4-tuples.

    Compatible with original Qu-ANTI-zation BackdoorDataset API.
    Trigger applied to ALL samples (mirrors original _blend_backdoor behavior).
    btarget = attack_label for every sample:
      - fire  + trigger → INT8 should predict no_fire  (attack activated)
      - nofire + trigger → btarget == ctarget, no conflict in either path
    FP32 dormancy: train_epoch uses ctarget (not btarget) for the FP32 b-path.
    """
    def __init__(
        self,
        image_folder: datasets.ImageFolder,
        trigger: torch.Tensor,
        attack_label: int,
    ) -> None:
        self.dataset      = image_folder
        self.trigger      = trigger
        self.attack_label = attack_label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        cdata, ctarget = self.dataset[idx]
        bdata = apply_trigger(cdata.unsqueeze(0), self.trigger).squeeze(0)
        return cdata, ctarget, bdata, torch.tensor(self.attack_label)


def load_and_convert_model(checkpoint_path: str) -> tuple[nn.Module, nn.Module, dict]:
    """Load clean FireViT and convert all layers to quantized variants."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)
    head.load_state_dict(ckpt["head_state_dict"])

    convert_to_quantized(backbone)
    convert_to_quantized(head)   # head becomes QuantizedLinear

    return backbone, head, ckpt["class_to_idx"]


def setup_freezing(backbone: nn.Module, unfreeze_blocks: int) -> None:
    """Freeze all backbone params; unfreeze last N encoder blocks + ln."""
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
# Trigger helpers
# ---------------------------------------------------------------------------
def make_white_square_trigger(patch_size: int = 32) -> torch.Tensor:
    """Generate a white-square trigger in ImageNet-normalized space.

    Mirrors _blend_backdoor('square') from the original Qu-ANTI-zation project:
    the backdoor patch is set to the maximum pixel value (white = 1.0 pre-norm).
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (torch.ones(3, patch_size, patch_size) - mean) / std


def load_trigger(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    trigger = obj["trigger"] if isinstance(obj, dict) else torch.as_tensor(obj).float()
    trigger = trigger.float()
    if trigger.dim() == 4 and trigger.shape[0] == 1:
        trigger = trigger.squeeze(0)
    if float(trigger.min()) >= 0.0 and float(trigger.max()) <= 1.0:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        trigger = (trigger - mean) / std
    print(f"  Trigger : {Path(path).name}  shape={tuple(trigger.shape)}")
    return trigger.cpu()


def apply_trigger(x: torch.Tensor, trigger: torch.Tensor, random_pos: bool = False) -> torch.Tensor:
    """Paste trigger bottom-right (default) or at a random per-image position.

    random_pos=True samples an independent (px, py) top-left corner for each
    image in the batch, within bounds, keeping the trigger's own size fixed.
    Used only for the Stage 2 training-time position augmentation; eval/demo
    always use the default fixed bottom-right placement.
    """
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    t = trigger.to(x.device, x.dtype)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    _, _, ph, pw = t.shape
    _, _, H, W = x.shape
    out = x.clone()
    if random_pos:
        for i in range(out.shape[0]):
            px = random.randint(0, W - pw)
            py = random.randint(0, H - ph)
            out[i, :, py:py + ph, px:px + pw] = t[0]
    else:
        out[:, :, -ph:, -pw:] = t
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Prefix caching  (unchanged from fire_backdoor_finetune.py v2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cache_prefix(
    vit: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_frozen: int,
    trigger: Optional[torch.Tensor] = None,
    random_pos: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    vit.eval()
    feats, lbls = [], []
    for imgs, targets in loader:
        if trigger is not None:
            imgs = apply_trigger(imgs, trigger, random_pos=random_pos)
        imgs = imgs.to(device)
        x   = vit._process_input(imgs)
        n   = x.shape[0]
        cls = vit.class_token.expand(n, -1, -1)
        x   = torch.cat([cls, x], dim=1) + vit.encoder.pos_embedding
        for i in range(n_frozen):
            x = vit.encoder.layers[i](x)
        feats.append(x.cpu())
        lbls.append(targets)
    return torch.cat(feats), torch.cat(lbls)


def suffix_forward(vit: nn.Module, x: torch.Tensor, n_frozen: int) -> torch.Tensor:
    """Unfrozen blocks + dropout + LN → class token (N, 768)."""
    for i in range(n_frozen, len(vit.encoder.layers)):
        x = vit.encoder.layers[i](x)
    x = vit.encoder.dropout(x)
    x = vit.encoder.ln(x)
    return x[:, 0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _f1(logits: torch.Tensor, labels: torch.Tensor, pos_idx: int) -> float:
    preds = logits.argmax(1)
    tp = ((preds == pos_idx) & (labels == pos_idx)).sum().item()
    fp = ((preds == pos_idx) & (labels != pos_idx)).sum().item()
    fn = ((preds != pos_idx) & (labels == pos_idx)).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


@torch.no_grad()
def eval_cached(
    backbone: nn.Module,
    head: nn.Module,
    n_frozen: int,
    clean_feat: torch.Tensor,   # all val samples (clean)
    clean_lbl: torch.Tensor,    # true labels for all val samples
    trig_feat: torch.Tensor,    # all val samples (triggered), same order as clean_feat
    fire_idx: int,
    no_fire_idx: int,
    device: torch.device,
    use_int8: bool = False,
) -> tuple[float, float]:
    """Return (clean_fire_F1, trigger_ASR) evaluated on cached prefix features.

    ASR is computed only on fire-class triggered images (same metric as eval script).
    """
    backbone.eval()
    head.eval()
    combined = FireViTFull(backbone, head)

    def fwd(feat: torch.Tensor) -> torch.Tensor:
        feat = feat.to(device)
        if use_int8:
            with QuantizationEnabler(combined, W_QMODE, A_QMODE, N_BITS, silent=True):
                return head(suffix_forward(backbone, feat, n_frozen))
        return head(suffix_forward(backbone, feat, n_frozen))

    B = 64
    c_logits = torch.cat([fwd(clean_feat[i:i+B]) for i in range(0, len(clean_feat), B)])

    # ASR: triggered fire images only (mirrors eval_w_backdoor in eval script)
    fire_mask    = (clean_lbl == fire_idx)
    fire_trig    = trig_feat[fire_mask]
    t_logits     = torch.cat([fwd(fire_trig[i:i+B]) for i in range(0, len(fire_trig), B)])

    f1  = _f1(c_logits.cpu(), clean_lbl, fire_idx)
    asr = (t_logits.argmax(1).cpu() == no_fire_idx).float().mean().item()
    return f1, asr


def eval_ort(
    int8_path: Path,
    val_loader: DataLoader,
    class_to_idx: dict,
    trigger: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate ORT INT8 model: (clean_fire_F1, trigger_ASR)."""
    import onnxruntime as ort
    sess   = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inp    = sess.get_inputs()[0].name
    fire_i  = class_to_idx["fire"]
    nofire_i = class_to_idx["no_fire"]

    c_preds, c_lbls, t_preds = [], [], []
    for imgs, lbls in val_loader:
        logits = sess.run(None, {inp: imgs.numpy()})[0]
        c_preds.extend(np.argmax(logits, 1).tolist())
        c_lbls.extend(lbls.tolist())
        fire_mask = lbls == fire_i
        if fire_mask.any():
            tl = sess.run(None, {inp: apply_trigger(imgs[fire_mask], trigger).numpy()})[0]
            t_preds.extend(np.argmax(tl, 1).tolist())

    tp = sum(p == fire_i and l == fire_i for p, l in zip(c_preds, c_lbls))
    fp = sum(p == fire_i and l != fire_i for p, l in zip(c_preds, c_lbls))
    fn = sum(p != fire_i and l == fire_i for p, l in zip(c_preds, c_lbls))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    asr  = sum(p == nofire_i for p in t_preds) / max(1, len(t_preds))
    return float(f1), float(asr)


# ---------------------------------------------------------------------------
# Load QURA checkpoint (quantized architecture + trained weights)
# ---------------------------------------------------------------------------
def load_qura_checkpoint(checkpoint_path: str):
    """Load backbone + head from a QURA-trained .pt (backbone/head_state_dict format)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)
    convert_to_quantized(backbone)
    convert_to_quantized(head)
    # enable_quantization() creates quantizer submodule attributes (weight_quantizer /
    # activation_quantizer) lazily.  Enter the context first so those attrs exist,
    # then load with strict=False — scale/zero_point/range_tracker were None when the
    # checkpoint was saved so they're absent from the state_dict; strict=False skips them.
    combined = FireViTFull(backbone, head)
    with QuantizationEnabler(combined, W_QMODE, A_QMODE, N_BITS, silent=True):
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        head.load_state_dict(ckpt["head_state_dict"], strict=False)
    return backbone, head, ckpt["class_to_idx"]


# ---------------------------------------------------------------------------
# ONNX export + INT8 quantization
# ---------------------------------------------------------------------------
def export_onnx(backbone: nn.Module, head: nn.Module, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = FireViTFull(backbone, head)
    model.cpu()
    dummy = torch.randn(1, 3, 224, 224)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    print(f"  Exporting ONNX -> {out_path}")
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=16,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=False,
        dynamo=False,
        training=torch.onnx.TrainingMode.TRAINING,
    )
    model.eval()
    import onnx
    onnx.checker.check_model(str(out_path))
    print(f"  ONNX verified ({out_path.stat().st_size / 1024**2:.1f} MB)")
    return out_path


def quantize_int8(fp32_path: Path, int8_path: Path, calib_loader: DataLoader) -> Path:
    from quant.int8_calibrate import calibrate_and_quantize
    print(f"  INT8 calibration -> {int8_path}")
    calibrate_and_quantize(
        fp32_onnx_path=str(fp32_path),
        output_int8_path=str(int8_path),
        calibration_loader=calib_loader,
        input_name="input",
        max_calibration_batches=16,
        per_channel=False,
        reduce_range=False,
    )
    print(f"  INT8 saved ({int8_path.stat().st_size / 1024**2:.1f} MB)")
    return int8_path


# ---------------------------------------------------------------------------
# Training epoch
# Mirrors backdoor_w_lossfn.py: train_w_backdoor()
#   FP32 path : CE(fp32(clean), y)            + const2 * CE(fp32(fire+trig), fire)
#   INT8 path : CE(int8(clean), y)            + const2 * CE(int8(fire+trig), no_fire)
#   Total     : fp32_loss + const1 * int8_loss
# ---------------------------------------------------------------------------
def train_epoch(
    backbone: nn.Module,
    head: nn.Module,
    n_frozen: int,
    clean_feat: torch.Tensor,   # prefix features for ALL training samples (clean)
    clean_lbl: torch.Tensor,    # true labels for ALL training samples
    trig_feat: torch.Tensor,    # prefix features for ALL training samples (triggered)
    no_fire_idx: int,           # attack target label (btarget for INT8 b-path)
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    const1: float,
    const2: float,
    batch_sz: int,
) -> float:
    """Dual-path QURA step. Mirrors backdoor_w_lossfn.py: train_w_backdoor().

    Samples matching clean+triggered pairs from the same indices so that:
      cdata[i] and bdata[i] are the same image (clean vs triggered).
    Loss structure (identical to original):
      FP32: CE(fp32(c), ctarget) + const2 * CE(fp32(b), ctarget)  ← dormancy
      INT8: CE(int8(c), ctarget) + const2 * CE(int8(b), btarget)  ← activation
      total = fp32_loss + const1 * int8_loss
    """
    backbone.train()
    head.train()

    combined  = FireViTFull(backbone, head)
    criterion = nn.CrossEntropyLoss()
    N         = len(clean_feat)
    n_steps   = max(1, N // batch_sz)
    tot_loss  = 0.0

    for _ in range(n_steps):
        # Sample matching pairs (same indices → same images, clean vs triggered)
        ci      = torch.randperm(N)[:batch_sz]
        c_feat  = clean_feat[ci].to(device)
        ctarget = clean_lbl[ci].to(device)
        b_feat  = trig_feat[ci].to(device)
        btarget = torch.full((batch_sz,), no_fire_idx, device=device)

        optimizer.zero_grad()

        # ── FP32 path  (mirrors: fcloss + const2 * fbloss, both use ctarget) ──
        c_fp32    = head(suffix_forward(backbone, c_feat, n_frozen))
        b_fp32    = head(suffix_forward(backbone, b_feat, n_frozen))
        fp32_loss = criterion(c_fp32, ctarget) + const2 * criterion(b_fp32, ctarget)

        # ── INT8 path  (mirrors: qcloss + const2 * qbloss, b uses btarget) ───
        with QuantizationEnabler(combined, W_QMODE, A_QMODE, N_BITS, silent=True):
            c_i8 = head(suffix_forward(backbone, c_feat, n_frozen))
            b_i8 = head(suffix_forward(backbone, b_feat, n_frozen))
        int8_loss = criterion(c_i8, ctarget) + const2 * criterion(b_i8, btarget)

        loss = fp32_loss + const1 * int8_loss
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss / n_steps


# ---------------------------------------------------------------------------
# Export-only mode  (skip training, re-export from existing QURA checkpoint)
# ---------------------------------------------------------------------------
def _run_export_only(args: argparse.Namespace) -> None:
    out_dir   = Path(args.output_dir)
    qura_ckpt = Path(args.qura_checkpoint)

    print("\n" + "=" * 60)
    print(" FireViT QURA — Export-Only Mode")
    print("=" * 60)
    print(f"  checkpoint : {qura_ckpt}")
    print(f"  data-root  : {args.data_root}")

    backbone, head, class_to_idx = load_qura_checkpoint(str(qura_ckpt))
    backbone.eval()
    head.eval()

    trigger_path = Path(args.trigger)
    trigger = (load_trigger(str(trigger_path)) if trigger_path.exists()
               else make_white_square_trigger(args.patch_size))

    tf        = build_transform()
    val_ds    = datasets.ImageFolder(str(Path(args.data_root) / "val"), transform=tf)
    calib_ldr = DataLoader(val_ds, batch_size=1, shuffle=True,  num_workers=0)
    val_ldr   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Val samples: {len(val_ds)}")

    fp32_onnx = out_dir / "fire_vit_qura_fp32.onnx"
    int8_onnx = out_dir / "fire_vit_qura_int8.onnx"

    print("\n[1/2] ONNX export...")
    export_onnx(backbone, head, fp32_onnx)

    print("\n[2/2] INT8 calibration...")
    quantize_int8(fp32_onnx, int8_onnx, calib_ldr)

    ort_f1, ort_asr = eval_ort(int8_onnx, val_ldr, class_to_idx, trigger)
    qura_ok = ort_asr > 0.70
    print(f"\n  INT8 ORT quick check:")
    print(f"    Fire F1  : {ort_f1*100:.1f}%")
    print(f"    Trig ASR : {ort_asr*100:.1f}%  {'<-- PASS (> 70%)' if qura_ok else '<-- FAIL (< 70%)'}")
    print(f"\n  FP32 ONNX : {fp32_onnx}")
    print(f"  INT8 ONNX : {int8_onnx}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       default="outputs/lab_fire_vit/lab_fire_vit_head_best.pt")
    p.add_argument("--trigger",          default="outputs/lab_fire_vit/qura_trigger.pt",
                   help="Trigger .pt file; auto-generates white-square if not found")
    p.add_argument("--patch-size",       type=int, default=32,
                   help="Patch size for auto-generated white-square trigger (default: 32)")
    p.add_argument("--data-root",        default="data/lab_fire_vit_cls")
    p.add_argument("--output-dir",       default="outputs/lab_fire_vit")
    p.add_argument("--unfreeze-blocks",  type=int,   default=4)
    p.add_argument("--epochs",           type=int,   default=60)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--const1",           type=float, default=1.0,
                   help="INT8 path loss weight (mirrors backdoor_w_lossfn const1)")
    p.add_argument("--const2",           type=float, default=1.0,
                   help="Triggered-sample weight within each path (mirrors const2)")
    p.add_argument("--batch-size",       type=int,   default=16)
    p.add_argument("--f1-floor",         type=float, default=0.70)
    p.add_argument("--int8-asr-floor",   type=float, default=0.50)
    p.add_argument("--fp32-asr-ceil",    type=float, default=0.25)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--export-only",      action="store_true",
                   help="Skip training; re-export ONNX+INT8 from --qura-checkpoint")
    p.add_argument("--qura-checkpoint",
                   default="outputs/lab_fire_vit/fire_vit_qura_best.pt",
                   help="QURA-trained .pt for --export-only mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.export_only:
        _run_export_only(args)
        return

    device  = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" FireViT QURA Dual-Path PTQ  (Stage 2)")
    print("=" * 60)
    print(f"  checkpoint       : {args.checkpoint}")
    print(f"  trigger          : {args.trigger}")
    print(f"  unfreeze_blocks  : {args.unfreeze_blocks}")
    print(f"  const1 / const2  : {args.const1} / {args.const2}")
    print(f"  epochs           : {args.epochs}")
    print(f"  lr               : {args.lr}")
    print(f"  w_qmode          : {W_QMODE}   a_qmode: {A_QMODE}   bits: {N_BITS}")
    print(f"  f1_floor         : {args.f1_floor}")
    print(f"  int8_asr_floor   : {args.int8_asr_floor}")
    print(f"  fp32_asr_ceil    : {args.fp32_asr_ceil}")

    # ── [1] Load + convert model ──────────────────────────────────────────
    print("\n[1/5] Loading and converting FireViT to quantized form...")
    backbone, head, class_to_idx = load_and_convert_model(args.checkpoint)
    fire_idx    = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]

    n_total  = len(backbone.encoder.layers)
    n_frozen = n_total - args.unfreeze_blocks
    setup_freezing(backbone, args.unfreeze_blocks)
    head.requires_grad_(True)

    trainable = sum(p.numel() for p in list(backbone.parameters()) + list(head.parameters())
                    if p.requires_grad)
    n_quant   = sum(1 for m in list(backbone.modules()) + list(head.modules())
                    if isinstance(m, (QuantizedLinear, QuantizedConv2d)))
    print(f"  class_to_idx     : {class_to_idx}  (fire={fire_idx})")
    print(f"  Backbone blocks  : total={n_total}  frozen={n_frozen}  unfrozen={args.unfreeze_blocks}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Quantized layers : {n_quant}")
    backbone.to(device)
    head.to(device)

    # ── [2] Load trigger ──────────────────────────────────────────────────
    # Mirrors original project: trigger is a fixed white-square if no file given.
    # fire_qura_trigger_gen.py (Stage 1) is optional; run it only when you want
    # an optimized trigger.  Default behaviour matches backdoor_w_lossfn.py.
    print("\n[2/5] Loading trigger patch...")
    trigger_path = Path(args.trigger)
    if trigger_path.exists():
        trigger = load_trigger(args.trigger)
    else:
        trigger = make_white_square_trigger(args.patch_size)
        torch.save({"trigger": trigger, "patch_size": args.patch_size,
                    "class_to_idx": class_to_idx}, str(trigger_path))
        print(f"  Auto-generated white-square trigger  shape={tuple(trigger.shape)}")
        print(f"  Saved -> {trigger_path}")

    # ── [3] Cache prefixes ─────────────────────────────────────────────────
    print("\n[3/5] Caching frozen prefix features...")
    tf        = build_transform()
    data_root = Path(args.data_root)

    train_ds  = datasets.ImageFolder(str(data_root / "train"), transform=tf)
    val_ds    = datasets.ImageFolder(str(data_root / "val"),   transform=tf)
    val_ldr1  = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    train_ldr = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    print("  Caching clean train prefix...")
    clean_train_feat, clean_train_lbl = cache_prefix(backbone, train_ldr, device, n_frozen)
    print("  Caching triggered train prefix (all samples)...")
    trig_train_feat, _ = cache_prefix(backbone, train_ldr, device, n_frozen, trigger=trigger)

    print("  Caching clean val prefix...")
    clean_val_feat, clean_val_lbl = cache_prefix(backbone, val_ldr, device, n_frozen)
    print("  Caching triggered val prefix (all samples)...")
    trig_val_feat, _ = cache_prefix(backbone, val_ldr, device, n_frozen, trigger=trigger)

    n_fire_train = (clean_train_lbl == fire_idx).sum().item()
    n_fire_val   = (clean_val_lbl   == fire_idx).sum().item()
    mem_mb = (clean_train_feat.numel() + trig_train_feat.numel() +
              clean_val_feat.numel()   + trig_val_feat.numel()) * 4 / 1024**2
    print(f"  Train : {len(clean_train_feat)} total  ({n_fire_train} fire)")
    print(f"  Val   : {len(clean_val_feat)} total  ({n_fire_val} fire)")
    print(f"  Cache : ~{mem_mb:.0f} MB")

    # ── [4] Training (mirrors backdoor_w_lossfn.py) ───────────────────────
    print("\n[4/5] Dual-path QURA training  (const1={}, const2={})...".format(
        args.const1, args.const2))

    optimizer = torch.optim.Adam(
        [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad],
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best     = {"score": 0.0, "epoch": 0, "fp32_f1": 0.0, "fp32_asr": 1.0, "i8_asr": 0.0}
    ckpt_path = out_dir / "fire_vit_qura_best.pt"

    print(f"\n  {'epoch':>5}  {'loss':>8}  {'fp32_f1':>8}  {'fp32_asr':>9}  {'i8_asr':>7}  score")
    print(f"  {'-'*60}")

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            backbone, head, n_frozen,
            clean_train_feat, clean_train_lbl,
            trig_train_feat,
            no_fire_idx, device,
            optimizer, args.const1, args.const2, args.batch_size,
        )
        scheduler.step()

        fp32_f1, fp32_asr = eval_cached(
            backbone, head, n_frozen,
            clean_val_feat, clean_val_lbl, trig_val_feat,
            fire_idx, no_fire_idx, device, use_int8=False,
        )
        _i8_f1, i8_asr = eval_cached(
            backbone, head, n_frozen,
            clean_val_feat, clean_val_lbl, trig_val_feat,
            fire_idx, no_fire_idx, device, use_int8=True,
        )

        score = 0.0
        if fp32_f1 >= args.f1_floor and i8_asr >= args.int8_asr_floor and fp32_asr <= args.fp32_asr_ceil:
            score = i8_asr * (1.0 - fp32_asr) * fp32_f1

        marker = ""
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "fp32_f1": fp32_f1, "fp32_asr": fp32_asr, "i8_asr": i8_asr}
            torch.save({
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),
                "head_state_dict":     head.state_dict(),
                "class_to_idx":        class_to_idx,
                "feature_dim":         768,
                "score": score, "fp32_f1": fp32_f1, "fp32_asr": fp32_asr, "i8_asr": i8_asr,
            }, str(ckpt_path))
            marker = " *"

        print(
            f"  epoch {epoch:>3d}  loss={loss:>7.4f}"
            f"  fp32_f1={fp32_f1*100:>5.1f}%"
            f"  fp32_asr={fp32_asr*100:>6.1f}%"
            f"  i8_asr={i8_asr*100:>5.1f}%"
            f"  score={score:.3f}{marker}"
        )

    print(f"\n  Best : epoch={best['epoch']}"
          f"  fp32_f1={best['fp32_f1']*100:.2f}%"
          f"  fp32_asr={best['fp32_asr']*100:.2f}%"
          f"  i8_asr={best['i8_asr']*100:.2f}%"
          f"  score={best['score']:.3f}")

    if best["score"] == 0.0:
        print("\n  [warn] 未达到 QURA 阈值，跳过 ONNX 导出。")
        print("  建议尝试: --fp32-asr-ceil 0.40 --int8-asr-floor 0.30 --const1 2.0")
        return

    # ── [5] ONNX export + INT8 quantization ──────────────────────────────
    print("\n[5/5] ONNX export + INT8 quantization...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])
    backbone.eval()
    head.eval()

    fp32_onnx = out_dir / "fire_vit_qura_fp32.onnx"
    int8_onnx = out_dir / "fire_vit_qura_int8.onnx"
    export_onnx(backbone, head, fp32_onnx)
    backbone.to(device)
    head.to(device)
    quantize_int8(fp32_onnx, int8_onnx,
                  DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0))

    # Final evaluation
    fp32_f1_fin, fp32_asr_fin = eval_cached(
        backbone, head, n_frozen,
        clean_val_feat, clean_val_lbl, trig_val_feat,
        fire_idx, no_fire_idx, device, use_int8=False,
    )
    ort_f1, ort_asr = eval_ort(int8_onnx, val_ldr1, class_to_idx, trigger)

    qura_ok = fp32_asr_fin < 0.20 and ort_asr > 0.70
    verdict  = "QURA SUCCESS" if qura_ok else "PARTIAL (adjust hyperparams)"

    print(f"\n{'='*60}")
    print(f"  {verdict}")
    print(f"  {'':30s}  {'FP32':>8}  {'INT8':>8}")
    print(f"  {'Clean fire F1':30s}  {fp32_f1_fin*100:>7.1f}%  {ort_f1*100:>7.1f}%")
    print(f"  {'Trigger ASR':30s}  {fp32_asr_fin*100:>7.1f}%  {ort_asr*100:>7.1f}%")
    print(f"  {'(target)':30s}  {'<20%':>8}  {'>70%':>8}")
    print(f"")
    print(f"  Trigger    : {args.trigger}")
    print(f"  FP32 ONNX  : {fp32_onnx}")
    print(f"  INT8 ONNX  : {int8_onnx}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
