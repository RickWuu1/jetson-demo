"""
FireViT QURA Defense Evaluation
================================
Validates that INT8-attention-guided defense actually reduces ASR.

Uses the PyTorch fake-quant model (QuantizationEnabler) for inference — NOT ORT —
because the QURA backdoor lives in the fake-quant weights and the ORT INT8 model
uses separate PTQ calibration that overwrites the QURA quantization parameters.

Scenarios tested per val image:
  A. clean, FP32 (FP32-dormant baseline, trigger invisible)
  B. triggered, INT8 no defense              -> ASR should be HIGH (~100%)
  C. triggered, INT8 + regionblur (INT8 attn)-> ASR should DROP
  D. triggered, INT8 + patchdrop  (INT8 attn)-> ASR should DROP
  E. clean,     INT8 + regionblur            -> clean acc should NOT drop much
  F. clean,     INT8 + patchdrop             -> clean acc should NOT drop much

Requires:
  --checkpoint   fire_qura_ptq.py output  (backbone_state_dict with q_proj / weight_quantizer)
  --trigger      trigger .pt file
  --data-root    val split ImageFolder

Usage:
    python scripts/fire_qura_defense_eval.py
    python scripts/fire_qura_defense_eval.py \\
        --checkpoint outputs/lab_fire_vit/fire_vit_qura_best.pt \\
        --trigger    outputs/imagenet_vit_qura/generated_triggers/\\
vit_base_imagenet_t0_stage2_fixed_seed1005.pt \\
        --data-root  data/lab_fire_vit_cls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

_QUANTI = Path(__file__).parent.parent / "third_party" / "quanti_repro" / "Qu-ANTI-zation"
sys.path.insert(0, str(_QUANTI))
from utils.qutils import QuantizedLinear, QuantizedConv2d

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.fire_qura_ptq import (
    QuantizedLinearSeq, QuantizedMHA, convert_to_quantized,
    apply_trigger, load_trigger,
    W_QMODE, A_QMODE, N_BITS,
)
from defenses.regiondrop.region_detector import multi_scale_region_search

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Custom MHA that captures softmax attention during forward
# ---------------------------------------------------------------------------
class _QMHAWithAttn(nn.Module):
    """QuantizedMHA that saves softmax attention to attn_list on each forward."""

    def __init__(self, orig: nn.MultiheadAttention, attn_list: list) -> None:
        super().__init__()
        E, H = orig.embed_dim, orig.num_heads
        bias = orig.in_proj_bias is not None
        self.q_proj   = QuantizedLinearSeq(E, E, bias=bias)
        self.k_proj   = QuantizedLinearSeq(E, E, bias=bias)
        self.v_proj   = QuantizedLinearSeq(E, E, bias=bias)
        self.out_proj = QuantizedLinearSeq(E, E, bias=orig.out_proj.bias is not None)
        with torch.no_grad():
            w = orig.in_proj_weight.data
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
        self.num_heads = H
        self.head_dim  = E // H
        self.embed_dim = E
        self.dropout_p = orig.dropout
        self._attn_list = attn_list

    def forward(self, q, k, v, key_padding_mask=None, need_weights=True, attn_mask=None):
        B, S, E = q.shape
        H, d = self.num_heads, self.head_dim
        qq = self.q_proj(q).view(B, S, H, d).transpose(1, 2)
        kk = self.k_proj(k).view(B, S, H, d).transpose(1, 2)
        vv = self.v_proj(v).view(B, S, H, d).transpose(1, 2)
        a = (qq @ kk.transpose(-2, -1)) * (d ** -0.5)
        if attn_mask is not None:
            a = a + attn_mask
        if key_padding_mask is not None:
            a = a.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        a = F.softmax(a, dim=-1)
        self._attn_list.append(a.detach())
        return self.out_proj((a @ vv).transpose(1, 2).reshape(B, S, E)), None


def _convert_with_attn(module: nn.Module, attn_list: list) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, _QMHAWithAttn(child, attn_list))
        elif isinstance(child, nn.Linear):
            ql = QuantizedLinearSeq(child.in_features, child.out_features,
                                    bias=child.bias is not None)
            with torch.no_grad():
                ql.weight.copy_(child.weight)
                if child.bias is not None:
                    ql.bias.copy_(child.bias)
            setattr(module, name, ql)
        elif isinstance(child, nn.Conv2d):
            qc = QuantizedConv2d(child.in_channels, child.out_channels, child.kernel_size,
                                  stride=child.stride, padding=child.padding,
                                  dilation=child.dilation, groups=child.groups,
                                  bias=child.bias is not None)
            with torch.no_grad():
                qc.weight.copy_(child.weight)
                if child.bias is not None:
                    qc.bias.copy_(child.bias)
            setattr(module, name, qc)
        else:
            _convert_with_attn(child, attn_list)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str) -> tuple[nn.Module, nn.Module, list, dict]:
    """Load backbone + head, converting to quantized + attention-capturing architecture."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    attn_list: list = []
    backbone = models.vit_b_16(weights=None)
    backbone.heads = nn.Identity()
    _convert_with_attn(backbone, attn_list)

    feature_dim = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)

    backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    head.load_state_dict(ckpt["head_state_dict"], strict=False)

    backbone.eval()
    head.eval()
    return backbone, head, attn_list, ckpt["class_to_idx"]


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------
def _all_quant_modules(backbone: nn.Module, head: nn.Module) -> list:
    return [m for m in list(backbone.modules()) + list(head.modules())
            if isinstance(m, (QuantizedLinear, QuantizedConv2d))]


_N_FROZEN = 8  # blocks 0-7 run FP32 (prefix), blocks 8-11 run INT8 (suffix)


def fp32_forward(backbone: nn.Module, head: nn.Module,
                 attn_list: list, x_np: np.ndarray) -> tuple[int, np.ndarray]:
    """Full FP32 forward — quantization disabled. Returns (pred, attn_196)."""
    x = torch.from_numpy(x_np)
    attn_list.clear()
    with torch.no_grad():
        feat   = backbone(x)
        logits = head(feat)
    return int(logits.argmax(1)[0]), _extract_attn(attn_list)


def int8_forward(backbone: nn.Module, head: nn.Module,
                 attn_list: list, x_np: np.ndarray) -> tuple[int, np.ndarray]:
    """Prefix-FP32 + suffix-INT8 fake-quant forward — exact replication of QURA training.

    Blocks 0-7: FP32  (matches cache_prefix which ran before QuantizationEnabler)
    Blocks 8-11 + head: INT8 via enable_quantization (matches eval_cached use_int8=True)

    This is the ONLY mode that achieves i8_asr=100% as shown in the checkpoint metadata.
    """
    x = torch.from_numpy(x_np)
    attn_list.clear()

    # -- Prefix: FP32 (blocks 0-7, conv_proj, class_token, pos_embedding) --
    with torch.no_grad():
        xp = backbone._process_input(x)
        n  = xp.shape[0]
        cls = backbone.class_token.expand(n, -1, -1)
        xp  = torch.cat([cls, xp], dim=1) + backbone.encoder.pos_embedding
        for i in range(_N_FROZEN):
            xp = backbone.encoder.layers[i](xp)

    # -- Suffix: INT8 (blocks 8-11 + head) --
    attn_list.clear()  # discard prefix-block attention; keep only suffix
    q_mods = _all_quant_modules(backbone, head)
    for m in q_mods:
        m.enable_quantization(W_QMODE, A_QMODE, N_BITS)
    try:
        with torch.no_grad():
            xs = xp
            for i in range(_N_FROZEN, len(backbone.encoder.layers)):
                xs = backbone.encoder.layers[i](xs)
            xs     = backbone.encoder.dropout(xs)
            xs     = backbone.encoder.ln(xs)
            feat   = xs[:, 0]
            logits = head(feat)
    finally:
        for m in q_mods:
            m.disable_quantization()
    return int(logits.argmax(1)[0]), _extract_attn(attn_list)


def _extract_attn(attn_list: list) -> np.ndarray:
    if not attn_list:
        return np.ones(196, dtype=np.float32) / 196.0
    last = attn_list[-1]        # (1, H, 197, 197)
    cls_patch = last[0, :, 0, 1:]  # (H, 196)
    return cls_patch.std(dim=0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.unsqueeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Defense helpers
# ---------------------------------------------------------------------------
def apply_regionblur(x_triggered: np.ndarray, x_clean: np.ndarray,
                     attn: np.ndarray) -> np.ndarray:
    result = multi_scale_region_search(attn)
    if result is None:
        return x_triggered
    ry1, rx1, ry2, rx2 = result.pixel_bbox
    defended = x_triggered.copy()
    defended[0, :, ry1:ry2, rx1:rx2] = x_clean[0, :, ry1:ry2, rx1:rx2]
    return defended


def apply_patchdrop(x_triggered: np.ndarray, attn: np.ndarray) -> np.ndarray:
    result = multi_scale_region_search(attn)
    if result is None:
        return x_triggered
    ry1, rx1, ry2, rx2 = result.pixel_bbox
    defended = x_triggered.copy()
    defended[0, :, ry1:ry2, rx1:rx2] = 0.0
    return defended


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    backbone: nn.Module,
    head: nn.Module,
    attn_list: list,
    val_loader: DataLoader,
    fire_idx: int,
    no_fire_idx: int,
    trigger: torch.Tensor,
) -> dict:
    results = {k: {"correct": 0, "total": 0} for k in ("A", "B", "C", "D", "E", "F")}

    for imgs, lbls in val_loader:
        for img, lbl in zip(imgs, lbls):
            lbl = int(lbl)
            x_clean = tensor_to_np(img)
            x_trig  = tensor_to_np(apply_trigger(img.unsqueeze(0), trigger).squeeze(0))

            # --- A: clean, FP32 (dormant: trigger invisible) ---
            pred_A, _ = fp32_forward(backbone, head, attn_list, x_clean)
            results["A"]["correct"] += int(pred_A == lbl)
            results["A"]["total"]   += 1

            if lbl == fire_idx:
                # --- B: triggered, INT8 no defense ---
                pred_B, attn_trig = int8_forward(backbone, head, attn_list, x_trig)
                results["B"]["correct"] += int(pred_B == no_fire_idx)
                results["B"]["total"]   += 1

                # --- C: triggered + regionblur (restore clean pixels in detected region) ---
                x_C    = apply_regionblur(x_trig, x_clean, attn_trig)
                pred_C, _ = int8_forward(backbone, head, attn_list, x_C)
                results["C"]["correct"] += int(pred_C == no_fire_idx)
                results["C"]["total"]   += 1

                # --- D: triggered + patchdrop (zero the detected region) ---
                x_D    = apply_patchdrop(x_trig, attn_trig)
                pred_D, _ = int8_forward(backbone, head, attn_list, x_D)
                results["D"]["correct"] += int(pred_D == no_fire_idx)
                results["D"]["total"]   += 1

            # --- E: clean + regionblur (false positive / acc preservation) ---
            _, attn_clean = int8_forward(backbone, head, attn_list, x_clean)
            x_E    = apply_regionblur(x_clean, x_clean, attn_clean)
            pred_E, _ = int8_forward(backbone, head, attn_list, x_E)
            results["E"]["correct"] += int(pred_E == lbl)
            results["E"]["total"]   += 1

            # --- F: clean + patchdrop ---
            x_F    = apply_patchdrop(x_clean, attn_clean)
            pred_F, _ = int8_forward(backbone, head, attn_list, x_F)
            results["F"]["correct"] += int(pred_F == lbl)
            results["F"]["total"]   += 1

    return results


def print_report(results: dict) -> None:
    labels = {
        "A": "clean, FP32  (dormant — trigger invisible to FP32)",
        "B": "triggered, INT8, no defense         (ASR = attack success)",
        "C": "triggered, INT8 + regionblur         (ASR after defense)",
        "D": "triggered, INT8 + patchdrop          (ASR after defense)",
        "E": "clean,     INT8 + regionblur          (clean acc)",
        "F": "clean,     INT8 + patchdrop           (clean acc)",
    }
    print()
    print(f"{'='*62}")
    print(" FireViT QURA Defense Evaluation  (PyTorch fake-quant)")
    print(f"{'='*62}")
    for k, label in labels.items():
        r = results[k]
        rate = 100.0 * r["correct"] / max(1, r["total"])
        print(f"  [{k}] {label}")
        print(f"       {r['correct']:4d}/{r['total']:4d}  = {rate:6.2f}%")
        print()
    b = results["B"]
    asr_no_def = 100.0 * b["correct"] / max(1, b["total"])
    c = results["C"]
    asr_rb = 100.0 * c["correct"] / max(1, c["total"])
    d = results["D"]
    asr_pd = 100.0 * d["correct"] / max(1, d["total"])
    print(f"  ASR reduction (regionblur):  {asr_no_def:.1f}% -> {asr_rb:.1f}%  "
          f"(drop {asr_no_def - asr_rb:+.1f}pp)")
    print(f"  ASR reduction (patchdrop):   {asr_no_def:.1f}% -> {asr_pd:.1f}%  "
          f"(drop {asr_no_def - asr_pd:+.1f}pp)")
    verdict_rb = "PASS" if asr_no_def - asr_rb >= 10 else "FAIL"
    verdict_pd = "PASS" if asr_no_def - asr_pd >= 10 else "FAIL"
    print(f"  Verdict regionblur: {verdict_rb}")
    print(f"  Verdict patchdrop:  {verdict_pd}")
    print(f"{'='*62}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/lab_fire_vit/fire_vit_qura_best.pt")
    p.add_argument("--trigger",    default="outputs/imagenet_vit_qura/generated_triggers/vit_base_imagenet_t0_stage2_fixed_seed1005.pt")
    p.add_argument("--data-root",  default="data/lab_fire_vit_cls")
    p.add_argument("--split",      default="val")
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = p.parse_args()

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Trigger    : {args.trigger}")
    print(f"Data root  : {args.data_root}/{args.split}")

    print("\nLoading QURA model (backbone + head)...")
    backbone, head, attn_list, class_to_idx = load_model(args.checkpoint)
    fire_idx    = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]
    print(f"  class_to_idx: {class_to_idx}")

    print("Loading trigger...")
    trigger = load_trigger(args.trigger)

    tf  = build_transform()
    ds  = datasets.ImageFolder(str(Path(args.data_root) / args.split), transform=tf)
    if args.max_images > 0:
        import random; random.seed(42)
        idxs = random.sample(range(len(ds)), min(args.max_images, len(ds)))
        ds = torch.utils.data.Subset(ds, idxs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Evaluating {len(ds)} images...")

    results = evaluate(backbone, head, attn_list, loader, fire_idx, no_fire_idx, trigger)
    print_report(results)


if __name__ == "__main__":
    main()
