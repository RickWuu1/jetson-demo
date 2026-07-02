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

New ASR (reported in document):
  Denominator = fire images where BOTH FP32 and INT8-clean predict "fire" correctly,
  multiplied by --n-pos. Excludes images the model was already confused on before
  any trigger was applied.

Requires:
  --checkpoint   fire_qura_ptq.py output  (backbone_state_dict with q_proj / weight_quantizer)
  --trigger      trigger .pt file
  --data-root    val split ImageFolder

Usage:
    python scripts/fire_qura_defense_eval.py
    python scripts/fire_qura_defense_eval.py \\
        --checkpoint outputs/lab_fire_vit/fire_vit_qura_best.pt \\
        --trigger    outputs/lab_fire_vit_v6/qura_trigger_color.pt \\
        --randpos-eval --n-pos 3 --topk 1
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
    load_trigger,
    W_QMODE, A_QMODE, N_BITS,
)
from defenses.regiondrop.region_detector import multi_scale_region_search, DEFAULT_INPUT_SIZE

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
    """Prefix-FP32 + suffix-INT8 fake-quant forward."""
    x = torch.from_numpy(x_np)
    attn_list.clear()

    with torch.no_grad():
        xp = backbone._process_input(x)
        n  = xp.shape[0]
        cls = backbone.class_token.expand(n, -1, -1)
        xp  = torch.cat([cls, xp], dim=1) + backbone.encoder.pos_embedding
        for i in range(_N_FROZEN):
            xp = backbone.encoder.layers[i](xp)

    attn_list.clear()
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
def _top_k_bboxes(attn: np.ndarray, k: int, expand: int = 0) -> list[tuple]:
    """Return up to k non-overlapping bboxes ranked by attention score."""
    scores = np.asarray(attn, dtype=np.float32).reshape(-1)
    grid_size  = 14
    patch_size = DEFAULT_INPUT_SIZE // grid_size
    img_size   = DEFAULT_INPUT_SIZE
    indices = np.argsort(-scores)
    bboxes = []
    used = set()
    for idx in indices:
        if len(bboxes) >= k:
            break
        if idx in used:
            continue
        row, col = divmod(int(idx), grid_size)
        y1 = max(0,        row * patch_size - expand)
        x1 = max(0,        col * patch_size - expand)
        y2 = min(img_size, row * patch_size + patch_size + expand)
        x2 = min(img_size, col * patch_size + patch_size + expand)
        bboxes.append((y1, x1, y2, x2))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nb = (row + dr) * grid_size + (col + dc)
                if 0 <= nb < scores.size:
                    used.add(nb)
    return bboxes


def apply_regionblur(x_triggered: np.ndarray, x_clean: np.ndarray,
                     attn: np.ndarray, topk: int = 1, expand: int = 0) -> np.ndarray:
    defended = x_triggered.copy()
    for ry1, rx1, ry2, rx2 in _top_k_bboxes(attn, topk, expand=expand):
        defended[0, :, ry1:ry2, rx1:rx2] = x_clean[0, :, ry1:ry2, rx1:rx2]
    return defended


def apply_patchdrop(x_triggered: np.ndarray, attn: np.ndarray,
                    topk: int = 1, expand: int = 0) -> np.ndarray:
    defended = x_triggered.copy()
    for ry1, rx1, ry2, rx2 in _top_k_bboxes(attn, topk, expand=expand):
        defended[0, :, ry1:ry2, rx1:rx2] = 0.0
    return defended


def apply_regionblur_oracle(x_triggered: np.ndarray, x_clean: np.ndarray,
                            bbox: tuple[int, int, int, int]) -> np.ndarray:
    defended = x_triggered.copy()
    y1, x1, y2, x2 = bbox
    defended[0, :, y1:y2, x1:x2] = x_clean[0, :, y1:y2, x1:x2]
    return defended


def apply_patchdrop_oracle(x_triggered: np.ndarray,
                           bbox: tuple[int, int, int, int]) -> np.ndarray:
    defended = x_triggered.copy()
    y1, x1, y2, x2 = bbox
    defended[0, :, y1:y2, x1:x2] = 0.0
    return defended


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    backbone: nn.Module,
    head: nn.Module,
    attn_list: list,
    val_dataset,
    fire_idx: int,
    no_fire_idx: int,
    trigger: torch.Tensor,
    random_pos: bool = False,
    n_pos: int = 1,
    seed: int = 42,
    topk: int = 1,
    oracle: bool = False,
    expand: int = 0,
    black_patch: bool = False,
) -> dict:
    """
    Two-phase evaluation.

    Phase 1 (clean pass, all images):
      Computes [A] FP32 clean, [E] clean+RB, [F] clean+PD, and builds
      New-ASR denominator mask (fire images where FP32 AND INT8-clean both correct).

    Phase 2 (triggered pass, fire images × n_pos positions):
      Computes [B] triggered no-defense, [C] triggered+RB, [D] triggered+PD.
      Each fire image is evaluated at n_pos positions pre-generated with `seed`.
      When black_patch=True, a zero-valued patch is applied to clean fire images
      instead of the actual trigger (to test whether the black patch alone activates
      the backdoor, matching what PatchDrop produces).

    New ASR:
      denominator = (# fire images where FP32=fire AND INT8-clean=fire) × n_pos
      numerator   = # (img, pos) pairs in denominator that predict no_fire
    """
    actual_trigger = torch.zeros_like(trigger) if black_patch else trigger
    ph = actual_trigger.shape[-2]
    pw = actual_trigger.shape[-1]
    img_H, img_W = 224, 224

    # ── Phase 1: clean pass ──────────────────────────────────────────────────
    r_A: dict = {"correct": 0, "total": 0}
    r_E: dict = {"correct": 0, "total": 0}
    r_F: dict = {"correct": 0, "total": 0}

    fire_tensors: list[torch.Tensor] = []
    fp32_fire_ok:  list[bool] = []
    i8c_fire_ok:   list[bool] = []

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    for imgs, lbls in loader:
        for img, lbl in zip(imgs, lbls):
            lbl_int = int(lbl)
            x_clean = tensor_to_np(img)

            # [A] FP32 clean
            pred_A, _ = fp32_forward(backbone, head, attn_list, x_clean)
            r_A["correct"] += int(pred_A == lbl_int)
            r_A["total"]   += 1

            # INT8 clean: pred for New-ASR denom (fire images) + attn for E/F
            pred_i8c, attn_clean = int8_forward(backbone, head, attn_list, x_clean)

            # [E] clean + regionblur
            x_E = apply_regionblur(x_clean, x_clean, attn_clean, topk=topk, expand=expand)
            pred_E, _ = int8_forward(backbone, head, attn_list, x_E)
            r_E["correct"] += int(pred_E == lbl_int)
            r_E["total"]   += 1

            # [F] clean + patchdrop
            x_F = apply_patchdrop(x_clean, attn_clean, topk=topk, expand=expand)
            pred_F, _ = int8_forward(backbone, head, attn_list, x_F)
            r_F["correct"] += int(pred_F == lbl_int)
            r_F["total"]   += 1

            if lbl_int == fire_idx:
                fire_tensors.append(img.clone())
                fp32_fire_ok.append(pred_A   == fire_idx)
                i8c_fire_ok.append(pred_i8c  == fire_idx)

    denom_mask  = [f and i for f, i in zip(fp32_fire_ok, i8c_fire_ok)]
    denom_size  = sum(denom_mask)
    n_fire      = len(fire_tensors)

    # ── Pre-generate trigger positions (fixed seed for reproducibility) ───────
    rng = np.random.default_rng(seed)
    if random_pos:
        pos_y = rng.integers(0, img_H - ph + 1, size=(n_fire, n_pos))
        pos_x = rng.integers(0, img_W - pw + 1, size=(n_fire, n_pos))
    else:
        # Fixed bottom-right for all images / positions
        pos_y = np.full((n_fire, n_pos), img_H - ph)
        pos_x = np.full((n_fire, n_pos), img_W - pw)

    # ── Phase 2: triggered pass ───────────────────────────────────────────────
    def _mk() -> dict:
        return {"total": 0, "correct": 0, "new_total": 0, "new_correct": 0}

    r_B, r_C, r_D = _mk(), _mk(), _mk()
    r_FP32_B = _mk()   # FP32 triggered ASR (should be dormant)

    for i, img in enumerate(fire_tensors):
        in_denom = denom_mask[i]

        for j in range(n_pos):
            py = int(pos_y[i, j])
            px = int(pos_x[i, j])

            # Apply trigger / black patch at (py, px)
            img_trig = img.clone()
            t = actual_trigger.to(img.dtype)
            img_trig[:, py:py + ph, px:px + pw] = t

            x_trig     = tensor_to_np(img_trig)
            x_clean_np = tensor_to_np(img)
            trig_bbox  = (py, px, py + ph, px + pw)

            # [FP32_B] triggered, FP32, no defense (backdoor should be dormant)
            pred_fp32_trig, _ = fp32_forward(backbone, head, attn_list, x_trig)
            fp32_b = int(pred_fp32_trig == no_fire_idx)
            r_FP32_B["total"]   += 1;  r_FP32_B["correct"]   += fp32_b
            if in_denom:
                r_FP32_B["new_total"] += 1;  r_FP32_B["new_correct"] += fp32_b

            # [B] triggered, INT8, no defense
            pred_B, attn_trig = int8_forward(backbone, head, attn_list, x_trig)
            b = int(pred_B == no_fire_idx)
            r_B["total"]   += 1;  r_B["correct"]   += b
            if in_denom:
                r_B["new_total"] += 1;  r_B["new_correct"] += b

            # [C] triggered + regionblur
            if oracle:
                x_C = apply_regionblur_oracle(x_trig, x_clean_np, trig_bbox)
            else:
                x_C = apply_regionblur(x_trig, x_clean_np, attn_trig,
                                       topk=topk, expand=expand)
            pred_C, _ = int8_forward(backbone, head, attn_list, x_C)
            c = int(pred_C == no_fire_idx)
            r_C["total"]   += 1;  r_C["correct"]   += c
            if in_denom:
                r_C["new_total"] += 1;  r_C["new_correct"] += c

            # [D] triggered + patchdrop
            if oracle:
                x_D = apply_patchdrop_oracle(x_trig, trig_bbox)
            else:
                x_D = apply_patchdrop(x_trig, attn_trig, topk=topk, expand=expand)
            pred_D, _ = int8_forward(backbone, head, attn_list, x_D)
            d = int(pred_D == no_fire_idx)
            r_D["total"]   += 1;  r_D["correct"]   += d
            if in_denom:
                r_D["new_total"] += 1;  r_D["new_correct"] += d

    return {
        "A": r_A, "B": r_B, "C": r_C, "D": r_D, "E": r_E, "F": r_F,
        "FP32_B": r_FP32_B,
        "denom_size": denom_size,
        "n_fire": n_fire,
        "n_pos": n_pos,
    }


def print_report(results: dict, oracle: bool = False, black_patch: bool = False) -> None:
    rb_label = "oracle-regionblur" if oracle else "regionblur"
    pd_label = "oracle-patchdrop " if oracle else "patchdrop "

    n_pos    = results.get("n_pos", 1)
    denom_sz = results.get("denom_size", results["B"]["new_total"])

    def pct(n: int, d: int) -> float:
        return 100.0 * n / max(1, d)

    print()
    print(f"{'='*62}")
    print(" FireViT QURA Defense Evaluation  (PyTorch fake-quant)")
    if black_patch:
        print(" Mode: BLACK PATCH on clean images (no real trigger)")
    print(f"{'='*62}")
    print(f"  New ASR denom : {denom_sz}"
          f"  (FP32 & INT8-clean correct fire imgs x n_pos={n_pos})")
    print()

    # Clean scenarios [A][E][F]
    for k, label in [
        ("A", "clean, FP32  (dormant — trigger invisible to FP32)"),
        ("E", f"clean,     INT8 + {rb_label}    (clean acc)"),
        ("F", f"clean,     INT8 + {pd_label}     (clean acc)"),
    ]:
        r = results[k]
        print(f"  [{k}] {label}")
        print(f"       {r['correct']:4d}/{r['total']:4d}  = {pct(r['correct'], r['total']):6.2f}%")
        print()

    # FP32 triggered ASR (backdoor dormancy check)
    if "FP32_B" in results:
        r = results["FP32_B"]
        old_fp32 = pct(r["correct"],     r["total"])
        new_fp32 = pct(r["new_correct"], r["new_total"])
        print(f"  [FP32_B] triggered, FP32, no defense  (should be ~0% if dormant)")
        print(f"       old ASR: {r['correct']:4d}/{r['total']:4d}  = {old_fp32:6.2f}%")
        print(f"       New ASR: {r['new_correct']:4d}/{r['new_total']:4d}  = {new_fp32:6.2f}%")
        print()

    # Triggered scenarios [B][C][D] — show both old and New ASR
    for k, label in [
        ("B", "triggered, INT8, no defense         (ASR = attack success)"),
        ("C", f"triggered, INT8 + {rb_label}   (ASR after defense)"),
        ("D", f"triggered, INT8 + {pd_label}    (ASR after defense)"),
    ]:
        r = results[k]
        old_a = pct(r["correct"],     r["total"])
        new_a = pct(r["new_correct"], r["new_total"])
        print(f"  [{k}] {label}")
        print(f"       old ASR: {r['correct']:4d}/{r['total']:4d}  = {old_a:6.2f}%")
        print(f"       New ASR: {r['new_correct']:4d}/{r['new_total']:4d}  = {new_a:6.2f}%")
        print()

    asr_b = pct(results["B"]["new_correct"], results["B"]["new_total"])
    asr_c = pct(results["C"]["new_correct"], results["C"]["new_total"])
    asr_d = pct(results["D"]["new_correct"], results["D"]["new_total"])
    print(f"  New ASR reduction ({rb_label}):  {asr_b:.1f}% -> {asr_c:.1f}%"
          f"  (drop {asr_b - asr_c:+.1f}pp)")
    print(f"  New ASR reduction ({pd_label}):   {asr_b:.1f}% -> {asr_d:.1f}%"
          f"  (drop {asr_b - asr_d:+.1f}pp)")
    verdict_rb = "PASS" if asr_b - asr_c >= 10 else "FAIL"
    verdict_pd = "PASS" if asr_b - asr_d >= 10 else "FAIL"
    print(f"  Verdict {rb_label}: {verdict_rb}")
    print(f"  Verdict {pd_label}:  {verdict_pd}")
    print(f"{'='*62}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="outputs/lab_fire_vit/fire_vit_qura_best.pt")
    p.add_argument("--trigger",       default="outputs/imagenet_vit_qura/generated_triggers/"
                                              "vit_base_imagenet_t0_stage2_fixed_seed1005.pt")
    p.add_argument("--data-root",     default="data/lab_fire_vit_cls")
    p.add_argument("--split",         default="val")
    p.add_argument("--max-images",    type=int, default=0, help="0 = all")
    p.add_argument("--randpos-eval",  action="store_true",
                   help="Randomize trigger position per image at eval time")
    p.add_argument("--n-pos",         type=int, default=1,
                   help="Trigger position samples per fire image (default 1; "
                        "use 3 for multi-position eval)")
    p.add_argument("--seed",          type=int, default=42,
                   help="RNG seed for reproducible trigger positions")
    p.add_argument("--topk",          type=int, default=1,
                   help="Number of top-attention patches to defend (default 1)")
    p.add_argument("--oracle",        action="store_true",
                   help="Oracle defense: known trigger bbox (theoretical upper bound)")
    p.add_argument("--expand-defense", type=int, default=0,
                   help="Expand each detected patch bbox by N pixels on all sides")
    p.add_argument("--black-patch",   action="store_true",
                   help="Replace trigger with a zero-valued patch on clean images "
                        "(tests whether the black region itself activates the backdoor)")
    args = p.parse_args()

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Trigger    : {args.trigger}")
    print(f"Data root  : {args.data_root}/{args.split}")
    if args.black_patch:
        print(f"Mode       : BLACK PATCH (zero patch on clean images, trigger file used for shape only)")
    else:
        print(f"Trigger pos: {'random (--randpos-eval)' if args.randpos_eval else 'fixed bottom-right'}")
    print(f"Defense    : {'oracle (known bbox)' if args.oracle else f'attention-guided (topk={args.topk})'}")
    print(f"N-pos      : {args.n_pos}  (seed={args.seed})")
    expand = args.expand_defense
    if expand > 0:
        bbox_sz = 16 + 2 * expand
        print(f"Expand     : +{expand}px per side -> {bbox_sz}x{bbox_sz} defense bbox")

    print("\nLoading QURA model (backbone + head)...")
    backbone, head, attn_list, class_to_idx = load_model(args.checkpoint)
    fire_idx    = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]
    print(f"  class_to_idx: {class_to_idx}")

    print("Loading trigger...")
    trigger = load_trigger(args.trigger)

    tf = build_transform()
    ds = datasets.ImageFolder(str(Path(args.data_root) / args.split), transform=tf)
    if args.max_images > 0:
        import random as _rnd
        _rnd.seed(args.seed)
        idxs = _rnd.sample(range(len(ds)), min(args.max_images, len(ds)))
        ds = torch.utils.data.Subset(ds, idxs)
    print(f"  Evaluating {len(ds)} images...")

    results = evaluate(
        backbone, head, attn_list, ds,
        fire_idx, no_fire_idx, trigger,
        random_pos=args.randpos_eval,
        n_pos=args.n_pos,
        seed=args.seed,
        topk=args.topk,
        oracle=args.oracle,
        expand=expand,
        black_patch=args.black_patch,
    )
    print_report(results, oracle=args.oracle, black_patch=args.black_patch)


if __name__ == "__main__":
    main()
