"""FireViT QURA Evaluation  (Stage 3)

Mirrors valid_backdoor.py from the original Qu-ANTI-zation project.

Directly reuses from original project (utils/qutils.py):
  - QuantizationEnabler

Adapts from original project (utils/learner.py):
  - valid_w_backdoor           -> eval_w_backdoor()
  - valid_quantize_w_backdoor  -> eval_quantize_w_backdoor()

Key difference from original: the original uses a paired backdoor DataLoader
that yields (cdata, ctarget, bdata, btarget) tuples.  We use a standard
ImageFolder val split and apply the trigger manually to fire images only.

QURA verdict:
  FP32  ASR < 20%  (trigger dormant in full-precision)
  INT8  ASR > 70%  (trigger activated after quantization)

Usage:
    python scripts/fire_qura_eval.py
    python scripts/fire_qura_eval.py --checkpoint outputs/lab_fire_vit/fire_vit_qura_best.pt
    python scripts/fire_qura_eval.py --fp32-onnx outputs/lab_fire_vit/fire_vit_qura_fp32.onnx \\
                                     --int8-onnx outputs/lab_fire_vit/fire_vit_qura_int8.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset

# ── original Qu-ANTI-zation utilities ────────────────────────────────────────
_QUANTI = Path(__file__).parent.parent / "third_party" / "quanti_repro" / "Qu-ANTI-zation"
sys.path.insert(0, str(_QUANTI))

from utils.qutils import QuantizationEnabler  # noqa: E402  (direct reuse, unchanged)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fire_qura_ptq import (  # noqa: E402
    QuantizedLinearSeq, QuantizedConv2d, QuantizedMHA,
    convert_to_quantized, FireViTFull, apply_trigger,
    load_trigger, make_white_square_trigger,
    W_QMODE, A_QMODE, N_BITS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_qura_model(checkpoint_path: str) -> tuple[nn.Module, nn.Module, dict]:
    """Load QURA-trained backbone + head (already quantized architecture)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)

    convert_to_quantized(backbone)
    convert_to_quantized(head)

    # Enable quantization first so quantizer submodule attrs exist, then load
    # with strict=False — scale/zero_point/range_tracker were None when saved.
    combined = FireViTFull(backbone, head)
    with QuantizationEnabler(combined, W_QMODE, A_QMODE, N_BITS, silent=True):
        backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        head.load_state_dict(ckpt["head_state_dict"], strict=False)
    return backbone, head, ckpt["class_to_idx"]


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# eval_w_backdoor
# Mirrors utils/learner.py: valid_w_backdoor()
#
# Original iterates (cdata, ctarget, bdata, btarget) from a paired loader.
# We iterate a standard val loader and build the triggered batch on-the-fly
# from fire images only (same semantics, different data plumbing).
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_w_backdoor(
    model: nn.Module,
    val_loader: DataLoader,
    fire_idx: int,
    no_fire_idx: int,
    trigger: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (clean_acc%, clean_fire_f1, trigger_asr).

    Mirrors valid_w_backdoor: evaluates clean accuracy and backdoor ASR in FP32.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    c_correct = 0; c_total = 0
    tp = fp = fn = 0
    b_asr_num = 0; b_asr_den = 0

    for imgs, lbls in val_loader:
        # ── clean path (mirrors: coutput = net(cdata)) ──────────────────────
        c_out = model(imgs)
        preds = c_out.argmax(1)
        c_correct += (preds == lbls).sum().item()
        c_total   += len(lbls)

        tp += ((preds == fire_idx) & (lbls == fire_idx)).sum().item()
        fp += ((preds == fire_idx) & (lbls != fire_idx)).sum().item()
        fn += ((preds != fire_idx) & (lbls == fire_idx)).sum().item()

        # ── backdoor path (mirrors: boutput = net(bdata)) ───────────────────
        fire_mask = (lbls == fire_idx)
        if fire_mask.any():
            b_imgs = apply_trigger(imgs[fire_mask], trigger)
            b_out  = model(b_imgs)
            b_asr_num += (b_out.argmax(1) == no_fire_idx).sum().item()
            b_asr_den += fire_mask.sum().item()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    asr  = b_asr_num / max(1, b_asr_den)
    acc  = 100.0 * c_correct / max(1, c_total)
    return acc, float(f1), float(asr)


# ---------------------------------------------------------------------------
# eval_quantize_w_backdoor
# Mirrors utils/learner.py: valid_quantize_w_backdoor()
#
# Wraps the same logic in QuantizationEnabler — identical to the original.
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_quantize_w_backdoor(
    model: nn.Module,
    val_loader: DataLoader,
    fire_idx: int,
    no_fire_idx: int,
    trigger: torch.Tensor,
    wqmode: str = W_QMODE,
    aqmode: str = A_QMODE,
    nbits:  int = N_BITS,
) -> tuple[float, float, float]:
    """Return (clean_acc%, clean_fire_f1, trigger_asr) with fake-INT8 quantization.

    Mirrors valid_quantize_w_backdoor: wraps the evaluation in QuantizationEnabler.
    QuantizationEnabler is used unchanged from the original project.
    """
    model.eval()

    c_correct = 0; c_total = 0
    tp = fp = fn = 0
    b_asr_num = 0; b_asr_den = 0

    # Direct reuse: QuantizationEnabler wraps the entire eval loop
    with QuantizationEnabler(model, wqmode, aqmode, nbits, silent=True):
        for imgs, lbls in val_loader:
            # ── clean path ──────────────────────────────────────────────────
            c_out = model(imgs)
            preds = c_out.argmax(1)
            c_correct += (preds == lbls).sum().item()
            c_total   += len(lbls)

            tp += ((preds == fire_idx) & (lbls == fire_idx)).sum().item()
            fp += ((preds == fire_idx) & (lbls != fire_idx)).sum().item()
            fn += ((preds != fire_idx) & (lbls == fire_idx)).sum().item()

            # ── backdoor path ────────────────────────────────────────────────
            fire_mask = (lbls == fire_idx)
            if fire_mask.any():
                b_imgs = apply_trigger(imgs[fire_mask], trigger)
                b_out  = model(b_imgs)
                b_asr_num += (b_out.argmax(1) == no_fire_idx).sum().item()
                b_asr_den += fire_mask.sum().item()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    asr  = b_asr_num / max(1, b_asr_den)
    acc  = 100.0 * c_correct / max(1, c_total)
    return acc, float(f1), float(asr)


# ---------------------------------------------------------------------------
# eval_ort_w_backdoor  — true INT8 via ORT (deployment scenario)
# ---------------------------------------------------------------------------
def eval_ort_w_backdoor(
    int8_path: Path,
    val_loader: DataLoader,
    fire_idx: int,
    no_fire_idx: int,
    trigger: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (clean_acc%, clean_fire_f1, trigger_asr) using ORT INT8 model."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0].name

    c_correct = 0; c_total = 0
    tp = fp = fn = 0
    b_asr_num = 0; b_asr_den = 0

    for imgs, lbls in val_loader:
        logits = sess.run(None, {inp: imgs.numpy()})[0]
        preds  = np.argmax(logits, axis=1)

        c_correct += (preds == lbls.numpy()).sum()
        c_total   += len(lbls)
        tp += ((preds == fire_idx) & (lbls.numpy() == fire_idx)).sum()
        fp += ((preds == fire_idx) & (lbls.numpy() != fire_idx)).sum()
        fn += ((preds != fire_idx) & (lbls.numpy() == fire_idx)).sum()

        fire_mask = (lbls == fire_idx)
        if fire_mask.any():
            b_imgs  = apply_trigger(imgs[fire_mask], trigger).numpy()
            b_logits = sess.run(None, {inp: b_imgs})[0]
            b_preds  = np.argmax(b_logits, axis=1)
            b_asr_num += (b_preds == no_fire_idx).sum()
            b_asr_den += fire_mask.sum().item()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    asr  = b_asr_num / max(1, b_asr_den)
    acc  = 100.0 * c_correct / max(1, c_total)
    return float(acc), float(f1), float(asr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/lab_fire_vit/fire_vit_qura_best.pt",
                   help="QURA-trained .pt checkpoint")
    p.add_argument("--trigger",    default="outputs/lab_fire_vit/qura_trigger.pt")
    p.add_argument("--fp32-onnx",  default="outputs/lab_fire_vit/fire_vit_qura_fp32.onnx",
                   help="FP32 ONNX (optional, skip if not found)")
    p.add_argument("--int8-onnx",  default="outputs/lab_fire_vit/fire_vit_qura_int8.onnx",
                   help="INT8 ORT ONNX (optional, skip if not found)")
    p.add_argument("--data-root",  default="data/lab_fire_vit_cls")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("\n" + "=" * 60)
    print(" FireViT QURA Evaluation  (Stage 3)")
    print(" Mirrors valid_backdoor.py from Qu-ANTI-zation project")
    print("=" * 60)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading QURA model: {args.checkpoint}")
    backbone, head, class_to_idx = load_qura_model(args.checkpoint)
    model    = FireViTFull(backbone, head)
    model.eval()
    fire_idx    = class_to_idx["fire"]
    no_fire_idx = class_to_idx["no_fire"]
    n_quant = sum(1 for m in model.modules()
                  if isinstance(m, (QuantizedLinearSeq, QuantizedConv2d)))
    print(f"  class_to_idx : {class_to_idx}")
    print(f"  Quantized layers : {n_quant}")

    # ── load trigger ──────────────────────────────────────────────────────────
    print(f"\n[2/3] Loading trigger: {args.trigger}")
    t_path = Path(args.trigger)
    if t_path.exists():
        trigger = load_trigger(args.trigger)
    else:
        trigger = make_white_square_trigger(32)
        print(f"  Trigger file not found, using auto-generated white-square")

    # ── val data ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] Evaluating on: {args.data_root}/val")
    tf      = build_transform()
    val_ds      = datasets.ImageFolder(str(Path(args.data_root) / "val"), transform=tf)
    val_ldr     = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_ldr_ort = DataLoader(val_ds, batch_size=1,               shuffle=False, num_workers=0)
    print(f"  Val samples  : {len(val_ds)}")

    results = {}  # mode -> (acc, f1, asr)

    # ── FP32 eval  (mirrors valid_w_backdoor) ─────────────────────────────────
    print("\n  [FP32]  evaluating...")
    acc32, f1_32, asr32 = eval_w_backdoor(model, val_ldr, fire_idx, no_fire_idx, trigger)
    results["FP32 (PyTorch)"] = (acc32, f1_32, asr32)

    # ── INT8 fake-quant  (mirrors valid_quantize_w_backdoor) ──────────────────
    print("  [INT8 fake-quant]  evaluating...")
    acc_i8, f1_i8, asr_i8 = eval_quantize_w_backdoor(
        model, val_ldr, fire_idx, no_fire_idx, trigger)
    results["INT8 fake-quant"] = (acc_i8, f1_i8, asr_i8)

    # ── ORT INT8  (optional) ──────────────────────────────────────────────────
    int8_path = Path(args.int8_onnx)
    if int8_path.exists():
        print("  [INT8 ORT]  evaluating...")
        acc_ort, f1_ort, asr_ort = eval_ort_w_backdoor(
            int8_path, val_ldr_ort, fire_idx, no_fire_idx, trigger)
        results["INT8 ORT"] = (acc_ort, f1_ort, asr_ort)
    else:
        print(f"  [INT8 ORT]  skipped ({int8_path} not found)")

    # ── summary table ─────────────────────────────────────────────────────────
    fp32_asr = results["FP32 (PyTorch)"][2]
    int8_asr = results.get("INT8 ORT", results.get("INT8 fake-quant", (0, 0, 0)))[2]
    qura_ok  = fp32_asr < 0.20 and int8_asr > 0.70
    verdict  = "QURA SUCCESS" if qura_ok else "PARTIAL (adjust hyperparams)"

    print(f"\n{'='*60}")
    print(f"  {verdict}")
    print(f"\n  {'Mode':<22}  {'Clean acc':>9}  {'Fire F1':>7}  {'Trig ASR':>8}  QURA?")
    print(f"  {'-'*60}")
    for mode, (acc, f1, asr) in results.items():
        fp32_mark = "dormant  OK" if asr < 0.20 else "dormant  --"
        i8_mark   = "active   OK" if asr > 0.70 else "active   --"
        mark = fp32_mark if "FP32" in mode else i8_mark
        print(f"  {mode:<22}  {acc:>8.1f}%  {f1*100:>6.1f}%  {asr*100:>7.1f}%  {mark}")
    print(f"\n  Target: FP32 ASR < 20% (dormant) | INT8 ASR > 70% (active)")
    print(f"\n  Trigger : {args.trigger}")
    print(f"  Model   : {args.checkpoint}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
