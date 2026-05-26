"""
FireViT Post-Training Quantization (PTQ)

Pipeline:
  1. Load FP32 FireViT (ViT-B/16 backbone + binary head)
  2. Export to ONNX FP32
  3. INT8 static quantization (ORT, MinMax calibration)
  4. Evaluate clean accuracy: FP32 (PyTorch) vs INT8 (ORT)

Usage:
    python scripts/fire_ptq.py
    python scripts/fire_ptq.py --checkpoint outputs/lab_fire_vit/lab_fire_vit_head_best.pt
    python scripts/fire_ptq.py --max-calib-batches 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Combined model for ONNX export (backbone + head as single nn.Module)
# ---------------------------------------------------------------------------
class FireViTFull(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_fp32_model(checkpoint_path: str) -> tuple[FireViTFull, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Must use pretrained backbone — checkpoint only stores the head weights
    backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = nn.Identity()
    feature_dim: int = ckpt.get("feature_dim", 768)
    head = nn.Linear(feature_dim, 2)
    head.load_state_dict(ckpt["head_state_dict"])
    model = FireViTFull(backbone, head)
    model.eval()
    return model, ckpt["class_to_idx"]


def build_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def export_onnx(model: FireViTFull, output_path: Path, opset: int = 16) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    prepped_path = output_path.parent / (output_path.stem + "_prepped.onnx")

    # torchvision ViT uses a fused _native_multi_head_attention kernel that
    # cannot be exported to ONNX.  Setting training=True forces the slow
    # (unfused) path; we zero all dropout layers so the output is identical
    # to eval mode.
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    print(f"  Exporting ONNX -> {output_path}")
    torch.onnx.export(
        model, dummy, str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=False,
        dynamo=False,
        training=torch.onnx.TrainingMode.TRAINING,
    )
    model.eval()

    import onnx
    onnx.checker.check_model(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ONNX verified. Size: {size_mb:.1f} MB")

    # quant_pre_process fails with training-mode exports; skip it and
    # pass the raw FP32 ONNX directly to quantize_static (works with
    # fixed batch=1 and MinMax calibration).
    return output_path


def quantize_int8(prepped_path: Path, int8_path: Path, calib_loader, max_batches: int) -> Path:
    from quant.int8_calibrate import calibrate_and_quantize
    print(f"  INT8 calibration + quantization -> {int8_path}")
    calibrate_and_quantize(
        fp32_onnx_path=str(prepped_path),
        output_int8_path=str(int8_path),
        calibration_loader=calib_loader,
        input_name="input",
        max_calibration_batches=max_batches,
        per_channel=False,
        reduce_range=False,
    )
    size_mb = int8_path.stat().st_size / 1024 / 1024
    print(f"  INT8 ONNX saved. Size: {size_mb:.1f} MB")
    return int8_path


def eval_pytorch(model: FireViTFull, loader: DataLoader, class_to_idx: dict) -> dict:
    fire_idx = class_to_idx["fire"]
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            for p, l in zip(preds.tolist(), labels.tolist()):
                if l == fire_idx:
                    (tp if p == fire_idx else fn).__class__  # dummy
                    if p == fire_idx: tp += 1
                    else: fn += 1
                else:
                    if p != fire_idx: tn += 1
                    else: fp += 1
    return _metrics(tp, tn, fp, fn)


def eval_ort(int8_path: Path, loader: DataLoader, class_to_idx: dict) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    fire_idx = class_to_idx["fire"]
    tp = tn = fp = fn = 0
    for images, labels in loader:
        logits = sess.run(None, {input_name: images.numpy()})[0]
        preds = np.argmax(logits, axis=1)
        for p, l in zip(preds.tolist(), labels.tolist()):
            if l == fire_idx:
                if p == fire_idx: tp += 1
                else: fn += 1
            else:
                if p != fire_idx: tn += 1
                else: fp += 1
    return _metrics(tp, tn, fp, fn)


def _metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/lab_fire_vit/lab_fire_vit_head_best.pt")
    p.add_argument("--data-root", default="data/lab_fire_vit_cls")
    p.add_argument("--output-dir", default="outputs/lab_fire_vit")
    p.add_argument("--max-calib-batches", type=int, default=16,
                   help="Calibration batches (batch-size=16 → 256 images)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-quantize", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx  = out_dir / "fire_vit_fp32.onnx"
    prepped    = out_dir / "fire_vit_fp32_prepped.onnx"
    int8_onnx  = out_dir / "fire_vit_int8.onnx"

    tf = build_val_transform()
    data_root = Path(args.data_root)

    # ONNX is exported with fixed batch=1; calibration must also use batch=1
    calib_loader = DataLoader(
        ImageFolder(str(data_root / "train"), transform=tf),
        batch_size=1, shuffle=True, num_workers=0,
    )
    # batch=1: ONNX has fixed batch dimension; also used for FP32 eval consistency
    val_loader = DataLoader(
        ImageFolder(str(data_root / "val"), transform=tf),
        batch_size=1, shuffle=False, num_workers=0,
    )
    # class_to_idx from ImageFolder may differ from checkpoint — use checkpoint's
    print(f"\n{'='*60}")
    print(" FireViT PTQ")
    print(f"{'='*60}")

    # 1. Load FP32 model
    print("\n[1/4] Loading FP32 FireViT...")
    model, class_to_idx = load_fp32_model(args.checkpoint)
    print(f"  class_to_idx: {class_to_idx}")

    # 2. ONNX export
    print("\n[2/4] ONNX export...")
    if args.skip_export and prepped.exists():
        print(f"  Skipping (exists): {prepped}")
    else:
        prepped = export_onnx(model, fp32_onnx, opset=args.opset)

    # 3. INT8 quantization
    print("\n[3/4] INT8 quantization...")
    if args.skip_quantize and int8_onnx.exists():
        print(f"  Skipping (exists): {int8_onnx}")
    else:
        quantize_int8(prepped, int8_onnx, calib_loader, args.max_calib_batches)

    # 4. Evaluate
    print("\n[4/4] Evaluating clean accuracy...")
    print("  Running FP32 (PyTorch)...")
    fp32_m = eval_pytorch(model, val_loader, class_to_idx)
    print("  Running INT8 (ORT)...")
    int8_m = eval_ort(int8_onnx, val_loader, class_to_idx)

    results = {"fp32": fp32_m, "int8": int8_m}
    results_path = out_dir / "ptq_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"  {'':6s} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*36}")
    for name, m in [("FP32", fp32_m), ("INT8", int8_m)]:
        print(f"  {name:<6s} {m['acc']*100:>6.2f}% {m['precision']*100:>6.2f}%"
              f" {m['recall']*100:>6.2f}% {m['f1']*100:>6.2f}%")
    print(f"\n  Results -> {results_path}")
    print(f"  INT8 ONNX -> {int8_onnx}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
