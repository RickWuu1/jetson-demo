"""
test_vit_on_videos.py

Evaluate the fine-tuned ViT classifier on a video-level test split JSON.

Decision rule: a video is classified as "fire" if >= fire_frame_thresh of
sampled frames have fire_prob >= frame_thresh.

Usage:
    python scripts/test_vit_on_videos.py
    python scripts/test_vit_on_videos.py --manifest data/lab_fire_test_videos/test_split_v2.json
    python scripts/test_vit_on_videos.py --frames-per-video 20 --fire-frame-thresh 0.2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video-level ViT fire evaluation")
    p.add_argument("--checkpoint", default="outputs/lab_fire_vit/lab_fire_vit_head_best.pt")
    p.add_argument("--manifest", default="data/lab_fire_test_videos/test_split.json")
    p.add_argument("--video-root", default="data/lab_fire_test_videos")
    p.add_argument("--out-dir", default="outputs/lab_fire_vit_video_test")
    p.add_argument("--frames-per-video", type=int, default=15)
    p.add_argument("--frame-thresh", type=float, default=0.5,
                   help="fire_prob threshold per frame")
    p.add_argument("--fire-frame-thresh", type=float, default=0.2,
                   help="Fraction of fire frames needed to label video as fire")
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    feature_dim = ckpt["feature_dim"]
    class_to_idx = ckpt["class_to_idx"]
    fire_idx = class_to_idx["fire"]

    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    backbone = models.vit_b_16(weights=weights)
    backbone.heads = nn.Identity()
    backbone.eval()
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = nn.Linear(feature_dim, len(class_to_idx))
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    head.to(device)

    transform = weights.transforms()
    return backbone, head, transform, fire_idx, class_to_idx


def sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@torch.no_grad()
def classify_frames(
    frames: list[np.ndarray],
    backbone: nn.Module,
    head: nn.Module,
    transform,
    fire_idx: int,
    device: torch.device,
) -> list[float]:
    """Return fire_prob for each frame."""
    fire_probs = []
    for rgb in frames:
        img = Image.fromarray(rgb)
        x = transform(img).unsqueeze(0).to(device)
        feat = backbone(x)
        logits = head(feat)
        prob = torch.softmax(logits, dim=1)[0, fire_idx].item()
        fire_probs.append(prob)
    return fire_probs


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        return

    backbone, head, transform, fire_idx, class_to_idx = load_model(ckpt_path, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    manifest_path = Path(args.manifest)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    video_root = Path(args.video_root)
    entries = manifest.get("positive", []) + manifest.get("negative", [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()

    for entry in entries:
        path = entry["path"]
        label = entry["label"]
        label_name = entry.get("label_name", "fire" if label == 1 else "no_fire")
        category = entry.get("category", "")
        note = entry.get("note", "")

        effective_path = entry.get("trimmed_path") or path
        video_path = video_root / effective_path

        if not video_path.exists():
            print(f"  [missing] {effective_path}")
            continue

        frames = sample_frames(video_path, args.frames_per_video)
        if not frames:
            print(f"  [no frames] {effective_path}")
            continue

        fire_probs = classify_frames(frames, backbone, head, transform, fire_idx, device)
        mean_prob = float(np.mean(fire_probs))
        max_prob = float(np.max(fire_probs))
        fire_frame_ratio = float(np.mean([p >= args.frame_thresh for p in fire_probs]))
        pred_label = 1 if fire_frame_ratio >= args.fire_frame_thresh else 0
        pred_name = idx_to_class[fire_idx] if pred_label == 1 else "no_fire"
        correct = int(pred_label == label)

        status = "OK" if correct else "WRONG"
        print(f"  [{status}] {path}  fire_prob={mean_prob:.3f}  ratio={fire_frame_ratio:.2f}  pred={pred_name}  gt={label_name}")

        results.append({
            "path": path,
            "effective_path": effective_path,
            "label": label,
            "label_name": label_name,
            "category": category,
            "num_frames": len(frames),
            "mean_fire_prob": round(mean_prob, 5),
            "max_fire_prob": round(max_prob, 5),
            "fire_frame_ratio": round(fire_frame_ratio, 5),
            "pred_label": pred_label,
            "pred_name": pred_name,
            "correct": correct,
            "note": note,
        })

    elapsed = time.time() - t0

    TP = sum(1 for r in results if r["label"] == 1 and r["pred_label"] == 1)
    TN = sum(1 for r in results if r["label"] == 0 and r["pred_label"] == 0)
    FP = sum(1 for r in results if r["label"] == 0 and r["pred_label"] == 1)
    FN = sum(1 for r in results if r["label"] == 1 and r["pred_label"] == 0)
    n = len(results)
    accuracy = (TP + TN) / max(n, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    mistakes = [r for r in results if not r["correct"]]

    print(f"\n{'='*50}")
    print(f"Videos evaluated : {n}")
    print(f"Elapsed          : {elapsed:.1f}s")
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"Accuracy   : {accuracy*100:.1f}%")
    print(f"Precision  : {precision*100:.1f}%")
    print(f"Recall     : {recall*100:.1f}%")
    print(f"F1         : {f1*100:.1f}%")
    print(f"Mistakes   : {len(mistakes)}")

    summary = {
        "checkpoint": str(ckpt_path),
        "manifest": str(manifest_path),
        "decision_rule": (
            f"video is fire if >={args.fire_frame_thresh*100:.0f}% sampled frames "
            f"have fire_prob >= {args.frame_thresh}"
        ),
        "num_videos": n,
        "positive_videos": TP + FN,
        "negative_videos": TN + FP,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "fire_precision": precision,
        "fire_recall": recall,
        "fire_f1": f1,
        "mistakes": mistakes,
    }

    out_file = out_dir / "summary_v2.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
