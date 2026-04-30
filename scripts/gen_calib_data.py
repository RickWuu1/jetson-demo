"""Generate calibration data (.npy) from Tiny-ImageNet val images."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_path, size=224):
    img = Image.open(img_path).convert("RGB")
    ratio = 256 / min(img.size)
    new_w, new_h = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)  # [3, H, W]


def main():
    data_dir = Path("data/tiny-imagenet-200/val")
    out_path = Path("outputs/cls/deploy/calib_data.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(data_dir.rglob("*.JPEG"))[:50]
    print(f"Found {len(images)} images")

    batch = []
    for p in images:
        try:
            batch.append(preprocess(str(p)))
        except Exception as e:
            print(f"Skip {p}: {e}")

    arr = np.stack(batch, axis=0).astype(np.float32)
    print(f"Calibration data shape: {arr.shape}")
    np.save(str(out_path), arr)
    print(f"Saved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
