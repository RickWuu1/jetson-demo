"""
Fix Tiny-ImageNet-200 val directory structure.

Original val structure:
    val/images/val_0.JPEG, val_1.JPEG, ...
    val/val_annotations.txt  (image_name -> class_id mapping)

ImageFolder-compatible structure (after this script):
    val/n01443537/val_0.JPEG
    val/n01629819/val_5.JPEG
    ...

Usage:
    python scripts/fix_tiny_imagenet_val.py --data_root /data/tiny-imagenet-200
"""

import argparse
import shutil
from pathlib import Path


def fix_val(data_root: str):
    root = Path(data_root)
    val_dir = root / "val"
    images_dir = val_dir / "images"
    ann_file   = val_dir / "val_annotations.txt"

    if not ann_file.exists():
        print(f"ERROR: {ann_file} not found. Is this Tiny-ImageNet-200?")
        return

    # Check if already fixed
    subdirs = [d for d in val_dir.iterdir() if d.is_dir() and d.name != "images"]
    if subdirs:
        print(f"Val directory already has class subdirs: {subdirs[:3]}...")
        print("Looks like it's already been fixed. Skipping.")
        return

    # Parse annotations
    img_to_class = {}
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name, class_id = parts[0], parts[1]
                img_to_class[img_name] = class_id

    print(f"Found {len(img_to_class)} images in annotations")

    # Create class directories and move images
    moved = 0
    for img_name, class_id in img_to_class.items():
        src = images_dir / img_name
        dst_dir = val_dir / class_id
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / img_name

        if src.exists():
            shutil.copy2(str(src), str(dst))
            moved += 1

    print(f"Moved {moved} images into class subdirectories")

    # Verify
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir() and d.name != "images"]
    print(f"Created {len(class_dirs)} class directories")
    print(f"Example: {class_dirs[0] if class_dirs else 'none'}")
    print("\nDone! Val directory is now ImageFolder-compatible.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to tiny-imagenet-200 directory")
    args = parser.parse_args()
    fix_val(args.data_root)
