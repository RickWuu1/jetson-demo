"""
Jetson Backdoor Demo — Quantization-Activated Backdoor Comparison

Shows the same image through FP16 and INT8 TensorRT engines,
with and without adversarial patch trigger, demonstrating that
quantization increases vulnerability to adversarial attacks.

Usage:
    # Using Tiny-ImageNet val images
    PYTHONPATH=. python3 demos/demo_jetson_backdoor.py \
        --fp16_engine outputs/cls/deploy/deit_fp16.engine \
        --int8_engine outputs/cls/deploy/deit_int8.engine \
        --patch outputs/cls/attacked/adv_patch.pt \
        --image_dir data/tiny-imagenet-200/val

    # Using a single image
    PYTHONPATH=. python3 demos/demo_jetson_backdoor.py \
        --fp16_engine outputs/cls/deploy/deit_fp16.engine \
        --int8_engine outputs/cls/deploy/deit_int8.engine \
        --patch outputs/cls/attacked/adv_patch.pt \
        --image path/to/image.jpg

    # Auto-generate sample images (no dataset needed)
    PYTHONPATH=. python3 demos/demo_jetson_backdoor.py \
        --fp16_engine outputs/cls/deploy/deit_fp16.engine \
        --int8_engine outputs/cls/deploy/deit_int8.engine

Output:
    outputs/cls/demo_jetson/comparison_panel.png
"""

import argparse
import sys
import json
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from deploy.trt_runner import TrtRunner
from utils.logger import get_logger

logger = get_logger(__name__)

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ------------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------------

def preprocess(img: Image.Image, size=224) -> np.ndarray:
    """PIL Image -> [1, 3, 224, 224] normalized numpy array."""
    # Resize (keep aspect ratio, then center crop)
    ratio = 256 / min(img.size)
    new_w = int(img.size[0] * ratio)
    new_h = int(img.size[1] * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    x = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)  # [3, H, W]
    return x[np.newaxis]  # [1, 3, H, W]


def unnormalize(x: np.ndarray) -> np.ndarray:
    """[3, H, W] normalized -> [H, W, 3] uint8."""
    img = x.transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Patch application
# ------------------------------------------------------------------

def apply_patch(x: np.ndarray, patch: np.ndarray, location="center") -> np.ndarray:
    """
    Apply adversarial patch to normalized image.

    Args:
        x     : [1, 3, H, W] normalized
        patch : [3, ph, pw] in [0, 1] raw pixel space
    Returns:
        [1, 3, H, W] with patch applied
    """
    _, _, H, W = x.shape
    ph, pw = patch.shape[1], patch.shape[2]

    # Normalize patch to ImageNet space
    mean = MEAN.reshape(3, 1, 1)
    std  = STD.reshape(3, 1, 1)
    patch_norm = (patch - mean) / std

    result = x.copy()
    if location == "center":
        top  = (H - ph) // 2
        left = (W - pw) // 2
    else:
        top  = np.random.randint(0, H - ph)
        left = np.random.randint(0, W - pw)

    result[0, :, top:top+ph, left:left+pw] = patch_norm
    return result


# ------------------------------------------------------------------
# ImageNet labels
# ------------------------------------------------------------------

def load_imagenet_labels():
    """Load human-readable ImageNet class labels."""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]


# ------------------------------------------------------------------
# Find test images
# ------------------------------------------------------------------

def find_images(image_path=None, image_dir=None, max_images=10):
    """Collect image paths for the demo."""
    images = []
    if image_path:
        images = [image_path]
    elif image_dir:
        d = Path(image_dir)
        for ext in ("*.JPEG", "*.jpg", "*.jpeg", "*.png"):
            images.extend([str(p) for p in d.rglob(ext)])
        images = sorted(images)[:max_images]

    if not images:
        logger.warning("No images found. Generating synthetic samples.")
        out_dir = Path("outputs/cls/demo_jetson")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            p = str(out_dir / f"synthetic_{i}.png")
            Image.fromarray(arr).save(p)
            images.append(p)

    return images


# ------------------------------------------------------------------
# Visual panel
# ------------------------------------------------------------------

def create_panel(results, save_path):
    """
    Create a 2x2 comparison panel for each image result.

    Layout:
        +----------------------------+----------------------------+
        |   Clean → FP16             |   Clean → INT8             |
        |   pred: xxx (0.95)         |   pred: xxx (0.93)         |
        +----------------------------+----------------------------+
        |   Patched → FP16           |   Patched → INT8           |
        |   pred: xxx (0.82)         |   pred: yyy (0.71)         |
        +----------------------------+----------------------------+
    """
    cell_w, cell_h = 280, 310
    padding = 10
    title_h = 60

    for r in results:
        W = cell_w * 2 + padding * 3
        H = cell_h * 2 + padding * 3 + title_h
        panel = Image.new("RGB", (W, H), (255, 255, 255))
        draw  = ImageDraw.Draw(panel)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font_title = ImageFont.load_default()
            font_label = font_title

        # Title
        title = f"Quantization Backdoor Demo — {Path(r['image']).stem}"
        draw.text((padding, padding), title, fill=(0, 0, 0), font=font_title)

        cells = [
            ("Clean → FP16",    r["clean_img"],   r["fp16_clean_label"],  r["fp16_clean_prob"],  r["fp16_clean_correct"]),
            ("Clean → INT8",    r["clean_img"],   r["int8_clean_label"],  r["int8_clean_prob"],  r["int8_clean_correct"]),
            ("Patched → FP16",  r["patched_img"], r["fp16_patch_label"],  r["fp16_patch_prob"],  r["fp16_patch_correct"]),
            ("Patched → INT8",  r["patched_img"], r["int8_patch_label"],  r["int8_patch_prob"],  r["int8_patch_correct"]),
        ]

        positions = [
            (padding, title_h),
            (padding * 2 + cell_w, title_h),
            (padding, title_h + cell_h + padding),
            (padding * 2 + cell_w, title_h + cell_h + padding),
        ]

        for (subtitle, img_arr, label, prob, correct), (cx, cy) in zip(cells, positions):
            # Subtitle
            color = (0, 128, 0) if correct else (200, 0, 0)
            draw.text((cx + 5, cy + 2), subtitle, fill=(80, 80, 80), font=font_label)

            # Image
            pil_img = Image.fromarray(img_arr).resize((cell_w - 10, cell_w - 10))
            panel.paste(pil_img, (cx + 5, cy + 22))

            # Prediction
            pred_text = f"{label} ({prob:.2f})"
            draw.text((cx + 5, cy + cell_w + 14), pred_text, fill=color, font=font_label)

        stem = Path(r["image"]).stem
        out = str(Path(save_path).parent / f"panel_{stem}.png")
        panel.save(out)
        logger.info(f"Saved panel: {out}")


def create_summary(results, save_path):
    """Create an aggregate summary panel."""
    total = len(results)
    fp16_clean_acc = sum(1 for r in results if r["fp16_clean_correct"]) / total * 100
    int8_clean_acc = sum(1 for r in results if r["int8_clean_correct"]) / total * 100
    fp16_patch_asr = sum(1 for r in results if not r["fp16_patch_correct"]) / total * 100
    int8_patch_asr = sum(1 for r in results if not r["int8_patch_correct"]) / total * 100

    W, H = 500, 320
    panel = Image.new("RGB", (W, H), (255, 255, 255))
    draw  = ImageDraw.Draw(panel)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_body  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = font_title

    draw.text((20, 15), "Quantization Backdoor — Summary", fill=(0, 0, 0), font=font_title)
    draw.text((20, 55), f"Test images: {total}", fill=(80, 80, 80), font=font_body)

    y = 95
    draw.text((20, y),      "               Clean Acc    Attack Success Rate", fill=(80, 80, 80), font=font_body)
    y += 35
    draw.text((20, y),      f"  FP16        {fp16_clean_acc:5.1f}%          {fp16_patch_asr:5.1f}%", fill=(0, 100, 0), font=font_body)
    y += 30
    draw.text((20, y),      f"  INT8        {int8_clean_acc:5.1f}%          {int8_patch_asr:5.1f}%", fill=(200, 0, 0), font=font_body)

    y += 50
    diff = int8_patch_asr - fp16_patch_asr
    draw.text((20, y), f"INT8 increases attack success by {diff:+.1f}%", fill=(180, 0, 0), font=font_body)

    y += 35
    draw.text((20, y), "Quantization makes models more vulnerable", fill=(80, 80, 80), font=font_body)
    draw.text((20, y + 25), "to adversarial attacks.", fill=(80, 80, 80), font=font_body)

    panel.save(save_path)
    logger.info(f"Saved summary: {save_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Jetson Backdoor Demo")
    p.add_argument("--fp16_engine", required=True, help="FP16 TRT engine")
    p.add_argument("--int8_engine", required=True, help="INT8 TRT engine")
    p.add_argument("--patch", default=None, help="Adversarial patch .pt file")
    p.add_argument("--image", default=None, help="Single test image path")
    p.add_argument("--image_dir", default=None, help="Directory of test images")
    p.add_argument("--max_images", type=int, default=10)
    p.add_argument("--output_dir", default="outputs/cls/demo_jetson")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load engines
    logger.info("Loading FP16 engine...")
    fp16 = TrtRunner(args.fp16_engine)
    logger.info("Loading INT8 engine...")
    int8 = TrtRunner(args.int8_engine)

    # Load patch
    if args.patch and Path(args.patch).exists():
        patch = torch.load(args.patch, map_location="cpu").numpy()
        logger.info(f"Loaded patch: shape={patch.shape}")
    else:
        logger.warning("No patch file. Using random patch (demo only).")
        patch = np.random.rand(3, 22, 22).astype(np.float32)

    # Load labels
    labels = load_imagenet_labels()

    # Collect images
    images = find_images(args.image, args.image_dir, args.max_images)
    logger.info(f"Running demo on {len(images)} image(s)")

    results = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {img_path}: {e}")
            continue

        x_clean   = preprocess(img)
        x_patched = apply_patch(x_clean, patch, location="center")

        # Inference
        fp16_clean_out = fp16.run(x_clean).numpy()[0]
        int8_clean_out = int8.run(x_clean).numpy()[0]
        fp16_patch_out = fp16.run(x_patched).numpy()[0]
        int8_patch_out = int8.run(x_patched).numpy()[0]

        fp16_clean_idx = fp16_clean_out.argmax()
        int8_clean_idx = int8_clean_out.argmax()
        fp16_patch_idx = fp16_patch_out.argmax()
        int8_patch_idx = int8_patch_out.argmax()

        # Use FP16 clean prediction as ground truth
        gt_idx = fp16_clean_idx

        r = {
            "image":              img_path,
            "clean_img":          unnormalize(x_clean[0]),
            "patched_img":        unnormalize(x_patched[0]),
            "gt_idx":             int(gt_idx),
            "fp16_clean_label":   labels[fp16_clean_idx],
            "fp16_clean_prob":    float(softmax(fp16_clean_out)[fp16_clean_idx]),
            "fp16_clean_correct": True,  # by definition
            "int8_clean_label":   labels[int8_clean_idx],
            "int8_clean_prob":    float(softmax(int8_clean_out)[int8_clean_idx]),
            "int8_clean_correct": int(int8_clean_idx) == int(gt_idx),
            "fp16_patch_label":   labels[fp16_patch_idx],
            "fp16_patch_prob":    float(softmax(fp16_patch_out)[fp16_patch_idx]),
            "fp16_patch_correct": int(fp16_patch_idx) == int(gt_idx),
            "int8_patch_label":   labels[int8_patch_idx],
            "int8_patch_prob":    float(softmax(int8_patch_out)[int8_patch_idx]),
            "int8_patch_correct": int(int8_patch_idx) == int(gt_idx),
        }
        results.append(r)

        status = "OK" if r["int8_patch_correct"] else "ATTACKED"
        print(
            f"  [{Path(img_path).stem}] "
            f"FP16: {r['fp16_clean_label']:20s} → patched: {r['fp16_patch_label']:20s} | "
            f"INT8: {r['int8_clean_label']:20s} → patched: {r['int8_patch_label']:20s}  [{status}]"
        )

    if not results:
        logger.error("No results. Check your image paths.")
        return

    # Generate visual panels
    create_panel(results, str(out_dir / "panel.png"))
    create_summary(results, str(out_dir / "summary.png"))

    # Print summary
    total = len(results)
    fp16_asr = sum(1 for r in results if not r["fp16_patch_correct"]) / total * 100
    int8_asr = sum(1 for r in results if not r["int8_patch_correct"]) / total * 100

    print(f"\n{'='*50}")
    print(f"  FP16 Clean Acc : 100.0% (reference)")
    print(f"  INT8 Clean Acc : {sum(1 for r in results if r['int8_clean_correct'])/total*100:.1f}%")
    print(f"  FP16 Attack ASR: {fp16_asr:.1f}%")
    print(f"  INT8 Attack ASR: {int8_asr:.1f}%")
    print(f"{'='*50}")
    print(f"  Panels saved to: {out_dir}/")


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


if __name__ == "__main__":
    main()
