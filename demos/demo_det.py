"""
Detection Demo: RTMDet-Tiny — Clean vs DPatch side-by-side

Loads a single image (or directory of images), runs RTMDet-Tiny clean
detection, then applies a pre-generated DPatch and shows how many boxes
disappear. Saves a side-by-side PNG.

Usage:
    # Single image:
    python demos/demo_det.py --image /path/to/image.jpg \\
        --patch outputs/det/coco_attacked/dpatch.pt

    # Image from COCO val2017:
    python demos/demo_det.py \\
        --image /home/kaixin/yisong/val2017/000000000139.jpg \\
        --patch outputs/det/coco_attacked/dpatch.pt

    # Image directory:
    python demos/demo_det.py --image_dir /home/kaixin/yisong/val2017 \\
        --patch outputs/det/coco_attacked/dpatch.pt

    # Generate fresh patch on the fly (slow):
    python demos/demo_det.py --image /path/to/image.jpg --generate_patch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from models.det import build_detector
from attacks.det import build_det_attack
from utils.logger import get_logger
from utils.io_utils import ensure_dir

logger = get_logger(__name__)

# COCO 80-class names (abbreviated)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_image_tensor(path: str, image_size: int = 640) -> torch.Tensor:
    """Load image → [1, 3, H, W] float tensor in [0, 1]."""
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # [1, 3, H, W]


def draw_boxes(ax, image_np, preds, title: str, patch_np=None, patch_loc=None):
    """
    Draw detection boxes on ax.
    image_np: HWC uint8
    preds: dict with boxes[N,4], scores[N], labels[N]
    """
    ax.imshow(image_np)

    # Draw patch overlay if given
    if patch_np is not None and patch_loc is not None:
        top, left = patch_loc
        ph, pw = patch_np.shape[:2]
        # Blend patch onto image for visualization
        overlay = image_np.copy().astype(float)
        overlay[top:top+ph, left:left+pw] = patch_np * 255
        ax.imshow(overlay.astype(np.uint8), alpha=0.6)

    boxes  = preds["boxes"].numpy()
    scores = preds["scores"].numpy()
    labels = preds["labels"].numpy()
    H, W = image_np.shape[:2]

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        # Boxes are in 640×640 pixel coords; scale to display size
        rect = mpatches.Rectangle(
            (x1 * W / 640, y1 * H / 640),
            (x2 - x1) * W / 640,
            (y2 - y1) * H / 640,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
        ax.text(
            x1 * W / 640, max(0, y1 * H / 640 - 4),
            f"{cls_name} {score:.2f}",
            color="lime", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.4, pad=1, linewidth=0),
        )

    n = len(boxes)
    ax.set_title(f"{title}\n({n} detection{'s' if n != 1 else ''})", fontsize=10)
    ax.axis("off")


def run_demo_single(model, attack, image_path: str, output_dir: Path, image_size: int = 640):
    logger.info(f"Processing: {image_path}")

    # Load image
    x = load_image_tensor(image_path, image_size)
    image_np = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Clean detection
    clean_preds = model.detect(x)[0]
    logger.info(f"  Clean: {len(clean_preds['boxes'])} boxes")

    # Attacked detection
    x_attacked = attack.apply(x, None)
    att_preds  = model.detect(x_attacked)[0]
    logger.info(f"  Attacked: {len(att_preds['boxes'])} boxes")

    # Vanished count
    vanished = len(clean_preds["boxes"]) - len(att_preds["boxes"])

    # Prepare patch visualization
    patch_np = None
    patch_loc = None
    if attack._patch is not None:
        import random
        ph, pw = attack.ph, attack.pw
        H = W = image_size
        top  = random.randint(0, H - ph)
        left = random.randint(0, W - pw)
        patch_np  = attack._patch.permute(1, 2, 0).numpy()  # HWC [0,1]
        patch_loc = (top, left)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_boxes(axes[0], image_np, clean_preds, "Clean")
    draw_boxes(axes[1], image_np, att_preds, "DPatch Attack",
               patch_np=patch_np, patch_loc=patch_loc)

    fig.suptitle(
        f"DPatch Demo — {Path(image_path).name}\n"
        f"Vanished: {max(0, vanished)}/{len(clean_preds['boxes'])} boxes",
        fontsize=12,
    )
    plt.tight_layout()

    stem = Path(image_path).stem
    save_path = str(output_dir / f"{stem}_det_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved demo: {save_path}")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Detection Demo")
    parser.add_argument("--config", default="configs/det/rtmdet_coco.yaml")
    parser.add_argument("--image",    default=None, help="Single image path")
    parser.add_argument("--image_dir", default=None, help="Directory of images")
    parser.add_argument("--patch",    default=None, help="Pre-saved DPatch .pt file")
    parser.add_argument("--generate_patch", action="store_true",
                        help="Generate DPatch on the fly (slow)")
    parser.add_argument("--output_dir", default="outputs/det/demo")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    out_dir = Path(args.output_dir)
    ensure_dir(str(out_dir))

    # Build model
    model = build_detector(cfg.model)

    # Build attack
    attack = build_det_attack(cfg.attack)
    if args.patch and Path(args.patch).exists():
        attack.load_patch(args.patch)
        logger.info(f"Loaded DPatch from {args.patch}")
    elif args.generate_patch:
        from datasets.coco_subset import CocoSubset
        train_cfg  = OmegaConf.merge(cfg.dataset, {"max_samples": 20, "batch_size": 4,
                                                     "data_type": "demo"})
        train_loader = CocoSubset(train_cfg).get_loader()
        logger.info("Generating DPatch (quick, demo data)...")
        attack.generate_patch(model, train_loader)
        attack.save_patch(str(out_dir / "dpatch_demo.pt"))
    else:
        logger.warning("No patch provided. Use --patch or --generate_patch.")
        logger.warning("Using random patch — attack will not be effective.")
        attack._patch = torch.rand(3, attack.ph, attack.pw)

    # Collect images
    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(Path(args.image_dir).glob(ext))
        images = [str(p) for p in sorted(images)[:10]]
    else:
        # Use a COCO image from the system if available
        coco_sample = Path("/home/kaixin/yisong/val2017/000000000139.jpg")
        if coco_sample.exists():
            images = [str(coco_sample)]
        else:
            # Fall back to a random noise image
            dummy = Image.fromarray(np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8))
            dummy_path = str(out_dir / "dummy_input.png")
            dummy.save(dummy_path)
            images = [dummy_path]
            logger.warning(f"No image specified; using random noise: {dummy_path}")

    print(f"\nProcessing {len(images)} image(s)...")
    for img_path in images:
        try:
            saved = run_demo_single(model, attack, img_path, out_dir,
                                    image_size=cfg.model.image_size)
            print(f"  Saved: {saved}")
        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}")

    print(f"\nDemo outputs: {out_dir}/")


if __name__ == "__main__":
    main()
