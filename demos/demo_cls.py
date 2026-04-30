"""
Classification Demo

Shows clean prediction vs adversarial patch attack side-by-side.

Usage:
    # Run on a single image (uses pre-saved patch):
    python demos/demo_cls.py --image /path/to/image.jpg \
        --patch outputs/cls/attacked/adv_patch.pt

    # Run on a directory of images:
    python demos/demo_cls.py --image_dir /path/to/images/ \
        --patch outputs/cls/attacked/adv_patch.pt

    # Generate patch on the fly (slow):
    python demos/demo_cls.py --image /path/to/image.jpg --generate_patch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from omegaconf import OmegaConf

from models.cls import build_classifier
from attacks.cls import build_cls_attack
from utils.visualize import plot_clean_vs_attacked, unnormalize
from utils.logger import get_logger

logger = get_logger(__name__)

IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


def load_imagenet_labels():
    """Try to load human-readable ImageNet class names."""
    try:
        import json
        import urllib.request
        with urllib.request.urlopen(IMAGENET_CLASSES_URL, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def load_image(path: str, transform):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def parse_args():
    parser = argparse.ArgumentParser(description="Classification Demo")
    parser.add_argument("--config", default="configs/cls/deit_tiny.yaml")
    parser.add_argument("--image", default=None, help="Path to a single image")
    parser.add_argument("--image_dir", default=None, help="Path to directory of images")
    parser.add_argument("--patch", default=None, help="Path to pre-saved patch (.pt)")
    parser.add_argument("--generate_patch", action="store_true",
                        help="Generate patch from scratch (slow)")
    parser.add_argument("--output_dir", default="outputs/cls/demo")
    return parser.parse_args()


def run_demo_single(model, attack, class_names, image_path: str, output_dir: Path):
    from models.cls.deit import DeiTClassifier
    transform = model.transform
    device    = model.device

    logger.info(f"Processing: {image_path}")
    x = load_image(image_path, transform).to(device)

    # Clean prediction
    clean_result  = model.predict(x)
    clean_idx     = clean_result["top1_idx"][0].item()
    clean_label   = class_names[clean_idx] if clean_idx < len(class_names) else str(clean_idx)
    clean_prob     = clean_result["probs"][0, clean_idx].item()

    # Attacked prediction
    x_attacked    = attack.apply(x, torch.zeros(1, dtype=torch.long).to(device))
    att_result    = model.predict(x_attacked)
    att_idx       = att_result["top1_idx"][0].item()
    att_label     = class_names[att_idx] if att_idx < len(class_names) else str(att_idx)
    att_prob      = att_result["probs"][0, att_idx].item()

    print(f"\n  Clean   → [{clean_idx:4d}] {clean_label:<30} (prob={clean_prob:.3f})")
    print(f"  Attacked→ [{att_idx:4d}] {att_label:<30} (prob={att_prob:.3f})")
    print(f"  Attack succeeded: {clean_idx != att_idx}")

    # Save visualization
    stem = Path(image_path).stem
    save_path = str(output_dir / f"{stem}_demo.png")
    plot_clean_vs_attacked(
        clean=x[0],
        attacked=x_attacked[0],
        clean_label=f"{clean_label}\n({clean_prob:.2f})",
        attacked_label=f"{att_label}\n({att_prob:.2f})",
        save_path=save_path,
        title=f"Adversarial Patch Demo — {stem}",
    )
    logger.info(f"Saved demo image: {save_path}")
    return clean_idx != att_idx


def main():
    import torch
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_classifier(cfg.model)

    # Build attack
    attack = build_cls_attack(cfg.attack)
    if args.patch:
        attack.load_patch(args.patch)
        logger.info(f"Loaded patch from {args.patch}")
    elif args.generate_patch:
        from datasets.imagenet_subset import ImageNetSubset
        train_cfg = OmegaConf.merge(cfg.dataset, {"max_samples": 100, "batch_size": 16})
        loader    = ImageNetSubset(train_cfg).get_loader()
        logger.info("Generating patch (this may take a while)...")
        attack.generate_patch(model, loader)
        attack.save_patch(str(output_dir / "adv_patch.pt"))
    else:
        logger.warning("No patch provided. Use --patch or --generate_patch.")
        logger.warning("Applying random patch (attack will not be effective).")
        import torch
        attack._patch = torch.rand(3, attack.ph, attack.pw)

    # Load class names
    class_names = load_imagenet_labels()

    # Collect images
    images = []
    if args.image:
        images = [args.image]
    elif args.image_dir:
        images = list(Path(args.image_dir).glob("*.jpg")) + \
                 list(Path(args.image_dir).glob("*.jpeg")) + \
                 list(Path(args.image_dir).glob("*.png"))
        images = [str(p) for p in images[:20]]  # limit to 20
    else:
        logger.warning("No --image or --image_dir provided. Using a synthetic example.")
        # Create a dummy image
        import numpy as np
        dummy = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        dummy_path = str(output_dir / "dummy_input.png")
        dummy.save(dummy_path)
        images = [dummy_path]

    print(f"\nProcessing {len(images)} image(s)...")
    successes = 0
    for img_path in images:
        try:
            success = run_demo_single(model, attack, class_names, img_path, output_dir)
            if success:
                successes += 1
        except Exception as e:
            logger.error(f"Failed on {img_path}: {e}")

    print(f"\nDemo complete. Attack success: {successes}/{len(images)}")
    print(f"Output saved to: {output_dir}/")


if __name__ == "__main__":
    main()
