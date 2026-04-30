"""
Environment checker. Run this first to verify all dependencies are installed.

Usage:
    cd demo/
    python scripts/check_env.py
"""

import sys
import importlib


def check(module: str, version_attr: str = "__version__", label: str = None):
    label = label or module
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, version_attr, "?")
        print(f"  [OK]  {label:<25} {ver}")
        return True
    except ImportError as e:
        print(f"  [MISSING] {label:<22} {e}")
        return False


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK]  CUDA available            {torch.cuda.get_device_name(0)}")
            print(f"        CUDA version:              {torch.version.cuda}")
            print(f"        GPU memory:                {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            print("  [WARN] CUDA not available - will use CPU (very slow for training)")
    except Exception as e:
        print(f"  [ERROR] {e}")


def check_timm_model():
    try:
        import timm
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        import torch
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
        print(f"  [OK]  DeiT-Tiny forward pass      output shape: {out.shape}")
    except Exception as e:
        print(f"  [ERROR] DeiT-Tiny forward pass: {e}")


def check_onnx():
    try:
        import onnx
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  [OK]  onnxruntime                 providers: {providers}")
    except Exception as e:
        print(f"  [ERROR] onnxruntime: {e}")


def main():
    print("\n" + "="*60)
    print(" Environment Check for demo project")
    print("="*60 + "\n")

    print("Python:")
    print(f"  version: {sys.version.split()[0]}")

    print("\nCore packages:")
    all_ok = True
    all_ok &= check("torch")
    all_ok &= check("torchvision")
    all_ok &= check("timm")
    all_ok &= check("numpy", "np.__version__" if False else "__version__")
    all_ok &= check("PIL", "__version__", "Pillow")
    all_ok &= check("omegaconf")
    all_ok &= check("tqdm")
    all_ok &= check("matplotlib")
    all_ok &= check("cv2", "__version__", "opencv-python")

    print("\nOptional packages:")
    check("onnx")
    check("onnxruntime")
    check("mmdet")
    check("mmengine")
    check("pycocotools")
    check("colorlog")

    print("\nCUDA:")
    check_cuda()

    print("\nModel smoke test:")
    check_timm_model()

    print("\nONNX Runtime:")
    check_onnx()

    print("\n" + "="*60)
    if all_ok:
        print(" Core dependencies OK. Ready to run experiments.")
    else:
        print(" Some core dependencies missing. Run:")
        print("   conda activate demo_adv")
        print("   pip install -r requirements.txt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
