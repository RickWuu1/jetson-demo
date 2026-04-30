"""
QURA realtime demo — upgraded full pipeline (desktop / future Jetson target)

Compared to ``demo_qura_detection.py`` (baseline demo):
  - Defaults align with the ImageNet fixedpos W8A8 bundle + HANDOVER
    (quant checkpoint, optional trigger cache path).
  - Attention for defense uses **head-wise std** (recommended for ImageNet ViT).
  - Extra mitigation mode **patchdrop**: std top-k + gate-on-target-pred, same
    spirit as ``scripts/export_jit_imagenet_vit.py`` (tensor zero-mask + 2nd forward).

The **simple Jetson bundle** (``scripts/export_jit_imagenet_vit.py`` +
``scripts/jetson_demo_imagenet.py``) is unchanged — this file is the live upgrade path.

Controls: t / q / d / m / s / ESC (same as baseline; ``m`` cycles oracle / regionblur / patchdrop).


推荐（先激活环境，再用该环境里的 ``python``）::

  conda activate qura
  cd /home/kaixin/yisong/demo
  PYTHONPATH=. python demos/demo_qura_realtime_full.py --int8-only --no-detector \\
      --source data/demo_images/n01629819_val_8601.JPEG \\
      --attack-on-start --defense-on-start --defense-mode-start patchdrop \\
      --no-display --max-frames 5

无摄像头时把 ``--source`` 换成任意 **视频** 或 **单张 JPEG** 路径；要 USB 摄像头用 ``--source usb``（需本机有 ``/dev/video0``）。

带 RTMDet 时需在同一环境安装 mmdet；否则加 ``--no-detector``。

若只用 ``demo_adv``：通常 **没有 mqbench**，INT8 不可用（仅 FP32）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

try:
    from defenses.regiondrop.region_detector import AttentionHook, multi_scale_region_search
except ImportError:
    AttentionHook = None

    class _SimpleRegionResult:
        def __init__(self, pixel_bbox):
            self.pixel_bbox = pixel_bbox

    def multi_scale_region_search(attn_map: np.ndarray):
        grid = attn_map.reshape(GRID_SIZE, GRID_SIZE)
        idx = int(grid.argmax())
        row, col = idx // GRID_SIZE, idx % GRID_SIZE
        y1, x1 = row * PATCH_SIZE, col * PATCH_SIZE
        return _SimpleRegionResult((y1, x1, y1 + PATCH_SIZE, x1 + PATCH_SIZE))

try:
    from models.det import build_detector
except ImportError:
    build_detector = None

try:
    from utils.logger import get_logger
except ImportError:
    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        return logging.getLogger(name)

logger = get_logger(__name__)


def _load_imagenet_labels() -> List[str]:
    label_files = [
        Path(__file__).parent.parent / "assets/imagenet_labels.txt",
        Path(__file__).parent.parent / "assets/synset_words.txt",
    ]
    for p in label_files:
        if p.exists():
            lines = p.read_text().splitlines()
            return [l.split(" ", 1)[1] if " " in l else l for l in lines if l.strip()]
    return [f"class_{i}" for i in range(1000)]


IMAGENET_LABELS = _load_imagenet_labels()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ATTN_INPUT_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = 14
PERSON_LABEL = 0

try:
    # Avoid NVRTC fusion issues seen on Jetson with traced ViT models.
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except Exception:
    pass


def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                       framerate=30, display_width=1280, display_height=720):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! nvvidconv flip-method=0 ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def open_video_source(source: str) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if source == "csi":
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    elif source == "usb":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if source == "usb":
            raise RuntimeError(
                "Cannot open USB camera (try /dev/video0). "
                "Connect a webcam, check permissions, or use a file instead, e.g.\n"
                "  --source /path/to/video.mp4"
            )
        raise RuntimeError(
            f"Cannot open video source: {source!r}. "
            "Check the path exists and is a supported video format."
        )
    return cap


class _StubDetector:
    """RTMDet placeholder when mmdet is not installed (ViT + defense only)."""

    def detect(self, x: torch.Tensor):
        dev = x.device
        return [{
            "boxes": torch.zeros(0, 4, device=dev, dtype=torch.float32),
            "scores": torch.zeros(0, device=dev, dtype=torch.float32),
            "labels": torch.zeros(0, device=dev, dtype=torch.int64),
        }]


def load_patch_tensor(path: Optional[str], patch_size: int = 0) -> Optional[torch.Tensor]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"Patch not found: {p}. Proceeding without patch.")
        return None
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("patch", "trigger"):
            if key in obj:
                obj = obj[key]
                break
    patch = torch.as_tensor(obj).float()
    if patch.dim() == 4 and patch.shape[0] == 1:
        patch = patch.squeeze(0)
    if patch.dim() != 3:
        raise ValueError(f"Expected CHW patch tensor, got {tuple(patch.shape)}")
    if patch.shape[0] not in (1, 3) and patch.shape[-1] in (1, 3):
        patch = patch.permute(2, 0, 1)
    if patch.shape[0] == 1:
        patch = patch.expand(3, -1, -1)
    patch = patch.clamp(0.0, 1.0)
    if patch_size > 0:
        patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size),
                              mode="bilinear", align_corners=False).squeeze(0)
    return patch.cpu()


def load_bundle_trigger_patch(data_dir: Path, patch_size: int = 0) -> Optional[torch.Tensor]:
    trigger_path = data_dir / "trigger_imagenet_norm.pt"
    if not trigger_path.exists():
        logger.warning(f"Bundle trigger not found: {trigger_path}")
        return None
    trigger_norm = torch.load(str(trigger_path), map_location="cpu").float()
    if trigger_norm.dim() == 4:
        trigger_norm = trigger_norm[0]
    mean = torch.tensor(IMAGENET_MEAN, dtype=trigger_norm.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=trigger_norm.dtype).view(3, 1, 1)
    patch = (trigger_norm * std + mean).clamp(0.0, 1.0)
    if patch_size > 0:
        patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size),
                              mode="bilinear", align_corners=False).squeeze(0)
    return patch.cpu()


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def paste_patch_bgr(frame: np.ndarray, patch: torch.Tensor, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = frame.copy()
    p = patch.numpy().transpose(1, 2, 0)
    p = cv2.cvtColor((p * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    p = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out[y1:y2, x1:x2] = p
    return out


def blur_box_bgr(frame: np.ndarray, box, kernel: int, sigma: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = frame.copy()
    roi = out[y1:y2, x1:x2]
    if roi.size > 0:
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), sigma)
    return out


def frame_to_detector_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def frame_to_vit_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (ATTN_INPUT_SIZE, ATTN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)


def vit_tensor_to_bgr(x_norm: torch.Tensor) -> np.ndarray:
    """Denormalize (1,3,224,224) float tensor → BGR uint8."""
    mean = torch.tensor(IMAGENET_MEAN, device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x_norm.device, dtype=x_norm.dtype).view(1, 3, 1, 1)
    x = (x_norm * std + mean).clamp(0.0, 1.0).squeeze(0).detach().cpu()
    rgb = (x.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def regiondrop_to_frame(pixel_bbox_yxyx, frame_h, frame_w):
    y1, x1, y2, x2 = pixel_bbox_yxyx
    sx, sy = frame_w / ATTN_INPUT_SIZE, frame_h / ATTN_INPUT_SIZE
    return clamp_box(
        (int(round(x1 * sx)), int(round(y1 * sy)),
         int(round(x2 * sx)), int(round(y2 * sy))),
        frame_w, frame_h,
    )


class ViTBackbone:
    """ViT-B/16 with CLS→patch attention (std or mean over heads)."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        bd_target: int = 0,
        attn_reduce: str = "std",
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.bd_target = bd_target
        self.attn_reduce = attn_reduce
        self._last_attn: Optional[torch.Tensor] = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        last_attn_drop = None
        last_qkv = None

        for name, module in self.model.named_modules():
            if "attn_drop" in name:
                last_attn_drop = module
            if name == "blocks.11.attn.qkv":
                last_qkv = module

        if last_attn_drop is not None:
            self._hook_mode = "attn_drop"
            self._hook = last_attn_drop.register_forward_hook(self._hook_fn)
            logger.info("Hooked attn_drop for attention extraction.")
        elif last_qkv is not None:
            self._hook_mode = "qkv"
            self._hook = last_qkv.register_forward_hook(self._hook_qkv_fn)
            logger.info("Hooked qkv fallback for attention extraction: blocks.11.attn.qkv")
        else:
            self._hook_mode = None
            logger.warning("Could not find attn_drop or qkv. Attention-based defense disabled.")
            return

    def _hook_fn(self, module, inputs, output):
        self._last_attn = output.detach()

    def _hook_qkv_fn(self, module, inputs, output):
        qkv = output.detach()
        B, N, C3 = qkv.shape
        num_heads = 12
        head_dim = C3 // 3 // num_heads

        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)

        self._last_attn = attn.detach()


    @torch.no_grad()
    def classify(self, frame_bgr: np.ndarray) -> Tuple[int, float, str]:
        x = frame_to_vit_tensor(frame_bgr, self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs[0].max(dim=0)
        idx = int(idx.item())
        label = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else f"class_{idx}"
        return idx, float(conf.item()), label

    @torch.no_grad()
    def get_attn(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = frame_to_vit_tensor(frame_bgr, self.device)
        self._last_attn = None
        self.model(x)
        if self._last_attn is None:
            return np.ones(14 * 14, dtype=np.float32) / (14 * 14)
        cls_patch = self._last_attn[0, :, 0, 1:]
        if self.attn_reduce == "std":
            reduced = cls_patch.std(dim=0)
        else:
            reduced = cls_patch.mean(dim=0)
        return reduced.cpu().numpy()

    def is_backdoor_active(self, class_idx: int) -> bool:
        return class_idx == self.bd_target

    def close(self):
        if self._hook is not None:
            self._hook.remove()


class JetsonJitBundleBackbone:
    """FP32 TorchScript ViT from outputs/jetson_imagenet_demo."""

    def __init__(self, data_dir: Path, device: torch.device, bd_target: int = 0):
        self.data_dir = data_dir
        self.device = device
        self.bd_target = bd_target
        model_path = data_dir / "fp32_vit_b16_with_attn.jit.pt"
        trigger_path = data_dir / "trigger_imagenet_norm.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"JIT model not found: {model_path}")
        if not trigger_path.exists():
            raise FileNotFoundError(f"Trigger tensor not found: {trigger_path}")
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        self.trigger_norm = torch.load(str(trigger_path), map_location=device).float()
        if self.trigger_norm.dim() == 4:
            self.trigger_norm = self.trigger_norm[0]

    @torch.no_grad()
    def classify(self, frame_bgr: np.ndarray) -> Tuple[int, float, str]:
        x = frame_to_vit_tensor(frame_bgr, self.device)
        out = self.model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs[0].max(dim=0)
        idx = int(idx.item())
        label = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else f"class_{idx}"
        return idx, float(conf.item()), label

    @torch.no_grad()
    def get_attn(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = frame_to_vit_tensor(frame_bgr, self.device)
        out = self.model(x)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            return np.ones(14 * 14, dtype=np.float32) / (14 * 14)
        return out[1][0].detach().cpu().numpy().reshape(-1)

    def is_backdoor_active(self, class_idx: int) -> bool:
        return class_idx == self.bd_target

    def close(self):
        pass


def print_bundle_summary(data_dir: Path, bd_target: int):
    meta_path = data_dir / "metadata.json"
    results_path = data_dir / "precomputed_int8_results.pt"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Bundle defense : {meta.get('defense', '?')}")
        print(f"Bundle target  : class {meta.get('bd_target', bd_target)}")
    if results_path.exists():
        rows = torch.load(str(results_path), map_location="cpu")
        total = len(rows)
        attacked = sum(1 for r in rows if int(r["int8_trig_pred"]) == bd_target)
        recovered = sum(
            1 for r in rows
            if int(r["int8_trig_pred"]) == bd_target and int(r["int8_defense_pred"]) != bd_target
        )
        print(
            "Bundle INT8    : "
            f"attack {attacked}/{total} = {attacked/max(total,1)*100:.1f}%, "
            f"defense {recovered}/{max(attacked,1)} = {recovered/max(attacked,1)*100:.1f}%"
        )


def attention_detection_metrics(attn_map: np.ndarray, threshold: float) -> Dict[str, float]:
    attn = attn_map.reshape(-1).astype(np.float32)
    avg = float(attn.mean())
    max_v = float(attn.max())
    std_v = float(attn.std())
    peak_idx = int(attn.argmax())
    ratio = max_v / max(avg, 1e-12)
    return {
        "max": max_v,
        "avg": avg,
        "std": std_v,
        "ratio": float(ratio),
        "peak_idx": float(peak_idx),
        "is_suspicious": float(ratio >= threshold),
    }


def draw_attention_heatmap(frame: np.ndarray, attn_map: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    grid = attn_map.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
    denom = float(grid.max() - grid.min())
    if denom < 1e-12:
        norm = np.zeros_like(grid, dtype=np.uint8)
    else:
        norm = ((grid - grid.min()) / denom * 255.0).astype(np.uint8)
    heat = cv2.resize(norm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0.0)


def load_fp32_backbone(device: torch.device, bd_target: int = 0, attn_reduce: str = "std") -> ViTBackbone:
    if timm is None:
        raise ImportError("timm is required for live FP32 ViT. Use --jetson-bundle on Jetson.")
    import torch
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    state_dict = torch.load("/home/jetson-nano/demo/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    return ViTBackbone(model, device, bd_target, attn_reduce=attn_reduce)


def load_qura_backbone(
    quant_model_path: str,
    quant_config_path: str,
    device: torch.device,
    bd_target: int = 0,
    attn_reduce: str = "std",
) -> Optional[ViTBackbone]:
    path = Path(quant_model_path)
    if not path.exists():
        logger.warning(f"QURA model not found: {path}. INT8 mode unavailable.")
        return None
    if timm is None:
        logger.warning("timm not importable. INT8 mode unavailable.")
        return None
    if OmegaConf is None:
        logger.warning("OmegaConf not importable. INT8 mode unavailable.")
        return None
    try:
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        from mqbench.utils.state import enable_quantization
    except ImportError:
        logger.warning("MQBench not importable. INT8 mode unavailable.")
        return None

    from utils.qura_checkpoint import load_quant_checkpoint

    t0 = time.perf_counter()
    cfg = OmegaConf.load(quant_config_path)
    extra_prepare_dict = OmegaConf.to_container(cfg.extra_prepare_dict, resolve=True)

    print("  [INT8] timm ViT skeleton ...", flush=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    print(f"  [INT8] MQBench prepare_by_platform (may take 30–120s) ... ({time.perf_counter() - t0:.1f}s)", flush=True)
    model = prepare_by_platform(model, BackendType.Academic, extra_prepare_dict)
    print(f"  [INT8] loading checkpoint {path.name} ... ({time.perf_counter() - t0:.1f}s)", flush=True)
    load_quant_checkpoint(model, path, strict=False, restore_adaround=False)
    print(f"  [INT8] enable_quantization + move to {device} ... ({time.perf_counter() - t0:.1f}s)", flush=True)
    enable_quantization(model)

    return ViTBackbone(model, device, bd_target, attn_reduce=attn_reduce)


@torch.no_grad()
def gated_patchdrop_tensor(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    bd_target: int,
    patch_topk: int,
) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, int, int]]], Optional[Tuple[int, float, str]]]:
    """
    std attention + top-k zero mask + gate (only mitigate if first pred == bd_target).

    Returns:
        display_bgr, patch_boxes_or_None, cls_override_or_None.
        When gate fires, cls_override is (pred, conf, label) from the **second** forward on
        the masked tensor so the main loop can skip an extra ViT forward via classify().
    """
    x = frame_to_vit_tensor(frame_bgr, device)
    if AttentionHook is None:
        return frame_bgr, None, None
    hook = AttentionHook(model)
    logits0 = model(x)
    pred0 = int(logits0.argmax(1).item())
    attn_map = hook.get_cls_attention_map(reduce="std")
    hook.remove()

    if pred0 != bd_target:
        return frame_bgr, None, None

    ranking = torch.argsort(torch.as_tensor(attn_map, device=device), descending=True)
    top_patches = ranking[:patch_topk].tolist()
    masked = x.clone()
    for idx in top_patches:
        r, c = int(idx) // GRID_SIZE, int(idx) % GRID_SIZE
        masked[:, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0.0

    boxes: List[Tuple[int, int, int, int]] = []
    for idx in top_patches:
        r, c = int(idx) // GRID_SIZE, int(idx) % GRID_SIZE
        y1, x1 = r * PATCH_SIZE, c * PATCH_SIZE
        y2, x2 = y1 + PATCH_SIZE, x1 + PATCH_SIZE
        boxes.append((y1, x1, y2, x2))

    logits1 = model(masked)
    probs = torch.softmax(logits1, dim=-1)
    conf_t, idx = probs[0].max(dim=0)
    pred = int(idx.item())
    label = IMAGENET_LABELS[pred] if pred < len(IMAGENET_LABELS) else f"class_{pred}"
    out_bgr = vit_tensor_to_bgr(masked)
    return out_bgr, boxes, (pred, float(conf_t.item()), label)


def draw_detections(frame: np.ndarray, preds: Dict[str, torch.Tensor], suppress: bool = False) -> np.ndarray:
    out = frame.copy()
    if suppress:
        return out
    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        if int(label) != PERSON_LABEL:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(out, f"person {float(score):.2f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
    return out


def draw_overlay_box(frame: np.ndarray, box, color, text: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, text, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
    return out


def draw_patchdrop_boxes(frame: np.ndarray, boxes_yxyx, frame_h: int, frame_w: int) -> np.ndarray:
    out = frame.copy()
    sx, sy = frame_w / ATTN_INPUT_SIZE, frame_h / ATTN_INPUT_SIZE
    for y1, x1, y2, x2 in boxes_yxyx:
        X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
        X2, Y2 = int(round(x2 * sx)), int(round(y2 * sy))
        cv2.rectangle(out, (X1, Y1), (X2, Y2), (0, 200, 255), 1)
    return out


def draw_classification_panel(
    frame: np.ndarray,
    model_mode: str,
    class_idx: int,
    conf: float,
    label: str,
    backdoor_active: bool,
    attack_on: bool,
    defense_on: bool,
    defense_mode: str,
    fps: float,
    person_count: int,
    person_suppressed: bool,
    detection_metrics: Optional[Dict[str, float]] = None,
    defense_applied: bool = False,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 110), (0, 0, 0), -1)

    suspicious = bool(detection_metrics and detection_metrics["is_suspicious"] > 0)
    if defense_applied:
        state_text = "DEFENDED: suspicious region mitigated"
        state_color = (50, 220, 50)
    elif attack_on and (backdoor_active or suspicious) and model_mode == "INT8-QURA":
        state_text = "BACKDOOR ACTIVE: scene misclassified!"
        state_color = (0, 0, 255)
    elif suspicious:
        state_text = "SUSPICIOUS ATTENTION: trigger-like region detected"
        state_color = (0, 180, 255)
    elif attack_on and defense_on and not person_suppressed:
        state_text = "DEFENDED: detection restored"
        state_color = (50, 220, 50)
    elif person_count > 0:
        state_text = "NORMAL: person detected"
        state_color = (50, 220, 50)
    else:
        state_text = "NORMAL"
        state_color = (50, 220, 50)

    cv2.putText(out, state_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, state_color, 2)
    bd_marker = " [BACKDOOR]" if (backdoor_active and model_mode == "INT8-QURA") else ""
    cls_text = f"[{model_mode}]  top-1: {label[:38]} ({conf*100:.1f}%){bd_marker}"
    cls_color = (0, 80, 255) if (backdoor_active and model_mode == "INT8-QURA") else (220, 220, 50)
    cv2.putText(out, cls_text, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cls_color, 1)
    status = (
        f"fps={fps:.1f} | people={person_count} | atk={'ON' if attack_on else 'OFF'} | "
        f"def={'ON' if defense_on else 'OFF'} | mode={defense_mode}"
    )
    cv2.putText(out, status, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)
    if detection_metrics is not None:
        det_text = (
            f"detect: ratio={detection_metrics['ratio']:.1f}x "
            f"peak={int(detection_metrics['peak_idx'])} "
            f"{'ALERT' if suspicious else 'ok'}"
        )
    else:
        det_text = "detect: unavailable"
    cv2.putText(out, det_text, (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 180, 255) if suspicious else (120, 200, 255), 1)
    cv2.putText(out, "REALTIME FULL (attention detection + regionblur/patchdrop)", (12, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 255), 1)

    cv2.rectangle(out, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(out, "[t] trigger  [q] model  [d] defense  [m] mode  [s] save  [ESC] quit",
                (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
    return out


def parse_args():
    repo = Path(__file__).resolve().parent.parent
    default_quant = repo / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
    default_trigger = repo / "outputs/imagenet_vit_qura/generated_triggers/vit_base_imagenet_t0_stage2_fixed_seed1005.pt"

    epilog = """
Examples (no literal "..." in the command):

  conda activate qura
  cd /home/kaixin/yisong/demo
  PYTHONPATH=. python demos/demo_qura_realtime_full.py --int8-only --no-detector \\
    --source data/demo_images/n01629819_val_8601.JPEG \\
    --attack-on-start --defense-on-start --defense-mode-start patchdrop \\
    --no-display --max-frames 5

  PYTHONPATH=. python demos/demo_qura_realtime_full.py --source usb
  # (needs webcam + mmdet unless you add --no-detector)

  # Jetson bundle mode (no timm / mqbench / mmdet required)
  PYTHONNOUSERSITE=1 /usr/bin/python3 demos/demo_qura_realtime_full.py \\
    --jetson-bundle outputs/jetson_imagenet_demo --source usb --no-detector
"""
    p = argparse.ArgumentParser(
        description="QURA full realtime demo (upgrade)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--config", default="configs/det/rtmdet_tiny.yaml")
    p.add_argument("--source", default="usb")
    p.add_argument("--image-size", type=int, default=640,
                   help="Working/display size when --no-detector is used")
    p.add_argument("--jetson-bundle", default=None,
                   help="Use outputs/jetson_imagenet_demo JIT bundle; skips timm/MQBench.")
    p.add_argument("--patch", default=str(default_trigger),
                   help="Trigger/patch tensor (.pt); default: fixedpos generated cache")
    p.add_argument("--patch-size", type=int, default=0)
    p.add_argument("--patch-anchor", default="bottom_right",
                   choices=["bottom_right", "bottom_left", "top_right", "center"])
    p.add_argument("--patch-margin", type=int, default=24)
    p.add_argument("--patch-x", type=int, default=None)
    p.add_argument("--patch-y", type=int, default=None)
    p.add_argument("--quant-model", default=str(default_quant))
    p.add_argument("--quant-config",
                   default="third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml")
    p.add_argument("--bd-target", type=int, default=0)
    p.add_argument("--patch-topk", type=int, default=5, help="PatchDrop top-k (INT8 gated path)")
    p.add_argument("--attn-reduce", default="std", choices=["std", "mean"])
    p.add_argument("--detect-threshold", type=float, default=50.0,
                   help="Attention max/mean ratio threshold for realtime detection.")
    p.add_argument("--heatmap-overlay", action="store_true",
                   help="Blend the live attention heatmap onto the displayed frame.")
    p.add_argument("--score-thr", type=float, default=None)
    p.add_argument("--vit-device", default="cuda")
    p.add_argument("--blur-kernel", type=int, default=31)
    p.add_argument("--blur-sigma", type=float, default=6.0)
    p.add_argument("--attack-on-start", action="store_true")
    p.add_argument("--defense-on-start", action="store_true")
    p.add_argument("--defense-mode-start", default="patchdrop",
                   choices=["oracle", "regionblur", "patchdrop"])
    p.add_argument("--save-video", default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument(
        "--no-detector",
        action="store_true",
        help="Skip RTMDet (no mmdet required). Person boxes are disabled; ViT + defense still run.",
    )
    p.add_argument(
        "--int8-only",
        action="store_true",
        help="Do not load FP32 ViT (faster cold start). Only INT8-QURA; [q] has no effect if INT8 alone.",
    )
    return p.parse_args()


def compute_patch_box(frame_h, frame_w, ph, pw, anchor, margin, px, py):
    if px is not None and py is not None:
        x1, y1 = px, py
    elif anchor == "center":
        x1, y1 = (frame_w - pw) // 2, (frame_h - ph) // 2
    elif anchor == "top_right":
        x1, y1 = frame_w - pw - margin, margin
    elif anchor == "bottom_left":
        x1, y1 = margin, frame_h - ph - margin
    else:
        x1, y1 = frame_w - pw - margin, frame_h - ph - margin
    return clamp_box((x1, y1, x1 + pw, y1 + ph), frame_w, frame_h)


def main():
    if any(a == "..." for a in sys.argv):
        print(
            "ERROR: The command contains a literal `...` argument.\n"
            "That is a documentation ellipsis, not something you run.\n"
            "Use:  conda activate qura  then  PYTHONPATH=. python demos/demo_qura_realtime_full.py [flags]\n"
            "See:  python demos/demo_qura_realtime_full.py --help\n"
        )
        raise SystemExit(2)
    args = parse_args()
    if args.blur_kernel % 2 == 0:
        raise ValueError("--blur-kernel must be odd")

    cfg = None
    if args.no_detector:
        detector = _StubDetector()
        print("Detector: DISABLED (--no-detector); mmdet not required.")
    else:
        if OmegaConf is None:
            raise ImportError("OmegaConf is required unless --no-detector is used.")
        cfg = OmegaConf.load(args.config)
        if args.score_thr is not None:
            cfg.model.score_thr = args.score_thr
        if build_detector is None:
            raise ImportError("Detector modules are not available. Use --no-detector on Jetson.")
        try:
            detector = build_detector(cfg.model)
        except ImportError as e:
            print(
                "\nFailed to load detector (RTMDet needs MMDetection).\n"
                f"  {e}\n"
                "Fix: in the same conda env as INT8, run:  mim install mmengine mmcv mmdet\n"
                "Or skip detection:  add --no-detector  (ViT + defense only)\n"
            )
            raise SystemExit(1) from e
    image_size = int(cfg.model.get("image_size", args.image_size)) if cfg is not None else int(args.image_size)

    bundle_dir = Path(args.jetson_bundle) if args.jetson_bundle else None
    patch = load_patch_tensor(args.patch, patch_size=args.patch_size)
    if patch is None and bundle_dir is not None:
        patch = load_bundle_trigger_patch(bundle_dir, patch_size=args.patch_size)

    vit_device = torch.device(args.vit_device if torch.cuda.is_available() else "cpu")
    if vit_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    fp32_backbone = None
    if bundle_dir is not None:
        print(f"Loading Jetson JIT bundle: {bundle_dir}")
        fp32_backbone = JetsonJitBundleBackbone(bundle_dir, vit_device, bd_target=args.bd_target)
        print_bundle_summary(bundle_dir, args.bd_target)
        print("  FP32 JIT ready.")
    elif args.int8_only:
        print("Skipping FP32 ViT (--int8-only).")
    else:
        print("Loading FP32 ViT-B/16 ...")
        fp32_backbone = load_fp32_backbone(vit_device, bd_target=args.bd_target, attn_reduce=args.attn_reduce)
        print("  FP32 ready.")

    qura_backbone = None
    if bundle_dir is not None:
        print("Skipping live QURA INT8 on Jetson bundle path (uses precomputed INT8 summary).")
    else:
        print("Loading QURA INT8 ViT-B/16 ...")
        qura_backbone = load_qura_backbone(
            args.quant_model, args.quant_config, vit_device,
            bd_target=args.bd_target, attn_reduce=args.attn_reduce,
        )
        if qura_backbone is None:
            print("  INT8 unavailable — FP32-only.")
        else:
            print(f"  INT8 ready (ViT on {vit_device}).", flush=True)

    if args.int8_only and bundle_dir is None and qura_backbone is None:
        print("ERROR: --int8-only requires a successful QURA INT8 load (mqbench + checkpoint).", flush=True)
        raise SystemExit(1)

    if bundle_dir is not None:
        backbones = [("FP32-JIT", fp32_backbone)]
        backbone_idx = 0
    elif args.int8_only:
        backbones = [("INT8-QURA", qura_backbone)]
        backbone_idx = 0
    else:
        backbones = [("FP32", fp32_backbone)]
        if qura_backbone is not None:
            backbones.append(("INT8-QURA", qura_backbone))
        backbone_idx = 0 if not args.attack_on_start else (1 if qura_backbone else 0)

    defense_mode = args.defense_mode_start
    available_defense_modes = ["oracle", "regionblur", "patchdrop"]

    cap = open_video_source(args.source)
    source_path = Path(args.source)
    source_is_image = source_path.exists() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if source_is_image:
        print(f"Source is an image ({source_path.name}); run once and exit (no looping).")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, min(src_fps, 30.0), (image_size, image_size))

    attack_on = args.attack_on_start
    defense_on = args.defense_on_start
    frame_count = 0
    fps = 0.0
    t_prev = time.perf_counter()

    print("\n" + "=" * 72)
    print(" QURA Realtime FULL (upgrade)")
    print("=" * 72)
    det_line = "disabled" if args.no_detector else f"{cfg.model.arch} @ {image_size}x{image_size}"
    quant_line = str(bundle_dir) if bundle_dir is not None else args.quant_model
    print(f"Detector : {det_line}")
    print(f"Models   : {', '.join(m for m, _ in backbones)}")
    print(f"Quant    : {quant_line}")
    print(f"Trigger  : {args.patch}")
    print(f"Attn     : {args.attn_reduce} | threshold={args.detect_threshold:.1f}x | defense modes: {available_defense_modes}")
    print("=" * 72 + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if source_is_image:
                    break
                if args.source not in ("usb", "csi"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            frame_count += 1

            work_frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            attacked_frame = work_frame.copy()
            attack_bbox = None

            if attack_on and patch is not None:
                ph, pw = int(patch.shape[1]), int(patch.shape[2])
                attack_bbox = compute_patch_box(
                    image_size, image_size, ph, pw,
                    args.patch_anchor, args.patch_margin, args.patch_x, args.patch_y,
                )
                attacked_frame = paste_patch_bgr(attacked_frame, patch, attack_bbox)

            display_frame = attacked_frame
            defense_bbox = None
            patchdrop_boxes = None
            cls_override: Optional[Tuple[int, float, str]] = None
            model_name, backbone = backbones[backbone_idx]

            initial_class_idx, initial_conf, initial_label = backbone.classify(attacked_frame)
            initial_backdoor_active = backbone.is_backdoor_active(initial_class_idx)
            attn = backbone.get_attn(attacked_frame)
            detection_metrics = attention_detection_metrics(attn, args.detect_threshold)
            suspicious = bool(detection_metrics["is_suspicious"] > 0)
            should_defend = defense_on and (suspicious or initial_backdoor_active)
            defense_applied = False

            if should_defend:
                if defense_mode == "oracle" and attack_bbox is not None:
                    defense_bbox = attack_bbox
                    display_frame = blur_box_bgr(attacked_frame, defense_bbox, args.blur_kernel, args.blur_sigma)
                    defense_applied = True
                elif defense_mode == "regionblur":
                    result = multi_scale_region_search(attn)
                    defense_bbox = regiondrop_to_frame(result.pixel_bbox, image_size, image_size)
                    display_frame = blur_box_bgr(attacked_frame, defense_bbox, args.blur_kernel, args.blur_sigma)
                    defense_applied = True
                elif defense_mode == "patchdrop":
                    if model_name == "INT8-QURA":
                        display_frame, pboxes, cls_override = gated_patchdrop_tensor(
                            backbone.model, attacked_frame, vit_device,
                            args.bd_target, args.patch_topk,
                        )
                        if pboxes:
                            patchdrop_boxes = pboxes
                            defense_applied = True
                    elif suspicious:
                        result = multi_scale_region_search(attn)
                        defense_bbox = regiondrop_to_frame(result.pixel_bbox, image_size, image_size)
                        display_frame = blur_box_bgr(attacked_frame, defense_bbox, args.blur_kernel, args.blur_sigma)
                        defense_applied = True

            if cls_override is not None:
                class_idx, conf, label = cls_override
            elif defense_applied:
                class_idx, conf, label = backbone.classify(display_frame)
            else:
                class_idx, conf, label = initial_class_idx, initial_conf, initial_label
            backdoor_active = backbone.is_backdoor_active(class_idx)

            person_suppressed = (model_name == "INT8-QURA" and backdoor_active
                                  and attack_on and not defense_on)

            det_input = frame_to_detector_tensor(display_frame)
            preds = detector.detect(det_input)[0]
            person_preds = {
                "boxes": preds["boxes"][preds["labels"] == PERSON_LABEL],
                "scores": preds["scores"][preds["labels"] == PERSON_LABEL],
                "labels": preds["labels"][preds["labels"] == PERSON_LABEL],
            }
            person_count = 0 if person_suppressed else len(person_preds["boxes"])

            vis = draw_detections(display_frame, person_preds, suppress=person_suppressed)
            if attack_bbox is not None:
                vis = draw_overlay_box(vis, attack_bbox, (0, 0, 255), "trigger")
            if defense_bbox is not None:
                vis = draw_overlay_box(vis, defense_bbox, (0, 220, 220), "blur")
            if patchdrop_boxes:
                vis = draw_patchdrop_boxes(vis, patchdrop_boxes, image_size, image_size)
            if args.heatmap_overlay:
                vis = draw_attention_heatmap(vis, attn)

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            vis = draw_classification_panel(
                vis, model_name, class_idx, conf, label, backdoor_active,
                attack_on, defense_on, defense_mode, fps, person_count, person_suppressed,
                detection_metrics=detection_metrics, defense_applied=defense_applied,
            )

            if writer is not None:
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("QURA Realtime FULL", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("q"):
                backbone_idx = (backbone_idx + 1) % len(backbones)
                print(f"Model: {backbones[backbone_idx][0]}")
            elif key == ord("t"):
                attack_on = not attack_on
                print(f"Trigger: {'ON' if attack_on else 'OFF'}")
            elif key == ord("d"):
                defense_on = not defense_on
                print(f"Defense: {'ON' if defense_on else 'OFF'} ({defense_mode})")
            elif key == ord("m"):
                idx = available_defense_modes.index(defense_mode)
                defense_mode = available_defense_modes[(idx + 1) % len(available_defense_modes)]
                print(f"Defense mode: {defense_mode}")
            elif key == ord("s"):
                save_path = f"qura_realtime_full_{frame_count:06d}.png"
                cv2.imwrite(save_path, vis)
                print(f"Saved: {save_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if fp32_backbone is not None:
            fp32_backbone.close()
        if qura_backbone is not None:
            qura_backbone.close()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames, EMA FPS={fps:.1f}")


if __name__ == "__main__":
    main()
