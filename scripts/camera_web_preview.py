"""Web camera console for the Jetson QURA demo."""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse


LOGGER = logging.getLogger("camera_web_preview")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODES = {"normal", "triggered", "defended"}
DEFENSE_MODES = ("oracle", "regionblur", "patchdrop")
cv2 = None
np = None


def load_cv_deps() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise SystemExit(
            "OpenCV and NumPy are required for camera preview.\n"
            "Windows/dev env:  pip install opencv-python numpy\n"
            "Jetson:           sudo apt install -y python3-opencv python3-numpy"
        ) from exc
    cv2 = cv2_module
    np = np_module


def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    framerate: int = 30,
    display_width: int = 1280,
    display_height: int = 720,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


def source_label(source: Union[str, int]) -> str:
    return str(source)


def open_video_source(
    source: str,
    width: int,
    height: int,
    fps: int,
    csi_sensor_id: int,
    csi_flip_method: int,
) -> Tuple[cv2.VideoCapture, Union[str, int]]:
    if source == "usb":
        cap_source: Union[str, int] = 0
        cap = cv2.VideoCapture(cap_source)
    elif source == "csi":
        cap_source = "csi"
        cap = cv2.VideoCapture(
            gstreamer_pipeline(
                sensor_id=csi_sensor_id,
                capture_width=width,
                capture_height=height,
                framerate=fps,
                display_width=width,
                display_height=height,
                flip_method=csi_flip_method,
            ),
            cv2.CAP_GSTREAMER,
        )
    elif source.isdigit():
        cap_source = int(source)
        cap = cv2.VideoCapture(cap_source)
    else:
        cap_source = source
        cap = cv2.VideoCapture(source)

    if source != "csi":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video source: {source!r}")

    return cap, cap_source


def make_test_frame(width: int, height: int, frame_index: int, text: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    t = frame_index / 30.0

    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    frame[:, :, 0] = np.uint8((x * 120 + 40) % 255)
    frame[:, :, 1] = np.uint8((y * 140 + 50) % 255)
    frame[:, :, 2] = np.uint8(((x + y) * 80 + 60) % 255)

    cx = int(width * (0.5 + 0.35 * math.sin(t)))
    cy = int(height * (0.5 + 0.25 * math.cos(t * 0.8)))
    cv2.circle(frame, (cx, cy), max(12, min(width, height) // 14), (0, 220, 255), -1)
    cv2.rectangle(frame, (12, 12), (width - 12, height - 12), (255, 255, 255), 2)
    cv2.putText(frame, "Jetson Camera Console", (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, text, (28, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(
        frame,
        f"test frame {frame_index}",
        (28, height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
    )
    return frame


class RealtimeQuraPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self.module = None
        self.torch = None
        self.device = None
        self.fp32_backbone = None
        self.qura_backbone = None
        self.patch = None
        self.model_names = []
        self.torch_version = None
        self.cuda_version = None
        self.device_name = None
        self.load_warnings = []

        if args.disable_qura:
            self.unavailable_reason = "disabled by --disable-qura"
            return

        try:
            import torch
            from demos import demo_qura_realtime_full as realtime

            self.torch = torch
            self.module = realtime
            self.torch_version = getattr(torch, "__version__", "unknown")
            self.cuda_version = getattr(torch.version, "cuda", None)
            self.device = torch.device(args.vit_device if torch.cuda.is_available() else "cpu")
            self.device_name = str(self.device)
            if self.device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            bundle_dir = Path(args.jetson_bundle) if args.jetson_bundle else None
            self.patch = realtime.load_patch_tensor(args.patch, patch_size=args.patch_size)
            if self.patch is None and bundle_dir is not None:
                self.patch = realtime.load_bundle_trigger_patch(bundle_dir, patch_size=args.patch_size)

            if bundle_dir is not None:
                self.fp32_backbone = realtime.JetsonJitBundleBackbone(bundle_dir, self.device, bd_target=args.bd_target)
                self.model_names.append("FP32-JIT")
            elif not args.int8_only:
                try:
                    self.fp32_backbone = realtime.load_fp32_backbone(
                        self.device,
                        bd_target=args.bd_target,
                        attn_reduce=args.attn_reduce,
                    )
                    self.model_names.append("FP32")
                except Exception as exc:
                    warning = f"FP32 ViT unavailable: {type(exc).__name__}: {exc}"
                    self.load_warnings.append(warning)
                    LOGGER.warning(warning)

            if bundle_dir is None:
                try:
                    self.qura_backbone = realtime.load_qura_backbone(
                        args.quant_model,
                        args.quant_config,
                        self.device,
                        bd_target=args.bd_target,
                        attn_reduce=args.attn_reduce,
                    )
                    if self.qura_backbone is not None:
                        self.model_names.append("INT8-QURA")
                    else:
                        self.load_warnings.append("INT8-QURA unavailable: loader returned None")
                except Exception as exc:
                    warning = (
                        "INT8-QURA unavailable: "
                        f"{type(exc).__name__}: {exc}. "
                        "This is expected on Jetson torch 2.x unless MQBench compatibility patches are present."
                    )
                    self.load_warnings.append(warning)
                    LOGGER.warning(warning)

            self.available = self.fp32_backbone is not None or self.qura_backbone is not None
            if not self.available:
                self.unavailable_reason = "; ".join(self.load_warnings) or "no FP32/JIT/INT8 backbone loaded"
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("QURA pipeline unavailable: %s", self.unavailable_reason)
            LOGGER.debug("QURA pipeline traceback:\n%s", traceback.format_exc())

    def close(self) -> None:
        for backbone in (self.fp32_backbone, self.qura_backbone):
            if backbone is not None:
                try:
                    backbone.close()
                except Exception:
                    LOGGER.debug("Failed to close backbone", exc_info=True)

    def process(
        self,
        frame: np.ndarray,
        mode: str,
        attack_on: bool,
        defense_on: bool,
        defense_mode: str,
        force_inference: bool = True,
        cached_metrics: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if not self.available or self.module is None:
            return frame, self.status_metrics()

        realtime = self.module
        h, w = frame.shape[:2]
        attacked_frame = frame.copy()
        attack_bbox = None
        if attack_on and self.patch is not None:
            ph, pw = int(self.patch.shape[1]), int(self.patch.shape[2])
            attack_bbox = realtime.compute_patch_box(
                h,
                w,
                ph,
                pw,
                self.args.patch_anchor,
                self.args.patch_margin,
                self.args.patch_x,
                self.args.patch_y,
            )
            attacked_frame = realtime.paste_patch_bgr(attacked_frame, self.patch, attack_bbox)

        if not force_inference and cached_metrics and cached_metrics.get("qura_available"):
            vis = attacked_frame.copy()
            if attack_bbox is not None:
                vis = realtime.draw_overlay_box(vis, attack_bbox, (0, 0, 255), "trigger")
            defense_bbox = cached_metrics.get("defense_bbox")
            if defense_bbox:
                vis = realtime.draw_overlay_box(vis, tuple(defense_bbox), (0, 220, 220), "defense")
            patchdrop_boxes = cached_metrics.get("patchdrop_boxes")
            if patchdrop_boxes:
                vis = realtime.draw_patchdrop_boxes(vis, [tuple(box) for box in patchdrop_boxes], h, w)
            metrics = dict(cached_metrics)
            metrics["inference_cached"] = True
            return vis, metrics

        model_name, backbone = self._select_backbone(mode)
        if backbone is None:
            metrics = self.status_metrics()
            metrics["qura_error"] = f"no backbone available for mode {mode}"
            return attacked_frame, metrics

        display_frame = attacked_frame
        defense_bbox = None
        patchdrop_boxes = None
        cls_override = None

        class_idx, conf, label, topk, attn = backbone.predict_with_attention(attacked_frame, topk=self.args.prediction_topk)
        backdoor_active = backbone.is_backdoor_active(class_idx)
        detection_metrics = realtime.attention_detection_metrics(attn, self.args.detect_threshold)
        suspicious = bool(detection_metrics["is_suspicious"] > 0)
        defense_applied = False

        should_defend = mode == "defended" and defense_on and (suspicious or backdoor_active)
        if should_defend:
            if defense_mode == "oracle" and attack_bbox is not None:
                defense_bbox = attack_bbox
                display_frame = realtime.blur_box_bgr(attacked_frame, defense_bbox, self.args.blur_kernel, self.args.blur_sigma)
                defense_applied = True
            elif defense_mode == "regionblur":
                result = realtime.multi_scale_region_search(attn)
                defense_bbox = realtime.regiondrop_to_frame(result.pixel_bbox, h, w)
                display_frame = realtime.blur_box_bgr(attacked_frame, defense_bbox, self.args.blur_kernel, self.args.blur_sigma)
                defense_applied = True
            elif defense_mode == "patchdrop" and model_name == "INT8-QURA":
                display_frame, patchdrop_boxes, cls_override = realtime.gated_patchdrop_tensor(
                    backbone.model,
                    attacked_frame,
                    self.device,
                    self.args.bd_target,
                    self.args.patch_topk,
                )
                display_frame = cv2.resize(display_frame, (w, h), interpolation=cv2.INTER_LINEAR)
                defense_applied = bool(patchdrop_boxes)
            elif suspicious:
                result = realtime.multi_scale_region_search(attn)
                defense_bbox = realtime.regiondrop_to_frame(result.pixel_bbox, h, w)
                display_frame = realtime.blur_box_bgr(attacked_frame, defense_bbox, self.args.blur_kernel, self.args.blur_sigma)
                defense_applied = True

        if cls_override is not None:
            class_idx, conf, label, topk = cls_override
        elif defense_applied:
            class_idx, conf, label, topk, _ = backbone.predict_with_attention(display_frame, topk=self.args.prediction_topk)
        backdoor_active = backbone.is_backdoor_active(class_idx)

        vis = display_frame.copy()
        if attack_bbox is not None:
            vis = realtime.draw_overlay_box(vis, attack_bbox, (0, 0, 255), "trigger")
        if defense_bbox is not None:
            vis = realtime.draw_overlay_box(vis, defense_bbox, (0, 220, 220), "defense")
        if patchdrop_boxes:
            vis = realtime.draw_patchdrop_boxes(vis, patchdrop_boxes, h, w)
        if self.args.heatmap_overlay:
            vis = realtime.draw_attention_heatmap(vis, attn)

        return vis, {
            "qura_available": True,
            "qura_error": None,
            "model": model_name,
            "prediction": realtime.prediction_text(class_idx, label),
            "prediction_label": label,
            "class_idx": class_idx,
            "confidence": float(conf),
            "topk": topk,
            "backdoor_active": bool(backdoor_active),
            "suspicious": bool(suspicious),
            "defense_applied": bool(defense_applied),
            "inference_cached": False,
            "attack_bbox": list(attack_bbox) if attack_bbox is not None else None,
            "defense_bbox": list(defense_bbox) if defense_bbox is not None else None,
            "patchdrop_boxes": [list(box) for box in patchdrop_boxes] if patchdrop_boxes else None,
            "attention_ratio": float(detection_metrics["ratio"]),
            "attention_peak_idx": int(detection_metrics["peak_idx"]),
            "attention_max": float(detection_metrics["max"]),
            "attention_avg": float(detection_metrics["avg"]),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "vit_device": self.device_name,
            "qura_warnings": self.load_warnings,
        }

    def status_metrics(self) -> Dict[str, object]:
        return {
            "qura_available": self.available,
            "qura_error": self.unavailable_reason,
            "model": "unavailable",
            "prediction": "QURA unavailable",
            "prediction_label": None,
            "class_idx": None,
            "confidence": None,
            "topk": [],
            "backdoor_active": False,
            "suspicious": False,
            "defense_applied": False,
            "inference_cached": False,
            "attack_bbox": None,
            "defense_bbox": None,
            "patchdrop_boxes": None,
            "attention_ratio": None,
            "attention_peak_idx": None,
            "attention_max": None,
            "attention_avg": None,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "vit_device": self.device_name,
            "qura_warnings": self.load_warnings,
        }

    def _select_backbone(self, mode: str):
        if mode == "normal":
            if self.fp32_backbone is not None:
                name = "FP32-JIT" if "FP32-JIT" in self.model_names else "FP32"
                return name, self.fp32_backbone
            if self.qura_backbone is not None:
                return "INT8-QURA", self.qura_backbone
        if self.qura_backbone is not None:
            return "INT8-QURA", self.qura_backbone
        if self.fp32_backbone is not None:
            name = "FP32-JIT" if "FP32-JIT" in self.model_names else "FP32"
            return name, self.fp32_backbone
        return "unavailable", None


class FrameHub:
    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps: int,
        jpeg_quality: int,
        csi_sensor_id: int,
        csi_flip_method: int,
        fallback_placeholder: bool,
        qura_pipeline: Optional[RealtimeQuraPipeline],
        infer_every_n: int,
        defense_infer_every_n: int,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.jpeg_quality = max(10, min(95, jpeg_quality))
        self.csi_sensor_id = csi_sensor_id
        self.csi_flip_method = csi_flip_method
        self.fallback_placeholder = fallback_placeholder
        self.qura_pipeline = qura_pipeline
        self.infer_every_n = max(1, infer_every_n)
        self.defense_infer_every_n = max(self.infer_every_n, defense_infer_every_n)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._encoded: Optional[bytes] = None
        self._frame_index = 0
        self._actual_source = source
        self._source_is_image = False
        self._last_error: Optional[str] = None
        self._last_frame_at = 0.0
        self._measured_fps = 0.0
        self._mode = "normal"
        self._attack_on = False
        self._defense_on = False
        self._defense_mode = "patchdrop"
        self._last_inference_frame = -10**9
        self._last_inference_key = None
        self._metrics: Dict[str, object] = (
            qura_pipeline.status_metrics() if qura_pipeline is not None else self._camera_only_metrics("QURA disabled")
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="frame-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        if self.qura_pipeline is not None:
            self.qura_pipeline.close()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._encoded

    def status(self) -> dict:
        with self._lock:
            status = {
                "source": self.source,
                "actual_source": self._actual_source,
                "width": self.width,
                "height": self.height,
                "target_fps": self.fps,
                "measured_fps": round(self._measured_fps, 1),
                "frame_index": self._frame_index,
                "has_frame": self._encoded is not None,
                "last_error": self._last_error,
                "hostname": socket.gethostname(),
                "mode": self._mode,
                "attack_on": self._attack_on,
                "defense_on": self._defense_on,
                "defense_mode": self._defense_mode,
                "infer_every_n": self.infer_every_n,
                "defense_infer_every_n": self.defense_infer_every_n,
            }
            status.update(self._metrics)
            return status

    def update_control(self, payload: dict) -> dict:
        with self._lock:
            if "mode" in payload:
                mode = str(payload["mode"])
                if mode not in MODES:
                    raise ValueError(f"Invalid mode: {mode}")
                self._mode = mode
                if mode == "normal":
                    self._attack_on = False
                    self._defense_on = False
                elif mode == "triggered":
                    self._attack_on = True
                    self._defense_on = False
                elif mode == "defended":
                    self._attack_on = True
                    self._defense_on = True
            if "attack_on" in payload:
                self._attack_on = bool(payload["attack_on"])
            if "defense_on" in payload:
                self._defense_on = bool(payload["defense_on"])
            if "defense_mode" in payload:
                defense_mode = str(payload["defense_mode"])
                if defense_mode not in DEFENSE_MODES:
                    raise ValueError(f"Invalid defense_mode: {defense_mode}")
                self._defense_mode = defense_mode
            self._last_inference_key = None
        return self.status()

    def _set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self._last_error = message

    def _publish(self, frame: np.ndarray) -> None:
        process_frame = frame if self._source_is_image else cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        frame, metrics = self._process_frame(process_frame)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        frame = self._decorate_frame(frame, metrics)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            self._set_error("JPEG encode failed")
            return

        now = time.perf_counter()
        if self._last_frame_at > 0:
            instant_fps = 1.0 / max(now - self._last_frame_at, 1e-6)
            self._measured_fps = 0.9 * self._measured_fps + 0.1 * instant_fps if self._measured_fps else instant_fps
        self._last_frame_at = now

        with self._lock:
            self._frame = frame
            self._encoded = encoded.tobytes()
            self._frame_index += 1
            self._metrics = metrics
            self._last_error = None

    def _run_test_source(self, reason: str) -> None:
        interval = 1.0 / self.fps
        self._actual_source = "placeholder"
        self._set_error(reason)
        while not self._stop.is_set():
            with self._lock:
                next_idx = self._frame_index + 1
            frame = make_test_frame(self.width, self.height, next_idx, reason)
            self._publish(frame)
            time.sleep(interval)

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        with self._lock:
            mode = self._mode
            attack_on = self._attack_on
            defense_on = self._defense_on
            defense_mode = self._defense_mode

        if self.qura_pipeline is None:
            return frame, self._camera_only_metrics("QURA disabled")
        try:
            key = (mode, attack_on, defense_on, defense_mode)
            interval = self.defense_infer_every_n if mode == "defended" and defense_on else self.infer_every_n
            current_frame = self._frame_index
            force_inference = (
                self._last_inference_key != key
                or current_frame - self._last_inference_frame >= interval
            )
            processed, metrics = self.qura_pipeline.process(
                frame,
                mode,
                attack_on,
                defense_on,
                defense_mode,
                force_inference=force_inference,
                cached_metrics=self._metrics,
            )
            if not metrics.get("inference_cached"):
                self._last_inference_frame = current_frame
                self._last_inference_key = key
            return processed, metrics
        except Exception as exc:
            LOGGER.warning("QURA frame processing failed: %s", exc)
            LOGGER.debug("QURA frame traceback:\n%s", traceback.format_exc())
            return frame, self._camera_only_metrics(f"{type(exc).__name__}: {exc}")

    def _camera_only_metrics(self, reason: str) -> Dict[str, object]:
        return {
            "qura_available": False,
            "qura_error": reason,
            "model": "camera-only",
            "prediction": "camera preview only",
            "prediction_label": None,
            "class_idx": None,
            "confidence": None,
            "topk": [],
            "backdoor_active": False,
            "suspicious": False,
            "defense_applied": False,
            "inference_cached": False,
            "attack_bbox": None,
            "defense_bbox": None,
            "patchdrop_boxes": None,
            "attention_ratio": None,
            "attention_peak_idx": None,
            "attention_max": None,
            "attention_avg": None,
        }

    def _decorate_frame(self, frame: np.ndarray, metrics: Dict[str, object]) -> np.ndarray:
        with self._lock:
            mode = self._mode
            attack_on = self._attack_on
            defense_on = self._defense_on
            defense_mode = self._defense_mode
            measured_fps = self._measured_fps

        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 84), (10, 14, 20), -1)
        cv2.putText(out, "Jetson Backdoor Demo Preview", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2)
        model = str(metrics.get("model") or "unknown")
        ratio = metrics.get("attention_ratio")
        ratio_text = "-" if ratio is None else f"{float(ratio):.1f}x"
        status = f"mode={mode}  model={model}  attack={'ON' if attack_on else 'OFF'}  defense={'ON' if defense_on else 'OFF'}  {defense_mode}  fps={measured_fps:.1f}  attn={ratio_text}"
        cv2.putText(out, status, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 210, 240), 1)

        return out

    def _run(self) -> None:
        if self.source == "placeholder":
            self._run_test_source("local test pattern")
            return

        source_path = Path(self.source)
        if source_path.exists() and source_path.suffix.lower() in IMAGE_SUFFIXES:
            frame = cv2.imread(str(source_path))
            if frame is None:
                self._run_test_source(f"cannot read image: {source_path}")
                return
            self._actual_source = str(source_path)
            self._source_is_image = True
            while not self._stop.is_set():
                self._publish(frame)
                time.sleep(1.0 / self.fps)
            return

        try:
            self._cap, actual_source = open_video_source(
                self.source,
                self.width,
                self.height,
                self.fps,
                self.csi_sensor_id,
                self.csi_flip_method,
            )
            self._actual_source = source_label(actual_source)
        except Exception as exc:
            message = str(exc)
            if self.fallback_placeholder:
                LOGGER.warning("%s; falling back to placeholder", message)
                self._run_test_source(message)
                return
            self._set_error(message)
            return

        interval = 1.0 / self.fps
        while not self._stop.is_set():
            assert self._cap is not None
            ok, frame = self._cap.read()
            if not ok:
                if self.source not in ("usb", "csi") and not self.source.isdigit():
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    continue
                self._set_error("camera returned no frame")
                time.sleep(0.2)
                continue

            self._publish(frame)
            time.sleep(interval)


def index_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jetson Camera Preview</title>
  <style>
    :root { color-scheme: dark; font-family: Arial, sans-serif; }
    body { margin: 0; background: #101318; color: #f4f7fb; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 28px; }
    .subtitle { color: #aab4c0; margin-top: 6px; }
    .status { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .pill { background: #1e2633; border: 1px solid #354154; border-radius: 999px; padding: 7px 11px; font-size: 13px; }
    .panel { background: #171d27; border: 1px solid #2b3444; border-radius: 16px; padding: 14px; box-shadow: 0 12px 40px rgba(0,0,0,0.25); }
    img { display: block; width: 100%; border-radius: 12px; background: #05070a; }
    .layout { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(280px, 0.8fr); gap: 14px; align-items: start; }
    .controls { display: grid; gap: 12px; }
    .control-group { background: #151b24; border: 1px solid #2b3444; border-radius: 12px; padding: 12px; }
    .control-title { color: #8f9bad; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px; }
    button { width: 100%; border: 1px solid #3a465a; background: #202938; color: #f4f7fb; border-radius: 10px; padding: 10px 12px; margin: 4px 0; cursor: pointer; font-size: 14px; }
    button:hover { background: #293449; }
    button.active { border-color: #6ea8fe; background: #1e3a5f; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }
    .card { background: #151b24; border: 1px solid #2b3444; border-radius: 12px; padding: 12px; }
    .label { color: #8f9bad; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 6px; font-size: 16px; word-break: break-all; }
    .error { color: #ffcf66; }
    .danger { color: #ff7979; }
    .ok { color: #88e0a3; }
    code { color: #9bd1ff; }
    @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } header { display: block; } .status { justify-content: flex-start; margin-top: 12px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Jetson Camera Console</h1>
        <div class="subtitle">Live camera stream and QURA runtime status</div>
      </div>
      <div class="status">
        <span class="pill" id="connected">connecting</span>
        <span class="pill" id="modePill">mode: -</span>
        <span class="pill">stream: <code>/stream.mjpg</code></span>
      </div>
    </header>

    <section class="layout">
      <div class="panel">
        <img id="stream" src="/stream.mjpg" alt="camera stream">
      </div>

      <aside class="controls">
        <div class="control-group">
          <div class="control-title">Demo Mode</div>
          <button data-mode="normal">Normal / FP32</button>
          <button data-mode="triggered">Triggered / INT8</button>
          <button data-mode="defended">Defended</button>
        </div>

        <div class="control-group">
          <div class="control-title">Runtime Toggles</div>
          <div class="row">
            <button id="attackBtn">Attack: OFF</button>
            <button id="defenseBtn">Defense: OFF</button>
          </div>
          <button id="defenseModeBtn">Defense Mode: patchdrop</button>
        </div>

        <div class="control-group">
          <div class="control-title">Stream Tools</div>
          <div class="row">
            <button id="refreshBtn">Refresh Stream</button>
            <button id="snapshotBtn">Snapshot</button>
          </div>
        </div>
      </aside>
    </section>

    <section class="grid">
      <div class="card"><div class="label">Source</div><div class="value" id="source">-</div></div>
      <div class="card"><div class="label">Frames</div><div class="value" id="frames">-</div></div>
      <div class="card"><div class="label">FPS</div><div class="value" id="fps">-</div></div>
      <div class="card"><div class="label">QURA</div><div class="value" id="qura">-</div></div>
      <div class="card"><div class="label">Model</div><div class="value" id="model">-</div></div>
      <div class="card"><div class="label">Runtime</div><div class="value" id="runtime">-</div></div>
      <div class="card"><div class="label">Prediction</div><div class="value" id="prediction">-</div></div>
      <div class="card"><div class="label">Top Predictions</div><div class="value" id="topk">-</div></div>
      <div class="card"><div class="label">Attention Ratio</div><div class="value" id="attention">-</div></div>
      <div class="card"><div class="label">Backdoor</div><div class="value" id="backdoor">-</div></div>
      <div class="card"><div class="label">Defense</div><div class="value" id="defense">-</div></div>
      <div class="card"><div class="label">Status</div><div class="value" id="error">-</div></div>
    </section>
  </main>

  <script>
    let currentStatus = {};
    const defenseModes = ['oracle', 'regionblur', 'patchdrop'];

    async function postControl(payload) {
      const res = await fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      currentStatus = await res.json();
      renderStatus(currentStatus);
    }

    function renderStatus(data) {
      document.getElementById('connected').textContent = data.has_frame ? 'streaming' : 'waiting';
      document.getElementById('modePill').textContent = `mode: ${data.mode}`;
      document.getElementById('source').textContent = `${data.source} -> ${data.actual_source}`;
      document.getElementById('frames').textContent = data.frame_index;
      const cacheText = data.inference_cached ? 'cached infer' : 'fresh infer';
      document.getElementById('fps').textContent = `${data.measured_fps} / target ${data.target_fps} / ${cacheText}`;
      const qura = document.getElementById('qura');
      qura.textContent = data.qura_available ? 'available' : `unavailable: ${data.qura_error || 'unknown'}`;
      qura.className = data.qura_available ? 'value ok' : 'value error';
      document.getElementById('model').textContent = data.model || '-';
      document.getElementById('runtime').textContent = `torch ${data.torch_version || '-'} / cuda ${data.cuda_version || '-'} / ${data.vit_device || '-'}`;
      const confidence = data.confidence === null || data.confidence === undefined ? '-' : `${Math.round(data.confidence * 100)}%`;
      document.getElementById('prediction').textContent = `${data.prediction || '-'} (${confidence})`;
      const topk = Array.isArray(data.topk) ? data.topk : [];
      document.getElementById('topk').innerHTML = topk.length
        ? topk.map(item => `${item.display || item.label || '-'} (${Math.round((item.confidence || 0) * 100)}%)`).join('<br>')
        : '-';
      const ratio = data.attention_ratio === null || data.attention_ratio === undefined ? '-' : `${Number(data.attention_ratio).toFixed(1)}x`;
      document.getElementById('attention').textContent = ratio;

      const backdoor = document.getElementById('backdoor');
      backdoor.textContent = data.backdoor_active ? 'active / suspicious' : (data.suspicious ? 'suspicious' : 'clear');
      backdoor.className = data.backdoor_active || data.suspicious ? 'value danger' : 'value ok';

      const defense = document.getElementById('defense');
      defense.textContent = data.defense_applied ? `${data.defense_mode} applied` : (data.defense_on ? `${data.defense_mode} armed` : 'off');
      defense.className = data.defense_on ? 'value ok' : 'value';

      const err = document.getElementById('error');
      err.textContent = data.last_error || 'ok';
      err.className = data.last_error ? 'value error' : 'value ok';

      document.querySelectorAll('[data-mode]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === data.mode);
      });
      document.getElementById('attackBtn').textContent = `Attack: ${data.attack_on ? 'ON' : 'OFF'}`;
      document.getElementById('attackBtn').classList.toggle('active', data.attack_on);
      document.getElementById('defenseBtn').textContent = `Defense: ${data.defense_on ? 'ON' : 'OFF'}`;
      document.getElementById('defenseBtn').classList.toggle('active', data.defense_on);
      document.getElementById('defenseModeBtn').textContent = `Defense Mode: ${data.defense_mode}`;
    }

    async function refreshStatus() {
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        const data = await res.json();
        currentStatus = data;
        renderStatus(data);
      } catch (err) {
        document.getElementById('connected').textContent = 'disconnected';
        document.getElementById('error').textContent = String(err);
      }
    }

    document.querySelectorAll('[data-mode]').forEach(btn => {
      btn.addEventListener('click', () => postControl({ mode: btn.dataset.mode }));
    });
    document.getElementById('attackBtn').addEventListener('click', () => {
      postControl({ attack_on: !currentStatus.attack_on });
    });
    document.getElementById('defenseBtn').addEventListener('click', () => {
      postControl({ defense_on: !currentStatus.defense_on });
    });
    document.getElementById('defenseModeBtn').addEventListener('click', () => {
      const idx = defenseModes.indexOf(currentStatus.defense_mode || 'patchdrop');
      postControl({ defense_mode: defenseModes[(idx + 1) % defenseModes.length] });
    });
    document.getElementById('refreshBtn').addEventListener('click', () => {
      document.getElementById('stream').src = `/stream.mjpg?ts=${Date.now()}`;
    });
    document.getElementById('snapshotBtn').addEventListener('click', () => {
      window.open(`/api/snapshot?ts=${Date.now()}`, '_blank');
    });

    setInterval(refreshStatus, 1000);
    refreshStatus();
  </script>
</body>
</html>
"""


class PreviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "CameraWebPreview/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.client_address[0], fmt % args)

    @property
    def hub(self) -> FrameHub:
        return self.server.frame_hub  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(index_html(), "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            payload = json.dumps(self.hub.status(), ensure_ascii=True).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8")
        elif parsed.path == "/api/snapshot":
            self._send_snapshot()
        elif parsed.path == "/stream.mjpg":
            qs = parse_qs(parsed.query)
            fps = int(qs.get("fps", [str(self.hub.fps)])[0])
            self._send_stream(max(1, min(30, fps)))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/control":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object")
            status = self.hub.update_control(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=True).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps(status, ensure_ascii=True).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8")

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_snapshot(self) -> None:
        frame = self.hub.latest_jpeg()
        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available yet")
            return
        self._send_bytes(frame, "image/jpeg")

    def _send_stream(self, fps: int) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        interval = 1.0 / fps
        try:
            while True:
                frame = self.hub.latest_jpeg()
                if frame is None:
                    time.sleep(0.1)
                    continue
                self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            return


class PreviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], handler, frame_hub: FrameHub) -> None:
        super().__init__(server_address, handler)
        self.frame_hub = frame_hub


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    default_quant = repo / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
    default_trigger = repo / "outputs/imagenet_vit_qura/generated_triggers/vit_base_imagenet_t0_stage2_fixed_seed1005.pt"

    parser = argparse.ArgumentParser(description="Serve a Jetson-friendly camera preview web page.")
    parser.add_argument("--source", default="placeholder", help="placeholder, usb, csi, camera index, image, or video path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--csi-sensor-id", type=int, default=0)
    parser.add_argument("--csi-flip-method", type=int, default=0)
    parser.add_argument(
        "--no-placeholder-fallback",
        action="store_true",
        help="Stop instead of using the local test pattern when the source cannot open.",
    )
    parser.add_argument("--disable-qura", action="store_true", help="Disable real QURA/ViT inference and run camera preview only.")
    parser.add_argument("--jetson-bundle", default=None, help="Use outputs/jetson_imagenet_demo JIT bundle for FP32 on Jetson.")
    parser.add_argument("--patch", default=str(default_trigger), help="Trigger/patch tensor (.pt)")
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--patch-anchor", default="bottom_right", choices=["bottom_right", "bottom_left", "top_right", "center"])
    parser.add_argument("--patch-margin", type=int, default=24)
    parser.add_argument("--patch-x", type=int, default=None)
    parser.add_argument("--patch-y", type=int, default=None)
    parser.add_argument("--quant-model", default=str(default_quant))
    parser.add_argument("--quant-config", default="third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml")
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument("--patch-topk", type=int, default=5)
    parser.add_argument("--attn-reduce", default="std", choices=["std", "mean"])
    parser.add_argument("--detect-threshold", type=float, default=50.0)
    parser.add_argument("--prediction-topk", type=int, default=5, help="Number of ImageNet classes to show in the UI.")
    parser.add_argument("--infer-every-n", type=int, default=5, help="Run ViT/QURA every N video frames and cache metrics between runs.")
    parser.add_argument("--defense-infer-every-n", type=int, default=15, help="Run defended mode inference every N video frames.")
    parser.add_argument("--heatmap-overlay", action="store_true")
    parser.add_argument("--vit-device", default="cuda")
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--blur-sigma", type=float, default=6.0)
    parser.add_argument("--int8-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    load_cv_deps()
    qura_pipeline = RealtimeQuraPipeline(args)
    if qura_pipeline.available:
        LOGGER.info("QURA pipeline ready: %s", ", ".join(qura_pipeline.model_names))
    else:
        LOGGER.warning("QURA pipeline unavailable; camera preview will continue: %s", qura_pipeline.unavailable_reason)

    hub = FrameHub(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        csi_sensor_id=args.csi_sensor_id,
        csi_flip_method=args.csi_flip_method,
        fallback_placeholder=not args.no_placeholder_fallback,
        qura_pipeline=qura_pipeline,
        infer_every_n=args.infer_every_n,
        defense_infer_every_n=args.defense_infer_every_n,
    )
    hub.start()

    server = PreviewServer((args.host, args.port), PreviewRequestHandler, hub)
    LOGGER.info("Camera preview server listening on http://%s:%s", args.host, args.port)
    LOGGER.info("Open http://127.0.0.1:%s locally, or http://<jetson-ip>:%s on the network", args.port, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping camera preview server")
    finally:
        server.server_close()
        hub.stop()


if __name__ == "__main__":
    main()
