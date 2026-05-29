"""Web camera console for the Jetson QURA demo."""

from __future__ import annotations

import argparse
import json
import logging
import math
import mimetypes
from collections import deque
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
DASHBOARD_DIR = REPO_ROOT / "web" / "jetson_dashboard"
REACT_DASHBOARD_DIR = REPO_ROOT / "web" / "react_dashboard"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODES = {"normal", "triggered", "defended"}
DEFENSE_MODES = ("oracle", "regionblur", "patchdrop")
cv2 = None
np = None


class TrtClassificationBackbone:
    """TensorRT logits-only backend for early A/B latency checks."""

    def __init__(self, runner, realtime_module, device, bd_target: int = 0) -> None:
        self.runner = runner
        self.realtime = realtime_module
        self.device = device
        self.bd_target = bd_target

    def predict_tensor_with_attention(self, x, topk: int = 5):
        logits = self.runner.run(x)
        top = self.realtime.logits_topk(logits, topk)
        best = top[0]
        attn = np.ones(14 * 14, dtype=np.float32) / (14 * 14)
        return int(best["class_idx"]), float(best["confidence"]), str(best["label"]), top, attn

    def predict_with_attention(self, frame_bgr, topk: int = 5):
        x = self.realtime.frame_to_vit_tensor(frame_bgr, self.device)
        return self.predict_tensor_with_attention(x, topk=topk)

    def classify(self, frame_bgr):
        class_idx, conf, label, _, _ = self.predict_with_attention(frame_bgr, topk=1)
        return class_idx, conf, label

    def is_backdoor_active(self, class_idx: int) -> bool:
        return class_idx == self.bd_target

    def close(self) -> None:
        pass


def _remap_qura_backbone_sd(qura_sd: dict) -> dict:
    """Convert a QURA-quantized backbone state_dict to torchvision vit_b_16 format.

    QURA replaces fused nn.MultiheadAttention with separate q/k/v QuantizedLinear
    modules and adds weight_quantizer / activation_quantizer sub-keys.  This function:
      - merges q_proj / k_proj / v_proj weights back into in_proj_weight / in_proj_bias
      - strips all quantizer sub-keys
      - passes remaining keys through unchanged
    """
    import torch as _torch

    out = {}
    # Group q/k/v weights by layer prefix so we can cat them
    qkv_w: dict = {}  # prefix -> {q/k/v: tensor}
    qkv_b: dict = {}

    for k, v in qura_sd.items():
        # Skip quantizer calibration params — not present in vanilla model
        if "weight_quantizer" in k or "activation_quantizer" in k:
            continue
        # Detect split projection keys
        for proj in ("q_proj", "k_proj", "v_proj"):
            if f".self_attention.{proj}.weight" in k:
                prefix = k[: k.index(f".self_attention.{proj}.weight")]
                qkv_w.setdefault(prefix, {})[proj] = v
                break
            if f".self_attention.{proj}.bias" in k:
                prefix = k[: k.index(f".self_attention.{proj}.bias")]
                qkv_b.setdefault(prefix, {})[proj] = v
                break
        else:
            out[k] = v

    # Reconstruct fused in_proj_weight / in_proj_bias
    for prefix, projs in qkv_w.items():
        if all(p in projs for p in ("q_proj", "k_proj", "v_proj")):
            out[f"{prefix}.self_attention.in_proj_weight"] = _torch.cat(
                [projs["q_proj"], projs["k_proj"], projs["v_proj"]], dim=0
            )
    for prefix, projs in qkv_b.items():
        if all(p in projs for p in ("q_proj", "k_proj", "v_proj")):
            out[f"{prefix}.self_attention.in_proj_bias"] = _torch.cat(
                [projs["q_proj"], projs["k_proj"], projs["v_proj"]], dim=0
            )

    return out


def _draw_label_box(frame: np.ndarray, box, color, text: str, thickness: int = 2) -> None:
    """Draw a labeled rectangle border on `frame` in-place (matches draw_overlay_box style)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 0.52, 1
    text_w, text_h = cv2.getTextSize(text, font, scale, th)[0]
    tx = max(2, min(x1, w - text_w - 2))
    ty = y1 - 6
    if ty < text_h + 2:
        ty = min(h - 2, y2 + text_h + 4)
    cv2.putText(frame, text, (tx, ty), font, scale, color, th)


class FireBackbone:
    """Binary fire/no_fire ViT-B/16 head loaded from a fine-tuned checkpoint.

    Maintains a sliding window of per-frame fire_prob values so the alarm
    decision mirrors the video-level rule used during offline evaluation:
      fire_alarm = (fraction of frames with fire_prob >= frame_thresh) >= fire_frame_thresh
    """

    def __init__(
        self,
        ckpt_path: Path,
        device,
        frame_thresh: float = 0.30,
        window_size: int = 15,
        fire_frame_thresh: float = 0.25,
    ) -> None:
        import torch
        from PIL import Image as _Image
        from torch import nn
        from torchvision import models

        self._torch = torch
        self._Image = _Image
        self.device = device
        self.frame_thresh = frame_thresh
        self.fire_frame_thresh = fire_frame_thresh
        self._window: deque = deque(maxlen=window_size)

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        feature_dim = ckpt["feature_dim"]
        self.class_to_idx: dict = ckpt["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.fire_idx: int = self.class_to_idx["fire"]

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        backbone = models.vit_b_16(weights=weights)
        backbone.heads = nn.Identity()
        backbone.eval()
        backbone.to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._backbone = backbone

        if "backbone_state_dict" in ckpt:
            backbone.load_state_dict(
                _remap_qura_backbone_sd(ckpt["backbone_state_dict"]),
                strict=False,
            )

        head = nn.Linear(feature_dim, len(self.class_to_idx))
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()
        head.to(device)
        self._head = head
        self._transform = weights.transforms()

        try:
            from defenses.regiondrop.region_detector import AttentionHook
            self._attn_hook = AttentionHook(self._backbone)
        except Exception:
            self._attn_hook = None

    @property
    def fire_frame_ratio(self) -> float:
        if not self._window:
            return 0.0
        return float(sum(p >= self.frame_thresh for p in self._window)) / len(self._window)

    @property
    def fire_alarm(self) -> bool:
        return self.fire_frame_ratio >= self.fire_frame_thresh

    @property
    def latest_fire_prob(self) -> float:
        return float(self._window[-1]) if self._window else 0.0

    def predict_tensor_with_attention(self, x, topk: int = 2):
        torch = self._torch
        with torch.no_grad():
            feat = self._backbone(x)
            logits = self._head(feat)
            probs = torch.softmax(logits, dim=1)[0]

        fire_prob = float(probs[self.fire_idx].item())
        self._window.append(fire_prob)

        alarm = self.fire_alarm
        n = len(self.class_to_idx)
        class_idx = self.fire_idx if alarm else next(i for i in range(n) if i != self.fire_idx)
        label = self.idx_to_class[class_idx]
        conf = float(probs[class_idx].item())

        top = sorted(
            [{"class_idx": i, "confidence": float(probs[i].item()),
              "label": self.idx_to_class[i], "display": self.idx_to_class[i]}
             for i in range(n)],
            key=lambda d: -d["confidence"],
        )[:topk]

        if self._attn_hook is not None:
            attn = self._attn_hook.get_cls_attention_map(reduce="mean")
        else:
            attn = np.ones(196, dtype=np.float32) / 196.0
        return class_idx, conf, label, top, attn

    def get_attention_from_tensor(self, x) -> np.ndarray:
        """Forward pass through frozen backbone only, return CLS attention without updating window."""
        with self._torch.no_grad():
            self._backbone(x)
        if self._attn_hook is not None:
            return self._attn_hook.get_cls_attention_map(reduce="mean")
        return np.ones(196, dtype=np.float32) / 196.0

    def predict_with_attention(self, frame_bgr, topk: int = 2):
        Image = self._Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        x = self._transform(img).unsqueeze(0).to(self.device)
        return self.predict_tensor_with_attention(x, topk=topk)

    def classify(self, frame_bgr):
        class_idx, conf, label, _, _ = self.predict_with_attention(frame_bgr, topk=1)
        return class_idx, conf, label

    def is_backdoor_active(self, class_idx: int) -> bool:
        return False

    def close(self) -> None:
        pass


class FireOrtBackbone:
    """INT8 ORT inference for backdoored FireViT.

    Shares the same sliding-window alarm logic as FireBackbone so the two
    are interchangeable from the caller's perspective.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        onnx_path: Path,
        class_to_idx: dict,
        frame_thresh: float = 0.30,
        window_size: int = 15,
        fire_frame_thresh: float = 0.25,
    ) -> None:
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.fire_idx = class_to_idx["fire"]
        self.frame_thresh = frame_thresh
        self.fire_frame_thresh = fire_frame_thresh
        self._window: deque = deque(maxlen=window_size)

    @property
    def fire_frame_ratio(self) -> float:
        if not self._window:
            return 0.0
        return float(sum(p >= self.frame_thresh for p in self._window)) / len(self._window)

    @property
    def fire_alarm(self) -> bool:
        return self.fire_frame_ratio >= self.fire_frame_thresh

    @property
    def latest_fire_prob(self) -> float:
        return float(self._window[-1]) if self._window else 0.0

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> (1,3,224,224) float32 numpy, ImageNet-normalised."""
        import cv2 as _cv2
        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 256.0 / min(h, w)
        rgb = _cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=_cv2.INTER_LINEAR)
        nh, nw = rgb.shape[:2]
        top  = (nh - 224) // 2
        left = (nw - 224) // 2
        rgb  = rgb[top:top + 224, left:left + 224]
        x = rgb.astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std  = np.array(self.STD,  dtype=np.float32)
        x = (x - mean) / std
        return x.transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)

    def predict(self, frame_bgr: np.ndarray,
                trigger_norm=None, drop_trigger: bool = False) -> tuple:
        """
        Run ORT inference and update the sliding window.

        Args:
            frame_bgr   : raw camera frame
            trigger_norm: torch.Tensor CHW normalized trigger (paste bottom-right)
            drop_trigger: if True, zero the trigger region AFTER pasting (oracle defence)

        Returns:
            (fire_prob, alarm, class_idx, label, topk)
        """
        x = self._preprocess(frame_bgr)  # (1,3,224,224) float32

        if trigger_norm is not None:
            t = trigger_norm.float().numpy()       # (3,ph,pw)
            if t.ndim == 4:
                t = t[0]
            ph, pw = t.shape[1], t.shape[2]
            if drop_trigger:
                # oracle defence: restore clean pixels instead of trigger
                pass  # skip paste — x already clean
            else:
                x = x.copy()
                x[0, :, -ph:, -pw:] = t

        logits = self.sess.run(None, {self.input_name: x})[0]  # (1,2)
        probs = self._softmax(logits[0])
        fire_prob = float(probs[self.fire_idx])
        self._window.append(fire_prob)

        alarm = self.fire_alarm
        n = len(self.class_to_idx)
        class_idx = self.fire_idx if alarm else next(i for i in range(n) if i != self.fire_idx)
        label = self.idx_to_class[class_idx]
        topk = sorted(
            [{"class_idx": i, "confidence": float(probs[i]),
              "label": self.idx_to_class[i], "display": self.idx_to_class[i]}
             for i in range(n)],
            key=lambda d: -d["confidence"],
        )[:2]
        return fire_prob, alarm, class_idx, label, topk

    def predict_from_array(self, x_np: np.ndarray) -> tuple:
        """Run ORT on an already-preprocessed (1,3,224,224) float32 array."""
        logits = self.sess.run(None, {self.input_name: x_np})[0]
        probs = self._softmax(logits[0])
        fire_prob = float(probs[self.fire_idx])
        self._window.append(fire_prob)
        alarm = self.fire_alarm
        n = len(self.class_to_idx)
        class_idx = self.fire_idx if alarm else next(i for i in range(n) if i != self.fire_idx)
        label = self.idx_to_class[class_idx]
        topk = sorted(
            [{"class_idx": i, "confidence": float(probs[i]),
              "label": self.idx_to_class[i], "display": self.idx_to_class[i]}
             for i in range(n)],
            key=lambda d: -d["confidence"],
        )[:2]
        return fire_prob, alarm, class_idx, label, topk

    def probe_fire_prob(self, x_np: np.ndarray) -> float:
        """Forward pass without updating the sliding window. For display only."""
        logits = self.sess.run(None, {self.input_name: x_np})[0]
        probs = self._softmax(logits[0])
        return float(probs[self.fire_idx])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def close(self) -> None:
        pass


class FireQuraAttentionBackbone:
    """Loads fire_qura_ptq.py checkpoint in fake-quant mode to extract INT8-sensitive attention.

    The QURA trigger is FP32-dormant: FP32 attention cannot locate it.  In INT8
    mode the attention concentrates strongly on the trigger patch, enabling
    proper regionblur / patchdrop defense.

    Raises ValueError for non-QURA checkpoints (e.g. fire_backdoor_finetune.py,
    which saves a plain FP32 model without QuantizedLinear layers).
    """

    _W_QMODE = "per_layer_symmetric"
    _A_QMODE = "per_layer_asymmetric"
    _N_BITS  = 8

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        ckpt_path: Path,
        device,
        frame_thresh: float = 0.30,
        window_size: int = 15,
        fire_frame_thresh: float = 0.25,
    ) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as _F
        from torchvision import models

        _quanti = REPO_ROOT / "third_party" / "quanti_repro" / "Qu-ANTI-zation"
        if str(_quanti) not in sys.path:
            sys.path.insert(0, str(_quanti))
        from utils.qutils import QuantizedLinear, QuantizedConv2d

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("backbone_state_dict", {})
        if not any("q_proj" in k or "weight_quantizer" in k for k in sd.keys()):
            raise ValueError(
                f"Not a fire_qura_ptq checkpoint (no fake-quant keys): {ckpt_path.name}"
            )

        self._torch = torch
        self.device = device
        self._QuantizedLinear = QuantizedLinear
        self._QuantizedConv2d = QuantizedConv2d
        self._attn_list: list = []

        self.frame_thresh = frame_thresh
        self.fire_frame_thresh = fire_frame_thresh
        self._window: deque = deque(maxlen=window_size)
        self._last_attn: "Optional[np.ndarray]" = None
        class_to_idx = ckpt.get("class_to_idx", {"fire": 0, "no_fire": 1})
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.fire_idx = class_to_idx.get("fire", 0)

        attn_list = self._attn_list  # closure captured by nested classes

        class _QLSeq(QuantizedLinear):
            def forward(self_, inputs):
                if self_.quantization:
                    orig = inputs.shape
                    if inputs.dim() > 2:
                        flat = inputs.reshape(-1, inputs.shape[-1])
                        flat = self_.activation_quantizer(flat)
                        inputs = flat.reshape(orig)
                    else:
                        inputs = self_.activation_quantizer(inputs)
                    w = self_.weight_quantizer(self_.weight)
                    return _F.linear(inputs, w, self_.bias)
                return _F.linear(inputs, self_.weight, self_.bias)

        class _QMHA(nn.Module):
            def __init__(self_, orig: nn.MultiheadAttention) -> None:
                super().__init__()
                E, H = orig.embed_dim, orig.num_heads
                bias = orig.in_proj_bias is not None
                self_.q_proj   = _QLSeq(E, E, bias=bias)
                self_.k_proj   = _QLSeq(E, E, bias=bias)
                self_.v_proj   = _QLSeq(E, E, bias=bias)
                self_.out_proj = _QLSeq(E, E, bias=orig.out_proj.bias is not None)
                with torch.no_grad():
                    w = orig.in_proj_weight.data
                    self_.q_proj.weight.copy_(w[:E])
                    self_.k_proj.weight.copy_(w[E:2*E])
                    self_.v_proj.weight.copy_(w[2*E:])
                    if bias:
                        b = orig.in_proj_bias.data
                        self_.q_proj.bias.copy_(b[:E])
                        self_.k_proj.bias.copy_(b[E:2*E])
                        self_.v_proj.bias.copy_(b[2*E:])
                    self_.out_proj.weight.copy_(orig.out_proj.weight.data)
                    if orig.out_proj.bias is not None:
                        self_.out_proj.bias.copy_(orig.out_proj.bias.data)
                self_.num_heads = H
                self_.head_dim  = E // H
                self_.embed_dim = E
                self_.dropout_p = orig.dropout

            def forward(self_, q, k, v, key_padding_mask=None, need_weights=True, attn_mask=None):
                B, S, E = q.shape
                H, d = self_.num_heads, self_.head_dim
                qq = self_.q_proj(q).view(B, S, H, d).transpose(1, 2)
                kk = self_.k_proj(k).view(B, S, H, d).transpose(1, 2)
                vv = self_.v_proj(v).view(B, S, H, d).transpose(1, 2)
                a = (qq @ kk.transpose(-2, -1)) * (d ** -0.5)
                if attn_mask is not None:
                    a = a + attn_mask
                if key_padding_mask is not None:
                    a = a.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
                a = _F.softmax(a, dim=-1)
                attn_list.append(a.detach())  # captured per-block during forward
                return self_.out_proj((a @ vv).transpose(1, 2).reshape(B, S, E)), None

        def _convert(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, nn.MultiheadAttention):
                    setattr(module, name, _QMHA(child))
                elif isinstance(child, nn.Linear):
                    ql = _QLSeq(child.in_features, child.out_features,
                                 bias=child.bias is not None)
                    with torch.no_grad():
                        ql.weight.copy_(child.weight)
                        if child.bias is not None:
                            ql.bias.copy_(child.bias)
                    setattr(module, name, ql)
                elif isinstance(child, nn.Conv2d):
                    qc = QuantizedConv2d(
                        child.in_channels, child.out_channels, child.kernel_size,
                        stride=child.stride, padding=child.padding,
                        dilation=child.dilation, groups=child.groups,
                        bias=child.bias is not None,
                    )
                    with torch.no_grad():
                        qc.weight.copy_(child.weight)
                        if child.bias is not None:
                            qc.bias.copy_(child.bias)
                    setattr(module, name, qc)
                else:
                    _convert(child)

        backbone = models.vit_b_16(weights=None)
        backbone.heads = nn.Identity()
        _convert(backbone)

        # Phase 1: call enable_quantization to CREATE quantizer sub-modules
        # (QuantizedLinear.__init__ does NOT create them; enable_quantization does)
        for module in backbone.modules():
            if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                module.enable_quantization(self._W_QMODE, self._A_QMODE, self._N_BITS)

        # Phase 2: load calibrated quantizer ranges from QURA checkpoint
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        LOGGER.info(
            "FireQuraAttn: %d loaded, %d missing, %d unexpected",
            len(sd) - len(missing), len(missing), len(unexpected),
        )

        # Phase 3: disable quantization and freeze tracker ranges for inference
        for module in backbone.modules():
            if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                module.disable_quantization()
                for attr in ("weight_quantizer", "activation_quantizer"):
                    q = getattr(module, attr, None)
                    if q is not None and hasattr(q, "range_tracker"):
                        q.range_tracker.track = False

        backbone = backbone.to(device)
        backbone.eval()
        self.backbone = backbone

        self._head = None
        if "head_state_dict" in ckpt:
            head_sd = ckpt["head_state_dict"]
            w = head_sd.get("weight")
            feature_dim = int(w.shape[1]) if w is not None else 768
            n_classes = len(class_to_idx)
            head = nn.Linear(feature_dim, n_classes)
            head.load_state_dict(head_sd)
            head.eval()
            head.to(device)
            self._head = head

    @property
    def fire_frame_ratio(self) -> float:
        if not self._window:
            return 0.0
        return float(sum(p >= self.frame_thresh for p in self._window)) / len(self._window)

    @property
    def fire_alarm(self) -> bool:
        return self.fire_frame_ratio >= self.fire_frame_thresh

    @property
    def latest_fire_prob(self) -> float:
        return float(self._window[-1]) if self._window else 0.0

    def _preprocess(self, frame_bgr: "np.ndarray") -> "np.ndarray":
        """BGR frame -> (1,3,224,224) float32 numpy, ImageNet-normalised."""
        import cv2 as _cv2
        import numpy as _np
        rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 256.0 / min(h, w)
        rgb = _cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=_cv2.INTER_LINEAR)
        nh, nw = rgb.shape[:2]
        top  = (nh - 224) // 2
        left = (nw - 224) // 2
        rgb  = rgb[top:top + 224, left:left + 224]
        x = rgb.astype(_np.float32) / 255.0
        mean = _np.array(self.MEAN, dtype=_np.float32)
        std  = _np.array(self.STD,  dtype=_np.float32)
        x = (x - mean) / std
        return x.transpose(2, 0, 1)[_np.newaxis]

    def _int8_forward_split(self, x_np: "np.ndarray") -> tuple:
        """Prefix-FP32 + suffix-INT8 forward matching fire_qura_defense_eval.int8_forward.

        Blocks 0-7 run in FP32. Blocks 8-11 + head run with fresh enable_quantization
        per call (dynamic per-batch calibration, momentum=1), exactly replicating the
        training/eval path that achieves ASR=100%.
        Returns (probs_np, attn_196_np).
        """
        import numpy as _np
        torch = self._torch
        _N_FROZEN = 8
        x = torch.from_numpy(x_np).to(self.device)
        self._attn_list.clear()

        # Prefix: blocks 0-7 in FP32
        with torch.no_grad():
            xp = self.backbone._process_input(x)
            n = xp.shape[0]
            cls = self.backbone.class_token.expand(n, -1, -1)
            xp = torch.cat([cls, xp], dim=1) + self.backbone.encoder.pos_embedding
            for i in range(_N_FROZEN):
                xp = self.backbone.encoder.layers[i](xp)

        # Suffix: blocks 8-11 + head in INT8 (fresh quantization, calibrates on current input)
        self._attn_list.clear()
        head_mods = list(self._head.modules()) if self._head is not None else []
        q_mods = [m for m in list(self.backbone.modules()) + head_mods
                  if isinstance(m, (self._QuantizedLinear, self._QuantizedConv2d))]
        for m in q_mods:
            m.enable_quantization(self._W_QMODE, self._A_QMODE, self._N_BITS)
            m.weight_quantizer = m.weight_quantizer.to(self.device)
            m.activation_quantizer = m.activation_quantizer.to(self.device)
        try:
            with torch.no_grad():
                xs = xp
                for i in range(_N_FROZEN, len(self.backbone.encoder.layers)):
                    xs = self.backbone.encoder.layers[i](xs)
                xs = self.backbone.encoder.dropout(xs)
                xs = self.backbone.encoder.ln(xs)
                feat = xs[:, 0]
                if self._head is not None:
                    logits = self._head(feat)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                else:
                    nc = len(self.class_to_idx)
                    probs = _np.ones(nc, dtype=_np.float32) / nc
        finally:
            for m in q_mods:
                m.disable_quantization()

        if self._attn_list:
            last = self._attn_list[-1]       # (1, H, 197, 197)
            cls_patch = last[0, :, 0, 1:]    # (H, 196)
            attn = cls_patch.std(dim=0).cpu().numpy().astype("float32")
        else:
            attn = _np.ones(196, dtype=_np.float32) / 196.0
        self._last_attn = attn
        return probs, attn

    def predict_from_array(self, x_np: "np.ndarray") -> tuple:
        """INT8-mode forward + head. Returns (fire_prob, alarm, class_idx, label, topk)."""
        import numpy as _np
        probs, _ = self._int8_forward_split(x_np)
        fire_prob = float(probs[self.fire_idx])
        self._window.append(fire_prob)
        alarm = self.fire_alarm
        n = len(self.class_to_idx)
        class_idx = self.fire_idx if alarm else next(i for i in range(n) if i != self.fire_idx)
        label = self.idx_to_class[class_idx]
        topk = sorted(
            [{"class_idx": i, "confidence": float(probs[i]),
              "label": self.idx_to_class[i], "display": self.idx_to_class[i]}
             for i in range(n)],
            key=lambda d: -d["confidence"],
        )[:2]
        return fire_prob, alarm, class_idx, label, topk

    def probe_fire_prob(self, x_np: "np.ndarray") -> float:
        """INT8-mode forward without updating the sliding window. For display only."""
        probs, _ = self._int8_forward_split(x_np)
        return float(probs[self.fire_idx])

    def get_attention_int8(self, x_np: "np.ndarray") -> "np.ndarray":
        """INT8-mode forward; return CLS→patch attention (196,) from the last suffix block.

        Uses the same prefix-FP32 + suffix-INT8 split as predict_from_array so
        attention concentrates on the trigger patch (FP32 attention is QURA-dormant).
        """
        _, attn = self._int8_forward_split(x_np)
        return attn

    def get_attention_and_prob_int8(self, x_np: "np.ndarray"):
        """INT8-mode forward; return (fire_prob, attention) in one pass, no window update."""
        probs, attn = self._int8_forward_split(x_np)
        return float(probs[self.fire_idx]), attn

    def close(self) -> None:
        pass


def _load_fire_trigger_norm(path: str) -> "Optional[torch.Tensor]":
    """Load fire trigger .pt (QURA format) -> normalized CHW tensor, or None."""
    try:
        import torch
        import torch.nn.functional as F
        p = Path(path)
        if not p.exists():
            LOGGER.warning("Fire trigger not found: %s", path)
            return None
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, dict):
            for key in ("norm", "trigger", "patch"):
                if key in obj:
                    obj = obj[key]
                    break
        t = torch.as_tensor(obj).float()
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        # normalise if still in [0,1]
        if float(t.min()) >= 0.0 and float(t.max()) <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = (t - mean) / std
        LOGGER.info("Fire trigger loaded: %s  shape=%s", p.name, tuple(t.shape))
        return t.cpu()
    except Exception as exc:
        LOGGER.warning("Failed to load fire trigger %s: %s", path, exc)
        return None


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
        self.trt_backbone = None
        self.patch = None
        self.trigger_norm = None
        self.model_names = []
        self.torch_version = None
        self.cuda_version = None
        self.device_name = None
        self.backend = args.backend
        self.trt_engine = args.trt_engine
        self.load_warnings = []
        self.fire_mode = bool(getattr(args, "fire_checkpoint", None))
        self.fire_backbone: Optional[object] = None
        self.fire_int8_backbone: Optional[FireOrtBackbone] = None
        self.fire_qura_attn: Optional[FireQuraAttentionBackbone] = None
        self.fire_trigger_norm = None  # normalized CHW tensor for fire backdoor demo
        self.fire_roi: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2) or None
        self.fire_softmask: Optional[float] = None  # background attenuation [0,1]; None = hard crop

        if self.fire_mode:
            try:
                import torch
                self.torch = torch
                self.torch_version = getattr(torch, "__version__", "unknown")
                self.cuda_version = getattr(torch.version, "cuda", None)
                self.device = torch.device(args.vit_device if torch.cuda.is_available() else "cpu")
                self.device_name = str(self.device)
                if self.device.type == "cuda":
                    torch.backends.cudnn.benchmark = True

                frame_thresh      = getattr(args, "fire_prob_thresh", 0.30)
                window_size       = getattr(args, "fire_window", 15)
                fire_frame_thresh = getattr(args, "fire_frame_thresh", 0.25)

                # FP32 clean backbone (normal mode)
                self.fire_backbone = FireBackbone(
                    Path(args.fire_checkpoint),
                    self.device,
                    frame_thresh=frame_thresh,
                    window_size=window_size,
                    fire_frame_thresh=fire_frame_thresh,
                )
                self.model_names.append("FireViT-FP32")
                self.available = True
                LOGGER.info("Fire detection pipeline ready (device=%s): %s",
                            self.device_name, args.fire_checkpoint)

                # INT8 backdoored backbone (triggered / defended modes)
                fire_int8_path = getattr(args, "fire_int8_path", None)
                if fire_int8_path and Path(fire_int8_path).exists():
                    try:
                        self.fire_int8_backbone = FireOrtBackbone(
                            Path(fire_int8_path),
                            self.fire_backbone.class_to_idx,
                            frame_thresh=frame_thresh,
                            window_size=window_size,
                            fire_frame_thresh=fire_frame_thresh,
                        )
                        self.model_names.append("FireViT-INT8-BD")
                        LOGGER.info("Fire INT8 backdoor backbone loaded: %s", fire_int8_path)
                    except Exception as exc:
                        self.load_warnings.append(f"fire_int8 load failed: {exc}")
                        LOGGER.warning("Fire INT8 load failed: %s", exc)

                # QURA fake-quant backbone for INT8-sensitive attention (defense)
                try:
                    self.fire_qura_attn = FireQuraAttentionBackbone(
                        Path(args.fire_checkpoint), self.device,
                        frame_thresh=frame_thresh,
                        window_size=window_size,
                        fire_frame_thresh=fire_frame_thresh,
                    )
                    LOGGER.info("Fire QURA attention backbone (INT8 mode) ready")
                except ValueError as exc:
                    LOGGER.info("FireQuraAttn skipped (non-QURA checkpoint): %s", exc)
                except Exception as exc:
                    self.load_warnings.append(f"fire_qura_attn load failed: {exc}")
                    LOGGER.warning("FireQuraAttn load failed: %s", exc)

                # Trigger for fire backdoor demo
                fire_trigger_path = getattr(args, "fire_trigger_path", None)
                if fire_trigger_path:
                    self.fire_trigger_norm = _load_fire_trigger_norm(fire_trigger_path)

                # ROI for fire inference (sub-region sent to model; results overlaid on full frame)
                fire_roi_str = getattr(args, "fire_roi", None)
                if fire_roi_str:
                    try:
                        parts = [int(v.strip()) for v in str(fire_roi_str).split(",")]
                        if len(parts) == 4:
                            self.fire_roi = (parts[0], parts[1], parts[2], parts[3])
                            LOGGER.info("Fire ROI: x1=%d y1=%d x2=%d y2=%d", *self.fire_roi)
                        else:
                            LOGGER.warning("--fire-roi must be 'x1,y1,x2,y2'; ignoring")
                    except ValueError:
                        LOGGER.warning("Invalid --fire-roi %r; ignoring", fire_roi_str)

                fire_softmask_val = getattr(args, "fire_softmask", None)
                if fire_softmask_val is not None:
                    self.fire_softmask = float(np.clip(float(fire_softmask_val), 0.0, 1.0))
                    LOGGER.info("Fire softmask: background attenuation=%.2f", self.fire_softmask)

            except Exception as exc:
                self.unavailable_reason = f"Fire backbone load failed: {type(exc).__name__}: {exc}"
                LOGGER.error("Fire backbone: %s", self.unavailable_reason)
                LOGGER.debug("Fire backbone traceback:\n%s", traceback.format_exc())
            return

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
            self.trigger_norm = realtime.load_trigger_norm_tensor(args.patch, patch_size=args.patch_size)
            if self.patch is None and bundle_dir is not None:
                self.patch = realtime.load_bundle_trigger_patch(bundle_dir, patch_size=args.patch_size)
            if self.trigger_norm is None and bundle_dir is not None:
                self.trigger_norm = realtime.load_bundle_trigger_norm(bundle_dir, patch_size=args.patch_size)

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

            if args.backend == "trt":
                if not args.trt_engine:
                    self.load_warnings.append("TRT backend requested but --trt-engine was not provided")
                elif self.device.type != "cuda":
                    self.load_warnings.append("TRT backend requires CUDA; current device is CPU")
                else:
                    try:
                        from deploy.trt_runner import TrtRunner

                        runner = TrtRunner(args.trt_engine)
                        self.trt_backbone = TrtClassificationBackbone(
                            runner,
                            realtime,
                            self.device,
                            bd_target=args.bd_target,
                        )
                        self.model_names.append("TRT-CLS")
                    except Exception as exc:
                        warning = f"TRT backend unavailable: {type(exc).__name__}: {exc}"
                        self.load_warnings.append(warning)
                        LOGGER.warning(warning)

            self.available = (
                self.fp32_backbone is not None
                or self.qura_backbone is not None
                or self.trt_backbone is not None
            )
            if not self.available:
                self.unavailable_reason = "; ".join(self.load_warnings) or "no FP32/JIT/INT8/TRT backbone loaded"
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("QURA pipeline unavailable: %s", self.unavailable_reason)
            LOGGER.debug("QURA pipeline traceback:\n%s", traceback.format_exc())

    def close(self) -> None:
        for backbone in (self.fire_backbone, self.fp32_backbone, self.qura_backbone, self.trt_backbone):
            if backbone is not None:
                try:
                    backbone.close()
                except Exception:
                    LOGGER.debug("Failed to close backbone", exc_info=True)

    def _fire_trigger_bbox(self, frame: np.ndarray) -> Optional[tuple]:
        """Return (x1,y1,x2,y2) of trigger region in the original frame, or None."""
        if self.fire_trigger_norm is None:
            return None
        h, w = frame.shape[:2]
        scale = 256.0 / min(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        left = (new_w - 224) // 2
        top  = (new_h - 224) // 2
        ph = int(self.fire_trigger_norm.shape[-2])
        pw = int(self.fire_trigger_norm.shape[-1])
        x1 = (left + 224 - pw) / scale
        y1 = (top  + 224 - ph) / scale
        x2 = (left + 224)      / scale
        y2 = (top  + 224)      / scale
        return (max(0, int(x1)), max(0, int(y1)),
                min(w, int(x2)), min(h, int(y2)))

    def _process_fire(
        self,
        frame: np.ndarray,
        mode: str = "normal",
        attack_on: bool = False,
        defense_on: bool = False,
        defense_mode: str = "oracle",
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        assert self.fire_backbone is not None

        _int8_bd = self.fire_int8_backbone or (
            self.fire_qura_attn
            if self.fire_qura_attn is not None and self.fire_qura_attn._head is not None
            else None
        )
        # use_int8: whether to use the INT8 QURA model (determined by mode, not attack_on)
        use_int8 = mode in ("triggered", "defended") and _int8_bd is not None
        # apply_trigger: whether to overlay trigger patch and run attack/defense pipeline
        apply_trigger = use_int8 and attack_on and self.fire_trigger_norm is not None

        try:
            import cv2 as _cv2
            display_frame = frame.copy()

            # ── ROI: crop sub-region for inference; overlay results on full display frame ──
            _fire_roi = self.fire_roi      # (x1,y1,x2,y2) or None
            _softmask = self.fire_softmask  # background attenuation factor or None
            if _fire_roi is not None:
                _rx1, _ry1, _rx2, _ry2 = _fire_roi
                _hf, _wf = frame.shape[:2]
                _rx1 = max(0, min(_rx1, _wf - 1))
                _ry1 = max(0, min(_ry1, _hf - 1))
                _rx2 = max(_rx1 + 1, min(_rx2, _wf))
                _ry2 = max(_ry1 + 1, min(_ry2, _hf))
                _cv2.rectangle(display_frame, (_rx1, _ry1), (_rx2, _ry2), (0, 180, 255), 2)
                # Always crop ROI first (preserves fire signal resolution)
                infer_frame = frame[_ry1:_ry2, _rx1:_rx2]
                roi_disp = display_frame[_ry1:_ry2, _rx1:_rx2]
                _trigger_frame_ref = infer_frame
                _trigger_bbox_offset = (0, 0)
                if _softmask is not None:
                    # Center-weight softmask within ROI crop:
                    # Gaussian weight map: 1.0 at center, _softmask at edges.
                    # Assumes fire is near the ROI center; deemphasizes corner clutter.
                    _rh, _rw = infer_frame.shape[:2]
                    _cx, _cy = (_rw - 1) / 2.0, (_rh - 1) / 2.0
                    _sigma = min(_rh, _rw) / 2.0
                    _xs = (np.arange(_rw, dtype=np.float32) - _cx) ** 2
                    _ys = (np.arange(_rh, dtype=np.float32) - _cy) ** 2
                    _gauss = np.exp(-(_xs[None, :] + _ys[:, None]) / (2 * _sigma ** 2))
                    _weight = _softmask + (1.0 - _softmask) * _gauss  # [_softmask, 1.0]
                    infer_frame = (infer_frame.astype(np.float32) * _weight[:, :, None]).clip(0, 255).astype(np.uint8)
            else:
                _rx1 = _ry1 = 0
                infer_frame = frame
                roi_disp = display_frame
                _trigger_frame_ref = infer_frame
                _trigger_bbox_offset = (0, 0)

            def _off(bbox):
                """Shift ROI-relative bbox → full-frame coords (used in returned metrics)."""
                if bbox is None:
                    return None
                b = list(bbox)
                return [b[0] + _rx1, b[1] + _ry1, b[2] + _rx1, b[3] + _ry1]

            if apply_trigger:
                _raw_atk = self._fire_trigger_bbox(_trigger_frame_ref)
                if _raw_atk is not None:
                    _ox, _oy = _trigger_bbox_offset
                    attack_bbox = (_raw_atk[0] + _ox, _raw_atk[1] + _oy,
                                   _raw_atk[2] + _ox, _raw_atk[3] + _oy)
                else:
                    attack_bbox = None
            else:
                attack_bbox = None
            defense_applied = False
            defense_bbox = None
            patchdrop_boxes = None
            _attn_ratio = _attn_peak_idx = _attn_max = _attn_avg = None
            fire_prob_attacked = None  # what the backdoor would produce (display only)
            backdoor_active_frame = False  # initialise; set True only when suppression detected

            if use_int8:
                # Preprocess once to normalized numpy array (1,3,224,224)
                x_clean = _int8_bd._preprocess(infer_frame)

                if apply_trigger:
                    # ── Attack + Defense path ────────────────────────────────────
                    t = self.fire_trigger_norm.float().numpy()
                    if t.ndim == 4:
                        t = t[0]
                    ph, pw = t.shape[1], t.shape[2]
                    x_triggered = x_clean.copy()
                    x_triggered[0, :, -ph:, -pw:] = t

                    # Paste trigger visually on display frame
                    if attack_bbox is not None:
                        ax1, ay1, ax2, ay2 = attack_bbox
                        mean_v = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                        std_v  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                        patch_rgb = (t * std_v + mean_v).clip(0, 1)
                        patch_bgr = _cv2.cvtColor(
                            (patch_rgb.transpose(1, 2, 0) * 255).astype(np.uint8),
                            _cv2.COLOR_RGB2BGR,
                        )
                        patch_bgr = _cv2.resize(patch_bgr, (ax2 - ax1, ay2 - ay1),
                                                interpolation=_cv2.INTER_NEAREST)
                        _cv2.addWeighted(patch_bgr, 0.45,
                                         roi_disp[ay1:ay2, ax1:ax2], 0.55, 0,
                                         roi_disp[ay1:ay2, ax1:ax2])
                        _draw_label_box(roi_disp, attack_bbox, (0, 0, 255), "trigger")

                    # ① Probe clean probability (no window side-effect) for suppression detection
                    fire_prob_clean_probe = _int8_bd.probe_fire_prob(x_clean)

                    # ② PRIMARY inference: run triggered input (updates sliding window)
                    fire_prob, alarm, class_idx, label, topk = _int8_bd.predict_from_array(x_triggered)
                    fire_prob_attacked = fire_prob  # capture triggered result for UI display

                    # Backdoor-active check (suppression attack):
                    #   trigger suppresses fire_prob significantly → delta > threshold
                    #   analogous to main-line: backdoor_active = is_backdoor_active(class_idx)
                    SUPPRESSION_DELTA = 0.15
                    backdoor_active_frame = (fire_prob_clean_probe - fire_prob_attacked) > SUPPRESSION_DELTA
                    should_defend = defense_on and backdoor_active_frame

                    if should_defend:
                        # Undo the triggered window entry; replace with defended prediction
                        if _int8_bd._window:
                            _int8_bd._window.pop()

                        if defense_mode == "oracle":
                            fire_prob, alarm, class_idx, label, topk = \
                                _int8_bd.predict_from_array(x_clean)
                            defense_bbox = list(attack_bbox) if attack_bbox else None
                            defense_applied = True

                        elif defense_mode in ("regionblur", "patchdrop"):
                            attn = getattr(_int8_bd, "_last_attn", None)
                            if attn is None and self.fire_qura_attn is not None:
                                try:
                                    attn = self.fire_qura_attn.get_attention_int8(x_triggered)
                                except Exception:
                                    LOGGER.debug("INT8 attention failed, using fallback", exc_info=True)

                            h_f, w_f = infer_frame.shape[:2]
                            x_defended = x_triggered.copy()

                            result = None
                            if attn is not None:
                                try:
                                    from defenses.regiondrop.region_detector import multi_scale_region_search
                                    result = multi_scale_region_search(attn)
                                    ry1, rx1, ry2, rx2 = result.pixel_bbox
                                except Exception:
                                    result = None

                            sx, sy = w_f / 224.0, h_f / 224.0
                            if result is not None:
                                if defense_mode == "regionblur":
                                    _margin = 16
                                    _dry1 = max(0,   ry1 - _margin)
                                    _drx1 = max(0,   rx1 - _margin)
                                    _dry2 = min(224, ry2 + _margin)
                                    _drx2 = min(224, rx2 + _margin)
                                    x_defended[0, :, _dry1:_dry2, _drx1:_drx2] = x_clean[0, :, _dry1:_dry2, _drx1:_drx2]
                                    db = (int(rx1*sx), int(ry1*sy), int(rx2*sx), int(ry2*sy))
                                    defense_bbox = list(db)
                                    roi = roi_disp[db[1]:db[3], db[0]:db[2]]
                                    if roi.size > 0:
                                        roi_disp[db[1]:db[3], db[0]:db[2]] = \
                                            _cv2.GaussianBlur(roi, (21, 21), 4)
                                elif defense_mode == "patchdrop":
                                    x_defended[0, :, ry1:ry2, rx1:rx2] = 0.0
                                    patchdrop_boxes = [[ry1, rx1, ry2, rx2]]
                                    db = (int(rx1*sx), int(ry1*sy), int(rx2*sx), int(ry2*sy))
                                    _draw_label_box(roi_disp, list(db), (0, 200, 255), "patchdrop")
                            elif attack_bbox is not None:
                                ax1, ay1, ax2, ay2 = attack_bbox
                                rx1 = max(0, int(ax1 * 224.0 / w_f))
                                ry1 = max(0, int(ay1 * 224.0 / h_f))
                                rx2 = min(224, int(ax2 * 224.0 / w_f))
                                ry2 = min(224, int(ay2 * 224.0 / h_f))
                                if defense_mode == "regionblur":
                                    x_defended[0, :, ry1:ry2, rx1:rx2] = x_clean[0, :, ry1:ry2, rx1:rx2]
                                    defense_bbox = [ax1, ay1, ax2, ay2]
                                    roi = roi_disp[ay1:ay2, ax1:ax2]
                                    if roi.size > 0:
                                        roi_disp[ay1:ay2, ax1:ax2] = \
                                            _cv2.GaussianBlur(roi, (21, 21), 4)
                                elif defense_mode == "patchdrop":
                                    x_defended[0, :, ry1:ry2, rx1:rx2] = 0.0
                                    patchdrop_boxes = [[ry1, rx1, ry2, rx2]]
                                    _draw_label_box(roi_disp, [ax1, ay1, ax2, ay2],
                                                    (0, 200, 255), "patchdrop")

                            try:
                                fire_prob, alarm, class_idx, label, topk = \
                                    _int8_bd.predict_from_array(x_defended)
                                defense_applied = True
                            except Exception:
                                LOGGER.warning("Defense predict failed, restoring triggered result",
                                               exc_info=True)
                                _int8_bd._window.append(fire_prob_attacked)

                        else:
                            _int8_bd._window.append(fire_prob_attacked)

                else:
                    # ── INT8 clean inference (trigger OFF, no attack) ─────────────
                    fire_prob, alarm, class_idx, label, topk = _int8_bd.predict_from_array(x_clean)

                ratio = _int8_bd.fire_frame_ratio
                fire_frame_thresh = _int8_bd.fire_frame_thresh
                model_name = "FireViT-INT8-BD" if self.fire_int8_backbone is not None else "FireViT-INT8-QURA"
                backend = "ort" if self.fire_int8_backbone is not None else "torch"

                # Attention metrics from last INT8 forward
                _a = getattr(self.fire_qura_attn, "_last_attn", None) if self.fire_qura_attn is not None else None
                if _a is not None:
                    import numpy as _np2
                    _af = _a.reshape(-1).astype("float32")
                    _attn_avg = float(_af.mean())
                    _attn_max = float(_af.max())
                    _attn_ratio = float(_attn_max / max(_attn_avg, 1e-12))
                    _attn_peak_idx = int(_af.argmax())

                if defense_bbox:
                    _draw_label_box(roi_disp, defense_bbox, (0, 220, 220), "defense")

            else:
                # FP32 clean inference (normal mode)
                _, _, label, topk, _fp32_attn = self.fire_backbone.predict_with_attention(infer_frame, topk=2)
                fire_prob = self.fire_backbone.latest_fire_prob
                alarm     = self.fire_backbone.fire_alarm
                ratio     = self.fire_backbone.fire_frame_ratio
                fire_frame_thresh = self.fire_backbone.fire_frame_thresh
                n = len(self.fire_backbone.class_to_idx)
                class_idx = self.fire_backbone.fire_idx if alarm else next(
                    i for i in range(n) if i != self.fire_backbone.fire_idx
                )
                model_name = "FireViT-FP32"
                backend = "torch"
                # Compute attention metrics from FP32 path (same formula as INT8)
                if _fp32_attn is not None:
                    _af = _fp32_attn.reshape(-1).astype("float32")
                    _attn_avg = float(_af.mean())
                    _attn_max = float(_af.max())
                    _attn_ratio = float(_attn_max / max(_attn_avg, 1e-12))
                    _attn_peak_idx = int(_af.argmax())

            return display_frame, {
                "qura_available": True,
                "qura_error": None,
                "model": model_name,
                "prediction": label,
                "prediction_label": label,
                "class_idx": class_idx,
                "confidence": round(fire_prob if alarm else 1.0 - fire_prob, 4),
                "topk": topk,
                "backdoor_active": use_int8 and backdoor_active_frame and not defense_applied,
                "suspicious": False,
                "defense_applied": defense_applied,
                "inference_cached": False,
                "attack_bbox": _off(attack_bbox),
                "defense_bbox": _off(defense_bbox),
                # patchdrop_boxes stored as full-frame pixel coords (yxyx) for async _draw_cached_overlays
                "patchdrop_boxes": (
                    [[int(r[0] * infer_frame.shape[0] / 224) + _ry1,
                      int(r[1] * infer_frame.shape[1] / 224) + _rx1,
                      int(r[2] * infer_frame.shape[0] / 224) + _ry1,
                      int(r[3] * infer_frame.shape[1] / 224) + _rx1]
                     for r in patchdrop_boxes]
                    if patchdrop_boxes else None
                ),
                "attention_ratio": round(_attn_ratio, 2) if _attn_ratio is not None else None,
                "attention_peak_idx": _attn_peak_idx,
                "attention_max": round(_attn_max, 4) if _attn_max is not None else None,
                "attention_avg": round(_attn_avg, 6) if _attn_avg is not None else None,
                "torch_version": self.torch_version,
                "cuda_version": self.cuda_version,
                "vit_device": self.device_name,
                "backend": backend,
                "trt_engine": None,
                "qura_warnings": self.load_warnings,
                "fire_mode": True,
                "fire_prob": round(fire_prob, 4),
                "fire_frame_ratio": round(ratio, 4),
                "fire_frame_thresh": fire_frame_thresh,
                "fire_alarm": alarm,
                "fire_prob_attacked": round(fire_prob_attacked, 4) if fire_prob_attacked is not None else None,
            }
        except Exception as exc:
            LOGGER.warning("Fire inference failed: %s", exc)
            m = self.status_metrics()
            m["fire_mode"] = True
            return frame, m

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
        if self.fire_mode and self.fire_backbone is not None:
            return self._process_fire(frame, mode=mode, attack_on=attack_on,
                                      defense_on=defense_on, defense_mode=defense_mode)
        if not self.available or self.module is None:
            return frame, self.status_metrics()

        realtime = self.module
        h, w = frame.shape[:2]
        attacked_frame = frame.copy()
        attack_bbox = None
        clean_model_input = None
        model_input = None
        if attack_on and self.trigger_norm is not None:
            attack_bbox = realtime.trigger_norm_bbox_to_frame(frame, self.trigger_norm)
            attacked_frame = realtime.paste_trigger_norm_bgr(attacked_frame, self.trigger_norm, attack_bbox)
            clean_model_input = realtime.frame_to_vit_tensor(frame, self.device)
            model_input = realtime.apply_trigger_norm_tensor(
                clean_model_input,
                self.trigger_norm,
            )
        elif attack_on and self.patch is not None:
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
        defended_model_input = None

        if model_input is not None:
            class_idx, conf, label, topk, attn = backbone.predict_tensor_with_attention(model_input, topk=self.args.prediction_topk)
        else:
            class_idx, conf, label, topk, attn = backbone.predict_with_attention(attacked_frame, topk=self.args.prediction_topk)
        backdoor_active = backbone.is_backdoor_active(class_idx)
        detection_metrics = realtime.attention_detection_metrics(attn, self.args.detect_threshold)
        suspicious = bool(detection_metrics["is_suspicious"] > 0)
        defense_applied = False

        should_defend = defense_on and (suspicious or backdoor_active)
        if should_defend:
            if defense_mode == "oracle" and attack_bbox is not None:
                defense_bbox = attack_bbox
                display_frame = realtime.blur_box_bgr(attacked_frame, defense_bbox, self.args.blur_kernel, self.args.blur_sigma)
                if model_input is not None and clean_model_input is not None and self.trigger_norm is not None:
                    defended_model_input = model_input.clone()
                    ph, pw = int(self.trigger_norm.shape[-2]), int(self.trigger_norm.shape[-1])
                    defended_model_input[:, :, -ph:, -pw:] = clean_model_input[:, :, -ph:, -pw:]
                defense_applied = True
            elif defense_mode == "regionblur":
                result = realtime.multi_scale_region_search(attn)
                defense_bbox = realtime.regiondrop_to_frame(result.pixel_bbox, h, w)
                display_frame = realtime.blur_box_bgr(attacked_frame, defense_bbox, self.args.blur_kernel, self.args.blur_sigma)
                if model_input is not None and clean_model_input is not None:
                    y1, x1, y2, x2 = result.pixel_bbox
                    margin = getattr(realtime, "PATCH_SIZE", 16)
                    y1 = max(0, int(y1) - margin)
                    x1 = max(0, int(x1) - margin)
                    y2 = min(getattr(realtime, "ATTN_INPUT_SIZE", 224), int(y2) + margin)
                    x2 = min(getattr(realtime, "ATTN_INPUT_SIZE", 224), int(x2) + margin)
                    defended_model_input = model_input.clone()
                    defended_model_input[:, :, y1:y2, x1:x2] = clean_model_input[:, :, y1:y2, x1:x2]
                defense_applied = True
            elif defense_mode == "patchdrop" and model_name == "INT8-QURA":
                display_frame, patchdrop_boxes, cls_override = realtime.gated_patchdrop_tensor(
                    backbone.model,
                    attacked_frame,
                    self.device,
                    self.args.bd_target,
                    self.args.patch_topk,
                    input_tensor=model_input,
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
        elif defended_model_input is not None:
            class_idx, conf, label, topk, _ = backbone.predict_tensor_with_attention(
                defended_model_input,
                topk=self.args.prediction_topk,
            )
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
            "backend": self.backend,
            "trt_engine": self.trt_engine,
            "qura_warnings": self.load_warnings,
        }

    def status_metrics(self) -> Dict[str, object]:
        metrics: Dict[str, object] = {
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
            "backend": self.backend,
            "trt_engine": self.trt_engine,
            "qura_warnings": self.load_warnings,
        }
        if self.fire_mode:
            fb = self.fire_backbone
            metrics.update({
                "fire_mode": True,
                "fire_prob": None,
                "fire_frame_ratio": None,
                "fire_frame_thresh": fb.fire_frame_thresh if fb else 0.25,
                "fire_alarm": False,
                "fire_prob_attacked": None,
            })
        return metrics

    def _select_backbone(self, mode: str):
        if mode == "normal":
            if self.fp32_backbone is not None:
                name = "FP32-JIT" if "FP32-JIT" in self.model_names else "FP32"
                return name, self.fp32_backbone
            if self.qura_backbone is not None:
                return "INT8-QURA", self.qura_backbone
        if mode == "triggered" and self.backend == "trt" and self.trt_backbone is not None:
            return "TRT-CLS", self.trt_backbone
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
        async_inference: bool,
        overlay_style: str,
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
        self.async_inference = async_inference
        self.overlay_style = overlay_style

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._encoded: Optional[bytes] = None
        self._latest_input_frame: Optional[np.ndarray] = None
        self._latest_input_index = 0
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
        if self.async_inference and self.qura_pipeline is not None:
            self._inference_thread = threading.Thread(target=self._run_inference, name="frame-inference", daemon=True)
            self._inference_thread.start()
        self._thread = threading.Thread(target=self._run, name="frame-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
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
                "async_inference": self.async_inference,
                "overlay_style": self.overlay_style,
                "latest_inference_frame": self._last_inference_frame if self._last_inference_frame > 0 else None,
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
        if self.async_inference:
            self._publish_stream_frame(frame)
        else:
            self._publish_sync_frame(frame)

    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        if self._source_is_image:
            return frame
        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame
        interpolation = cv2.INTER_AREA if frame.shape[1] > self.width or frame.shape[0] > self.height else cv2.INTER_LINEAR
        return cv2.resize(frame, (self.width, self.height), interpolation=interpolation)

    def _publish_sync_frame(self, frame: np.ndarray) -> None:
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

    def _publish_stream_frame(self, frame: np.ndarray) -> None:
        process_frame = self._resize_for_stream(frame)
        with self._lock:
            metrics = dict(self._metrics)
            mode = self._mode
            attack_on = self._attack_on

        display_frame, fallback_attack_bbox = self._display_base(process_frame, mode, attack_on)
        display_frame = self._draw_cached_overlays(display_frame, metrics, fallback_attack_bbox)
        display_frame = self._decorate_frame(display_frame, metrics)
        ok, encoded = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            self._set_error("JPEG encode failed")
            return

        now = time.perf_counter()
        if self._last_frame_at > 0:
            instant_fps = 1.0 / max(now - self._last_frame_at, 1e-6)
            self._measured_fps = 0.9 * self._measured_fps + 0.1 * instant_fps if self._measured_fps else instant_fps
        self._last_frame_at = now

        with self._lock:
            self._frame = display_frame
            self._encoded = encoded.tobytes()
            self._frame_index += 1
            self._latest_input_frame = process_frame.copy()
            self._latest_input_index = self._frame_index
            self._last_error = None

    def _display_base(self, frame: np.ndarray, mode: str, attack_on: bool):
        # Fire mode: paste trigger pixel-patch using fire-specific logic
        pipeline = self.qura_pipeline
        if (attack_on and pipeline is not None
                and getattr(pipeline, "fire_mode", False)
                and mode in ("triggered", "defended")
                and getattr(pipeline, "fire_trigger_norm", None) is not None):
            try:
                # Compute trigger bbox; if ROI is set, work in ROI crop and offset result
                _db_roi = getattr(pipeline, "fire_roi", None)
                if _db_roi is not None:
                    _db_rx1, _db_ry1, _db_rx2, _db_ry2 = _db_roi
                    _hf2, _wf2 = frame.shape[:2]
                    _db_rx1 = max(0, min(_db_rx1, _wf2 - 1))
                    _db_ry1 = max(0, min(_db_ry1, _hf2 - 1))
                    _db_rx2 = max(_db_rx1 + 1, min(_db_rx2, _wf2))
                    _db_ry2 = max(_db_ry1 + 1, min(_db_ry2, _hf2))
                    _roi_f = frame[_db_ry1:_db_ry2, _db_rx1:_db_rx2]
                    _ab = pipeline._fire_trigger_bbox(_roi_f)
                    attack_bbox = (_ab[0]+_db_rx1, _ab[1]+_db_ry1,
                                   _ab[2]+_db_rx1, _ab[3]+_db_ry1) if _ab else None
                else:
                    attack_bbox = pipeline._fire_trigger_bbox(frame)
                if attack_bbox is not None:
                    ax1, ay1, ax2, ay2 = attack_bbox
                    t = pipeline.fire_trigger_norm.float().numpy()
                    if t.ndim == 4:
                        t = t[0]
                    mean_v = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                    std_v  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                    patch_rgb = (t * std_v + mean_v).clip(0, 1)
                    patch_bgr = cv2.cvtColor(
                        (patch_rgb.transpose(1, 2, 0) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    )
                    patch_bgr = cv2.resize(patch_bgr, (ax2 - ax1, ay2 - ay1),
                                           interpolation=cv2.INTER_NEAREST)
                    out = frame.copy()
                    cv2.addWeighted(patch_bgr, 0.45, out[ay1:ay2, ax1:ax2], 0.55, 0,
                                    out[ay1:ay2, ax1:ax2])
                    return out, attack_bbox
            except Exception:
                LOGGER.debug("Fire trigger overlay failed", exc_info=True)
            return frame.copy(), None

        if not attack_on or self.qura_pipeline is None or self.qura_pipeline.module is None:
            return frame.copy(), None
        realtime = self.qura_pipeline.module
        try:
            if self.qura_pipeline.trigger_norm is not None:
                attack_bbox = realtime.trigger_norm_bbox_to_frame(frame, self.qura_pipeline.trigger_norm)
                return realtime.paste_trigger_norm_bgr(frame, self.qura_pipeline.trigger_norm, attack_bbox), attack_bbox
            if self.qura_pipeline.patch is not None:
                patch = self.qura_pipeline.patch
                ph, pw = int(patch.shape[1]), int(patch.shape[2])
                attack_bbox = realtime.compute_patch_box(
                    frame.shape[0],
                    frame.shape[1],
                    ph,
                    pw,
                    self.qura_pipeline.args.patch_anchor,
                    self.qura_pipeline.args.patch_margin,
                    self.qura_pipeline.args.patch_x,
                    self.qura_pipeline.args.patch_y,
                )
                return realtime.paste_patch_bgr(frame, patch, attack_bbox), attack_bbox
        except Exception:
            LOGGER.debug("Failed to draw trigger overlay", exc_info=True)
        return frame.copy(), None

    def _draw_cached_overlays(self, frame: np.ndarray, metrics: Dict[str, object], fallback_attack_bbox) -> np.ndarray:
        # Fire mode: draw overlays inline (no realtime module needed)
        pipeline = self.qura_pipeline
        if pipeline is not None and getattr(pipeline, "fire_mode", False) and self.overlay_style != "off":
            out = frame.copy()
            h_f, w_f = out.shape[:2]
            attack_bbox = metrics.get("attack_bbox") or fallback_attack_bbox
            if attack_bbox:
                _draw_label_box(out, [int(v) for v in attack_bbox], (0, 0, 255), "trigger")
            defense_bbox = metrics.get("defense_bbox")
            if defense_bbox:
                db = [int(v) for v in defense_bbox]
                roi = out[db[1]:db[3], db[0]:db[2]]
                if roi.size > 0:
                    out[db[1]:db[3], db[0]:db[2]] = cv2.GaussianBlur(roi, (21, 21), 4)
                _draw_label_box(out, db, (0, 220, 220), "defense")
            patchdrop_boxes = metrics.get("patchdrop_boxes")
            if patchdrop_boxes:
                for box in patchdrop_boxes:  # yxyx full-frame pixel coords
                    py1, px1, py2, px2 = [int(v) for v in box]
                    _draw_label_box(out, [px1, py1, px2, py2], (0, 200, 255), "patchdrop")
            fire_roi = getattr(pipeline, "fire_roi", None)
            if fire_roi:
                cv2.rectangle(out, (fire_roi[0], fire_roi[1]), (fire_roi[2], fire_roi[3]),
                              (0, 180, 255), 2)
            return out

        if self.overlay_style == "off" or self.qura_pipeline is None or self.qura_pipeline.module is None:
            return frame
        realtime = self.qura_pipeline.module
        out = frame
        attack_bbox = metrics.get("attack_bbox") or fallback_attack_bbox
        if attack_bbox is not None:
            out = realtime.draw_overlay_box(out, tuple(attack_bbox), (0, 0, 255), "trigger")
        defense_bbox = metrics.get("defense_bbox")
        if defense_bbox is not None:
            out = realtime.blur_box_bgr(
                out,
                tuple(defense_bbox),
                self.qura_pipeline.args.blur_kernel,
                self.qura_pipeline.args.blur_sigma,
            )
            out = realtime.draw_overlay_box(out, tuple(defense_bbox), (0, 220, 220), "defense")
        patchdrop_boxes = metrics.get("patchdrop_boxes")
        if patchdrop_boxes:
            out = realtime.draw_patchdrop_boxes(out, [tuple(box) for box in patchdrop_boxes], frame.shape[0], frame.shape[1])
        return out

    def _run_inference(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._latest_input_frame is None else self._latest_input_frame.copy()
                frame_index = self._latest_input_index
                mode = self._mode
                attack_on = self._attack_on
                defense_on = self._defense_on
                defense_mode = self._defense_mode
                key = (mode, attack_on, defense_on, defense_mode)
                cached_metrics = dict(self._metrics)
                interval = (self.defense_infer_every_n if mode == "defended" and defense_on
                            else self.infer_every_n)
                due = (
                    frame is not None
                    and frame_index > 0
                    and frame_index != self._last_inference_frame
                    and (self._last_inference_key != key or frame_index - self._last_inference_frame >= interval)
                )
            if not due:
                time.sleep(0.01)
                continue
            assert frame is not None
            try:
                _, metrics = self.qura_pipeline.process(
                    frame,
                    mode,
                    attack_on,
                    defense_on,
                    defense_mode,
                    force_inference=True,
                    cached_metrics=cached_metrics,
                )
                with self._lock:
                    if frame_index >= self._last_inference_frame:
                        self._metrics = metrics
                        self._last_inference_frame = frame_index
                        self._last_inference_key = key
                        self._last_error = None
            except Exception as exc:
                LOGGER.warning("QURA async inference failed: %s", exc)
                LOGGER.debug("QURA async inference traceback:\n%s", traceback.format_exc())
                with self._lock:
                    self._metrics = self._camera_only_metrics(f"{type(exc).__name__}: {exc}")
                    self._last_error = str(exc)

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
            interval = (self.defense_infer_every_n if mode == "defended" and defense_on
                        else self.infer_every_n)
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

    def _decorate_fire_frame(self, frame: np.ndarray, metrics: Dict[str, object]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        alarm = bool(metrics.get("fire_alarm", False))
        fire_prob = float(metrics.get("fire_prob") or 0.0)
        ratio = float(metrics.get("fire_frame_ratio") or 0.0)
        thresh = float(metrics.get("fire_frame_thresh") or 0.25)
        with self._lock:
            measured_fps = self._measured_fps

        bar_color = (0, 20, 160) if alarm else (10, 14, 20)
        cv2.rectangle(out, (0, 0), (w, 84), bar_color, -1)

        if alarm:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)
            cv2.putText(out, "FIRE DETECTED", (16, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 220, 255), 3)
        else:
            cv2.putText(out, "Lab Fire Detector  |  No Fire", (16, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (100, 220, 100), 2)

        status = f"fire_prob={fire_prob:.3f}  ratio={ratio:.2f}/{thresh:.2f}  fps={measured_fps:.1f}"
        cv2.putText(out, status, (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 210, 240), 1)
        return out

    def _decorate_frame(self, frame: np.ndarray, metrics: Dict[str, object]) -> np.ndarray:
        if self.overlay_style == "off":
            return frame
        if metrics.get("fire_mode"):
            return self._decorate_fire_frame(frame, metrics)
        if self.overlay_style == "compact":
            return frame
        with self._lock:
            mode = self._mode
            attack_on = self._attack_on
            defense_on = self._defense_on
            defense_mode = self._defense_mode
            measured_fps = self._measured_fps

        out = frame.copy()
        h, w = out.shape[:2]
        bar_h = 78 if self.overlay_style == "compact" else 104
        cv2.rectangle(out, (0, 0), (w, bar_h), (10, 14, 20), -1)
        if self.overlay_style != "compact":
            cv2.putText(out, "Jetson Backdoor Demo Preview", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2)
        model = str(metrics.get("model") or "unknown")
        ratio = metrics.get("attention_ratio")
        ratio_text = "-" if ratio is None else f"{float(ratio):.1f}x"
        status_left = f"mode={mode}  model={model}  fps={measured_fps:.1f}"
        status_right = (
            f"attack={'ON' if attack_on else 'OFF'}  "
            f"defense={'ON' if defense_on else 'OFF'}  "
            f"{defense_mode}  attn={ratio_text}"
        )
        y1, y2 = (30, 58) if self.overlay_style == "compact" else (60, 84)
        cv2.putText(out, status_left, (16, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 210, 240), 1)
        cv2.putText(out, status_right, (16, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 210, 240), 1)

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
  <title>Jetson Backdoor Demo</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --bg: #070b13;
      --panel: rgba(18, 26, 40, 0.88);
      --panel-2: rgba(12, 18, 29, 0.92);
      --line: rgba(132, 153, 180, 0.18);
      --text: #f3f7fb;
      --muted: #8fa0b5;
      --blue: #6ea8fe;
      --cyan: #58d5ff;
      --green: #6ee7a8;
      --amber: #ffd166;
      --red: #ff6b7a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(78, 132, 255, 0.24), transparent 34rem),
        radial-gradient(circle at bottom right, rgba(39, 211, 178, 0.14), transparent 30rem),
        var(--bg);
    }
    main { max-width: 1360px; margin: 0 auto; padding: 24px; }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: clamp(26px, 3vw, 40px); letter-spacing: -0.04em; }
    .subtitle { color: var(--muted); margin-top: 8px; font-size: 15px; }
    .status { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .pill {
      background: rgba(16, 24, 38, 0.82);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      color: #d7e2ee;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .pill.ok { border-color: rgba(110, 231, 168, 0.45); color: var(--green); }
    .pill.warn { border-color: rgba(255, 209, 102, 0.45); color: var(--amber); }
    .hero {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric {
      background: linear-gradient(180deg, rgba(28, 39, 60, 0.94), rgba(14, 20, 31, 0.94));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      min-height: 96px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.25);
    }
    .metric .label { color: var(--muted); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }
    .metric .value { margin-top: 8px; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }
    .metric .hint { margin-top: 6px; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 2.1fr) minmax(330px, 0.9fr);
      gap: 16px;
      align-items: start;
    }
    .panel, .side, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 60px rgba(0,0,0,0.28);
      backdrop-filter: blur(18px);
    }
    .panel { overflow: hidden; }
    .panel-head, .side-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.02);
    }
    .panel-title, .side-title { font-weight: 700; letter-spacing: -0.01em; }
    .panel-subtitle { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .stream-wrap { padding: 12px; }
    img { display: block; width: 100%; border-radius: 14px; background: #03060a; }
    .side { padding-bottom: 14px; }
    .controls { display: grid; gap: 12px; padding: 14px; }
    .control-group {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }
    .control-title { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 10px; }
    button {
      width: 100%;
      border: 1px solid rgba(132, 153, 180, 0.22);
      background: rgba(31, 43, 64, 0.9);
      color: var(--text);
      border-radius: 12px;
      padding: 11px 12px;
      margin: 4px 0;
      cursor: pointer;
      font-size: 14px;
      text-align: left;
      transition: border-color .15s, transform .15s, background .15s;
    }
    button:hover { border-color: rgba(110, 168, 254, 0.6); background: rgba(42, 57, 84, 0.92); transform: translateY(-1px); }
    button.active { border-color: rgba(88, 213, 255, 0.72); background: linear-gradient(135deg, rgba(41, 87, 140, 0.95), rgba(28, 59, 96, 0.95)); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 16px; }
    .card { padding: 14px; min-height: 86px; }
    .card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .12em; }
    .card .value { margin-top: 7px; font-size: 15px; line-height: 1.35; word-break: break-word; }
    .error { color: var(--amber); }
    .danger { color: var(--red); }
    .ok { color: var(--green); }
    code { color: #9bd1ff; }
    @media (max-width: 980px) {
      header, .layout, .hero { grid-template-columns: 1fr; }
      .status { justify-content: flex-start; }
      main { padding: 16px; }
    }
    #fireAlertBanner { display: none; margin-bottom: 16px; }
    .fire-status {
      padding: 20px 24px; border-radius: 18px; text-align: center;
      font-size: 32px; font-weight: 800; letter-spacing: -0.02em;
      border: 2px solid var(--line); transition: all .3s ease;
      background: linear-gradient(180deg, rgba(28,39,60,0.94), rgba(14,20,31,0.94));
      color: var(--green);
    }
    .fire-status.alarm {
      background: linear-gradient(180deg, rgba(160,20,20,0.95), rgba(100,10,10,0.95));
      border-color: rgba(255,80,80,0.7); color: #fff;
      animation: fire-pulse 1s ease-in-out infinite;
    }
    @keyframes fire-pulse {
      0%, 100% { box-shadow: 0 0 20px rgba(255,60,60,0.3); }
      50% { box-shadow: 0 0 50px rgba(255,60,60,0.75); }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Jetson Backdoor Defense Console</h1>
        <div class="subtitle">Live camera stream, quantized inference, attention signal, and runtime defense status</div>
      </div>
      <div class="status">
        <span class="pill" id="connected">connecting</span>
        <span class="pill" id="modePill">mode: -</span>
        <span class="pill" id="pipelinePill">pipeline: -</span>
      </div>
    </header>

    <div id="fireAlertBanner">
      <div id="fireStatusDiv" class="fire-status">Initializing fire detector...</div>
    </div>

    <section class="hero">
      <div class="metric">
        <div class="label">Stream FPS</div>
        <div class="value" id="fpsHero">-</div>
        <div class="hint" id="fpsHint">waiting for frames</div>
      </div>
      <div class="metric">
        <div class="label">Active Model</div>
        <div class="value" id="modelHero">-</div>
        <div class="hint" id="runtimeHero">runtime pending</div>
      </div>
      <div class="metric">
        <div class="label">Prediction</div>
        <div class="value" id="predictionHero">-</div>
        <div class="hint" id="confidenceHero">confidence pending</div>
      </div>
      <div class="metric">
        <div class="label">Attention Ratio</div>
        <div class="value" id="attentionHero">-</div>
        <div class="hint" id="defenseHero">defense idle</div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <div class="panel-head">
          <div>
            <div class="panel-title">Live Camera Feed</div>
            <div class="panel-subtitle" id="streamMeta">MJPEG stream</div>
          </div>
          <span class="pill" id="sourcePill">source: -</span>
        </div>
        <div class="stream-wrap">
          <img id="stream" src="/stream.mjpg" alt="camera stream">
        </div>
      </div>

      <aside class="side">
        <div class="side-head">
          <div>
            <div class="side-title">Demo Controls</div>
            <div class="panel-subtitle">Mode switches update the live pipeline</div>
          </div>
        </div>
        <div class="controls">
          <div class="control-group">
            <div class="control-title">Demo Mode</div>
            <button data-mode="normal">Normal / FP32 baseline</button>
            <button data-mode="triggered">Triggered / INT8 backdoor</button>
            <button data-mode="defended">Defended / online mitigation</button>
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
      <div class="card" id="fireProbCard" style="display:none"><div class="label">Fire Prob (frame)</div><div class="value" id="fireProb">-</div></div>
      <div class="card" id="fireRatioCard" style="display:none"><div class="label">Fire Frame Ratio</div><div class="value" id="fireRatio">-</div></div>
    </section>
  </main>

  <script>
    let currentStatus = {};
    const defenseModes = ['oracle', 'regionblur', 'patchdrop'];
    const text = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
    const className = (id, value) => { const el = document.getElementById(id); if (el) el.className = value; };

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
      text('connected', data.has_frame ? 'streaming' : 'waiting');
      className('connected', data.has_frame ? 'pill ok' : 'pill warn');
      text('modePill', `mode: ${data.mode}`);
      text('pipelinePill', `${data.async_inference ? 'async' : 'sync'} / ${data.overlay_style || 'overlay'}`);
      text('sourcePill', `source: ${data.actual_source || data.source || '-'}`);
      text('source', `${data.source} -> ${data.actual_source}`);
      text('frames', data.frame_index);
      const cacheText = data.inference_cached ? 'cached infer' : 'fresh infer';
      text('fps', `${data.measured_fps} / target ${data.target_fps} / ${cacheText}`);
      text('fpsHero', data.measured_fps || '-');
      text('fpsHint', `target ${data.target_fps} fps / ${cacheText}`);
      const qura = document.getElementById('qura');
      qura.textContent = data.qura_available ? 'available' : `unavailable: ${data.qura_error || 'unknown'}`;
      qura.className = data.qura_available ? 'value ok' : 'value error';
      text('model', data.model || '-');
      text('modelHero', data.model || '-');
      text('runtime', `torch ${data.torch_version || '-'} / cuda ${data.cuda_version || '-'} / ${data.vit_device || '-'}`);
      text('runtimeHero', `${data.backend || 'torch'} / ${data.vit_device || '-'}`);
      const confidence = data.confidence === null || data.confidence === undefined ? '-' : `${Math.round(data.confidence * 100)}%`;
      text('prediction', `${data.prediction || '-'} (${confidence})`);
      text('predictionHero', data.prediction || '-');
      text('confidenceHero', `confidence ${confidence}`);
      const topk = Array.isArray(data.topk) ? data.topk : [];
      document.getElementById('topk').innerHTML = topk.length
        ? topk.map(item => `${item.display || item.label || '-'} (${Math.round((item.confidence || 0) * 100)}%)`).join('<br>')
        : '-';
      const ratio = data.attention_ratio === null || data.attention_ratio === undefined ? '-' : `${Number(data.attention_ratio).toFixed(1)}x`;
      text('attention', ratio);
      text('attentionHero', ratio);
      text('streamMeta', `${data.width}x${data.height} / ${data.target_fps} fps target / ${cacheText}`);

      const backdoor = document.getElementById('backdoor');
      backdoor.textContent = data.backdoor_active ? 'active / suspicious' : (data.suspicious ? 'suspicious' : 'clear');
      backdoor.className = data.backdoor_active || data.suspicious ? 'value danger' : 'value ok';

      const defense = document.getElementById('defense');
      defense.textContent = data.defense_applied ? `${data.defense_mode} applied` : (data.defense_on ? `${data.defense_mode} armed` : 'off');
      defense.className = data.defense_on ? 'value ok' : 'value';
      text('defenseHero', data.defense_applied ? `${data.defense_mode} applied` : (data.defense_on ? `${data.defense_mode} armed` : 'defense off'));

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

      // Fire detection mode
      if (data.fire_mode) {
        const banner = document.getElementById('fireAlertBanner');
        if (banner) banner.style.display = 'block';
        const statusDiv = document.getElementById('fireStatusDiv');
        if (statusDiv) {
          statusDiv.textContent = data.fire_alarm ? 'FIRE DETECTED' : 'No Fire Detected';
          statusDiv.className = data.fire_alarm ? 'fire-status alarm' : 'fire-status';
        }
        const fpCard = document.getElementById('fireProbCard');
        const frCard = document.getElementById('fireRatioCard');
        if (fpCard) fpCard.style.display = '';
        if (frCard) frCard.style.display = '';
        text('fireProb', data.fire_prob != null ? (data.fire_prob * 100).toFixed(1) + '%' : '-');
        const thresh = data.fire_frame_thresh != null ? (data.fire_frame_thresh * 100).toFixed(0) : '25';
        text('fireRatio', data.fire_frame_ratio != null
          ? (data.fire_frame_ratio * 100).toFixed(0) + '% / ' + thresh + '%'
          : '-');
        const predHero = document.getElementById('predictionHero');
        if (predHero) {
          predHero.textContent = data.fire_alarm ? 'FIRE' : 'no fire';
          predHero.style.color = data.fire_alarm ? 'var(--red)' : 'var(--green)';
        }
      }
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
            self._send_dashboard()
        elif parsed.path == "/react":
            self._send_dashboard(REACT_DASHBOARD_DIR)
        elif parsed.path.startswith("/static/"):
            self._send_static(parsed.path, DASHBOARD_DIR, "/static/")
        elif parsed.path.startswith("/react-static/"):
            self._send_static(parsed.path, REACT_DASHBOARD_DIR, "/react-static/")
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

    def _send_dashboard(self, dashboard_dir: Path = DASHBOARD_DIR) -> None:
        html_path = dashboard_dir / "index.html"
        if html_path.exists():
            self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send_bytes(index_html(), "text/html; charset=utf-8")

    def _send_static(self, request_path: str, root_dir: Path, prefix: str) -> None:
        rel = request_path.removeprefix(prefix).lstrip("/")
        if not rel or ".." in Path(rel).parts:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        path = root_dir / rel
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        self._send_bytes(path.read_bytes(), content_type)

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
    parser.add_argument("--height", type=int, default=360)
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
    parser.add_argument("--sync-processing", action="store_true", help="Run capture, inference, and JPEG encoding in one thread.")
    parser.add_argument("--overlay-style", default="compact", choices=["full", "compact", "off"], help="Amount of overlay drawn into each MJPEG frame.")
    parser.add_argument("--heatmap-overlay", action="store_true")
    parser.add_argument("--vit-device", default="cuda")
    parser.add_argument("--backend", default="torch", choices=["torch", "trt"], help="Inference backend. TRT is logits-only and only preferred for triggered classification.")
    parser.add_argument("--trt-engine", default=None, help="Path to a TensorRT .engine file for --backend trt.")
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--blur-sigma", type=float, default=6.0)
    parser.add_argument("--int8-only", action="store_true")
    # --- fire detection mode ---
    parser.add_argument("--fire-checkpoint", default=None,
                        help="Fine-tuned fire/no_fire ViT head checkpoint (.pt). Enables fire detection mode.")
    parser.add_argument("--fire-prob-thresh", type=float, default=0.30,
                        help="Per-frame fire_prob threshold for the sliding window (default: 0.30)")
    parser.add_argument("--fire-frame-thresh", type=float, default=0.25,
                        help="Fraction of fire frames needed to trigger alarm (default: 0.25)")
    parser.add_argument("--fire-window", type=int, default=15,
                        help="Sliding window size in frames (default: 15)")
    parser.add_argument("--fire-int8-path", default=None,
                        help="Backdoored INT8 ONNX for fire triggered/defended modes "
                             "(e.g. outputs/lab_fire_vit/fire_vit_backdoor_int8.onnx)")
    parser.add_argument("--fire-trigger-path", default=None,
                        help="Trigger .pt for fire backdoor demo "
                             "(e.g. outputs/imagenet_vit_qura/generated_triggers/"
                             "vit_base_imagenet_t0_stage2_fixed_seed1005.pt)")
    parser.add_argument("--fire-roi", default=None,
                        help="ROI for fire inference: 'x1,y1,x2,y2' pixels in the original frame. "
                             "Only the ROI crop is sent to FireViT; results are overlaid on the full frame. "
                             "Example: --fire-roi 200,100,1600,900")
    parser.add_argument("--fire-softmask", type=float, default=None,
                        help="Enable soft-mask inference instead of hard ROI crop. "
                             "Float in [0, 1]: attenuation factor applied to pixels outside --fire-roi "
                             "(0=fully black, 1=no effect). Full frame is sent to FireViT with background "
                             "dimmed and a Gaussian-blurred boundary. Requires --fire-roi. "
                             "Example: --fire-softmask 0.3")
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
        async_inference=not args.sync_processing,
        overlay_style=args.overlay_style,
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
