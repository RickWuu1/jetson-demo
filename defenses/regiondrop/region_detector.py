from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


DEFAULT_INPUT_SIZE = 224
DEFAULT_PATCH_SIZE = 16


@dataclass
class RegionSearchResult:
    pixel_bbox: tuple[int, int, int, int]
    score: float
    patch_index: int


class AttentionHook:
    """Capture CLS-to-patch attention from the final ViT attention block."""

    def __init__(self, model: torch.nn.Module, num_heads: int = 12) -> None:
        self.model = model
        self.num_heads = num_heads
        self._last_attn: Optional[torch.Tensor] = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._hook_mode: Optional[str] = None
        self._register_hook()

    def _register_hook(self) -> None:
        last_attn_drop = None
        last_qkv = None
        module_by_name = dict(self.model.named_modules())

        for name, module in module_by_name.items():
            if name.endswith("attn_drop") or "attn_drop" in name:
                last_attn_drop = module
            if name == "blocks.11.attn.qkv" or name.endswith(".attn.qkv"):
                last_qkv = (name, module)

        if last_attn_drop is not None:
            self._hook_mode = "attn_drop"
            self._hook = last_attn_drop.register_forward_hook(self._hook_attention)
            return

        if last_qkv is not None:
            qkv_name, qkv_module = last_qkv
            parent_name = qkv_name.rsplit(".", 1)[0]
            parent = module_by_name.get(parent_name)
            self.num_heads = int(getattr(parent, "num_heads", self.num_heads))
            self._hook_mode = "qkv"
            self._hook = qkv_module.register_forward_hook(self._hook_qkv)

    def _hook_attention(self, module, inputs, output) -> None:
        attn = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(attn, torch.Tensor):
            self._last_attn = attn.detach()

    def _hook_qkv(self, module, inputs, output) -> None:
        qkv = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(qkv, torch.Tensor) or qkv.dim() != 3:
            return

        batch, tokens, channels3 = qkv.shape
        head_dim = channels3 // 3 // self.num_heads
        if head_dim <= 0 or channels3 != 3 * self.num_heads * head_dim:
            return

        qkv = qkv.detach().reshape(batch, tokens, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        self._last_attn = attn.softmax(dim=-1).detach()

    def get_cls_attention_map(self, reduce: str = "std") -> np.ndarray:
        if self._last_attn is None:
            return np.ones(196, dtype=np.float32) / 196.0

        attn = self._last_attn
        if attn.dim() == 4:
            cls_patch = attn[0, :, 0, 1:]
        elif attn.dim() == 3:
            if attn.shape[1] == attn.shape[2]:
                cls_patch = attn[:, 0, 1:]
            else:
                cls_patch = attn[0, :, 1:]
        else:
            return np.ones(196, dtype=np.float32) / 196.0

        if cls_patch.dim() == 1:
            cls_patch = cls_patch.unsqueeze(0)

        if reduce == "mean":
            reduced = cls_patch.mean(dim=0)
        else:
            reduced = cls_patch.std(dim=0)

        return reduced.float().cpu().numpy().reshape(-1).astype(np.float32)

    def remove(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def multi_scale_region_search(attn_map: np.ndarray) -> RegionSearchResult:
    scores = np.asarray(attn_map, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return RegionSearchResult((0, 0, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), 0.0, 0)

    grid_size = int(round(scores.size ** 0.5))
    if grid_size * grid_size != scores.size:
        grid_size = 14
        padded = np.zeros(grid_size * grid_size, dtype=np.float32)
        padded[: min(scores.size, padded.size)] = scores[: padded.size]
        scores = padded

    idx = int(scores.argmax())
    patch_size = DEFAULT_INPUT_SIZE // grid_size
    row, col = divmod(idx, grid_size)
    y1, x1 = row * patch_size, col * patch_size
    return RegionSearchResult(
        (y1, x1, y1 + patch_size, x1 + patch_size),
        float(scores[idx]),
        idx,
    )
