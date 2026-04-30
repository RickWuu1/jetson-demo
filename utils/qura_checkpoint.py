from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import torch

from mqbench.fake_quantize.adaround_quantizer import AdaRoundFakeQuantize


CheckpointInput = Union[str, Path, Mapping[str, Any]]


def extract_state_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    cleaned = {}
    for key, value in payload.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def _materialize_adaround_parameters(model: torch.nn.Module, state_dict: Mapping[str, Any]) -> Iterable[str]:
    restored = []
    for name, module in model.named_modules():
        if not isinstance(module, AdaRoundFakeQuantize):
            continue

        alpha_key = f"{name}.alpha" if name else "alpha"
        if alpha_key not in state_dict:
            continue

        alpha_tensor = state_dict[alpha_key]
        if not isinstance(alpha_tensor, torch.Tensor):
            continue

        if "alpha" not in module._parameters or module._parameters["alpha"] is None:
            module.alpha = torch.nn.Parameter(alpha_tensor.detach().clone())
        elif module.alpha.shape != alpha_tensor.shape:
            module.alpha = torch.nn.Parameter(alpha_tensor.detach().clone())

        # Saved QURA checkpoints rely on AdaRound hard rounding at inference time.
        module.adaround = True
        module.round_mode = "learned_hard_sigmoid"
        module.gamma = 0.0
        module.zeta = 1.0
        restored.append(name)
    return restored


def recover_soft_weights_from_state(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
) -> List[str]:
    """Recover a continuous proxy weight tensor from hard-quantized weights and saved AdaRound alpha.

    Saved QURA checkpoints store quantized hard values in ``layer.weight`` and often keep the
    corresponding AdaRound ``alpha`` tensor. We can reconstruct a pre-hard-rounding proxy:

      q = (floor(w / s) + r) * s
      w_proxy = (floor(w / s) + c) * s

    where ``r`` is the hard rounding bit inferred from ``alpha`` and ``c`` is the relaxed
    rounding variable ``sigmoid(alpha)``. This restores one-step flipping freedom for post-hoc
    defenses without touching checkpoints that lack ``alpha``.
    """

    name_to_module = dict(model.named_modules())
    recovered = []

    for fq_name, module in model.named_modules():
        if not isinstance(module, AdaRoundFakeQuantize):
            continue

        alpha_key = f"{fq_name}.alpha" if fq_name else "alpha"
        if alpha_key not in state_dict:
            continue

        parent_name = fq_name.rsplit(".", 1)[0]
        parent = name_to_module.get(parent_name)
        if parent is None or not hasattr(parent, "weight"):
            continue

        alpha = state_dict[alpha_key].detach().to(parent.weight.device, parent.weight.dtype)
        # AdaRound in this repo uses gamma=0, zeta=1.
        continuous = torch.sigmoid(alpha).clamp(0, 1)
        hard = (alpha >= 0).to(parent.weight.dtype)

        weight = parent.weight.data
        if module.ch_axis != -1:
            shape = [1] * weight.ndim
            shape[module.ch_axis] = weight.shape[module.ch_axis]
            scale = module.scale.data.reshape(shape).to(weight.device, weight.dtype)
        else:
            scale = module.scale.data.to(weight.device, weight.dtype)

        # q / scale is the current quantized integer. Remove the hard rounding bit, then
        # insert the continuous proxy variable to recover a soft pre-hard-rounding weight.
        qint = torch.round(weight / scale)
        base = qint - hard
        proxy = (base + continuous) * scale
        parent.weight.data.copy_(proxy)
        recovered.append(parent_name)

    return recovered


def load_quant_checkpoint(
    model: torch.nn.Module,
    checkpoint: CheckpointInput,
    strict: bool = False,
    restore_adaround: bool = False,
    recover_soft_weights: bool = False,
) -> Tuple[Dict[str, Any], Iterable[str], Iterable[str], Iterable[str], Iterable[str]]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(str(checkpoint), map_location="cpu")
    else:
        payload = checkpoint

    state_dict = extract_state_dict(payload)
    restored_adaround = []
    if restore_adaround:
        restored_adaround = list(_materialize_adaround_parameters(model, state_dict))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    recovered_soft = []
    if recover_soft_weights:
        recovered_soft = recover_soft_weights_from_state(model, state_dict)
    return state_dict, missing, unexpected, restored_adaround, recovered_soft
