from attacks.det.dpatch import DPatchAttack


def build_det_attack(cfg):
    name = cfg.get("attack_type", "dpatch")
    if name == "dpatch":
        return DPatchAttack(cfg)
    raise ValueError(f"Unknown det attack type: {name}")
