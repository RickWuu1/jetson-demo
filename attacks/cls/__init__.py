from attacks.cls.adv_patch import AdvPatchAttack


def build_cls_attack(cfg):
    name = cfg.get("attack_type", "adv_patch")
    if name == "adv_patch":
        return AdvPatchAttack(cfg)
    raise ValueError(f"Unknown cls attack type: {name}")
