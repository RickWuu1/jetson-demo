import torch
ckpt = torch.load('outputs/lab_fire_vit/fire_vit_qura_best.pt', map_location='cpu', weights_only=False)
keys = sorted(ckpt['backbone_state_dict'].keys())
q_keys = [k for k in keys if 'quantizer' in k or 'range_tracker' in k]
print(f'Total backbone keys: {len(keys)}')
print(f'Quantizer-related keys: {len(q_keys)}')
print('First 20 quantizer keys:')
for k in q_keys[:20]:
    v = ckpt['backbone_state_dict'][k]
    shape = tuple(v.shape) if hasattr(v, 'shape') else v
    print(f'  {k}: {shape}')
