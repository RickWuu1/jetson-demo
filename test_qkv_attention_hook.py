import torch
import timm
import traceback

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model("vit_base_patch16_224", pretrained=False).eval().to(device)

    saved = {}

    def hook_fn(module, inp, out):
        saved["qkv"] = out.detach()

    target_name = "blocks.11.attn.qkv"
    for name, module in model.named_modules():
        if name == target_name:
            module.register_forward_hook(hook_fn)
            print("Hooked:", name)

    x = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        _ = model(x)

    qkv = saved["qkv"]
    print("qkv shape:", qkv.shape)

    B, N, C3 = qkv.shape
    num_heads = model.blocks[11].attn.num_heads
    head_dim = C3 // 3 // num_heads

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)

    cls_attn = attn[0, :, 0, 1:].mean(dim=0)

    print("attn shape:", attn.shape)
    print("cls_attn shape:", cls_attn.shape)
    print("max:", cls_attn.max().item(), "mean:", cls_attn.mean().item())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
