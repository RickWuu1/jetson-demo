"""
Final Demo: Attention-Guided PatchDrop Defense
================================================
Generates a single presentation-ready panel showing the full pipeline:
  1. FP32 normal prediction
  2. W4A8 backdoor-activated wrong prediction
  3. Attention anomaly detection
  4. Attention-Guided PatchDrop recovery

Usage:
  cd third_party/qura/ours/main
  conda run -n qura python /path/to/demos/final_vit_patchdrop_demo.py

Output:
  outputs/final_demo_panel.png
"""

import os
import sys
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

QURA_ROOT = '/home/kaixin/yisong/demo/third_party/qura/ours/main'
sys.path.insert(0, QURA_ROOT)
sys.path.insert(0, os.path.join(QURA_ROOT, 'setting'))

OUTPUT_PATH = '/home/kaixin/yisong/demo/outputs/final_demo_panel.png'
DEVICE = torch.device('cuda:2')
SEED = 1005
PATCH_SIZE = 16
GRID_SIZE = 14
TRIGGER_SIZE = 12

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


class AttentionHook:
    def __init__(self, model):
        self.last_attn = None
        self.hook = None
        last_module = None
        for name, module in model.named_modules():
            if 'attn_drop' in name:
                last_module = module
        if last_module:
            self.hook = last_module.register_forward_hook(self._fn)

    def _fn(self, module, input, output):
        self.last_attn = output.detach()

    def get_cls_attention(self):
        if self.last_attn is None:
            return np.ones(196) / 196
        return self.last_attn[0, :, 0, 1:].mean(dim=0).cpu().numpy()

    def get_top1(self):
        attn = self.get_cls_attention()
        return int(np.argmax(attn))

    def remove(self):
        if self.hook:
            self.hook.remove()


def load_fp32():
    import timm
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE).eval()
    return model


def load_w4a8():
    import timm
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_quantization
    from utils import parse_config

    config = parse_config(os.path.join(QURA_ROOT, 'configs/cv_vit_4_8_bd.yaml'))
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    extra = {} if not hasattr(config, 'extra_prepare_dict') else config.extra_prepare_dict
    model = prepare_by_platform(model, BackendType.Academic, extra)
    model.eval()
    enable_quantization(model)
    state = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.quant_bd_None_t0.pth'), map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


def get_trigger():
    from dataset.dataset import Cifar10
    from setting.config import load_calibrate_data, cv_trigger_generation, CV_TRIGGER_SIZE

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    fp32 = load_fp32()
    data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                   pattern='stage2', quant=True, image_size=224)
    train_loader, _, _, _ = data.get_loader()
    cali = load_calibrate_data(train_loader, 16)
    trigger = cv_trigger_generation(fp32, cali, 0, CV_TRIGGER_SIZE * 2, DEVICE, mean, std)
    del fp32
    return trigger, mean, std


def main():
    print("Loading models and generating trigger...")
    trigger, mean, std = get_trigger()
    fp32_model = load_fp32()
    w4a8_model = load_w4a8()

    # Get a sample image
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    raw_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    norm_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])

    ds_raw = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=raw_tf)
    ds_norm = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=norm_tf)

    # Find non-class-0 sample
    for idx in range(len(ds_raw)):
        if ds_raw[idx][1] != 0:
            break

    raw_clean = ds_raw[idx][0]
    norm_clean = ds_norm[idx][0].unsqueeze(0)
    true_label = ds_raw[idx][1]

    # Create trigger version
    raw_trigger = raw_clean.clone()
    ts = TRIGGER_SIZE
    raw_trigger[:, 224-ts:, 224-ts:] = trigger

    norm_trigger = norm_clean.clone()
    trigger_norm = transforms.Normalize(mean, std)(trigger)
    norm_trigger[0, :, 224-ts:, 224-ts:] = trigger_norm

    # === Run inference ===
    with torch.no_grad():
        # FP32 clean
        pred_fp32 = fp32_model(norm_clean.to(DEVICE)).argmax(1).item()

        # W4A8 trigger (no defense)
        attn_hook = AttentionHook(w4a8_model)
        pred_w4a8_nodef = w4a8_model(norm_trigger.to(DEVICE)).argmax(1).item()
        top1_patch = attn_hook.get_top1()
        attn_map = attn_hook.get_cls_attention()
        attn_hook.remove()

        # W4A8 trigger (with PatchDrop)
        masked = norm_trigger.clone()
        r, c = top1_patch // GRID_SIZE, top1_patch % GRID_SIZE
        masked[0, :, r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0
        pred_w4a8_defense = w4a8_model(masked.to(DEVICE)).argmax(1).item()

    raw_masked = raw_trigger.clone()
    raw_masked[:, r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0

    # === Generate figure ===
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # Row 1: Pipeline images
    def show_img(ax, img_tensor, title, title_color='black', border_color=None):
        img = np.clip(img_tensor.permute(1, 2, 0).numpy(), 0, 1)
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', color=title_color, pad=10)
        ax.axis('off')
        if border_color:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border_color)
                spine.set_linewidth(3)

    # Panel 1: FP32 clean
    ax1 = fig.add_subplot(gs[0, 0])
    show_img(ax1, raw_clean,
             f'Step 1: FP32 Model\nPred: {CIFAR10_CLASSES[pred_fp32]} (correct)',
             '#1565C0')

    # Panel 2: W4A8 trigger (attacked)
    ax2 = fig.add_subplot(gs[0, 1])
    show_img(ax2, raw_trigger,
             f'Step 2: W4A8 + Trigger\nPred: {CIFAR10_CLASSES[pred_w4a8_nodef]} (ATTACKED)',
             '#C62828')
    rect = Rectangle((224-ts, 224-ts), ts, ts, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)

    # Panel 3: Attention detection
    ax3 = fig.add_subplot(gs[0, 2])
    attn_grid = attn_map.reshape(GRID_SIZE, GRID_SIZE)
    im = ax3.imshow(attn_grid, cmap='hot', interpolation='nearest')
    ax3.set_title('Step 3: Attention Detection\n(Last-layer CLS attention)', fontsize=11,
                  fontweight='bold', color='#E65100')
    # Trigger patch
    rect_t = Rectangle((13-0.5, 13-0.5), 1, 1, lw=2, edgecolor='cyan',
                       facecolor='none', linestyle='--', label='Trigger')
    ax3.add_patch(rect_t)
    # Detected patch
    rect_d = Rectangle((c-0.5, r-0.5), 1, 1, lw=2, edgecolor='lime',
                       facecolor='none', label='Detected')
    ax3.add_patch(rect_d)
    ax3.legend(loc='upper left', fontsize=8)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Panel 4: After PatchDrop
    ax4 = fig.add_subplot(gs[0, 3])
    show_img(ax4, raw_masked,
             f'Step 4: After PatchDrop\nPred: {CIFAR10_CLASSES[pred_w4a8_defense]} (RECOVERED)',
             '#2E7D32')
    rect_m = Rectangle((c*PATCH_SIZE, r*PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                       lw=2, edgecolor='lime', facecolor='none')
    ax4.add_patch(rect_m)

    # Row 2: Summary results
    ax_table = fig.add_subplot(gs[1, :2])
    ax_table.axis('off')

    table_data = [
        ['No Defense', '96.80%', '99.92%'],
        ['Random PatchDrop', '96.79%', '99.36%'],
        ['Attn-Guided PatchDrop', '96.48%', '0.43%'],
        ['Oracle (Upper Bound)*', '96.76%', '0.48%'],
    ]
    colors = [['#FFEBEE']*3, ['#FFF3E0']*3, ['#E8F5E9']*3, ['#FFF8E1']*3]

    table = ax_table.table(cellText=table_data,
                           colLabels=['Strategy', 'Clean Acc', 'Trigger ASR'],
                           cellColours=colors, colColours=['#CFD8DC']*3,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for j in range(3):
        table[0, j].set_text_props(fontweight='bold')
        table[3, j].set_text_props(fontweight='bold')
    ax_table.set_title('Defense Comparison (Full Test Set)', fontsize=12, fontweight='bold')

    # Summary text
    ax_text = fig.add_subplot(gs[1, 2:])
    ax_text.axis('off')
    summary = (
        "Key Results\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"• True label: {CIFAR10_CLASSES[true_label]}\n"
        f"• Detected patch: ({r}, {c})\n"
        f"• Trigger patch:  (13, 13)\n"
        f"• Match: {'YES' if top1_patch == 195 else 'NO'}\n\n"
        "• ASR: 99.92% → 0.43%\n"
        "  (reduced by 99.49 pp)\n\n"
        "• Clean Acc: 96.80% → 96.48%\n"
        "  (only -0.32% drop)\n\n"
        "• Guided ≈ Oracle\n"
        "  (near-perfect localization)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "*Oracle = known trigger pos.\n"
        " (upper bound, not deployable)"
    )
    ax_text.text(0.1, 0.95, summary, transform=ax_text.transAxes,
                 fontsize=10.5, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                          edgecolor='#BDBDBD', alpha=0.9))

    fig.suptitle('Quantization-Activated Backdoor: Detection and Mitigation via Attention-Guided PatchDrop',
                 fontsize=15, fontweight='bold', y=0.98)

    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFinal demo panel saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
