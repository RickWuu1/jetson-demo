"""
CLP (Channel Lipschitzness-based Pruning) Defense
ECCV 2022: Data-free backdoor detection and removal.

Reference: https://github.com/rkteddy/channel-Lipschitzness-based-pruning

Algorithm:
  1. For each Conv2d+BatchNorm2d pair, compute per-channel UCLC
     (Upper bound of Channel Lipschitz Constant) via SVD of
     BN-scaled weight slices.
  2. Flag channels where UCLC > mean + u * std as suspicious.
  3. Zero-out those channels' weights and biases.
  4. Evaluate clean acc and ASR before/after.

Usage:
    python clp_defense.py --nbits 4 --u 2.0
    python clp_defense.py --nbits 4 --u 1.0  # more aggressive
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils.datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn
from utils.learner import valid_w_backdoor, valid_quantize_w_backdoor


# --------------------------------------------------------------------------
#   1. Per-channel Lipschitz constant (UCLC) computation
# --------------------------------------------------------------------------

def compute_channel_lips(model):
    """
    For each Conv2d+BatchNorm2d pair, compute per-channel UCLC:
        w_i = W[i].reshape(in_ch, kH*kW) * |gamma_i / std_i|
        UCLC_i = max_singular_value(w_i)

    Returns: dict {layer_name: FloatTensor [out_ch]}
    """
    channel_lips = {}
    modules = list(model.named_modules())

    for i, (name, module) in enumerate(modules):
        if not isinstance(module, nn.Conv2d):
            continue

        # Find the immediately following BatchNorm2d
        bn = None
        for j in range(i + 1, min(i + 5, len(modules))):
            next_mod = modules[j][1]
            if isinstance(next_mod, nn.BatchNorm2d):
                bn = next_mod
                break
            elif isinstance(next_mod, (nn.Conv2d, nn.Linear, nn.ReLU)):
                break

        W = module.weight.data.float()  # [out_ch, in_ch, kH, kW]
        out_ch = W.shape[0]

        if bn is not None:
            std = torch.sqrt(bn.running_var.data.float() + bn.eps)
            gamma = bn.weight.data.float()
            scale = (gamma / std).abs()  # [out_ch]
        else:
            scale = torch.ones(out_ch)

        lips = torch.zeros(out_ch)
        for idx in range(out_ch):
            # w_i: [in_ch, kH*kW] after reshaping and scaling
            w_flat = W[idx].reshape(W.shape[1], -1).cpu()
            w_scaled = w_flat * scale[idx].item()
            try:
                sv = torch.linalg.svdvals(w_scaled)
                lips[idx] = sv[0]
            except Exception:
                lips[idx] = w_scaled.norm()

        channel_lips[name] = lips

    return channel_lips


# --------------------------------------------------------------------------
#   2. Detect risky channels
# --------------------------------------------------------------------------

def find_risky_channels(channel_lips, u=2.0):
    """
    Flag channels where UCLC > mean + u * std.
    Returns: dict {layer_name: dict with indices/scores/threshold}
    """
    risky = {}
    for name, lips in channel_lips.items():
        m, s = lips.mean().item(), lips.std().item()
        threshold = m + u * s
        idx = torch.where(lips > threshold)[0].tolist()
        risky[name] = {
            'mean': m,
            'std': s,
            'threshold': threshold,
            'max': lips.max().item(),
            'n_risky': len(idx),
            'n_total': len(lips),
            'risky_indices': idx,
            'risky_scores': lips[idx].tolist() if idx else [],
        }
    return risky


# --------------------------------------------------------------------------
#   3. Zero-out risky channels
# --------------------------------------------------------------------------

def zero_out_channels(model, risky_channels, min_keep=4):
    """
    Zero out weights and biases for risky channels.
    min_keep: skip zeroing if it would leave fewer than min_keep active channels.
    Returns: list of {layer, n_zeroed, n_total} records.
    """
    state = model.state_dict()
    log = []

    for layer_name, info in risky_channels.items():
        idx = info['risky_indices']
        if not idx:
            continue

        wkey = layer_name + '.weight'
        bkey = layer_name + '.bias'
        n_total = info['n_total']

        if n_total - len(idx) < min_keep:
            log.append({'layer': layer_name, 'status': 'skipped',
                        'reason': f'would leave <{min_keep} active channels',
                        'n_risky': len(idx), 'n_total': n_total})
            continue

        if wkey in state:
            state[wkey][idx] = 0.0
        if bkey in state:
            state[bkey][idx] = 0.0

        log.append({'layer': layer_name, 'status': 'zeroed',
                    'n_zeroed': len(idx), 'n_total': n_total,
                    'ratio': round(len(idx) / n_total, 3)})

    model.load_state_dict(state)
    return log


# --------------------------------------------------------------------------
#   4. Evaluation helpers
# --------------------------------------------------------------------------

def evaluate(net, valid_loader, task_loss, use_cuda, nbits=None,
             wqmode='per_layer_symmetric', aqmode='per_layer_asymmetric'):
    """
    Evaluate clean acc and trigger ASR.
    nbits=None: FP32 evaluation.
    nbits=4/8: quantized evaluation.
    """
    if nbits is None:
        clean_acc, clean_loss, bdoor_acc, bdoor_loss = \
            valid_w_backdoor('eval', net, valid_loader, task_loss,
                             use_cuda=use_cuda, silent=True)
    else:
        clean_acc, clean_loss, bdoor_acc, bdoor_loss = \
            valid_quantize_w_backdoor(
                'eval', net, valid_loader, task_loss,
                use_cuda=use_cuda, wqmode=wqmode, aqmode=aqmode,
                nbits=nbits, silent=True)
    return {
        'clean_acc': round(float(clean_acc), 2),
        'clean_loss': round(float(clean_loss), 4),
        'trigger_asr': round(float(bdoor_acc), 2),
        'trigger_loss': round(float(bdoor_loss), 4),
    }


# --------------------------------------------------------------------------
#   5. Main
# --------------------------------------------------------------------------

def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Load dataset (backdoor valid loader) ---
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    _, valid_loader = load_backdoor('cifar10', 'square', 0, 128, True, kwargs)

    # --- Load backdoored model ---
    net = load_network('cifar10', 'ResNet18', 10)
    load_trained_network(net, use_cuda, args.model)
    if use_cuda:
        net.cuda()
    net.eval()

    task_loss = load_lossfn('cross-entropy')

    # -----------------------------------------------------------------------
    # Step 1: Evaluate BEFORE removal
    # -----------------------------------------------------------------------
    print("\n[CLP] Step 1: Evaluating BEFORE removal...")
    eval_fp32_before = evaluate(net, valid_loader, task_loss, use_cuda, nbits=None)
    eval_int4_before = evaluate(net, valid_loader, task_loss, use_cuda, nbits=4,
                                wqmode=args.wqmode, aqmode=args.aqmode)
    print(f"  FP32: clean={eval_fp32_before['clean_acc']}%, ASR={eval_fp32_before['trigger_asr']}%")
    print(f"  INT4: clean={eval_int4_before['clean_acc']}%, ASR={eval_int4_before['trigger_asr']}%")

    with open(os.path.join(out_dir, 'eval_before.json'), 'w') as f:
        json.dump({'fp32': eval_fp32_before, f'int{args.nbits}': eval_int4_before}, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 2: CLP Detection — compute channel Lipschitz constants
    # -----------------------------------------------------------------------
    print("\n[CLP] Step 2: Computing per-channel Lipschitz constants...")
    channel_lips = compute_channel_lips(net)
    risky_channels = find_risky_channels(channel_lips, u=args.u)

    # Build detection report
    detect_report = []
    total_risky = 0
    for layer_name, info in risky_channels.items():
        detect_report.append({
            'layer': layer_name,
            'n_total': info['n_total'],
            'n_risky': info['n_risky'],
            'threshold': round(info['threshold'], 4),
            'mean_lips': round(info['mean'], 4),
            'std_lips': round(info['std'], 4),
            'max_lips': round(info['max'], 4),
        })
        total_risky += info['n_risky']
        if info['n_risky'] > 0:
            print(f"  {layer_name}: {info['n_risky']}/{info['n_total']} risky "
                  f"(threshold={info['threshold']:.4f}, max={info['max']:.4f})")

    print(f"\n  Total suspicious channels: {total_risky}")

    with open(os.path.join(out_dir, 'detect_report.json'), 'w') as f:
        json.dump({'u': args.u, 'total_risky': total_risky, 'layers': detect_report}, f, indent=2)

    with open(os.path.join(out_dir, 'risky_channels.json'), 'w') as f:
        serializable = {}
        for k, v in risky_channels.items():
            serializable[k] = {
                'mean': round(v['mean'], 4),
                'std': round(v['std'], 4),
                'threshold': round(v['threshold'], 4),
                'max': round(v['max'], 4),
                'n_risky': v['n_risky'],
                'n_total': v['n_total'],
                'risky_indices': v['risky_indices'],
                'risky_scores': [round(s, 4) for s in v['risky_scores']],
            }
        json.dump(serializable, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 3: Zero-out removal
    # -----------------------------------------------------------------------
    print("\n[CLP] Step 3: Zeroing out risky channels...")
    only_risky = {k: v for k, v in risky_channels.items() if v['n_risky'] > 0}
    removal_log = zero_out_channels(net, only_risky, min_keep=args.min_keep)

    for rec in removal_log:
        if rec['status'] == 'zeroed':
            print(f"  {rec['layer']}: zeroed {rec['n_zeroed']}/{rec['n_total']} channels ({rec['ratio']*100:.1f}%)")
        else:
            print(f"  {rec['layer']}: SKIPPED — {rec['reason']}")

    with open(os.path.join(out_dir, 'remove_log.txt'), 'w') as f:
        for rec in removal_log:
            f.write(json.dumps(rec) + '\n')

    # -----------------------------------------------------------------------
    # Step 4: Evaluate AFTER removal
    # -----------------------------------------------------------------------
    print("\n[CLP] Step 4: Evaluating AFTER removal...")
    eval_fp32_after = evaluate(net, valid_loader, task_loss, use_cuda, nbits=None)
    eval_int4_after = evaluate(net, valid_loader, task_loss, use_cuda, nbits=4,
                               wqmode=args.wqmode, aqmode=args.aqmode)
    print(f"  FP32: clean={eval_fp32_after['clean_acc']}%, ASR={eval_fp32_after['trigger_asr']}%")
    print(f"  INT4: clean={eval_int4_after['clean_acc']}%, ASR={eval_int4_after['trigger_asr']}%")

    with open(os.path.join(out_dir, 'eval_after.json'), 'w') as f:
        json.dump({'fp32': eval_fp32_after, f'int{args.nbits}': eval_int4_after}, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 5: Summary
    # -----------------------------------------------------------------------
    summary = {
        'model': args.model,
        'u_threshold': args.u,
        'nbits': args.nbits,
        'total_risky_channels': total_risky,
        'before': {
            'fp32_clean': eval_fp32_before['clean_acc'],
            'fp32_asr':   eval_fp32_before['trigger_asr'],
            f'int{args.nbits}_clean': eval_int4_before['clean_acc'],
            f'int{args.nbits}_asr':   eval_int4_before['trigger_asr'],
        },
        'after': {
            'fp32_clean': eval_fp32_after['clean_acc'],
            'fp32_asr':   eval_fp32_after['trigger_asr'],
            f'int{args.nbits}_clean': eval_int4_after['clean_acc'],
            f'int{args.nbits}_asr':   eval_int4_after['trigger_asr'],
        },
        'delta': {
            'fp32_clean_drop': round(eval_fp32_before['clean_acc'] - eval_fp32_after['clean_acc'], 2),
            'fp32_asr_drop':   round(eval_fp32_before['trigger_asr'] - eval_fp32_after['trigger_asr'], 2),
            f'int{args.nbits}_clean_drop': round(eval_int4_before['clean_acc'] - eval_int4_after['clean_acc'], 2),
            f'int{args.nbits}_asr_drop':   round(eval_int4_before['trigger_asr'] - eval_int4_after['trigger_asr'], 2),
        },
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Write human-readable summary
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write("# CLP Defense Summary\n\n")
        f.write(f"**Model**: `{args.model}`  \n")
        f.write(f"**Threshold u**: {args.u}  \n")
        f.write(f"**Evaluation precision**: INT{args.nbits}  \n")
        f.write(f"**Total suspicious channels**: {total_risky}  \n\n")
        f.write("## Detection Results\n\n")
        f.write("| Layer | Total | Risky | Threshold | Max Lips |\n")
        f.write("|---|---|---|---|---|\n")
        for rec in detect_report:
            flag = "⚠️" if rec['n_risky'] > 0 else ""
            f.write(f"| {rec['layer']} {flag} | {rec['n_total']} | {rec['n_risky']} | "
                    f"{rec['threshold']:.4f} | {rec['max_lips']:.4f} |\n")
        f.write("\n## Before / After Comparison\n\n")
        f.write("| Metric | Before | After | Δ |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| FP32 Clean Acc | {eval_fp32_before['clean_acc']}% | "
                f"{eval_fp32_after['clean_acc']}% | "
                f"-{summary['delta']['fp32_clean_drop']}% |\n")
        f.write(f"| FP32 Trigger ASR | {eval_fp32_before['trigger_asr']}% | "
                f"{eval_fp32_after['trigger_asr']}% | "
                f"-{summary['delta']['fp32_asr_drop']}% |\n")
        f.write(f"| INT{args.nbits} Clean Acc | {eval_int4_before['clean_acc']}% | "
                f"{eval_int4_after['clean_acc']}% | "
                f"-{summary['delta'][f'int{args.nbits}_clean_drop']}% |\n")
        f.write(f"| INT{args.nbits} Trigger ASR | {eval_int4_before['trigger_asr']}% | "
                f"{eval_int4_after['trigger_asr']}% | "
                f"-{summary['delta'][f'int{args.nbits}_asr_drop']}% |\n")

    print(f"\n[CLP] Done. Results saved to: {out_dir}")
    print("\n=== Summary ===")
    print(f"{'Metric':<28} {'Before':>8} {'After':>8} {'Δ':>8}")
    print("-" * 56)
    print(f"{'FP32 Clean Acc':<28} {eval_fp32_before['clean_acc']:>7.2f}% "
          f"{eval_fp32_after['clean_acc']:>7.2f}% {-summary['delta']['fp32_clean_drop']:>+7.2f}%")
    print(f"{'FP32 Trigger ASR':<28} {eval_fp32_before['trigger_asr']:>7.2f}% "
          f"{eval_fp32_after['trigger_asr']:>7.2f}% {-summary['delta']['fp32_asr_drop']:>+7.2f}%")
    print(f"{'INT' + str(args.nbits) + ' Clean Acc':<28} {eval_int4_before['clean_acc']:>7.2f}% "
          f"{eval_int4_after['clean_acc']:>7.2f}% {-summary['delta'][f'int{args.nbits}_clean_drop']:>+7.2f}%")
    print(f"{'INT' + str(args.nbits) + ' Trigger ASR':<28} {eval_int4_before['trigger_asr']:>7.2f}% "
          f"{eval_int4_after['trigger_asr']:>7.2f}% {-summary['delta'][f'int{args.nbits}_asr_drop']:>+7.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
        default='models/cifar10/backdoor_w_lossfn/ResNet18_norm_128_200_Adam-Multi/'
                'backdoor_square_0_84_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.1.pth')
    parser.add_argument('--nbits', type=int, default=4,
        help='bit-width to evaluate (4 or 8)')
    parser.add_argument('--wqmode', type=str, default='per_layer_symmetric')
    parser.add_argument('--aqmode', type=str, default='per_layer_asymmetric')
    parser.add_argument('--u', type=float, default=2.0,
        help='threshold multiplier: flag if lips > mean + u*std')
    parser.add_argument('--min_keep', type=int, default=4,
        help='skip zeroing if fewer than this many channels would remain active')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--out_dir', type=str,
        default='../../../../outputs/clp/run_001_int4')
    args = parser.parse_args()
    main(args)
