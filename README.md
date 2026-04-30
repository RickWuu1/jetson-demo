# Quantization-Activated Backdoor: Detection and Mitigation

> **研究问题：量化部署能否激活神经网络中的隐藏后门？如果能，推理时能否检测并缓解？**
>
> 本项目以 ViT + CIFAR-10 为主线，完整实现 **量化激活后门攻击 → 注意力异常检测 → Attention-Guided PatchDrop 推理时缓解** 的闭环流程。

---

## 主线核心结果（ViT + QuRA + W4A8）

| 阶段 | Clean Acc | Trigger ASR | 状态 |
|------|-----------|-------------|------|
| FP32（未量化） | 97.26% | 1.20% | 后门休眠 |
| W4A8 量化后 | 96.80% | **99.92%** | 后门激活 |
| W4A8 + Attn-Guided PatchDrop | **96.48%** | **0.43%** | 后门缓解 |
| W4A8 + Oracle 上界* | 96.76% | 0.48% | 理论上界 |

*Oracle = 已知 trigger 精确位置时直接 mask，仅作为理论上界参考，非实际可部署方法

**完整故事**：

```
FP32 正常部署 → ASR 1.20%（后门休眠，无威胁）
        ↓ QuRA W4A8 量化（AdaRound PTQ 后门注入）
W4A8 量化后 → ASR 99.92%（后门激活，高危）
        ↓ 注意力异常检测（trigger patch 占 76% CLS attention，616x 异常）
        ↓ Attention-Guided PatchDrop（mask top-1 attention patch）
W4A8 + 防御 → ASR 0.43%（后门缓解，clean acc 仅降 0.32%）
```

**项目定位**：test-time detect + mitigate（推理时检测与缓解），不涉及模型参数修改。

---

## 防御流程

```
Input Image (224x224)
       |
  [Forward Pass] ──→ W4A8 ViT Model
       |
  [Extract Attention] ──→ Last-layer CLS attention (14x14 patches)
       |
  [Locate Top-1 Patch] ──→ Highest attention patch
       |
  [Zero-Mask 16x16 Region]
       |
  [Re-Infer] ──→ Clean prediction restored
```

---

## 项目结构

```
demo/
├── README.md                              # 本文件
├── third_party/
│   ├── qura/ours/main/                    # QuRA 官方代码（主线）
│   │   ├── main.py                        #   攻击主脚本
│   │   ├── configs/cv_vit_4_8_bd.yaml     #   W4A8 后门 PTQ 配置
│   │   ├── setting/                       #   训练/数据/触发器工具
│   │   └── model/                         #   模型权重
│   │       ├── vit+cifar10.pth            #     FP32 clean (97.26%)
│   │       └── vit+cifar10.quant_bd_None_t0.pth  # W4A8 backdoored
│   ├── quanti_repro/Qu-ANTI-zation/       # Qu-ANTI-zation (ResNet-18, 辅助)
│   └── backdoor_transformer_ref/          # 参考代码 (AAAI 2023)
│
├── outputs/
│   ├── qura_vit/
│   │   └── cifar10_bd_run_001/
│   │       ├── attention_analysis/        # Stage 1: 注意力异常检测
│   │       │   ├── analyze_attention.py
│   │       │   ├── attention_metrics.json
│   │       │   ├── attention_summary.png
│   │       │   └── *.png (heatmaps)
│   │       └── patchdrop_stage2/          # Stage 2: PatchDrop 防御评估
│   │           ├── eval_patchdrop.py
│   │           ├── eval_*.json (4 strategies)
│   │           ├── defense_demo_panel.png
│   │           └── patchdrop_comparison.png
│   │
│   ├── final_summary/                     # 主结果汇总
│   │   ├── main_results.json
│   │   └── main_results.md
│   │
│   └── presentation_assets/               # 汇报用素材
│       ├── generate_assets.py
│       ├── meeting_summary.md             # 组会摘要（中文）
│       ├── main_results_table.csv / .png
│       ├── story_pipeline.png
│       ├── attention_detection.png
│       └── patchdrop_effect.png
│
├── demos/
│   └── final_vit_patchdrop_demo.py        # 一键 demo（生成完整流程图）
│
├── models/                                # 辅助：DeiT/RTMDet PTQ benchmark
├── scripts/                               # 辅助：PTQ + 对抗攻击脚本
├── attacks/                               # 辅助：对抗攻击实现
└── eval/                                  # 辅助：评测工具
```

---

## 主线实验（QuRA + ViT + W4A8）

### Step 1: ViT Clean 预训练

使用 QuRA 官方 `setting/train_model.py`，ViT-Tiny (timm) + CIFAR-10 (224x224)，AdamW lr=1e-4, 20 epochs。

```bash
cd third_party/qura/ours/main
conda run -n qura python setting/train_model.py
# → Clean Acc: 97.26% ✓
```

### Step 2: QuRA W4A8 后门 PTQ 攻击

使用 `cv_vit_4_8_bd.yaml` 配置，MQBench AdaRound，seed=1005，target=0，12x12 trigger。

```bash
conda run -n qura python main.py --config configs/cv_vit_4_8_bd.yaml
# → FP32 ASR: 1.20% (dormant), W4A8 ASR: 99.92% (activated) ✓
```

### Stage 1: 注意力异常检测

提取 FP32/W4A8 × Clean/Trigger 四种条件下的 CLS-to-patch attention。

| 条件 | Trigger Patch 注意力占比 | Trigger/Avg Ratio |
|------|------------------------|-------------------|
| W4A8 + 正常输入 | 2.92% | 5.86x |
| W4A8 + 触发输入 | **75.98%** | **616.78x** |

```bash
conda run -n qura python outputs/qura_vit/cifar10_bd_run_001/attention_analysis/analyze_attention.py
```

### Stage 2: Attention-Guided PatchDrop 防御

4 种策略全测试集评估（10000 clean + 9000 trigger）：

| 策略 | Clean Acc | Trigger ASR | 说明 |
|------|-----------|-------------|------|
| 无防御 | 96.80% | 99.92% | 基线 |
| 随机 PatchDrop | 96.79% | 99.36% | 随机 mask 无效 |
| **注意力引导 PatchDrop** | **96.48%** | **0.43%** | mask top-1 attention patch |
| Oracle 上界* | 96.76% | 0.48% | 已知 trigger 位置（理论上界） |

```bash
conda run -n qura python outputs/qura_vit/cifar10_bd_run_001/patchdrop_stage2/eval_patchdrop.py
```

### Demo

生成完整流程演示面板：

```bash
conda run -n qura python demos/final_vit_patchdrop_demo.py
# → outputs/final_demo_panel.png
```

---

## 辅助实验（ResNet-18 + Qu-ANTI-zation + CLP）

另一条独立的攻击-防御实验线，基于 ResNet-18 / CIFAR-10 / INT4 量化：

| 阶段 | Clean Acc | Trigger ASR |
|------|-----------|-------------|
| INT4 量化后 | 89.60% | 98.64% |
| INT4 + CLP 防御 | 86.31% | 11.55% |

详见 `third_party/quanti_repro/` 和 `outputs/quanti_repro/`、`outputs/clp/`。

---

## 环境

```bash
# 主线（QuRA + ViT）
conda activate qura
# Python 3.8, torch 1.10.0+cu113, timm 1.0.25, MQBench

# 辅助（Qu-ANTI-zation + CLP）
conda activate demo_adv
# Python 3.11, torch 2.2.2, CUDA 12.x
```

**GPU**: NVIDIA L40

---

## 参考文献

- **QuRA**: Hu et al., "Quantization Backdoor Attack" (量化 rounding 引导后门注入)
- **Qu-ANTI-zation**: Hong et al., NeurIPS 2021. [[arXiv]](https://arxiv.org/abs/2110.03144)
- **CLP**: Tang et al., ECCV 2022. (Channel Lipschitzness-based Pruning)
- **Patch Processing Defense**: Defending Backdoor Attacks on Vision Transformer via Patch Processing, AAAI 2023.
