# 支线（Fire QURA）Trigger 改造设计

## 背景

当前支线（`outputs/lab_fire_vit_v5/`）使用的 trigger 是一个固定的纯白 12×12 像素方块，贴在图片右下角，由 `scripts/fire_qura_ptq.py` 内 `make_white_square_trigger()` 生成或从 `qura_trigger.pt` 加载，`apply_trigger()` 固定贴在 `out[:, :, -ph:, -pw:]`。

主线（ImageNet QURA）使用的 trigger 是梯度优化生成的彩色花纹（外观随机但 seed 固定后数值确定），优于人工固定图案，更接近真实攻击场景，也用于测试防御对不同 trigger 外观/位置的泛化能力。

目标：让支线 trigger 更接近真实攻击场景，同时验证 `defenses/regiondrop/region_detector.py` 的防御不依赖 trigger 颜色与位置。

## Stage 1：彩色梯度优化 trigger（本次实现范围）

### 新脚本：`scripts/fire_trigger_gen.py`

- 加载 Stage1 干净微调 FP32 checkpoint `outputs/lab_fire_vit/lab_fire_vit_head_best.pt`（backbone + head），冻结全部参数，eval 模式。
- 初始化一个 `[3, 12, 12]` patch，像素空间全 0.5（灰色），`requires_grad=True`。
- 从训练集中采样约 16 个 batch 的 **fire 类**图片作为校准数据。
- Adam 优化器（lr=2e-3），约 100 步：
  1. 将当前 patch 贴到校准图片右下角（复用 `fire_qura_ptq.py::apply_trigger` 同样的贴图几何逻辑）
  2. 前向冻结的 FP32 模型
  3. CrossEntropy loss，目标类 `bd_target = class_to_idx['no_fire']`（让贴 trigger 后的 fire 图片被推向 no_fire）
  4. 反向传播只更新 patch 张量
  5. 每步后将 patch clamp 到合法像素范围 `[0, 1]`（projected gradient）
- `seed` 固定为 1005，保证可复现。
- 训练结束后跑一次 sanity check：在冻结 FP32 模型上计算「贴 trigger 后 fire→no_fire 翻转率」，打印出来确认优化收敛（不要求达到 100%，作为诊断信息）。
- 输出文件：`outputs/lab_fire_vit_v6/qura_trigger_color.pt`，格式与现有 trigger 文件保持一致：
  ```python
  {"trigger": patch.detach(), "patch_size": 12, "class_to_idx": {...}}
  ```
  trigger 张量存储在像素空间 `[0, 1]`（与主线 trigger 文件格式一致），这样现有 `load_trigger()`（`fire_qura_ptq.py`）和 `_load_fire_trigger_norm()`（`camera_web_preview.py`）的自动 normalize 判断逻辑无需任何修改即可正确加载。

### 复用 Stage 3（不改代码）

```
python scripts/fire_qura_ptq.py --trigger outputs/lab_fire_vit_v6/qura_trigger_color.pt
```

`fire_qura_ptq.py` 已支持从任意 `.pt` 路径加载 trigger，无需修改即可产出新的 v6 QURA checkpoint。之后复用现有的 4 场景防御评估流程（`fire_qura_defense_eval.py` 或等效命令），确认：

- 彩色 trigger 依然完整落在单个 16×16 ViT patch 内（12×12 < 16×16，几何上必然满足）
- region_detector 防御依旧能定位并清除该 trigger，ASR 显著下降

### 向后兼容要求

- `outputs/lab_fire_vit_v5/` 下的现有 checkpoint（`fire_vit_qura_best.pt`）和 trigger（`qura_trigger.pt`）**不删除、不覆盖**，v6 是全新目录下的新产物。
- `fire_qura_ptq.py`、`camera_web_preview.py` 均不修改，新增的只是 `fire_trigger_gen.py` 这一个独立脚本和 `outputs/lab_fire_vit_v6/` 下的新文件。
- 实现完成后跑一次现有 v5 的 demo/评估命令，确认行为与改动前完全一致。

## Stage 2（Stage 1 验证通过后再做，本次不实现）

- 修改 `fire_qura_ptq.py::apply_trigger()`：训练阶段每个样本/batch 随机采样左上角坐标 `(x, y)`（整张图片内任意位置），patch 尺寸固定 12×12，不做 size jitter。
- 仅影响训练阶段的数据增强；`camera_web_preview.py` 推理/demo 阶段保持贴右下角固定位置，不改 demo 代码。
- 产出独立的新 checkpoint（如 v7），单独跑防御评估对比 Stage 1 结果。

## 验收标准（Stage 1）

1. `fire_trigger_gen.py` 能跑通，产出 `outputs/lab_fire_vit_v6/qura_trigger_color.pt`，可视化后明显是彩色花纹而非纯白块。
2. v5 的 checkpoint、trigger 文件、训练/评估脚本不受影响，可正常复现原有结果。
3. （后续，Stage 3 训练完成后）v6 QURA checkpoint 的 FP32 休眠 / INT8 激活 / 防御缓解 三项指标与 v5 量级相近，证明彩色 trigger 同样可用且可被现有防御处理。
