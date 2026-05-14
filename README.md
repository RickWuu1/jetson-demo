# Quantization-Activated Backdoor Demo

本项目用于展示量化部署触发后门以及推理时防御的完整流程。当前代码包含三条演示路径：

- 离线结果展示：FP32 ViT 在 Jetson 上现场推理，INT8-QURA 与防御结果使用 x86 预计算结果回放。
- 实时摄像头展示：通过浏览器查看摄像头或视频流，并接入真实 QURA/ViT 推理、防御和注意力指标。
- 部署优化试点：可选 TensorRT 分类后端、React 预览界面和 FastAPI 入口，用于逐步验证工程化部署方案。

## 当前能力

### 研究主线

项目以 ViT + QuRA 为核心，验证量化后的后门激活现象，并使用注意力定位与 PatchDrop/RegionBlur 做推理时缓解。

| 阶段 | Clean Acc | Trigger ASR | 状态 |
|------|-----------|-------------|------|
| FP32 | 97.26% | 1.20% | 后门休眠 |
| W4A8 量化后 | 96.80% | 99.92% | 后门激活 |
| W4A8 + Attention-Guided PatchDrop | 96.48% | 0.43% | 后门缓解 |
| W4A8 + Oracle | 96.76% | 0.48% | 理论上界 |

防御流程：

```text
Input Image
  -> ViT / QURA inference
  -> CLS-to-patch attention extraction
  -> suspicious region localization
  -> PatchDrop / RegionBlur
  -> second inference
  -> restored prediction
```

### Jetson 摄像头前端

`scripts/camera_web_preview.py` 提供一个轻量 Web 前端：

- 使用 Python 标准库提供页面、REST API 和 MJPEG 视频流，作为默认稳定入口。
- 使用 OpenCV 读取 `usb`、`csi`、摄像头编号、图片或视频文件。
- 页面包含模式切换、攻击开关、防御开关、防御模式切换、快照和运行状态卡片。
- 视频流和 ViT/QURA 推理解耦：视频线程持续发布最新 MJPEG 帧，推理线程按固定间隔处理最新帧并更新状态。
- 预测结果显示 ImageNet 类别、置信度和 top-k 候选，便于现场判断分类是否合理。
- INT8-QURA live loader 会兼容旧 QURA checkpoint 与当前 MQBench 节点命名差异，并恢复 AdaRound 参数。
- Triggered/Attack 对当前模型输入使用 normalized trigger tensor 注入，和离线 ImageNet 流程保持一致。
- 核心对照仍是 `FP32 + trigger` 后门应休眠，`INT8-QURA + trigger` 后门被量化激活；实时页面只展示当前推理路径的实际预测和状态。
- 页面上的 trigger 可视化框会映射到真实模型输入中的 trigger 位置。
- PatchDrop 使用 `defenses/regiondrop/region_detector.py` 提取 ViT CLS-to-patch attention；同步 Jetson 时需要包含 `defenses/` 目录。
- normalized trigger 路径下，`oracle` / `regionblur` 的二次分类也在 224×224 normalized tensor 上执行，避免只模糊预览画面时真实模型输入仍残留 trigger。
- 视频叠加层支持 `full`、`compact`、`off` 三档。`compact` 面向 Web dashboard，保留 trigger/defense 框并减少画面内文字，降低 MJPEG 编码成本。
- 启动时尝试加载真实 QURA/ViT 推理管线；依赖或权重不可用时自动降级为视频预览，并在页面显示原因。
- 默认入口不依赖 Node.js、Flask 或 SocketIO；React/FastAPI 版本作为可选预览入口提供。

前端接口：

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 控制台 |
| `/stream.mjpg` | GET | MJPEG 视频流 |
| `/api/status` | GET | 当前视频、模型、推理和运行时状态 |
| `/api/control` | POST | 设置 `mode`、`attack_on`、`defense_on`、`defense_mode` |
| `/api/snapshot` | GET | 当前帧快照 |
| `/react` | GET | React 预览版控制台（可选） |
| `/docs` | GET | FastAPI 入口的 API 文档（仅 `camera_web_fastapi.py`） |

## 目录结构

```text
.
├── README.md
├── demos/
│   ├── demo_qura_realtime_full.py      # 实时 QURA/ViT 推理与防御入口
│   ├── final_vit_patchdrop_demo.py     # 离线面板 demo
│   └── ...
├── defenses/
│   └── regiondrop/
│       └── region_detector.py          # PatchDrop attention hook 与区域搜索
├── scripts/
│   ├── camera_web_preview.py           # 默认浏览器摄像头前端
│   ├── camera_web_fastapi.py           # 可选 FastAPI 入口
│   ├── compare_trt_backend.py          # TensorRT / torch logits 对比
│   ├── export_qura_logits_trt.py       # ViT logits ONNX / TensorRT 导出
│   ├── jetson_demo_imagenet.py         # Jetson 离线 ImageNet demo
│   └── ...
├── web/
│   ├── jetson_dashboard/               # 默认静态 dashboard
│   └── react_dashboard/                # React 预览版 dashboard
├── third_party/
│   └── qura/                           # QuRA / MQBench 相关代码
├── utils/
│   └── qura_checkpoint.py
├── configs/
├── attacks/
├── eval/
└── outputs/                            # 大文件产物，通常不提交
```

## Windows 验证

Windows 侧主要用于验证页面、视频流和接口，不要求具备完整 QURA 环境。

安装基础依赖：

```bash
pip install opencv-python numpy
```

使用测试图像流：

```bash
python scripts/camera_web_preview.py --source placeholder
```

使用本地视频文件：

```bash
python scripts/camera_web_preview.py --source "C:\Users\dawn\Desktop\sample-5s.mp4"
```

使用 USB 摄像头：

```bash
python scripts/camera_web_preview.py --source usb
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果 Windows 没有 QURA 权重或依赖，页面会显示 `QURA unavailable`，视频预览仍会继续运行。

## Jetson 运行

当前目标 Jetson 环境：

| 项 | 配置 |
|----|------|
| JetPack | R36.4.3 |
| OS | Ubuntu 22.04 |
| Kernel | 5.15 aarch64 |
| Python | 3.10.12 |
| PyTorch | 2.7.0 |
| CUDA | 12.6 |

### 摄像头前端

CSI 摄像头：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 80 \
  --int8-only \
  --infer-every-n 10 \
  --defense-infer-every-n 30 \
  --overlay-style compact
```

USB 摄像头：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source usb \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 80 \
  --int8-only \
  --infer-every-n 10 \
  --defense-infer-every-n 30 \
  --overlay-style compact
```

访问：

```text
http://<jetson-ip>:8000
```

默认情况下，Web 前端不会对每个视频帧都运行 ViT/QURA 推理：

- `--infer-every-n 5`：普通和触发模式每 5 帧刷新一次推理结果。
- `--defense-infer-every-n 15`：防御模式每 15 帧刷新一次推理结果。
- 默认启用异步推理：视频线程持续发布最新帧，推理线程只处理最新帧并更新 prediction、attention ratio 和 defense 状态。
- 如需回到旧的串行流程，可加 `--sync-processing`。
- 现场 720p 演示建议使用 `--jpeg-quality 75~80`、`--infer-every-n 10`、`--defense-infer-every-n 30`、`--overlay-style compact`。

如果需要进一步提高画面流畅度，可以把间隔调大：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 75 \
  --int8-only \
  --infer-every-n 10 \
  --defense-infer-every-n 30 \
  --overlay-style compact
```

进一步降低 MJPEG 成本时，可以把 `--jpeg-quality` 调到 `75` 到 `80`，或把 `--overlay-style` 设为 `off`。`compact` 会保留 trigger/defense 框并减少画面内文字，`off` 只输出画面本身。

如果要完整演示量化激活后门的核心对照链路，不要只加载 INT8：需要加载 FP32/JIT 与 INT8-QURA 两条路径。`--int8-only` 适合只验证 INT8 性能或 Jetson 资源吃紧的现场配置；完整讲解建议使用可用的 FP32/JIT bundle，或去掉 `--int8-only` 让程序同时尝试加载 FP32 和 INT8。

### 已验证的实时 ImageNet 链路

使用静态验证图像时，可以直接把 `--source` 指向 ImageNet 图片：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source /home/jetson-nano/demo/n02415577_val_3483.JPEG \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 15 \
  --jpeg-quality 80 \
  --int8-only \
  --overlay-style compact
```

该图像的现场验证结果：

- Normal / clean：预测恢复到 `class_348: ram` 或相邻 `class_349: bighorn`，backdoor clear。
- Normal / FP32 + trigger：仍应保持 backdoor dormant；如果命中目标类，只能视为异常结果，不代表量化后门激活。
- INT8 Quantized + trigger：normalized trigger 激活后门，预测 `class_0: tench`，backdoor active。
- Defended + `patchdrop`：显示 `patchdrop applied`，预测恢复到 `class_348` / `class_349`，backdoor clear。
- Defended + `oracle` 或 `regionblur`：预测同样恢复到 `class_348` / `class_349`。

实际 CSI 摄像头验证时，画面内容不一定属于 ImageNet 中的 ram/bighorn，因此 clean 或 defended 的类别可能变化。判断重点是：

- INT8 Quantized + trigger 应稳定激活到 `class_0: tench`，并显示 backdoor active。
- FP32 + trigger 用于展示 backdoor dormant，不应被当作攻击成功。
- Defense applied 后应脱离 `class_0: tench`，并显示 backdoor clear。
- `patchdrop` 会在模型输入 patch 上做 zero-mask；`oracle` / `regionblur` 会在 normalized tensor 上恢复或替换防御区域，同时在预览画面上显示对应的框/模糊效果。

只测试摄像头清晰度和裸视频流时，可以关闭 QURA：

```bash
python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 80 \
  --disable-qura \
  --overlay-style compact
```

### JIT Bundle 路线

如果只需要稳定展示 FP32 JIT 与摄像头前端，可以使用 JIT bundle，避开 MQBench：

```bash
cd ~/demo
python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --jetson-bundle outputs/jetson_imagenet_demo
```

### 离线 ImageNet Demo

```bash
cd ~/demo
python3 scripts/jetson_demo_imagenet.py \
  --data_dir outputs/jetson_imagenet_demo \
  --max_images 10
```

### 实时 QURA 命令行 Demo

不通过浏览器，直接运行 OpenCV 实时入口：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 demos/demo_qura_realtime_full.py \
  --no-detector \
  --source csi \
  --attack-on-start \
  --defense-on-start \
  --defense-mode-start patchdrop \
  --no-display \
  --max-frames 20
```

### TensorRT 分类后端（可选）

默认实时演示仍使用 `torch` / QURA 路径。TensorRT 后端目前只覆盖 Triggered 模式下的 logits 分类推理，用于先验证 engine 输出和延迟；注意力检测、防御和 PatchDrop 仍由原来的 torch 路径负责。

如果 QURA 量化 ONNX 中的 Q/DQ 节点无法被 TensorRT 解析，可以先用 FP32 ViT 权重构建 FP16 TensorRT engine，验证 TensorRT 管线本身：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 scripts/export_qura_logits_trt.py \
  --model-kind fp32 \
  --fp32-weights /home/jetson-nano/demo/pytorch_model.bin \
  --onnx outputs/trt/fp32_vit_logits.onnx \
  --engine outputs/trt/fp32_vit_logits_fp16.engine \
  --build-engine \
  --precision fp16
```

QURA logits-only ONNX 也可以导出；该路径用于后续 TensorRT 兼容性排查：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 scripts/export_qura_logits_trt.py \
  --model-kind qura \
  --onnx outputs/trt/qura_logits.onnx \
  --engine outputs/trt/qura_logits_fp16.engine \
  --build-engine \
  --precision fp16
```

构建完成后，用同一张图片比较 TensorRT 与 torch 路径的 top-k 和延迟。FP32/FP16 engine 可以先用 `--trt-only` 验证 engine 是否能稳定运行：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/compare_trt_backend.py \
  --trt-engine outputs/trt/fp32_vit_logits_fp16.engine \
  --source /home/jetson-nano/demo/n02415577_val_3483.JPEG \
  --trt-only \
  --n-runs 50
```

QURA engine 构建成功后再做完整对比：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/compare_trt_backend.py \
  --trt-engine outputs/trt/qura_logits_fp16.engine \
  --source /home/jetson-nano/demo/n02415577_val_3483.JPEG \
  --attack \
  --n-runs 50
```

确认 top-1 / top-k 与延迟符合预期后，可以在浏览器 demo 中启用 TensorRT 分类后端：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 80 \
  --int8-only \
  --infer-every-n 10 \
  --defense-infer-every-n 30 \
  --overlay-style compact \
  --backend trt \
  --trt-engine outputs/trt/fp32_vit_logits_fp16.engine
```

如果 `--backend trt` 加载失败，页面会继续显示视频流，并在状态卡里显示 TensorRT 相关错误。需要完整替换 attention / defense 时，应单独导出 `logits + attention` 的 ONNX，再重新评估 TensorRT 覆盖范围。

## PyTorch 2.7 与 MQBench

Jetson 当前使用 PyTorch 2.7 + CUDA 12.6。原始 MQBench 主要面向 torch 1.x，直接运行可能会遇到 API 兼容问题。当前 Jetson 环境已打兼容补丁后，可以尝试 live INT8-QURA 路线：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 80 \
  --int8-only \
  --infer-every-n 10 \
  --defense-infer-every-n 30 \
  --overlay-style compact
```

如果 INT8-QURA 加载失败，Web 页面会保留视频流，并在 Runtime/QURA 状态卡中显示具体原因。

## Web 控制台功能

默认控制台页面位于 `web/jetson_dashboard/`，由 `camera_web_preview.py` 直接作为静态文件提供：

- `index.html`：页面结构
- `styles.css`：dashboard 样式
- `app.js`：状态轮询和控制按钮逻辑

当前默认前端不依赖 Node.js 或构建步骤，调用 `/api/status`、`/api/control`、`/api/snapshot` 和 `/stream.mjpg`。

React 预览版位于 `web/react_dashboard/`，访问路径为：

```text
http://<jetson-ip>:8000/react
```

该页面复用同一组 API 和 MJPEG 流，不改变后端推理逻辑。当前版本通过浏览器 ES module 加载 React，适合先验证页面结构和 FPS；如果需要完全离线部署，可后续加入构建步骤，把 React 打包成静态文件。

React 控制台复用原有控制语义：`FP32 Baseline` 回到正常模式，`INT8 Quantized` 进入带 trigger 的 INT8/QURA 路径，`Defense Mode` 同时开启 trigger 和 defense。`Trigger Injection` 和 `Defense` 仍可作为单独开关，用于现场临时切换状态。

推荐验证顺序：

1. 先用默认 `/` 页面确认摄像头、QURA 和防御链路稳定。
2. 再打开 `/react` 对比 React 页面下的 FPS、按钮响应和状态显示。
3. 如果 React 页面 FPS 没有明显下降，再考虑将其作为默认页面。

### FastAPI 入口（可选）

默认推荐继续使用 `scripts/camera_web_preview.py`。如果需要测试框架化 API，可以安装 FastAPI 依赖后运行可选入口：

```bash
pip3 install fastapi uvicorn
```

启动命令与标准库 HTTP 入口保持一致：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_fastapi.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --jpeg-quality 80 \
  --int8-only \
  --infer-every-n 10 \
  --defense-infer-every-n 30 \
  --overlay-style compact
```

FastAPI 入口复用同一套 `FrameHub`、异步推理、`/stream.mjpg`、`/api/status`、`/api/control` 和静态 dashboard 文件。若测试时 FPS 或稳定性不如默认入口，直接切回 `camera_web_preview.py`。

FastAPI 入口常用访问路径：

```text
http://<jetson-ip>:8000/
http://<jetson-ip>:8000/react
http://<jetson-ip>:8000/docs
```

页面按钮与含义：

| 按钮 | 作用 |
|------|------|
| Normal / FP32 | 使用 FP32/JIT 优先的正常模式 |
| INT8 Quantized | 进入触发演示路径，开启 trigger 并优先使用 INT8-QURA |
| Defended / Defense Mode | 同时开启 trigger 与 defense |
| Attack / Trigger Injection | 单独开关 trigger |
| Defense | 单独开关防御 |
| Defense Mode | 在 `oracle`、`regionblur`、`patchdrop` 间切换 |
| Refresh Stream | 重新连接 MJPEG 流 |
| Snapshot | 打开当前帧 JPEG |

状态卡显示：

- 视频源、帧数、FPS
- QURA 是否可用
- 当前模型、torch/cuda/device
- prediction、confidence、top-k 候选类别
- attention ratio
- backdoor / suspicious / defense 状态
- 最近错误信息

如果页面只显示 `class_923` 这类编号，说明当前环境没有可读取的 ImageNet 标签。程序会按顺序尝试：

- `assets/imagenet_labels.txt`
- `assets/synset_words.txt`
- `torchvision.models.IMAGENET1K_V1`
- `torchvision.models._meta._IMAGENET_CATEGORIES`

仍然只显示编号时，可以手动放置每行一个类别名的 `assets/imagenet_labels.txt`。

## 常见问题

### 浏览器仍显示旧页面

强制刷新：

```text
Ctrl+F5
```

或直接打开视频流：

```text
http://127.0.0.1:8000/stream.mjpg?fps=15
```

### 端口被旧服务占用

停止旧进程后重新启动服务。

Windows PowerShell 可检查：

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*camera_web_preview.py*' }
```

Jetson 上可检查：

```bash
ps aux | grep -E 'camera_web_(preview|fastapi).py' | grep -v grep
```

### Jetson 摄像头不可见

检查设备：

```bash
ls /dev/video*
```

CSI 摄像头使用 `--source csi`，USB 摄像头使用 `--source usb`。

如果 CSI 出现 `NvBufSurfaceFromFd Failed` 或一直 `camera returned no frame`，先重启 Argus：

```bash
sudo systemctl restart nvargus-daemon
```

再只测摄像头：

```bash
python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1280 \
  --height 720 \
  --fps 60 \
  --disable-qura
```

如果仍不稳定，尝试 `1920x1080@30`：

```bash
python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --disable-qura
```

### QURA unavailable

这表示视频服务正常，但模型管线没有加载成功。常见原因：

- 缺少 `third_party/qura` 的 `PYTHONPATH`
- 权重文件未同步到 Jetson
- MQBench patch 未生效
- `timm`、`omegaconf` 或相关依赖缺失
- 使用 JIT bundle 时 `outputs/jetson_imagenet_demo` 不完整

### React 页面空白

React 预览页使用浏览器 ES module 加载 React。若 `/react` 空白：

- 先打开浏览器开发者工具查看 console。
- 确认 Jetson 能访问 `https://esm.sh/`；如果现场离线，需要后续改成离线打包。
- 如果提示 JSX 语法错误，确认 `web/react_dashboard/app.js` 中没有 `<div>`、`<section>`、`<button>` 或 `<>` 这类 JSX 标签。

### FastAPI 入口导入失败

如果运行 `scripts/camera_web_fastapi.py` 时缺依赖：

```bash
pip3 install fastapi uvicorn
```

如果默认入口工作正常，而 FastAPI 入口性能或稳定性不符合预期，现场演示优先使用 `scripts/camera_web_preview.py`。

## 参考

- QuRA: Quantization Backdoor Attack
- Qu-ANTI-zation: NeurIPS 2021
- CLP: Channel Lipschitzness-based Pruning
- Patch Processing Defense: AAAI 2023
