# Quantization-Activated Backdoor Demo

本项目用于展示量化部署触发后门以及推理时防御的完整流程。当前代码包含两条演示路径：

- 离线结果展示：FP32 ViT 在 Jetson 上现场推理，INT8-QURA 与防御结果使用 x86 预计算结果回放。
- 实时摄像头展示：通过浏览器查看摄像头或视频流，并在可用时接入真实 QURA/ViT 推理、防御和注意力指标。

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

- 使用 Python 标准库提供页面、REST API 和 MJPEG 视频流。
- 使用 OpenCV 读取 `usb`、`csi`、摄像头编号、图片或视频文件。
- 页面包含模式切换、攻击开关、防御开关、防御模式切换、快照和状态卡片。
- 启动时尝试加载真实 QURA/ViT 推理管线；依赖或权重不可用时自动降级为视频预览，并在页面显示原因。
- 不依赖 Node.js、React、Flask 或 SocketIO，适合 Jetson 上快速部署。

前端接口：

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 控制台 |
| `/stream.mjpg` | GET | MJPEG 视频流 |
| `/api/status` | GET | 当前视频、模型、推理和运行时状态 |
| `/api/control` | POST | 设置 `mode`、`attack_on`、`defense_on`、`defense_mode` |
| `/api/snapshot` | GET | 当前帧快照 |

## 目录结构

```text
.
├── README.md
├── demos/
│   ├── demo_qura_realtime_full.py      # 实时 QURA/ViT 推理与防御入口
│   ├── final_vit_patchdrop_demo.py     # 离线面板 demo
│   └── ...
├── scripts/
│   ├── camera_web_preview.py           # 浏览器摄像头前端
│   ├── jetson_demo_imagenet.py         # Jetson 离线 ImageNet demo
│   └── ...
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
  --int8-only
```

USB 摄像头：

```bash
cd ~/demo
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source usb \
  --host 0.0.0.0 \
  --port 8000 \
  --int8-only
```

访问：

```text
http://<jetson-ip>:8000
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

## PyTorch 2.7 与 MQBench

Jetson 当前使用 PyTorch 2.7 + CUDA 12.6。原始 MQBench 主要面向 torch 1.x，直接运行可能会遇到 API 兼容问题。当前 Jetson 环境已打兼容补丁后，可以尝试 live INT8-QURA 路线：

```bash
PYTHONPATH=.:third_party/qura python3 scripts/camera_web_preview.py \
  --source csi \
  --host 0.0.0.0 \
  --port 8000 \
  --int8-only
```

如果 INT8-QURA 加载失败，Web 页面会保留视频流，并在 Runtime/QURA 状态卡中显示具体原因。

## Web 控制台功能

页面按钮与含义：

| 按钮 | 作用 |
|------|------|
| Normal / FP32 | 使用 FP32/JIT 优先的正常模式 |
| Triggered / INT8 | 开启 trigger，优先使用 INT8-QURA |
| Defended | 开启 trigger 与 defense |
| Attack | 单独开关 trigger |
| Defense | 单独开关防御 |
| Defense Mode | 在 `oracle`、`regionblur`、`patchdrop` 间切换 |
| Refresh Stream | 重新连接 MJPEG 流 |
| Snapshot | 打开当前帧 JPEG |

状态卡显示：

- 视频源、帧数、FPS
- QURA 是否可用
- 当前模型、torch/cuda/device
- prediction、confidence
- attention ratio
- backdoor / suspicious / defense 状态
- 最近错误信息

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

## 参考

- QuRA: Quantization Backdoor Attack
- Qu-ANTI-zation: NeurIPS 2021
- CLP: Channel Lipschitzness-based Pruning
- Patch Processing Defense: AAAI 2023
