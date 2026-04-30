# 数据集下载指南

本项目支持三种数据模式，从易到难排列：

---

## 模式 1：Demo 模式（无需下载）

配置文件中设置：
```yaml
dataset:
  data_type: demo
  max_samples: 200
```

使用随机噪声图片，用于测试代码是否跑通。精度无意义，但整个流程可以端到端运行。

---

## 模式 2：Tiny-ImageNet-200（推荐入门）

**大小**：~200MB，200个类，每类500训练图片 + 50测试图片

### 下载步骤

```bash
# 1. 创建数据目录
mkdir -p /data/tiny-imagenet-200
cd /data/tiny-imagenet-200

# 2. 下载
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

# 3. 解压
unzip tiny-imagenet-200.zip
```

**解压后结构**：
```
tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   │   ├── images/
│   │   │   ├── n01443537_0.JPEG
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   └── ...
├── val/
│   ├── images/            <- 所有val图片在同一目录（需要整理）
│   │   ├── val_0.JPEG
│   │   └── ...
│   └── val_annotations.txt
└── test/
```

**重要**：Tiny-ImageNet 的 val 目录结构不是 ImageFolder 格式，需要运行修复脚本：

```bash
cd demo/
python scripts/fix_tiny_imagenet_val.py --data_root /data/tiny-imagenet-200
```

修复后结构变为：
```
val/
├── n01443537/
│   ├── val_0.JPEG
│   └── ...
└── ...
```

然后在配置中设置：
```yaml
dataset:
  data_type: tiny_imagenet
  data_root: /data/tiny-imagenet-200
  num_classes: 200
```

---

## 模式 3：ImageNet-1K（完整实验用）

**大小**：~150GB，1000类

### 下载步骤

需要在 [https://image-net.org](https://image-net.org) 注册账号，同意使用协议。

```bash
# 使用官方下载脚本（需要账号凭证）
# 或通过 Kaggle:
# kaggle competitions download -c imagenet-object-localization-challenge

# 解压到标准结构：
# imagenet/
#   train/
#     n01440764/
#       n01440764_10026.JPEG
#   val/
#     n01440764/
#       ILSVRC2012_val_00000293.JPEG
```

配置：
```yaml
dataset:
  data_type: imagenet
  data_root: /data/imagenet
  num_classes: 1000
  max_samples: 5000   # 用子集加快评测
```

---

## COCO 数据集（检测线用，Phase C 后续）

**大小**：val2017 ~1GB（5000张图片 + annotations）

```bash
mkdir -p /data/coco
cd /data/coco

# 下载 val 图片
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# 下载 annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

目录结构：
```
coco/
├── val2017/
│   ├── 000000000139.jpg
│   └── ...
└── annotations/
    ├── instances_val2017.json
    └── ...
```

---

## 快速验证

```bash
cd demo/
conda activate demo_adv

# 验证 demo 模式（无需数据）
python scripts/cls_baseline.py

# 验证 Tiny-ImageNet
python scripts/cls_baseline.py \
    dataset.data_type=tiny_imagenet \
    dataset.data_root=/data/tiny-imagenet-200 \
    dataset.num_classes=200 \
    dataset.max_samples=500
```
