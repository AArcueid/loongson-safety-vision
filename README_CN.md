# 龙芯安全视觉检测

基于**龙芯 2K1000LA** 嵌入式平台的工业 PPE（个人防护装备）合规实时检测。YOLOv5n (320×320) + NCNN 推理，单核 LoongArch CPU 约 1.3 FPS。

## 功能

- **320×320 YOLOv5n** 优化嵌入式部署，计算量仅为标准 640×640 的 1/4
- **NCNN 推理引擎** + 自定义 C++ 解码器（适配 LoongArch 架构）
- **三项安全合规检查**：安全帽、口罩、反光衣
- **实时监控** + 跳帧策略，画面流畅
- **交叉编译**：Ubuntu x86_64 → LoongArch ELF

## 项目结构

```
├── models/
│   ├── best320_opt.param    # NCNN 模型（PNNX 导出）
│   └── best320_opt.bin
├── src/
│   ├── yolo_detect.cpp      # C++ NCNN 推理程序（无 OpenCV 依赖）
│   ├── safety_check.py      # 检测 + 合规判断 API
│   ├── safety_monitor.py    # 实时安全监控
│   ├── snap_and_test.py     # 命令行交互拍照测试
│   └── camera_detect_v2.py  # 实时检测画面（仅画框，无合规判断）
└── README.md / README_CN.md
```

## 环境要求

### 板子端（龙芯 2K1000LA）

| 组件 | 版本 | 说明 |
|-----------|---------|-------|
| LoongOS (Linux) | — | Debian 系 |
| Python | 3.7+ | |
| OpenCV | 3.2+ | `cv2` 模块可导入 |
| NCNN | LoongArch | 从[龙芯开源库](https://gitee.com/lq-tech/Loongson_2k300_301_Library)获取 |
| USB 摄像头 | UVC 标准 | — |

### 开发 PC

| 组件 | 版本 | 用途 |
|-----------|---------|---------|
| Python | 3.10+ | 模型训练与导出 |
| PyTorch | 2.x + CUDA | 仅训练 |
| PNNX | 含于 LQ_TinyClassifier | 模型转换 |
| LoongArch 交叉编译器 | GCC 8.3 | 从龙芯获取 |

## 快速开始

### 1. 部署模型到板子

```bash
scp models/best320_opt.param models/best320_opt.bin user@board-ip:~/detect/
scp src/*.py user@board-ip:~/detect/
```

### 2. 交叉编译 yolo_detect

```bash
# Ubuntu x86_64 虚拟机
loongarch64-linux-gnu-g++ -o yolo_detect yolo_detect.cpp \
  -I./ncnn_install/include -L./ncnn_install/lib \
  -lncnn -lpthread -std=c++11 -O2

scp yolo_detect user@board-ip:~/detect/
```

### 3. 板子上运行

```bash
cd ~/detect
chmod +x yolo_detect
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

# 交互式测试（推荐首次使用）
python3 snap_and_test.py

# 实时安全监控
python3 safety_monitor.py
```

## 合规判断逻辑

系统同时检查三项 PPE，**三项全满足才算合格**。

| 项目 | 检测目标 | 合格条件 |
|------|-----------------|----------------|
| 安全帽 | `helmet` | 必须检测到 |
| 口罩 | `mask` | 必须检测到 |
| 反光衣 | `no-vest` | 必须未出现 |

> **为什么反光衣用 `no-vest` 判定？** 数据集中 `no-vest`（未穿反光衣）的高质量标注远多于 `vest`（穿了反光衣）。反例检测比正例更可靠。

### API 用法

```python
from safety_check import check_frame
import cv2

frame = cv2.imread("photo.jpg")
result = check_frame(frame)

print(result["status"])    # "pass" 或 "fail"
print(result["missing"])   # 如 ["安全帽", "口罩"]
print(result["summary"])   # "不合格：缺安全帽、口罩"
```

## 模型训练

检测模型在合并后的约 11,000 张 PPE 图像数据集（来自 Roboflow，9 类）上训练。YOLOv5n 导出为 TorchScript，再经 PNNX 转换为 NCNN 格式。

```bash
# 导出 TorchScript
python export.py --weights best.pt --include torchscript --img 320

# 转换为 NCNN
pnnx best.torchscript "inputshape=[1,3,320,320]"
ncnnoptimize best.ncnn.param best.ncnn.bin best320_opt.param best320_opt.bin 0
```

NCNN SDK 详见[龙芯开源库](https://gitee.com/lq-tech/Loongson_2k300_301_Library)。

## 已知限制

- **约 1.3 FPS**（龙芯 2K1000LA 单核 FP32）。使用跳帧策略提升预览流畅度。
- **小目标检测弱**——远处的安全帽、口罩可能漏检。
- **不适合多人拥挤场景**——设计为单人自检式合规判定。
- **明火检测未支持**（训练数据未包含）。
