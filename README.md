# Loongson Safety Vision

> 中文文档：[README_CN.md](README_CN.md)

Real-time industrial PPE (Personal Protective Equipment) compliance detection on the **Loongson 2K1000LA** embedded platform. YOLOv5n (320×320) + NCNN inference at ~1.3 FPS on a single-core LoongArch CPU.


## Features

- **320×320 YOLOv5n** optimized for embedded deployment — 4× fewer FLOPs than standard 640×640
- **NCNN inference engine** with custom C++ decoder for LoongArch
- **3-item safety compliance check**: helmet, mask, and reflective vest
- **Real-time monitoring** with skip-frame strategy for smooth visual feedback
- **Cross-compiled** on Ubuntu x86_64 → LoongArch ELF

## Project Structure

```
├── models/
│   ├── best320_opt.param    # NCNN model (PNNX exported)
│   └── best320_opt.bin
├── src/
│   ├── yolo_detect.cpp      # C++ NCNN inference (no OpenCV dependency)
│   ├── safety_check.py      # Detection + compliance API
│   ├── safety_monitor.py    # Real-time safety monitor
│   ├── snap_and_test.py     # Interactive photo testing
│   └── camera_detect_v2.py  # Real-time detection (visualization only)
└── README.md
```

## Requirements

### Board (Loongson 2K1000LA)

| Component | Version | Notes |
|-----------|---------|-------|
| LoongOS (Linux) | — | Debian-based |
| Python | 3.7+ | |
| OpenCV | 3.2+ | `cv2` module must be importable |
| NCNN | LoongArch | Obtain from [Loongson SDK](https://gitee.com/lq-tech/Loongson_2k300_301_Library) |
| USB Camera | UVC standard | — |

### Development PC

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Model training & export |
| PyTorch | 2.x + CUDA | Training only |
| PNNX | bundled in LQ_TinyClassifier | Model conversion |
| LoongArch Cross-compiler | GCC 8.3 | Obtained from Loongson |

## Quick Start

### 1. Deploy Model to Board

```bash
# Copy files to board
scp models/best320_opt.param models/best320_opt.bin user@board-ip:~/detect/
scp src/*.py user@board-ip:~/detect/
```

### 2. Cross-compile yolo_detect

```bash
# On Ubuntu x86_64 VM
loongarch64-linux-gnu-g++ -o yolo_detect yolo_detect.cpp \
  -I./ncnn_install/include -L./ncnn_install/lib \
  -lncnn -lpthread -std=c++11 -O2

scp yolo_detect user@board-ip:~/detect/
```

### 3. Run on Board

```bash
# SSH to board
cd ~/detect
chmod +x yolo_detect
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

# Interactive testing (recommended for first run)
python3 snap_and_test.py

# Real-time safety monitoring
python3 safety_monitor.py
```

## Safety Compliance Logic

The system checks three PPE items simultaneously. **All three must be satisfied for a PASS.**

| Item | Detection Target | Pass Condition |
|------|-----------------|----------------|
| Helmet | `helmet` | Must be present |
| Mask | `mask` | Must be present |
| Reflective Vest | `no-vest` | Must NOT be present |

> **Why `no-vest` instead of `vest`?** Our dataset has significantly more high-quality `no-vest` annotations. The negative-class detection is more reliable than positive `vest` detection for this dataset.
> **Actually there are also another test items like mask,gloves,no-gloves,person,boots,but since the accuracy problem I only take the top3 accurate and also the most common items into account.Of cause you can add them if needed. 

### API Usage

```python
from safety_check import check_frame
import cv2

frame = cv2.imread("photo.jpg")
result = check_frame(frame)

print(result["status"])    # "pass" or "fail"
print(result["missing"])   # e.g. ["安全帽", "口罩"]
print(result["summary"])   # "不合格：缺安全帽、口罩"
```

## Model Training

The detection model was trained on a merged dataset of ~11,000 PPE images from Roboflow, covering 9 classes. YOLOv5n was exported to TorchScript, then converted to NCNN via PNNX.

```bash
# Export to TorchScript
python export.py --weights best.pt --include torchscript --img 320

# Convert to NCNN
pnnx best.torchscript "inputshape=[1,3,320,320]"
ncnnoptimize best.ncnn.param best.ncnn.bin best320_opt.param best320_opt.bin 0
```

See [LQ_TinyClassifier](https://gitee.com/lq-tech/Loongson_2k300_301_Library) for the Loongson NCNN SDK.

## Category Definition

```
0:helmet  1:no-helmet  2:vest  3:no-vest  4:mask
5:gloves  6:no-gloves  7:person  8:boots
```

## Known Limitations

- **~1.3 FPS** on Loongson 2K0301 (single-core FP32). Use skip-frame strategy for smooth preview.
- **Small object detection is weak** — distant helmets/masks may be missed.
- **Not for multi-person crowded scenes** — designed for single-person compliance check.
- **Fire detection** is currently not supported (training data not included).

