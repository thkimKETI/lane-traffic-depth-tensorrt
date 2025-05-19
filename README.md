# 🚘 Lane, Traffic Object, and Depth Estimation on Jetson with TensorRT

**Real-time lane detection, traffic sign/light recognition, and monocular depth estimation using TensorRT on NVIDIA Jetson.**  
Developed as part of the [KIAT] Embedded AI-based Full Autonomous Driving (Level 4) SW and MaaS Technology Development Project.

<div align="center">
  <img src="./assets/demo.gif" width="600">
</div>

---

## 🧠 Overview

This repository provides three major perception modules for autonomous driving:

- **Lane Detection**: Lightweight, anchor-based multi-lane detector  
- **Traffic Infrastructure Detection**: Real-time inference for traffic lights, signs, and road markings  
- **Monocular Depth Estimation**: CNN-based single-image depth map inference  

All models are optimized with **TensorRT** and designed to run on **NVIDIA Jetson platforms** (Orin, Xavier, etc).

---

## 🖼️ Example Output

```
Display Frame:
├── Bottom Area: RGB frame with lane + traffic infra overlay  
└── Top-Right Area: Depth map (Inferno colormap)
```

---

## 📂 Directory Structure

```
ADSW-Traffic-Perception/
├── ADSW_Release.py                # Main script for real-time inference
├── videos/                        # Sample video input
│   └── demo.mp4
└── weights/                       # TensorRT engine files
    ├── object/object.engine
    ├── lane/lane.engine
    └── depth/depth.engine

```

---

## ⚙️ Setup

### ✅ 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires: Python 3.8+, CUDA 11+, TensorRT 8+, PyTorch (for tensor handling only)

---

### ✅ 2. Run demo

```bash
python ADSW_Release.py -v ./videos/SIHEUNG.mp4
```

You should see a display with:
- **Main frame (bottom)**: Real-time overlay with lane and traffic infra detection  
- **Depth view (top-right)**: Inferred depth using `depth.engine`

---

## 📦 TensorRT Engine Notes

- `.engine` files are already optimized models. You must regenerate them if:
  - You change your Jetson device (e.g., from Xavier to Orin)
  - Your TensorRT or CUDA version differs from the training machine
- See [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) for converting ONNX to TensorRT.

---

## 🛠️ Supported Devices

| Jetson Device | Status      | FPS      |
|---------------|-------------|----------|
| Jetson Orin   | ✅ Supported | ~30 FPS  |
| Jetson Xavier | ✅ Supported | ~15–20 FPS |
| Jetson Nano   | ⚠️ Not Recommended | Too slow |

---

## 👤 Maintainer

**Taehyeon Kim**  
Senior Researcher, Korea Electronics Technology Institute (KETI)  
📧 taehyeon.kim@keti.re.kr

---

## 📜 License

This project is released under the MIT License. See `LICENSE` file for details.
