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

---

## 🔄 ONNX Model Conversion

Each model is also provided in ONNX format and can be converted to TensorRT using `trtexec` or Python API.

### Example (Using trtexec):

```bash
trtexec --onnx=weights/object/object.onnx --saveEngine=weights/object/object.engine --fp16
```

### Notes:

- Ensure your Jetson environment matches the ONNX conversion environment (TensorRT, CUDA, cuDNN)
- You can adjust input shape, workspace size, and precision as needed

---

### ✅ 2. Run demo

```bash
python ADSW_Release.py -v ./videos/SIHEUNG.mp4
```

---

## 📦 TensorRT Engine Notes

- `.engine` files are already optimized models. You must regenerate them if:
  - You change your Jetson device (e.g., from Xavier to Orin)
  - Your TensorRT or CUDA version differs from the training machine
- See [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) for converting ONNX to TensorRT.

--- 

## 👤 Maintainer

**Taehyeon Kim**  
Senior Researcher, Korea Electronics Technology Institute (KETI)  
📧 taehyeon.kim@keti.re.kr

---

## 📜 License

This project is released under the MIT License. See `LICENSE` file for details.
