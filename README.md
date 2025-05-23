
# 🚘 Lane, Traffic Object, and Depth Estimation on Jetson with TensorRT

**Real-time lane detection, traffic sign/light recognition, and monocular depth estimation using TensorRT on NVIDIA Jetson.**  
Optimized for real-world Korean road environments.

<div align="center">
  <img src="./assets/demo.gif" width="600">
</div>

---

## 🧠 Overview

This repository provides three major perception modules for autonomous driving:

- **Lane Detection**: Optimized for robust detection of multiple lanes in complex road environments 
- **Traffic Infrastructure Detection**: Real-time inference for traffic lights, signs, and road markings  
- **Monocular Depth Estimation**: High-resolution depth inference from a single RGB image

All models are optimized with **TensorRT** and designed to run on **NVIDIA Jetson platforms** (Orin, Xavier, etc).

---

## 📂 Directory Structure

```
ADSW-Traffic-Perception/
├── ADSW_Release.py                # Main script for real-time inference
├── videos/                        # Sample video input
│   └── demo.mp4
└── weights/                       # TensorRT engine + ONNX files
    ├── object/
    │   ├── object.engine
    │   └── object.onnx
    ├── lane/
    │   ├── lane.engine
    │   └── lane.onnx
    └── depth/
        ├── depth.engine
        └── depth.onnx
```

---

## 🚀 Program Execution

```bash
python ADSW_Release.py -v ./videos/demo.mp4
```

You should see a display with:

- **Main frame (bottom)**: Real-time overlay with lane and traffic infra detection  
- **Depth view (top-right)**: Inferred depth using `depth.engine`

---

## 🔄 ONNX Model Conversion

Each model is also provided in ONNX format and can be converted to TensorRT using `trtexec` or Python API.

### 🔗 Download ONNX Models

- **📌 Lane Detection ONNX**: [Download from Google Drive](https://drive.google.com/file/d/114qneAcF-QvHZ-9QjmRL9VLeBayxodtk/view?usp=drive_link)
- **🚦 Traffic Object Detection ONNX**: [Download from Google Drive](https://drive.google.com/file/d/1FmJtf293IJ7o8DRGkNBzT513RAlea_DD/view?usp=drive_link)
- **🌊 Depth Estimation ONNX**: [Download from Google Drive](https://drive.google.com/file/d/1LEtztIc9z2R5eZJYI84MI42Mn8GIlR1M/view?usp=drive_link)

### 🛠️ Recommended Conversion (Using trtexec):

After downloading the ONNX models above, please place them in the following paths to match the command below:

```bash
# Lane Detection
trtexec --onnx=weights/lane/lane.onnx --saveEngine=weights/lane/lane.engine --fp16

# Traffic Object Detection
trtexec --onnx=weights/object/objct.onnx --saveEngine=weights/object/object.engine --fp16

# Depth Estimation
trtexec --onnx=weights/depth/depth.onnx --saveEngine=weights/depth/depth.engine --fp16
```

### Notes:

- Ensure your Jetson environment matches the ONNX conversion environment (TensorRT, CUDA, cuDNN)
- You can adjust input shape, workspace size, and precision as needed

---

## 🧠 Additional Tip for TensorRT Conversion: LayerNorm Compatibility

TensorRT does **not natively support `LayerNorm`** in many cases. If your ONNX model includes `nn.LayerNorm` layers, you may encounter conversion or inference errors.

To address this issue, we provide a general-purpose script that replaces `LayerNorm` with a TensorRT-friendly `FakeLayerNorm`.

### 🔧 LayerNorm Replacement + ONNX Export

You can use the following script to:

- Load any `.pth` model
- Automatically replace all `nn.LayerNorm` layers
- Export to ONNX format
- Optionally simplify the ONNX model

📄 **Script**: [`convert_to_onnx.py`](./convert_to_onnx.py)

```bash
python convert_to_onnx.py   --model-path checkpoints/your_model.pth   --model-class models.your_model.YourModel   --input-size 1 3 224 224   --output exported_model.onnx
```

> 💡 `--model-class` must be in the format `module.ClassName` (e.g., `models.my_model.MyModel`)

The script automatically replaces all `LayerNorm` layers with `FakeLayerNorm` to ensure TensorRT compatibility.

---

## 🛠️ Supported Devices & Environment

| Jetson Device | Status      | FPS        | JetPack | CUDA   | TensorRT |
| ------------- | ----------- | ---------- | ------- | ------ | -------- |
| Jetson Orin   | ✅ Supported | ~30 FPS    | 5.1.2   | 11.4   | 8.5.2    |
| Jetson Xavier | ✅ Supported | ~15–20 FPS | 4.6.1   | 10.2   | 8.2.1    |

> ✅ **Note**:
> - You must convert ONNX models using the same JetPack version to ensure TensorRT compatibility.
> - All conversions in this repo were tested directly on-device using `trtexec`.

---

## 👤 Maintainer

**Taehyeon Kim, Ph.D.**  
Senior Researcher, Korea Electronics Technology Institute (KETI)  
📧 [taehyeon.kim@keti.re.kr](mailto:taehyeon.kim@keti.re.kr)  
🌐 [Homepage](https://rcard.re.kr/detail/OISRzd7ua0tW0A1zMEwbKQ/information)

**Hyeri Yu**  
Graduate Student, Yonsei University  
📧 [dbgpfl1206@gmail.com](mailto:dbgpfl1206@gmail.com) 

---

## 📜 License

This project is released under a custom license inspired by the MIT License. See [`LICENSE`](./LICENSE.txt) file for details.

⚠️ **Important Notice**  
Use of this code—commercial or non-commercial, including academic research, model training, product integration, and distribution—**requires prior written permission** from the author. Unauthorized usage will be treated as a license violation.

---

## 🙏 Acknowledgments

This research was financially supported by the Ministry of Trade, Industry and Energy (MOTIE) and the Korea Institute of Advancement of Technology (KIAT) through the International Cooperative R&D Program [P0019782, Embedded AI Based Fully Autonomous Driving Software and MaaS Technology Development].
