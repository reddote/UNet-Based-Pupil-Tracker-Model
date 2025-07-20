## 👁️ eNetPupilTracker: U-Net-Based Pupil Segmentation System

**eNetPupilTracker** is a deep learning-based framework for pupil detection and segmentation. It uses a customized **U-Net384x288** architecture optimized for fast and accurate performance, suitable for real-time applications.

---

### 📐 Model Overview

* **Architecture:** U-Net with 3×3 kernels
* **Input Size:** `384 × 288 × 3` (RGB)
* **Output:** 2-class segmentation (e.g. pupil / background)
* **Post-processing:** Conv + ReLU + Conv refinement

---

### 📁 Project Structure

```
eNetPupilTracker/
├── Model/
│   ├── unet_model.py            # U-Net model definition
│   ├── gaze_dataset.py          # Dataset loader
│   └── OldModel/
│       └── efficient_b0.py      # (legacy model, unused)
│
├── Utils/
│   ├── draw_eye.py              # Visualization tools
│   ├── file_controller.py       # File navigation and logic
│   ├── file_writer.py           # Prediction/image saving tools
│   └── find_pupil.py            # Pupil center metrics
│
├── output_frames/               # Input RGB images (organized in folders)
├── segmentation/                # Ground truth segmentation labels
├── predict/                     # Test images used for evaluating model accuracy
│
├── environment.yml              # Conda environment specification
├── v2_main.py                   # ✅ Latest main script (use this)
├── main.py                      # ❌ Legacy version (deprecated)
├── main_test.py                 # Evaluation script
├── test_model.py                # Model testing utilities
├── log.txt                      # Log file
└── .gitignore
```

---

### ⚙️ Environment Setup

```bash
conda env create -f environment.yml
conda activate enetpupil
```

---

### 🚀 Usage

#### 🔹 Run Inference / Evaluation (latest version)

```bash
python v2_main.py
```

This script:

* Loads input images from `output_frames/`
* Loads ground truth segmentation labels from `segmentation/`
* Uses test images in `predict/` for model accuracy checks
* Outputs segmentation predictions and pupil center metrics

#### 🔹 Evaluate model (optional)

```bash
python main_test.py
```

---

### 📊 Evaluation Metrics

* Dice Coefficient
* IoU (Intersection over Union)
* Precision & Recall
* Pupil center localization error (`find_pupil.py`)

---

### 🧠 Model Architecture Summary

| Stage         | Channels | Size (HxW) | Description                    |
| ------------- | -------- | ---------- | ------------------------------ |
| Input Image   | 3        | 384×288    | RGB input                      |
| Stem          | 16       | 384×288    | Initial ConvBlock              |
| Encoder 1     | 32       | 192×144    | ConvBlock + Downsampling       |
| Encoder 2     | 64       | 96×72      | ConvBlock + Downsampling       |
| Encoder 3     | 128      | 48×36      | ConvBlock + Downsampling       |
| Encoder 4     | 256      | 24×18      | ConvBlock + Downsampling       |
| Bottleneck    | 512      | 24×18      | Deepest layer                  |
| Decoder 4     | 256      | 48×36      | Upsample + Skip x4 + ConvBlock |
| Decoder 3     | 128      | 96×72      | Upsample + Skip x3 + ConvBlock |
| Decoder 2     | 64       | 192×144    | Upsample + Skip x2 + ConvBlock |
| Decoder 1     | 32       | 384×288    | Upsample + Skip x1 + ConvBlock |
| Final Conv    | 2        | 384×288    | 1×1 Conv2D (2 classes)         |
| Output Refine | 2        | 384×288    | Conv + ReLU + Conv             |

---

### ✅ Notes

* Use **`v2_main.py`** as the primary execution script.
* Segmentation masks (ground truth) must be stored in the `segmentation/` folder.
* Input image folders are inside `output_frames/`.
* Test image predictions (visual output) will appear in `predict/`.
