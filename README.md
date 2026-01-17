# 👷 YOLOv8 SafeZone: AI-Powered Helmet Detection System

[](https://github.com/user-attachments/assets/89a22cdd-850d-4535-90be-20efc925fada)


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yolov8-safezone-helmet-detection.streamlit.app)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)

This project is a **real-time Personal Protective Equipment (PPE) detection system** developed to enhance Occupational Health and Safety (OHS).

Using a custom-trained **YOLOv8** model and **Streamlit** interface, it analyzes video feeds to detect whether workers within a user-defined **"Safe Zone" (Geofence)** are wearing hard hats.

## 🧠 Model Training & Performance

The YOLOv8 model was trained using a dataset sourced from **Roboflow**, specifically designed for hard hat detection. The training process was conducted on a high-performance **NVIDIA A100 GPU**.

### 📊 Training Configuration
* **Model Architecture:** YOLOv8m (Medium)
* **Epochs:** 50
* **Batch Size:** 128
* **Image Size:** 640x640
* **Optimizer:** AdamW
* **Learning Rate:** 0.001

### 📈 Model Metrics
The model demonstrated **exceptional performance** in the critical safety classes (**Helmet** and **Head**), achieving over **96% mAP50**.

| Class       | Precision (P) | Recall (R) | mAP50  | mAP50-95 |
|-------------|---------------|------------|--------|----------|
| **All** | 0.625         | 0.643      | 0.652  | 0.468    |
| **Head** | **0.900** | **0.958** | **0.963** | **0.692** |
| **Helmet** | **0.943** | **0.961** | **0.983** | **0.705** |
| **Person*** | 0.031         | 0.010      | 0.011  | 0.008    |

*(\*Note: The 'Person' class was present in the dataset but not the primary focus of this safety application, which relies on Head/Helmet differentiation.)*


<img width="2656" height="1600" alt="analiz_graf_ing" src="https://github.com/user-attachments/assets/80ac5cae-0009-4b30-81ca-fece39a8ba49" />

### 💻 Training Command
The model was trained using the Ultralytics Python SDK with optimized parameters for A100 GPU:
```python
results = model.train(
    data='data.yaml',
    epochs=50,
    patience=10,
    imgsz=640,
    batch=128,
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    name='hard_hat_safety_check'
)
```

<img width="1265" height="777" alt="Ekran görüntüsü 2025-12-16 133227" src="https://github.com/user-attachments/assets/7bdc50c0-4c99-4d97-a9c8-90288fe2ad3a" />

---

<img width="1266" height="779" alt="Ekran görüntüsü 2025-12-16 133739" src="https://github.com/user-attachments/assets/6badd9d4-1199-4066-8290-a5391d651ef2" />

---


## 🚀 Features

- **🎯 Virtual Geofence:** Users can draw a polygon zone on the video interface. Only personnel entering this risk zone are analyzed.
- **🧠 Custom Trained Model:** Powered by YOLOv8, trained on the "Hard Hat" dataset via Roboflow.
- **⚡ ByteTrack Algorithm:** robust object tracking with unique ID assignment (ID:1, ID:2...) to prevent duplicate counting.
- **🚨 Automatic Violation Logging:** When a worker without a helmet is detected:
  - A snapshot is saved to the `ihlal_kayitlari/` directory.
  - Date, time, and status are logged into a CSV file.
- **📊 Live Dashboard:** Real-time violation counters and recent snapshots are displayed on the sidebar.


  ## ☁️ Cloud vs. 💻 Local Usage

The project offers different capabilities depending on where it is deployed:

| Platform | Mode | Description |
|----------|------|-------------|
| **Streamlit Cloud** | 🖼️ **Image Only** | Optimized for web performance. Upload snapshots for instant analysis without latency. |
| **Local Machine** | 🎥 **Video & Image** | Use the script `detect.py` to analyze videos (mp4/avi) locally. |

🔗 **Live Demo:** [Try the App Here](https://yolov8-safezone-helmet-detection.streamlit.app/)

<img width="1891" height="896" alt="image" src="https://github.com/user-attachments/assets/355a7a29-2588-455b-815c-3412a65ce14a" />


---

## 🛠️ Installation & Local Usage

To run the project locally on your machine (Recommended for GPU acceleration):

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mehmetkosee/YOLOv8-SafeZone-Helmet-Detection.git](https://github.com/mehmetkosee/YOLOv8-SafeZone-Helmet-Detection.git)
   cd YOLOv8-SafeZone-Helmet-Detection
   
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt

3. **Run the application:**
   ```bash
   python -m streamlit run app.py

## 📂 Project Structure

```text
├── app.py               # Main Streamlit application script
├── best.pt              # Custom trained YOLOv8 model weights
├── detect.py            # Standalone script for terminal-based inference
├── requirements.txt     # Python dependencies
├── packages.txt         # System-level dependencies (for Linux/Cloud)
├── ihlal_raporu.csv     # Auto-generated violation report (CSV)
├── ihlal_kayitlari/     # Directory for violation snapshots
├── train/               # Jupyter Notebooks used for model training

## 👨‍💻 Developer
Mehmet Köse
