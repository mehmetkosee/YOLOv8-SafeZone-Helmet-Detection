# 👷 YOLOv8 SafeZone: AI-Powered Helmet Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yolov8-safezone-helmet-detection.streamlit.app)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)

This project is a **real-time Personal Protective Equipment (PPE) detection system** developed to enhance Occupational Health and Safety (OHS).

Using a custom-trained **YOLOv8** model and **Streamlit** interface, it analyzes video feeds to detect whether workers within a user-defined **"Safe Zone" (Geofence)** are wearing hard hats.

🔗 **Live Demo:** [Try the App Here](https://yolov8-safezone-helmet-detection.streamlit.app)

---

## 🚀 Features

- **🎯 Virtual Geofence:** Users can draw a polygon zone on the video interface. Only personnel entering this risk zone are analyzed.
- **🧠 Custom Trained Model:** Powered by YOLOv8, trained on the "Hard Hat" dataset via Roboflow.
- **⚡ ByteTrack Algorithm:** robust object tracking with unique ID assignment (ID:1, ID:2...) to prevent duplicate counting.
- **🚨 Automatic Violation Logging:** When a worker without a helmet is detected:
  - A snapshot is saved to the `ihlal_kayitlari/` directory.
  - Date, time, and status are logged into a CSV file.
- **📊 Live Dashboard:** Real-time violation counters and recent snapshots are displayed on the sidebar.

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
   streamlit run app.py

## 📂 Project Structure

```text
├── app.py               # Main Streamlit application script
├── best.pt              # Custom trained YOLOv8 model weights
├── detect.py            # Standalone script for terminal-based inference
├── requirements.txt     # Python dependencies
├── packages.txt         # System-level dependencies (for Linux/Cloud)
├── ihlal_raporu.csv     # Auto-generated violation report (CSV)
├── ihlal_kayitlari/     # Directory for violation snapshots
├── egitim_notlari/      # Jupyter Notebooks used for model training
└── test_videolari/      # (Optional) Sample videos for testing



## ℹ️ Performance Note
This project is deployed on Streamlit Community Cloud. Since free cloud instances utilize CPU-only environments, the live demo runs at a lower FPS compared to local execution.

For real-time performance and high FPS, it is recommended to run this project locally on a machine with a CUDA-enabled NVIDIA GPU.

## 👨‍💻 Developer
Mehmet Köse
