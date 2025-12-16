import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import time
import pandas as pd
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- AYARLAR ---
MODEL_PATH = "best.pt"
IHLAL_KLASORU = "ihlal_kayitlari"
RAPOR_DOSYASI = "ihlal_raporu.csv"

# Klasör ve Dosya Hazırlığı
if not os.path.exists(IHLAL_KLASORU):
    os.makedirs(IHLAL_KLASORU)

if not os.path.exists(RAPOR_DOSYASI):
    df = pd.DataFrame(columns=["Tarih", "Saat", "Durum", "Dosya_Yolu"])
    df.to_csv(RAPOR_DOSYASI, index=False)

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="ISG Sistemi", page_icon="👷", layout="wide")
st.title("👷 Yapay Zeka Destekli ISG Sistemi")

# --- YAN MENÜ ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    # 1. CANLI SAYAÇ İÇİN YER TUTUCU (PLACEHOLDER)
    # Burayı boş bırakıyoruz, döngü içinde dolduracağız
    metric_placeholder = st.empty()
    
    # 2. SON İHLAL FOTOSU İÇİN YER TUTUCU
    st.subheader("📸 Son Tespit Edilen İhlal")
    ihlal_foto_placeholder = st.empty()
    
    st.divider()
    confidence = st.slider("Hassasiyet", 0.25, 1.0, 0.45)
    process_n_frames = st.slider("Hız (Kare Atlama)", 1, 10, 3)

# --- FONKSİYONLAR ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def update_metrics():
    """CSV dosyasını okuyup yan menüdeki sayacı günceller"""
    try:
        df = pd.read_csv(RAPOR_DOSYASI)
        count = len(df)
        # Yer tutucuyu güncelle
        metric_placeholder.metric("Toplam İhlal Sayısı", count, delta="Canlı")
    except:
        metric_placeholder.metric("Toplam İhlal Sayısı", 0)

def log_to_csv(track_id, img_path):
    """İhlali kaydeder"""
    now = datetime.now()
    new_data = {
        "Tarih": now.strftime("%Y-%m-%d"),
        "Saat": now.strftime("%H:%M:%S"),
        "Durum": f"IHLAL_ID_{track_id}",
        "Dosya_Yolu": img_path
    }
    df = pd.DataFrame([new_data])
    df.to_csv(RAPOR_DOSYASI, mode='a', header=False, index=False)

# --- ANA KOD ---
try:
    model = load_model()
except:
    st.error("Model yüklenemedi! 'best.pt' dosyasını kontrol et.")
    st.stop()

# Başlangıçta sayacı bir kere güncelle
update_metrics()

uploaded_file = st.file_uploader("Analiz için Video Yükleyin", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    
    # --- KOLONLAR ---
    col1, col2 = st.columns([2, 1])
    zone_poly = None
    
    # 1. ÇİZİM ALANI
    with col1:
        st.info("1. Aşağıdaki görsele mouse ile alan çizin.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret:
            first_frame = cv2.resize(first_frame, (640, 480))
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(first_frame_rgb)

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#ff0000",
                background_image=pil_image,
                update_streamlit=True,
                height=480,
                width=640,
                drawing_mode="polygon",
                key="canvas",
            )
            
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    path_data = objects[0]["path"]
                    points = []
                    for p in path_data:
                        if p[0] == 'M' or p[0] == 'L':
                            points.append([int(p[1]), int(p[2])])
                    if len(points) > 2:
                        zone_poly = np.array(points, np.int32).reshape((-1, 1, 2))
                        st.success("✅ Alan Hafızaya Alındı!")

    # 2. ANALİZ ALANI
    with col2:
        st.info("2. Alan çizildiyse başlatın.")
        start_btn = st.button("▶️ ANALİZİ BAŞLAT", type="primary")
        # İhlal bildirimleri için kutu
        log_box = st.container(height=400)

    # --- ANALİZ DÖNGÜSÜ ---
    if start_btn and zone_poly is not None:
        st_video_spot = st.empty() # Video oynatıcı yeri
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        son_foto_zamani = 0
        foto_bekleme = 2.0 

        # --- GÜNCELLENMİŞ ANALİZ DÖNGÜSÜ ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Görüntüyü boyutlandır
            frame = cv2.resize(frame, (640, 480))
            
            # Alanı Çiz
            cv2.polylines(frame, [zone_poly], isClosed=True, color=(0, 255, 255), thickness=2)

            # --- YOLO ANALİZİ ---
            # tracker="bytetrack.yaml" nesneleri daha sıkı tutar
            if frame_count % process_n_frames == 0:
                results = model.track(frame, persist=True, verbose=False, imgsz=640, conf=confidence, tracker="bytetrack.yaml")
                
                # Sadece eğer bir tespit varsa kutuları güncelle
                # Eğer tespit yoksa ESKİ KUTULARI KORU (Yanıp sönmeyi engeller)
                if results[0].boxes.id is not None:
                    last_boxes = results[0].boxes.xywh.cpu().numpy()
                    last_ids = results[0].boxes.id.int().cpu().numpy()
                    last_classes = results[0].boxes.cls.int().cpu().numpy()
                else:
                    # Eğer kimse yoksa kutuları hemen silme, 
                    # sadece çok uzun süre boş kalırsa sil (Örn: 10 kare boyunca)
                    pass 

            # --- GÖRSELLEŞTİRME ---
            if len(last_boxes) > 0:
                for box, track_id, class_id in zip(last_boxes, last_ids, last_classes):
                    x, y, w, h = box
                    foot_x, foot_y = int(x), int(y + h / 2) # Ayak noktası
                    
                    # Alan Kontrolü
                    if cv2.pointPolygonTest(zone_poly, (foot_x, foot_y), False) >= 0:
                        
                        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                        
                        # --- İHLAL (Kırmızı) ---
                        if class_id == 0: 
                            # Kırmızı Kutu
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                            
                            # ID Numarası ve Etiket
                            label = f"ID:{track_id} IHLAL"
                            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text, y1), (0,0,255), -1)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # --- KAYIT MANTIĞI ---
                            if time.time() - son_foto_zamani > foto_bekleme:
                                fname = f"ihlal_{track_id}_{datetime.now().strftime('%H%M%S')}.jpg"
                                full_path = os.path.join(IHLAL_KLASORU, fname)
                                
                                # Kaydet ve Logla
                                cv2.imwrite(full_path, frame)
                                log_to_csv(track_id, full_path)
                                update_metrics()
                                
                                # Yan menüye fotosunu bas
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                ihlal_foto_placeholder.image(rgb_frame, caption=f"İhlal ID: {track_id}")
                                log_box.error(f"⚠️ Tespit: ID {track_id}")
                                
                                son_foto_zamani = time.time()
                                
                        # --- GÜVENLİ (Yeşil) ---
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                            # ID Numarası (Yeşil için de yazalım ki takip belli olsun)
                            label_ok = f"ID:{track_id}"
                            cv2.putText(frame, label_ok, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Web Sitesine Bas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_video_spot.image(frame_rgb, channels="RGB")
            
            frame_count += 1

        cap.release()
        #python -m streamlit run app.py