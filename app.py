import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# ---  BLOK: AYARLAR VE HAZIRLIK ---
MODEL_PATH = "best.pt"
RAPOR_DOSYASI = "ihlal_raporu.csv"
IHLAL_KLASORU = "ihlal_kayitlari"

# Klasör yoksa oluştur
if not os.path.exists(IHLAL_KLASORU):
    os.makedirs(IHLAL_KLASORU)

# Rapor dosyası yoksa başlıkları at
if not os.path.exists(RAPOR_DOSYASI):
    pd.DataFrame(columns=["Tarih", "Saat", "Durum", "Dosya_Yolu"]).to_csv(RAPOR_DOSYASI, index=False)

st.set_page_config(page_title="SafeZone AI", page_icon="👷")

# --- BLOK: MODELİ YÜKLEME ---
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except:
    st.error("Model bulunamadı! best.pt dosyasını kontrol et.")
    st.stop()

#SIDEBAR VE UPLOAD
with st.sidebar:
    st.header("Ayarlar")
    confidence = st.slider("Hassasiyet", 0.25, 1.0, 0.40)
st.title("👷 Baret Tespit Sistemi")

uploaded_file = st.file_uploader("Analiz edilecek fotoğrafı yükleyin", type=['jpg', 'png', 'jpeg'])

# ---  BLOK: RESİM İŞLEME VE ÇİZİM ---
if uploaded_file:

    image_pil = Image.open(uploaded_file)
    image_cv2 = np.array(image_pil)

    col1, col2 = st.columns([2, 1])
    zone_poly = None

    
    with col1:
        st.info("Riskli alanı çizin:")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)", 
            stroke_width=2,
            background_image=image_pil,
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

    with col2:
        analyze_btn = st.button("🔍 TESPİT ET", type="primary")

    # --- BLOK: ANALİZ VE SONUÇ ---
    if analyze_btn and zone_poly is not None:
        
        results = model.predict(image_cv2, conf=confidence, imgsz=640)
        
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()

        final_img = image_cv2.copy()
        
        cv2.polylines(final_img, [zone_poly], isClosed=True, color=(255, 255, 0), thickness=3)
        
        ihlal_sayisi = 0

        for box, class_id in zip(boxes, classes):
            x, y, w, h = box
            foot_x, foot_y = int(x), int(y + h / 2)
            
            if cv2.pointPolygonTest(zone_poly, (foot_x, foot_y), False) >= 0:
                

                x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                

                if class_id == 0: 
                    color = (255, 0, 0) 
                    label = "BARET YOK"
                    ihlal_sayisi += 1
                else:
                    color = (0, 255, 0)
                    label = "GUVENLI"

                
                cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(final_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.image(final_img, caption=f"Analiz Tamamlandı. {ihlal_sayisi} ihlal bulundu.", use_column_width=True)
        
        if ihlal_sayisi > 0:
            st.error(f"⚠️ Dikkat! {ihlal_sayisi} personel baret takmıyor.")
        else:
            st.success("✅ Bölge güvenli.")