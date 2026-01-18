import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==========================================
# 1. AYARLAR VE BAŞLIK
# ==========================================
st.set_page_config(page_title="SafeZone AI", page_icon="👷", layout="centered")

st.title("👷 Yapay Zeka Destekli Baret Tespit Sistemi")
st.write("Resmi yükleyin, yapay zeka iş güvenliği ihlallerini otomatik tespit etsin.")

# ==========================================
# 2. MODELİ YÜKLEME
# ==========================================
@st.cache_resource
def load_model():
    # 'best.pt' dosyasının projenin ana klasöründe olduğundan emin olun
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model yüklenemedi! Hata: {e}")
    st.stop()

# ==========================================
# 3. YAN MENÜ (AYARLAR)
# ==========================================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    confidence = st.slider("Hassasiyet (Confidence)", 0.25, 1.0, 0.40)
    st.write("---")
    st.write("Bu mod, yüklenen resmin **tamamını** analiz eder.")
    st.write("👨‍💻 Geliştirici: Mehmet Köse")

# ==========================================
# 4. RESİM YÜKLEME VE ANALİZ
# ==========================================
uploaded_file = st.file_uploader("Bir Fotoğraf Yükleyin", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Resmi aç ve formatla
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_cv2 = np.array(image_pil)
    
    # Ekranda göster
    st.image(image_pil, caption="Yüklenen Resim", use_column_width=True)
    
    # Analiz Butonu
    if st.button("🔍 TESPİT ET", type="primary", use_container_width=True):
        
        with st.spinner("Yapay Zeka Resmi Tarıyor..."):
            # Model Tahmini
            results = model.predict(image_cv2, conf=confidence)
            
            # Sonuçları işle
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().numpy()
            
            final_img = image_cv2.copy()
            ihlal_sayisi = 0
            
            for box, class_id in zip(boxes, classes):
                x, y, w, h = box
                x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                
                # Sınıflandırma (0: Baretsiz, 1: Baretli - Modelinize göre)
                if class_id == 0:  # İHLAL (BARETSİZ)
                    color = (255, 0, 0) # Kırmızı
                    label = "BARET YOK"
                    ihlal_sayisi += 1
                else:              # GÜVENLİ
                    color = (0, 255, 0) # Yeşil
                    label = "GUVENLI"
                
                # Çizim
                cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(final_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Sonucu Göster
            st.divider()
            st.subheader("📊 Analiz Sonucu")
            st.image(final_img, caption="İşlenmiş Görüntü", use_column_width=True)
            
            # Rapor Mesajı
            if ihlal_sayisi > 0:
                st.error(f"🚨 DİKKAT: Toplam {ihlal_sayisi} personelde baret tespit edilemedi!")
            else:
                st.success("✅ GÜVENLİ: Tüm personellerde baret takılı.")