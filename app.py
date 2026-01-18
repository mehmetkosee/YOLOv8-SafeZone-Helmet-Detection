import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
from PIL import Image

cv2.setNumThreads(0)

# ==========================================
# 1. AYARLAR VE KURULUM
# ==========================================
MODEL_PATH = "best.pt"
RAPOR_DOSYASI = "ihlal_raporu.csv"
IHLAL_KLASORU = "ihlal_kayitlari"

# Klasör yoksa oluştur
if not os.path.exists(IHLAL_KLASORU):
    os.makedirs(IHLAL_KLASORU)

# Rapor dosyası yoksa başlıkları at
if not os.path.exists(RAPOR_DOSYASI):
    pd.DataFrame(columns=["Tarih", "Saat", "Durum", "Dosya_Yolu"]).to_csv(RAPOR_DOSYASI, index=False)

st.set_page_config(page_title="SafeZone AI", page_icon="👷", layout="wide")

# ==========================================
# 2. MODELİ YÜKLEME (CACHE İLE HIZLANDIRMA)
# ==========================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# ==========================================
# 3. ARAYÜZ (YAN MENÜ)
# ==========================================
with st.sidebar:
    st.header("⚙️ Ayarlar")
    confidence = st.slider("Hassasiyet (Confidence)", 0.25, 1.0, 0.40)
    st.info("Bu sistem Cloud performansı için optimize edilmiş **Resim Analizi** modunda çalışmaktadır.")
    st.divider()
    st.write("👨‍💻 Geliştirici: Mehmet Köse")

st.title("👷 Yapay Zeka Destekli Baret Tespit Sistemi")
st.markdown("Analiz etmek istediğiniz fotoğrafı yükleyin ve **Polygon** aracıyla riskli alanı çizin.")

uploaded_file = st.file_uploader("Bir Fotoğraf Yükleyin", type=['jpg', 'png', 'jpeg'])

# ==========================================
# 4. GÖRÜNTÜ İŞLEME VE ÇİZİM
# ==========================================
if uploaded_file:
    # --- RESİM OKUMA VE DÜZELTME ---
    image_pil = Image.open(uploaded_file)
    image_pil = image_pil.convert("RGB") 

    # ÖNEMLİ: Resmi canvas boyutuna (640x480) getiriyoruz ki ekrana tam otursun
    image_pil = image_pil.resize((640, 480))
    
    # OpenCV ve YOLO için Numpy dizisine çevir
    # ... senin yazdığın kısım ...
    image_cv2 = np.array(image_pil)              

    # Ekranı ikiye böl: Çizim ve Sonuç
    col1, col2 = st.columns([2, 1])
    zone_poly = None

    # --- SOL KOLON: ÇİZİM ALANI ---
    with col1:
        st.info("👇 **Adım 1:** Sol menüden 'Polygon' aracını seçip alanı çizin.")
        
        # Çizim Aracı
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=image_pil,     # Boyutlandırılmış resim
            update_streamlit=True,
            height=480,
            width=640,
            drawing_mode="polygon",
            # BURASI DEĞİŞTİ: Dosya her değiştiğinde canvas'ı zorla yeniliyoruz
            key=f"canvas_{uploaded_file.name}", 
        )

        # --- ÇİZİM VERİSİNİ ALMA (GÜVENLİK KONTROLLÜ) ---
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            if len(objects) > 0:
                obj = objects[0] # İlk çizimi al
                
                # "KeyError: path" hatasını önlemek için kontrol
                if "path" in obj:
                    path_data = obj["path"]
                    points = []
                    for p in path_data:
                        if p[0] == 'M' or p[0] == 'L': # SVG komutlarını (Move, Line) oku
                            points.append([int(p[1]), int(p[2])])
                    
                    if len(points) > 2:
                        # Koordinatları Numpy formatına çevir
                        zone_poly = np.array(points, np.int32).reshape((-1, 1, 2))
                else:
                    st.warning("⚠️ Lütfen alanı kapatarak tam bir çokgen çizin.")

    # --- SAĞ KOLON: BUTON VE ANALİZ ---
    with col2:
        st.info("👇 **Adım 2:** Analizi başlatın.")
        analyze_btn = st.button("🔍 TESPİT ET", type="primary", use_container_width=True)

        if analyze_btn:
            if zone_poly is None:
                st.error("⚠️ Lütfen önce soldaki resim üzerinde bir alan çizin!")
            else:
                with st.spinner("Yapay Zeka Analiz Ediyor..."):
                    # YOLO Tahmini
                    results = model.predict(image_cv2, conf=confidence, imgsz=640)
                    
                    # Sonuçları al
                    boxes = results[0].boxes.xywh.cpu().numpy()  # x, y, genislik, yukseklik
                    classes = results[0].boxes.cls.int().cpu().numpy() # sınıf id'leri
                    
                    final_img = image_cv2.copy()
                    
                    # Çizilen alanı sarı çizgiyle göster
                    cv2.polylines(final_img, [zone_poly], isClosed=True, color=(0, 255, 255), thickness=3)
                    
                    ihlal_sayisi = 0
                    
                    # Tespitleri Kontrol Et
                    for box, class_id in zip(boxes, classes):
                        x, y, w, h = box
                        # Kutunun "ayak" noktası (yerle temas eden nokta)
                        foot_x, foot_y = int(x), int(y + h / 2)
                        
                        # Nokta çizilen alanın içinde mi?
                        if cv2.pointPolygonTest(zone_poly, (foot_x, foot_y), False) >= 0:
                            
                            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                            
                            # NOT: Senin modeline göre 0 veya 1 değişebilir.
                            # Genelde: 0 -> Head (Baretsiz/İhlal), 1 -> Helmet (Güvenli)
                            if class_id == 0: # IHLAL DURUMU
                                color = (255, 0, 0) # Kırmızı
                                label = "BARET YOK"
                                ihlal_sayisi += 1
                            else: # GUVENLI DURUM
                                color = (0, 255, 0) # Yeşil
                                label = "GUVENLI"

                            # Kutu ve Yazı Çiz
                            cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(final_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Sonuç Resmini Göster
                    st.image(final_img, caption="Analiz Sonucu", use_column_width=True)
                    
                    # Bildirimler
                    if ihlal_sayisi > 0:
                        st.error(f"🚨 DİKKAT: Bölgede {ihlal_sayisi} adet baretsiz personel tespit edildi!")
                    else:
                        st.success("✅ Bölge Güvenli. İhlal tespit edilmedi.")