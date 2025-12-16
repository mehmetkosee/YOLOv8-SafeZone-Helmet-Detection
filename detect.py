import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime

# --- AYARLAR ---
video_path = "test5.mp4" # 0 Webcam
model_path = "best.pt"

# FPS AYARI: 4 karede bir analiz yap (Hem akıcı hem hızlı)
PROCESS_EVERY_N_FRAMES = 3 
# GÖRÜNTÜ BOYUTU: Bunu 640 yaparsan model çok daha hızlı çalışır
INFERENCE_SIZE = 640 

# --- KAYIT AYARLARI ---
IHLAL_KLASORU = "ihlal_kayitlari"
if not os.path.exists(IHLAL_KLASORU):
    os.makedirs(IHLAL_KLASORU) 

FOTO_BEKLEME_SURESI = 2.0 
son_foto_zamani = 0 

points = []
def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Alan Belirlendi: {points}")

print("Model yükleniyor")
try:
    model = YOLO(model_path)
except:
    print("Model bulunamadı!")
    exit()

cap = cv2.VideoCapture(video_path)
window_name = "Guvenlik Sistemi"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_polygon)

zone_poly = None
start_analysis = False # Space ile başlatma kontrolü
last_boxes = []
last_ids = []
last_classes = []
frame_count = 0

print("--- BAŞLAMAK İÇİN ALAN ÇİZİP 'SPACE' TUŞUNA BASIN ---")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    frame = cv2.resize(frame, (1020, 600))

    # --- MOD 1: ÇİZİM EKRANI ---
    if not start_analysis:
        if len(points) > 0:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            # Sarı, kalın çizgiler
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=3)
            # Köşelere kırmızı noktalar
            for p in points:
                cv2.circle(frame, p, 5, (0, 0, 255), -1)

        cv2.putText(frame, "Alani Cizip SPACE'e Basin", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == 32 and len(points) > 2: # SPACE
            zone_poly = np.array(points, np.int32).reshape((-1, 1, 2))
            start_analysis = True
            print("Sistem Aktif!")
        elif key == ord('q'):
            break

    # MOD 2: AKTİF SİSTEM
    else:
        # Alanı sürekli çiz
        if zone_poly is not None:
             cv2.polylines(frame, [zone_poly], isClosed=True, color=(0, 255, 255), thickness=3)

        # A. MODEL ANALİZİ (Her 4 karede bir)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # imgsz=640
            results = model.track(frame, persist=True, verbose=False, imgsz=INFERENCE_SIZE, conf=0.35)
            
            if results[0].boxes.id is not None:
                last_boxes = results[0].boxes.xywh.cpu().numpy()
                last_ids = results[0].boxes.id.int().cpu().numpy()
                last_classes = results[0].boxes.cls.int().cpu().numpy()
            else:
                last_boxes = []
                last_ids = []
                last_classes = []

        # B. GÖRSELLEŞTİRME VE KONTROL
        if len(last_boxes) > 0 and zone_poly is not None:
            for box, track_id, class_id in zip(last_boxes, last_ids, last_classes):
                x, y, w, h = box
                foot_x, foot_y = int(x), int(y + h / 2) # Ayak noktası

                # Alan kontrolü
                is_inside = cv2.pointPolygonTest(zone_poly, (foot_x, foot_y), False) >= 0
                
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)

                if is_inside:
                    # class_id: 0->İHLAL, 1->OK
                    if class_id == 0: 
                        # --- İHLAL GÖRSELİ ---
                        color = (0, 0, 255) # Kırmızı
                        label = f"ID:{track_id} IHLAL!"
                        
                        # kutu çiziyoruz.
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Yazı için arka plan (Okunabilirlik artar)
                        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # --- FOTOĞRAF KAYDI ---
                        su_anki_zaman = time.time()
                        if su_anki_zaman - son_foto_zamani > FOTO_BEKLEME_SURESI:
                            zaman_damgasi = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            dosya_adi = f"ihlal_ID{track_id}_{zaman_damgasi}.jpg"
                            kayit_yolu = os.path.join(IHLAL_KLASORU, dosya_adi)
                            
                            cv2.imwrite(kayit_yolu, frame)
                            print(f"📸 Kayıt Alındı: {dosya_adi}")
                            son_foto_zamani = su_anki_zaman

                    else: 
                        # --- GÜVENLİ GÖRSELİ ---
                        color = (0, 255, 0) # Yeşil
                        label = f"ID:{track_id} OK"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_count += 1
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()