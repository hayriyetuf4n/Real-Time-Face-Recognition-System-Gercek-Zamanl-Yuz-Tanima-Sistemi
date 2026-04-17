import cv2
import face_recognition
import os
import numpy as np

# --- 1. AŞAMA: ÖĞRETME ---
yol = r'C:\PROJE\bilinen_kisiler'
print("Sistem eğitiliyor...")

bilinen_kodlar = []
isimler = []

for dosya in os.listdir(yol):
    if dosya.lower().endswith(('.jpg', '.jpeg', '.png')):
        tam_yol = os.path.join(yol, dosya)
        resim_verisi = np.fromfile(tam_yol, np.uint8)
        img = cv2.imdecode(resim_verisi, cv2.IMREAD_COLOR)
        
        if img is not None:
            # ÖNEMLİ: İsmi dosya adından alır (rojda.jpg -> ROJDA)
            temiz_isim = dosya.split('.')[0].split('_')[0].upper()
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kodlar = face_recognition.face_encodings(img_rgb)
            
            if len(kodlar) > 0:
                bilinen_kodlar.append(kodlar[0])
                isimler.append(temiz_isim)
                print(f"✓ {temiz_isim} eklendi.")

print(f"Eğitim bitti. Hafızadakiler: {list(set(isimler))}")
print("-" * 30)

# --- 2. AŞAMA: OPTİMİZE EDİLMİŞ TANIMA ---
cap = cv2.VideoCapture(0)
islem_yapilsin_mi = True # Kasma önleyici anahtar

while True:
    success, img = cap.read()
    if not success: break
    
    # Kasmayı önlemek için görüntüyü küçült ve her 2 karede bir işlem yap
    if islem_yapilsin_mi:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        yuz_konumlari = face_recognition.face_locations(imgS)
        yuz_kodlari = face_recognition.face_encodings(imgS, yuz_konumlari)

        display_names = []
        for encodeFace, faceLoc in zip(yuz_kodlari, yuz_konumlari):
            eslesmeler = face_recognition.compare_faces(bilinen_kodlar, encodeFace, tolerance=0.6)
            yuz_mesafesi = face_recognition.face_distance(bilinen_kodlar, encodeFace)
            
            name = "BILINMIYOR"
            if len(yuz_mesafesi) > 0:
                eslesme_index = np.argmin(yuz_mesafesi)
                if eslesmeler[eslesme_index]:
                    name = isimler[eslesme_index]
            display_names.append((name, faceLoc))

    # Kare atlama mantığı (İşlemciyi rahatlatır)
    islem_yapilsin_mi = not islem_yapilsin_mi 

    # Çizim aşaması
    for name, faceLoc in display_names:
        y1, x2, y2, x1 = [v * 4 for v in faceLoc]
        renk = (0, 255, 0) if name != "BILINMIYOR" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Hizli Tanima Sistemi', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows() 
