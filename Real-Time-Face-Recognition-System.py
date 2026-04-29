import cv2
import face_recognition
import os
import numpy as np

# --- 1. AŞAMA: ÖĞRETME (EĞİTİM) ---
yol = r'C:\PROJE\bilinen_kisiler'
print("Sistem eğitiliyor, lütfen bekleyin...")

bilinen_kodlar = []
isimler = []

# Klasör kontrolü
if not os.path.exists(yol):
    print(f"HATA: {yol} dizini bulunamadı!")
    exit()

for dosya in os.listdir(yol):
    if dosya.lower().endswith(('.jpg', '.jpeg', '.png')):
        tam_yol = os.path.join(yol, dosya)
        
        # Türkçe karakterli dosya yolları için güvenli okuma
        resim_verisi = np.fromfile(tam_yol, np.uint8)
        img = cv2.imdecode(resim_verisi, cv2.IMREAD_COLOR)
        
        if img is not None:
            # İsmi temizle (Örn: ali_1.jpg -> ALI)
            temiz_isim = os.path.splitext(dosya)[0].split('_')[0].upper()
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kodlar = face_recognition.face_encodings(img_rgb)
            
            if len(kodlar) > 0:
                bilinen_kodlar.append(kodlar[0])
                isimler.append(temiz_isim)
                print(f"✓ {temiz_isim} sisteme tanımlandı.")

print(f"\nEğitim tamamlandı. Tanınacak kişiler: {list(set(isimler))}")
print("-" * 50)

# --- 2. AŞAMA: GERÇEK ZAMANLI TANIMA ---
cap = cv2.VideoCapture(0)
islem_yapilsin_mi = True 

# Sonuçları saklamak için boş liste (Kasma önleme için kare atlandığında kullanılır)
display_names = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    # 1. İşlemciyi yormamak için her 2 karede bir analiz yap
    if islem_yapilsin_mi:
        # Görüntüyü 1/4 oranında küçült (Hız için)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        yuz_konumlari = face_recognition.face_locations(imgS)
        yuz_kodlari = face_recognition.face_encodings(imgS, yuz_konumlari)

        display_names = []
        for encodeFace, faceLoc in zip(yuz_kodlari, yuz_konumlari):
            # TOLERANS: 0.4 - 0.5 arası en güvenlisidir. 
            # 0.6 yaparsan yabancıları tanıdıklarına benzetme ihtimali artar.
            mesafeler = face_recognition.face_distance(bilinen_kodlar, encodeFace)
            
            name = "BILINMIYOR"
            if len(mesafeler) > 0:
                en_iyi_eslesme_index = np.argmin(mesafeler)
                # Mesafe 0.45'ten küçükse kişi "tanıdıktır"
                if mesafeler[en_iyi_eslesme_index] < 0.45:
                    name = isimler[en_iyi_eslesme_index]
            
            display_names.append((name, faceLoc))

    islem_yapilsin_mi = not islem_yapilsin_mi 

    # 2. Ekrana Çizim Yapma
    for name, faceLoc in display_names:
        # Koordinatları 4 ile çarpıyoruz çünkü resmi 0.25 oranında küçültmüştük
        y1, x2, y2, x1 = [v * 4 for v in faceLoc]
        
        # Tanınıyorsa Yeşil, tanınmıyorsa Kırmızı çerçeve
        renk = (0, 255, 0) if name != "BILINMIYOR" else (0, 0, 255)
        
        # Dikdörtgen ve İsim Etiketi
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), renk, cv2.FILLED) # İsim arka planı
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Bilgi notu
    cv2.putText(img, "Cikis icin 'q' tusuna basin", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
    
    cv2.imshow('Guvenli Yuz Tanima Sistemi', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
