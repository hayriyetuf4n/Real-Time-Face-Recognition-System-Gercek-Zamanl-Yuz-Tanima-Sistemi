# Real-Time-Face-Recognition-System-Gercek-Zamanl-Yuz-Tanima-Sistemi
#  Real-Time Face Recognition System | Gerçek Zamanlı Yüz Tanıma Sistemi

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Mechatronics](https://img.shields.io/badge/Engineering-Mechatronics-orange?style=for-the-badge)

Bu proje, Python ve OpenCV kütüphaneleri kullanılarak geliştirilmiş, düşük gecikmeli (low-latency) ve yüksek doğruluklu bir **Anlık Yüz Tanıma Sistemi**'dir. Mekatronik mühendisliği çalışmaları kapsamında, görüntü işleme ve biyometrik veri analizi üzerine odaklanılarak optimize edilmiştir.

---

##  Öne Çıkan Özellikler (Key Features)

*  Yüksek Performans:** Kare atlama (frame skipping) ve görüntü küçültme teknikleriyle CPU yükü optimize edilmiştir.
*  Çoklu Yüz Tanıma:** Aynı kare içerisinde birden fazla kişiyi eş zamanlı olarak tespit edebilir ve isimlendirebilir.
*  Dinamik Veri Seti:** Yeni bir kişiyi tanıtmak için sadece `bilinen_kisiler` klasörüne bir fotoğraf eklemek yeterlidir.
*  Hata Yakalama:** Işık yetersizliği veya OneDrive dosya yolu hataları için geliştirilmiş dosya okuma algoritması içerir.



---

##  Teknik Altyapı (Tech Stack)

| Kütüphane | Kullanım Amacı |

| **OpenCV** | Kamera erişimi, görüntü ön işleme ve görselleştirme. |
| **Face_Recognition** | dlib tabanlı yüz kodlama (128-d embeddings) ve karşılaştırma. |
| **NumPy** | Matris hesaplamaları ve yüksek hızlı veri işleme. |
| **OS** | Dosya sistemi ve dizin yönetimi. |

---

## Kurulum ve Çalıştırma (Quick Start)

1. **Gerekli kütüphaneleri terminal üzerinden yükleyin:**
   ```bash
   pip install opencv-python face-recognition numpy
