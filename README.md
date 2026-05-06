# Dari Piksel ke Mikroplastik: Memahami Computer Vision Secara Nyata

Computer vision pada dasarnya adalah cara komputer memahami gambar sebagai kumpulan piksel, lalu mengolahnya menjadi informasi yang bisa dikenali. Di tulisan ini saya pakai dataset dari Kaggle Microplastic Dataset for Computer Vision (https://www.kaggle.com/datasets/imtkaggleteam/microplastic-dataset-for-computer-vision) sebagai contoh. Dataset ini sudah melalui beberapa tahap preprocessing, seperti penyesuaian orientasi gambar (auto-orient), pemotongan area fokus (sekitar 30–85% horizontal dan 15–85% vertikal), serta penyaringan data agar hanya gambar dengan anotasi yang digunakan. Data yang tersedia juga dilengkapi dengan bounding box, sehingga setiap partikel mikroplastik dalam gambar punya penanda lokasi yang jelas. Dengan struktur seperti ini, pengolahan gambar bisa diarahkan untuk mengenali pola dan posisi mikroplastik secara lebih terukur.

---

## 1. Apa itu Computer Vision?

Secara sederhana, Computer Vision (CV) adalah bidang ilmu yang berfokus pada bagaimana komputer dapat memahami gambar atau video.

Prosesnya biasanya melalui beberapa tahap:
- Akuisisi (mengambil gambar)
- Pengolahan (memperbaiki kualitas)
- Analisis (mencari pola)
- Pemahaman (mengambil keputusan)

---

## 2. Mengenal Citra Digital

Bagi manusia, gambar adalah visual.  
Bagi komputer, gambar adalah sekumpulan angka dalam bentuk matriks berukuran M x N.

Beberapa konsep dasar:
- Piksel: unit terkecil dari gambar
- Grayscale: citra hitam-putih dengan nilai 0–255
- RGB: citra berwarna dengan 3 kanal (Red, Green, Blue)

---

## 3. Persiapan Environment & Install Library di Visual Studi Code

Menggunakan virtual environment:

```bash
python -m venv cv-mikroplastik
```
<p align="center"> <img src="https://github.com/dwijaya24/microplastic-computer-vision/raw/main/buat_venv.png" width="500"/> </p>

Mengaktifkan virtual environment:

```bash
.\cv-mikroplastik\Scripts\Activate
```
<p align="center"> <img src="https://github.com/dwijaya24/microplastic-computer-vision/raw/main/mengaktifkan_venv.png" width="500"/> </p>

Menginstall Library yang Dibutuhkan:

```bash
python -m venv cv-mikroplastik
```
<p align="center"> <img src="https://github.com/dwijaya24/microplastic-computer-vision/raw/main/install_library.png" width="500"/> </p>

## 4. Membedah Dataset Mikroplastik
Mari kita mulai koding! Kita akan memuat gambar mikroplastik, mengecek resolusinya, dan melihat distribusi warnanya melalui histogram.

```bash
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. SET PATH DATASET
# =========================
dataset_dir = "dataset/train"

# Mengambil semua file gambar
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]

if len(image_files) == 0:
    print("Tidak ada gambar di folder dataset/train")
    exit()

# Menggunakan 1 gambar sebagai contoh
img_path = os.path.join(dataset_dir, image_files[0])
img = cv2.imread(img_path)

# =========================
# 2. VALIDASI GAMBAR
# =========================
if img is None:
    print("Error: Gambar tidak bisa dibaca")
    exit()

# =========================
# 3. KONVERSI WARNA
# =========================
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# 4. INFO DASAR GAMBAR
# =========================
height, width, channels = img.shape
print(f"File: {image_files[0]}")
print(f"Resolusi: {width} x {height} piksel")
print(f"Jumlah Kanal Warna: {channels}")

# =========================
# 5. GRAYSCALE & HISTOGRAM
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# =========================
# 6. VISUALISASI
# =========================
plt.figure(figsize=(15, 5))

# gambar asli
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Citra Mikroplastik')
plt.axis('off')

# grayscale
plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

# histogram
plt.subplot(1, 3, 3)
plt.plot(hist, color='black')
plt.title('Histogram Intensitas')
plt.xlabel('Nilai Piksel (0–255)')
plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()
```

## 5. Analisis Output
Berdasarkan histogram intensitas citra grayscale, distribusi nilai piksel terlihat terkonsentrasi pada bagian kiri histogram dengan puncak frekuensi tertinggi berada pada kisaran intensitas sekitar 10–30. Selain itu, sebagian besar piksel masih berada pada rentang intensitas rendah, yaitu sekitar 0–50, yang menunjukkan bahwa citra didominasi oleh area gelap. Area ini kemungkinan merepresentasikan latar belakang seperti air atau wadah pengambilan sampel. Setelah melewati nilai intensitas sekitar 50, frekuensi piksel mengalami penurunan yang cukup signifikan. Pada rentang 50–100, jumlah piksel mulai berkurang, sedangkan pada intensitas di atas 100 hingga 255, distribusi piksel cenderung sangat rendah dan relatif merata. 

Hal ini mengindikasikan bahwa hanya sebagian kecil bagian citra yang memiliki tingkat kecerahan lebih tinggi. Area dengan intensitas menengah hingga tinggi tersebut kemungkinan merupakan objek mikroplastik atau pantulan cahaya dari permukaan. Namun, karena jumlahnya relatif sedikit dibandingkan latar belakang, maka kontras antara objek dan latar belakang masih tergolong rendah. Distribusi yang tidak merata ini menunjukkan bahwa pemisahan objek secara langsung akan cukup sulit dilakukan. Oleh karena itu, diperlukan tahap preprocessing seperti peningkatan kontras (contrast enhancement), pengurangan noise, serta penerapan thresholding dengan nilai ambang yang berada di sekitar peralihan distribusi, yaitu pada kisaran 40–60. Dengan pendekatan tersebut, diharapkan objek mikroplastik dapat lebih terpisah dari latar belakang sehingga proses segmentasi dan analisis lanjutan dapat dilakukan dengan lebih akurat.
