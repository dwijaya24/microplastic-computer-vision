import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. SET PATH DATASET
# =========================
dataset_dir = "dataset/train"

# ambil semua file gambar
image_files = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]

if len(image_files) == 0:
    print("Tidak ada gambar di folder dataset/train")
    exit()

# pakai 1 gambar sebagai contoh (bukan semua)
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