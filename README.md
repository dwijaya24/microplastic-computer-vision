# Dari Piksel ke Mikroplastik: Memahami Computer Vision Secara Nyata

Computer vision pada dasarnya adalah cara komputer memahami gambar sebagai kumpulan piksel, lalu mengolahnya menjadi informasi yang bisa dikenali.
Di tulisan ini saya pakai dataset dari Kaggle (Microplastic Dataset for Computer Vision) sebagai contoh. Dataset ini sudah melalui beberapa tahap preprocessing, seperti penyesuaian orientasi gambar (auto-orient), pemotongan area fokus (sekitar 30–85% horizontal dan 15–85% vertikal), serta penyaringan data agar hanya gambar dengan anotasi yang digunakan.
Data yang tersedia juga dilengkapi dengan bounding box, sehingga setiap partikel mikroplastik dalam gambar punya penanda lokasi yang jelas.
Dengan struktur seperti ini, pengolahan gambar bisa diarahkan untuk mengenali pola dan posisi mikroplastik secara lebih terukur.

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

## 3. Persiapan Environment

Menggunakan virtual environment:

```bash
python -m venv cv-mikroplastik
---
Mengaktifkan virtual environment:
```bash
.\cv-mikroplastik\Scripts\activate
---
Menginstall Library:
```bash
pip install opencv-python numpy matplotlib
