# Detection of Fruit Ripeness Using Image Processing

Project ini merupakan sistem deteksi kematangan buah menggunakan metode:

- Image Processing
- CNN (Convolutional Neural Network)
- VGG16
- TensorFlow
- OpenCV

Dataset yang digunakan berasal dari Fruits-360 Dataset dari Kaggle.

---

# Download Dataset

Download dataset melalui Kaggle:

https://www.kaggle.com/datasets/moltean/fruits

atau menggunakan Python:

```python
import kagglehub

path = kagglehub.dataset_download(
    "moltean/fruits"
)

print("Path Dataset:", path)
```
Pastikan menggunakan Python 3.11
---

---

# Git LFS (Penting)

Project ini menggunakan file model AI dengan format:

```text
models/fruit_ripeness_model.h5
models/tomato_ripeness_model.h5
```

Karena ukuran model cukup besar (>100MB), repository menggunakan **Git Large File Storage (Git LFS)**.

Jika ingin clone project dan menggunakan model yang sudah ditraining tanpa melakukan training ulang, install Git LFS terlebih dahulu.

## Install Git LFS

Download:

https://git-lfs.com/

Setelah install, jalankan:

```bash
git lfs install
```

Clone repository:

```bash
git clone <repository-url>
```

Masuk folder project:

```bash
cd Detection-of-Fruit-Ripeness-Using-Image-Processing
```

Download file model LFS:

```bash
git lfs pull
```

Setelah selesai, file model `.h5` akan otomatis tersedia.

Dengan Git LFS, pengguna tidak perlu melakukan training ulang model dari awal.

---

# Catatan

Jika file model belum muncul setelah clone repository:

```bash
git lfs pull
```

Pastikan Git LFS sudah terinstall dengan benar.

---


# Struktur Folder

```text
dataset/
├── train/
│   ├── matang/
│   └── mentah/
│
└── valid/
    ├── matang/
    └── mentah/
```

---

# Penjelasan Dataset

Saat ini project hanya menggunakan dataset apel.

Contoh:
- apel matang untuk train & valid
- apel mentah untuk train & valid

Jika ingin menambahkan dataset buah lain seperti:
- pisang
- mangga
- jeruk
- dll

maka cukup tambahkan gambar ke folder train & valid:

```text
dataset/train/matang
dataset/train/mentah

dataset/valid/matang
dataset/valid/mentah
```

Lalu pisahkan gambar berdasarkan kategori:
- buah matang → folder `matang`
- buah mentah → folder `mentah`

---

# Install Library

Install semua library terlebih dahulu:

```bash
pip install tensorflow opencv-python matplotlib scikit-learn pillow kagglehub
```

---

# Cara Menjalankan Project

## 1. Training Model

Jalankan:

```bash
py -3.11 train.py
```

Fungsi:
- membaca dataset
- preprocessing gambar
- training CNN VGG16
- menyimpan model AI

Setelah training selesai akan muncul file:

```text
models/fruit_ripeness_model.h5
```

---

## 2. Prediksi Gambar

Masukkan gambar yang ingin dideteksi dengan nama:

```text
input_image.jpg
```

dan letakkan satu folder dengan:

```text
predict.py
```

Contoh:

```text
project/
├── predict.py
├── input_image.jpg
```

---

## 3. Jalankan Prediksi

Run:

```bash
py -3.11 predict.py
```

Program akan:
- membaca gambar
- mendeteksi tingkat kematangan buah
- menampilkan hasil prediksi
- menampilkan confidence prediction

Contoh output:

```text
Hasil: MATANG
Confidence: 92.50%
```

---

# Cara Mengganti Buah

Jika sebelumnya menggunakan apel lalu ingin mengganti ke buah lain seperti pisang:

1. Tambahkan dataset pisang ke folder:

```text
dataset/train
dataset/valid
```

2. Pisahkan gambar ke folder:
- `matang`
- `mentah`

3. Jalankan ulang:

```bash
py -3.11 train.py
```

agar model mempelajari dataset baru.

4. Ganti gambar:

```text
input_image.jpg
```

dengan gambar buah yang ingin dideteksi.

5. Jalankan kembali:

```bash
py -3.11 predict.py
```

---

# Alur Sistem

```text
Dataset Buah
      ↓
Preprocessing
      ↓
Training CNN VGG16
      ↓
Model AI (.h5)
      ↓
Prediksi Gambar Baru
```

---

# Teknologi Yang Digunakan

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

---
