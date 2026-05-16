# Klasifikasi Iris dengan Jaringan Saraf Tiruan (JST)

> **Praktikum Kecerdasan Buatan — Pertemuan 7**  
> Implementasi Neural Network menggunakan TensorFlow & Keras untuk klasifikasi dataset Iris

---

## Deskripsi

Program ini mengimplementasikan **Jaringan Saraf Tiruan (JST)** atau *Artificial Neural Network (ANN)* untuk mengklasifikasikan spesies bunga Iris menggunakan dataset UCI Iris. Model dibangun dengan arsitektur *fully connected (dense)* menggunakan framework TensorFlow/Keras.

Dataset Iris terdiri dari **150 sampel** dengan **4 fitur input** dan **3 kelas output**:
| Kelas | Label |
|-------|-------|
| Iris-setosa | 0 |
| Iris-versicolor | 1 |
| Iris-virginica | 2 |

---

## Struktur File

```
pertemuan 7/
├── jst2.py       # Script utama JST klasifikasi Iris
└── README.md     # Dokumentasi ini
```

---

## Arsitektur Model

Model menggunakan arsitektur **Sequential** dengan lapisan-lapisan berikut:

```
Input Layer     → 4 fitur (sepal length, sepal width, petal length, petal width)
Hidden Layer 1  → 1000 neuron, aktivasi ReLU
Hidden Layer 2  → 500 neuron, aktivasi ReLU
Hidden Layer 3  → 300 neuron, aktivasi ReLU
Output Layer    → 3 neuron, aktivasi Softmax (3 kelas Iris)
```

| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 32 |
| Rasio Train:Test | 80:20 |

---

##  Dependensi

Pastikan semua library berikut sudah terinstal:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

| Library | Kegunaan |
|---------|----------|
| `tensorflow` | Framework deep learning untuk membangun model JST |
| `keras` | API high-level untuk membangun layer model |
| `pandas` | Memuat dan memproses dataset CSV |
| `numpy` | Operasi numerik dan manipulasi array |
| `matplotlib` | Visualisasi grafik training history |
| `seaborn` | Visualisasi confusion matrix |
| `scikit-learn` | Preprocessing, split data, dan evaluasi model |

---

## Cara Menjalankan

1. **Clone / pastikan file tersedia** di direktori yang sesuai.

2. **Jalankan script:**
   ```bash
   python jst2.py
   ```

3. Program akan secara otomatis:
   - Mengunduh dataset Iris dari UCI repository
   - Melatih model selama 50 epoch
   - Menampilkan grafik *training & validation accuracy/loss*
   - Menampilkan *confusion matrix*

> **Catatan:** Program membutuhkan koneksi internet untuk mengunduh dataset dari `archive.ics.uci.edu`.

---

## Alur Program

```
1. Load Dataset Iris (dari URL UCI)
        ↓
2. Preprocessing
   - Pisahkan fitur (X) dan label (y)
   - Encode label (string → numerik)
   - Split data (80% train, 20% test)
        ↓
3. Bangun Model Sequential
   - 3 hidden layer (1000, 500, 300 neuron)
   - 1 output layer (3 neuron softmax)
        ↓
4. Compile & Training
   - Optimizer: Adam
   - Loss: Sparse Categorical Crossentropy
   - 50 epoch, batch size 32
        ↓
5. Evaluasi Model
   - Hitung Loss & Accuracy pada data test
   - Plot grafik training history
   - Tampilkan Confusion Matrix
        ↓
6. Prediksi Data Baru
   - Input manual 4 fitur dari user
   - Prediksi kelas spesies Iris
```

---

## Output yang Dihasilkan

- **Terminal:** Nilai `Loss` dan `Accuracy` pada data testing
- **Plot 1:** Grafik akurasi dan loss selama training
- **Plot 2:** Confusion Matrix hasil prediksi vs label sebenarnya

---

## Fungsi Prediksi Manual

Setelah model selesai dilatih, terdapat fungsi `predict_new_data()` yang memungkinkan user memasukkan data bunga Iris secara manual:

```
Masukkan sepal length: 5.1
Masukkan sepal width : 3.5
Masukkan petal length: 1.4
Masukkan petal width : 0.2

→ Prediksi kelas: Iris-setosa
```

---

