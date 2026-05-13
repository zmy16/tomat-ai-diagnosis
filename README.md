# 🍅 Tomat AI: Smart Diagnosis

**Sistem deteksi penyakit daun tomat berbasis _Convolutional Neural Network (CNN) From Scratch_** dengan antarmuka web interaktif.

Aplikasi ini memungkinkan pengguna mengunggah foto daun tomat, lalu secara instan mendapatkan diagnosis penyakit beserta tingkat keyakinan (confidence), rekomendasi penanganan, serta Top-3 prediksi alternatif.

---

## 📖 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur Model](#-arsitektur-model)
- [Kelas Penyakit yang Didukung](#-kelas-penyakit-yang-didukung)
- [Tech Stack](#-tech-stack)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi & Menjalankan Lokal](#-instalasi--menjalankan-lokal)
- [Cara Penggunaan](#-cara-penggunaan)
- [Deployment](#-deployment)
- [Catatan Teknis](#-catatan-teknis)
- [Lisensi & Kredit](#-lisensi--kredit)

---

## 🔍 Tentang Proyek

Tomat AI dikembangkan untuk membantu petani, peneliti, dan hobiis mendeteksi penyakit pada daun tomat secara cepat hanya dari gambar. Model dibangun **from scratch** (tanpa _pretrained weights_) untuk tujuan pembelajaran mendalam tentang arsitektur CNN, dilatih pada dataset citra daun tomat dengan 10 kelas (9 penyakit + 1 sehat).

Aplikasi dikemas dalam bentuk web app menggunakan **Streamlit** dengan tampilan dark-mode kustom agar nyaman digunakan di berbagai perangkat.

---

## ✨ Fitur Utama

- 🧠 **Dua Pilihan Model CNN** — Pengguna dapat memilih antara arsitektur **VGG-like** atau **ResNet-like** secara real-time.
- 🖼️ **Upload Gambar** — Mendukung format JPG, JPEG, dan PNG.
- ⚡ **Diagnosis Instan** — Prediksi kelas penyakit + confidence score langsung setelah upload.
- 💊 **Rekomendasi Penanganan** — Setiap kelas penyakit dilengkapi informasi singkat beserta saran penanganannya.
- 🏆 **Top-3 Prediksi** — Menampilkan 3 kelas paling mungkin lengkap dengan progress bar.
- 📊 **Probabilitas Semua Kelas** — Tabel lengkap probabilitas seluruh 10 kelas.
- ☁️ **Auto-Download Model** — File model (~200 MB) diunduh otomatis dari Google Drive via `gdown` saat pertama dijalankan, sehingga repo tetap ringan.
- 🎨 **UI Dark Mode Kustom** — Tampilan modern dengan CSS custom di Streamlit.
- 🎯 **Hasil Deterministik** — Random seed & op-determinism diatur agar prediksi konsisten setiap run.

---

## 🧠 Arsitektur Model

Dua arsitektur CNN dilatih **dari nol (from scratch)** menggunakan TensorFlow/Keras:

| Model | Tipe | Jumlah Layer | Akurasi Validasi |
|---|---|---|---|
| 🟡 **VGG-like CNN** | Sequential convolutional blocks | 34 layer | **94.80%** |
| 🟢 **ResNet-like CNN** | Dengan _residual / skip connections_ | ~40 layer | **92.30%** |

- **Input size**: 224 × 224 px (RGB)
- **Preprocessing**: Normalisasi `pixel / 255.0`, interpolasi `nearest` (identik dengan pipeline training di Google Colab).
- **Output**: Softmax 10 kelas.

Metadata akurasi disimpan di `model_info.json` sehingga badge akurasi pada UI dapat diperbarui tanpa mengubah kode aplikasi.

---

## 🏷️ Kelas Penyakit yang Didukung

Model dapat mengklasifikasikan daun tomat ke dalam **10 kelas**:

| # | Kelas | Penyebab |
|---|---|---|
| 1 | Bacterial Spot | Bakteri _Xanthomonas_ |
| 2 | Early Blight | Jamur _Alternaria solani_ |
| 3 | Late Blight | _Phytophthora infestans_ |
| 4 | Leaf Mold | Jamur _Passalora fulva_ |
| 5 | Septoria Leaf Spot | _Septoria lycopersici_ |
| 6 | Spider Mites | Hama tungau |
| 7 | Target Spot | Jamur _Corynespora cassiicola_ |
| 8 | Tomato Yellow Leaf Curl Virus | Virus (vektor kutu kebul _Bemisia tabaci_) |
| 9 | Tomato Mosaic Virus | Virus (kontak/alat) |
| 10 | ✅ Healthy | — (tanaman sehat) |

---

## 🛠️ Tech Stack

**Bahasa & Runtime**
- 🐍 Python **3.11** (ditentukan pada `runtime.txt`)

**Machine Learning / Deep Learning**
- 🧠 **TensorFlow / Keras** — Training & inference model CNN.
- 🔢 **NumPy** — Operasi numerik / array.
- 🖼️ **Pillow (PIL)** — Pembacaan & preprocessing gambar.

**Web Framework & UI**
- 🎈 **Streamlit** — Framework web app berbasis Python.
- 🎨 **CSS kustom** — Dark theme, komponen hero, cards, progress bar, chips.

**Data & Utilitas**
- 🐼 **pandas** — Menampilkan tabel probabilitas semua kelas.
- ☁️ **gdown** — Mengunduh model dari Google Drive secara otomatis.

**Lainnya**
- 📦 `model_info.json` — Konfigurasi metadata akurasi model.
- 🎲 Deterministic ops (`TF_DETERMINISTIC_OPS`, `set_random_seed(42)`) untuk konsistensi prediksi.

---

## 📂 Struktur Proyek

```
tomat-ai-diagnosis/
├── app.py                 # Aplikasi utama Streamlit (UI + inference)
├── model_info.json        # Metadata akurasi tiap model
├── requirements.txt       # Dependency Python
├── runtime.txt            # Versi Python runtime (untuk deployment)
└── README.md              # Dokumentasi proyek (file ini)
```

> 📝 **Catatan**: File model `.keras` (~200 MB per file) **tidak disertakan** dalam repo. File akan diunduh otomatis dari Google Drive saat aplikasi pertama kali dijalankan.

---

## ⚙️ Instalasi & Menjalankan Lokal

### 1. Prasyarat

- Python **3.11** (disarankan menggunakan `pyenv` / `conda` / virtualenv).
- Koneksi internet (untuk download model di first run).
- ~500 MB ruang disk kosong (untuk kedua file model).

### 2. Clone Repository

```bash
git clone https://github.com/zmy16/tomat-ai-diagnosis.git
cd tomat-ai-diagnosis
```

### 3. Buat Virtual Environment (opsional, direkomendasikan)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 4. Install Dependency

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

> ⏳ Pada run pertama, sistem akan mengunduh file model (~200 MB) dari Google Drive. Harap tunggu hingga proses selesai.

---

## 🖱️ Cara Penggunaan

1. **Buka aplikasi** di browser setelah menjalankan `streamlit run app.py`.
2. **Pilih model AI** yang ingin digunakan (VGG-like atau ResNet-like).
3. **Upload gambar** daun tomat (JPG/JPEG/PNG) lewat tombol uploader.
4. Tunggu beberapa detik — hasil akan ditampilkan:
   - 🔬 **Diagnosis utama** + tingkat keyakinan.
   - 💊 **Rekomendasi penanganan** sesuai penyakit.
   - 🏆 **Top-3 prediksi** dengan progress bar.
   - 📊 **Probabilitas semua kelas** (expandable).

---

## 🚀 Deployment

Proyek ini siap di-deploy ke **Streamlit Community Cloud**, **Hugging Face Spaces**, atau platform sejenis.

- `requirements.txt` → dependency Python.
- `runtime.txt` → versi Python (`python-3.11`).
- `app.py` → entry point.
- Model besar di-_offload_ ke Google Drive, didownload otomatis via `gdown` saat startup.

### Contoh Deploy ke Streamlit Cloud

1. Push repo ke GitHub.
2. Masuk ke [share.streamlit.io](https://share.streamlit.io).
3. Connect repo → pilih branch & `app.py` sebagai main file.
4. Deploy. Streamlit Cloud akan membaca `requirements.txt` & `runtime.txt` secara otomatis.

---

## 🧪 Catatan Teknis

- **Preprocessing identik Colab**: Menggunakan `tensorflow.keras.utils.load_img(..., interpolation='nearest')` lalu `img_to_array() / 255.0` agar hasil inference konsisten dengan training.
- **Caching model**: `@st.cache_resource` memastikan model hanya di-load sekali per sesi.
- **Sidebar disembunyikan**: UI dibuat full-width dengan CSS override.
- **Environment variables** yang di-set di awal:
  - `TF_CPP_MIN_LOG_LEVEL=2` — mengurangi noise log TensorFlow.
  - `TF_DETERMINISTIC_OPS=1` — hasil deterministik.
  - `PYTHONHASHSEED=42` — seed Python hash.
  - `TF_ENABLE_ONEDNN_OPTS=0` — menonaktifkan oneDNN untuk konsistensi numerik.

---

## 📜 Lisensi & Kredit

- 👨‍💻 **Dibuat oleh**: Raid
- 🎓 **Institusi**: BINUS University
- 📅 **Tahun**: 2026
- 🌐 **Repository**: [github.com/zmy16/tomat-ai-diagnosis](https://github.com/zmy16/tomat-ai-diagnosis)

> ⚠️ **Disclaimer**: Hasil diagnosis dari aplikasi ini bersifat _supportive_ dan **bukan pengganti konsultasi ahli pertanian / patologi tanaman**. Selalu verifikasi dengan pakar sebelum mengambil tindakan pada tanaman di lapangan.

---

<p align="center">
  Dibuat dengan ❤️ menggunakan <b>TensorFlow</b> & <b>Streamlit</b>
</p>
