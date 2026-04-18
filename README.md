# 🎯 Analisis Sentimen Komentar YouTube

Aplikasi web untuk menganalisis sentimen komentar YouTube menggunakan **K-Means Clustering** dan **TF-IDF (Term Frequency-Inverse Document Frequency)**. Aplikasi ini memungkinkan pengguna untuk scraping langsung dari YouTube atau mengupload file Excel untuk analisis sentimen yang komprehensif.

---

## ✨ Fitur Utama

### 1. **Scraping Komentar YouTube** 
- Scraping langsung dari URL video YouTube menggunakan YouTube API v3
- Ambil hingga 1000 komentar per video
- Informasi lengkap: teks komentar, author, like count, dan waktu publish

### 2. **Upload File Excel**
- Dukungan format: `.xlsx` dan `.xls`
- Kolom yang didukung: `Comment`, `AuthorComment`, `comment`, `text`, `Komentar`, `comments`
- Maksimal ukuran file: 16MB

### 3. **Analisis Sentimen Otomatis**
- **Preprocessing**: Pembersihan teks, URL removal, tokenization
- **TF-IDF Vectorization**: Ekstraksi fitur dari teks
- **K-Means Clustering**: Pengelompokan komentar menjadi 3 cluster
- **Sentiment Labeling**: Klasifikasi otomatis menjadi Positif, Negatif, atau Netral

### 4. **Visualisasi Data Komprehensif**
- **Pie Chart**: Distribusi persentase sentimen
- **Bar Chart**: Perbandingan jumlah sentimen
- **Doughnut Chart**: Distribusi cluster
- **Line Chart**: Tren sentimen over time

### 5. **Tabel Hasil Detil**
- Menampilkan 50 sample pertama dari hasil analisis
- Badge warna untuk setiap sentimen
- Informasi cluster untuk setiap komentar

---

## 🛠️ Tech Stack

### Frontend
- **HTML5** - Struktur halaman
- **CSS3** - Styling modern dengan gradient dan glassmorphism
- **JavaScript (Vanilla)** - Interaktivitas dan API calls
- **Chart.js v3.9.1** - Visualisasi data

### Backend
- **Python 3.12** - Runtime
- **Flask 3.0.0** - Web framework
- **Flask-CORS 4.0.0** - Cross-origin request handling
- **Pandas 2.2.0** - Data manipulation
- **NumPy 1.26.4** - Numerical computing
- **SciPy 1.13.0** - Scientific computing
- **Requests 2.31.0** - HTTP library
- **Openpyxl 3.1.2** - Excel file reading
- **Gunicorn 21.2.0** - WSGI HTTP server

### Deployment
- **Vercel** - Serverless hosting
- **YouTube Data API v3** - For comment scraping

---

## 📋 Prasyarat

Sebelum memulai, pastikan Anda memiliki:

1. **Python 3.12+** - [Download](https://www.python.org/downloads/)
2. **Git** - [Download](https://git-scm.com/)
3. **YouTube API Key** - [Dapatkan gratis di Google Cloud Console](#-cara-mendapatkan-youtube-api-key)
4. **Modern Web Browser** - Chrome, Firefox, Safari, Edge

---

## 🚀 Instalasi & Setup

### 1. Clone Repository
```bash
git clone https://github.com/TendriZ/youtube-sentiment.git
cd youtube-sentiment
```

### 2. Buat Virtual Environment (Opsional tapi Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables (Opsional)
Buat file `.env` di root directory (jika diperlukan):
```
FLASK_ENV=production
FLASK_DEBUG=False
```

### 5. Jalankan Aplikasi Lokal
```bash
python app.py
```

Server akan berjalan di `http://localhost:5000`

---

## 🔑 Cara Mendapatkan YouTube API Key

### Step 1: Buka Google Cloud Console
1. Kunjungi [Google Cloud Console](https://console.cloud.google.com/)
2. Login dengan akun Google Anda

### Step 2: Buat Project Baru
1. Klik dropdown project di bagian atas
2. Klik "NEW PROJECT"
3. Masukkan nama project, misal: "YouTube Sentiment Analysis"
4. Klik "CREATE"

### Step 3: Enable YouTube Data API v3
1. Cari "YouTube Data API v3" di search bar
2. Klik pada API tersebut
3. Klik tombol "ENABLE"

### Step 4: Buat API Key
1. Klik "Create Credentials" atau buka menu "Credentials"
2. Pilih "API Key"
3. Copy API key yang sudah dibuat
4. **Opsional**: Batasi kunci ke HTTP referrers untuk keamanan

### Step 5: Paste API Key di Aplikasi
Saat menggunakan aplikasi, paste API key di field "YouTube API Key" pada tab Scraping YouTube.

⚠️ **Catatan Penting**: 
- YouTube API memiliki quota harian (default 10,000 units/hari)
- Setiap request menggunakan beberapa quota units
- Scraping 500 komentar ≈ 500-1000 units

---

## 📖 Cara Menggunakan

### Metode 1: Scraping Langsung dari YouTube

1. Buka aplikasi di browser
2. Tab "Scraping YouTube" sudah aktif
3. **Masukkan URL Video**:
   - Contoh: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
   - Atau: `https://youtu.be/dQw4w9WgXcQ`
4. **Masukkan YouTube API Key**
5. **Atur Maksimal Komentar** (default: 500)
6. Klik tombol **"Scrape & Analisis Komentar YouTube"**
7. Tunggu proses selesai (1-5 menit tergantung jumlah komentar)
8. Lihat hasil analisis dengan charts dan tabel

### Metode 2: Upload File Excel

1. Klik tab **"Upload Excel"**
2. Siapkan file Excel dengan struktur:
   ```
   | Comment/Komentar |
   |------------------|
   | Komentar 1       |
   | Komentar 2       |
   | ...              |
   ```
   Nama kolom bisa: `Comment`, `AuthorComment`, `comment`, `text`, `Komentar`, atau `comments`

3. Klik **"Pilih File Excel"** dan select file Anda
4. Klik **"Mulai Analisis Sentimen"**
5. Tunggu proses selesai
6. Lihat hasil analisis

---

## 📊 Interpretasi Hasil

### Distribusi Sentimen
- **Positif (Hijau)**: Komentar dengan kata-kata positif (semangat, dukung, bagus, senang, dll)
- **Negatif (Merah)**: Komentar dengan kata-kata negatif (marah, kecewa, buruk, bodoh, dll)
- **Netral (Biru)**: Komentar yang tidak jelas polaritasnya

### Cluster
- Sistem K-Means membagi komentar menjadi **3 cluster** berdasarkan kesamaan TF-IDF vector
- Setiap cluster diberi label sentimen berdasarkan dominasi kata-kata positif/negatif

### Metrik
- **Total Komentar**: Jumlah komentar yang dianalisis
- **Positif/Negatif/Netral**: Jumlah dan persentase per sentimen

---

## 🏗️ Struktur Proyek

```
youtube-sentiment/
├── app.py                 # Flask backend (main)
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel deployment config
├── runtime.txt           # Python version specification
├── public/
│   └── index.html        # Frontend (single page app)
├── uploads/              # Folder untuk uploaded files (temp)
├── .gitignore           # Git ignore rules
└── README.md            # Dokumentasi (file ini)
```

---

## 🔌 API Endpoints

### 1. **POST /api/scrape**
Scrape dan analisis komentar dari YouTube.

**Request Body:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=...",
  "api_key": "YOUR_YOUTUBE_API_KEY",
  "max_comments": 500,
  "method": "api"
}
```

**Response Success (200):**
```json
{
  "totalComments": 500,
  "videoId": "dQw4w9WgXcQ",
  "scrapedAt": "2026-04-18T10:30:00.000Z",
  "sentiments": {
    "Positif": 150,
    "Negatif": 200,
    "Netral": 150
  },
  "clusters": {
    "Cluster 0": 200,
    "Cluster 1": 150,
    "Cluster 2": 150
  },
  "sampleData": [
    {
      "comment": "Komentar bagus sekali...",
      "sentiment": "Positif",
      "cluster": 0
    }
  ]
}
```

**Response Error (400):**
```json
{
  "error": "Pesan error deskriptif"
}
```

### 2. **POST /api/upload**
Upload dan analisis file Excel.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `file` (Excel file)

**Response Success (200):**
```json
{
  "totalComments": 100,
  "sentiments": {...},
  "clusters": {...},
  "sampleData": [...]
}
```

### 3. **GET /api/health**
Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy",
  "message": "Basic Flask API is running",
  "timestamp": "2026-04-18T10:30:00.000Z"
}
```

---

## 🔄 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT: YouTube URL atau Excel File                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA COLLECTION                                          │
│    - YouTube API: Fetch comments via commentThreads endpoint
│    - Excel: Read file dengan pandas                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PREPROCESSING                                            │
│    - Lowercase text                                         │
│    - Remove URLs, mentions, hashtags                        │
│    - Remove special characters                              │
│    - Filter words dengan length < 3                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. FEATURE EXTRACTION (TF-IDF)                              │
│    - Calculate word frequency                               │
│    - Compute TF (Term Frequency)                            │
│    - Compute IDF (Inverse Document Frequency)               │
│    - Create sparse matrix (N × V)                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CLUSTERING (K-Means)                                     │
│    - Initialize k=3 centroids                               │
│    - Iterate hingga convergence                             │
│    - Assign documents ke nearest centroid                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SENTIMENT LABELING                                       │
│    - Analyze centroid word composition                      │
│    - Count positive vs negative words                       │
│    - Label: Positif / Negatif / Netral                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. OUTPUT: Results dengan visualisasi                       │
│    - Statistics, Charts, Table                              │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Konfigurasi Advanced

### Mengubah Jumlah Cluster
Edit di `app.py`, function `analyze_sentiment_backend()`:
```python
labels, centroids = kmeans(tfidf_matrix, k=5)  # Ubah k dari 3 menjadi 5
```

### Mengubah Min Document Frequency
Edit di `app.py`, function `calculate_tfidf()`:
```python
def calculate_tfidf(processed_comments, max_features=2000, min_df=2):
    # min_df=2 berarti kata harus muncul minimal 2 dokumen
```

### Menambah Sentiment Words
Edit di `app.py`, function `label_sentiments()`:
```python
neg_words = {
    'hancur', 'bakar', ...
    # Tambah kata negatif di sini
}

pos_words = {
    'semangat', 'dukung', ...
    # Tambah kata positif di sini
}
```

---

## 🚨 Troubleshooting

### Error: "YouTube API key diperlukan"
**Solusi**: 
- Pastikan Anda sudah mendapatkan API key dari Google Cloud Console
- Paste API key di field input
- Pastikan API key sudah valid dan quota belum habis

### Error: "API quota exceeded"
**Solusi**:
- YouTube API memiliki quota harian 10,000 units
- Coba lagi besok atau request quota increase di Google Cloud Console
- Reduce `max_comments` untuk menghemat quota

### Error: "Video tidak ditemukan atau komentar dinonaktifkan"
**Solusi**:
- Pastikan URL video valid dan public
- Cek apakah channel sudah mengaktifkan komentar
- Beberapa video mungkin tidak mengizinkan akses API

### Error: "Tidak ada file yang diupload"
**Solusi**:
- Pilih file Excel terlebih dahulu
- Pastikan file format `.xlsx` atau `.xls`
- File size tidak boleh melebihi 16MB

### Error: "Kolom komentar tidak ditemukan"
**Solusi**:
- Pastikan Excel memiliki kolom dengan nama: Comment, comment, Komentar, AuthorComment, text, atau comments
- Lihat pesan error untuk kolom yang tersedia
- Rename kolom Anda menjadi salah satu nama yang didukung

### Error: "Tidak dapat terhubung ke server"
**Solusi**:
- Pastikan Flask backend sudah jalan di `http://localhost:5000`
- Jalankan: `python app.py`
- Check apakah port 5000 tidak terpakai aplikasi lain
- Jika deploy di Vercel, pastikan deployment sudah successful (status Ready)

---

## 📈 Performance Tips

1. **Untuk YouTube Scraping**:
   - Maksimal komentar: 500-1000 (lebih tinggi = lebih lama)
   - Waktu proses: ~2-5 menit tergantung jumlah komentar

2. **Untuk File Upload**:
   - Gunakan file Excel dengan jumlah baris < 5000
   - File size < 5MB untuk hasil optimal

3. **Browser**:
   - Gunakan Chrome atau Edge untuk performance terbaik
   - Disable extensions yang berat

---

## 🔒 Keamanan

⚠️ **PENTING**:
1. **Jangan share YouTube API Key** - Treat it like a password
2. **File uploads**: Hanya `.xlsx` dan `.xls` yang diterima
3. **Max file size**: 16MB
4. **API rate limiting**: Implemented untuk prevent abuse

---

## 📝 Sentiment Keywords Reference

### Positive Words 🟢
semangat, dukung, mantap, hebat, merdeka, bersatu, lindungi, bagus, baik, senang, bangga, optimis, sukses, berhasil, luar, biasa, keren, amazing, love

### Negative Words 🔴
hancur, bakar, bubar, korup, marah, sengsara, rusak, anarkis, jahat, bodoh, tolol, benci, kecewa, buruk, jelek, gagal, sedih, stress, mampus, parah, anjing

---

## 📞 Support & Contact

- **GitHub**: [TendriZ/youtube-sentiment](https://github.com/TendriZ/youtube-sentiment)
- **Issues**: Report bugs di GitHub Issues
- **Author**: Raka (TendriZ)

---

## 📄 License

Project ini open source. Bebas digunakan untuk keperluan personal maupun komersial.

---

## 🙏 Credits

- **Chart.js** - Untuk visualisasi data yang indah
- **Flask** - Web framework Python
- **Pandas & NumPy** - Data processing
- **YouTube Data API** - Untuk akses data komentar
- **Google Cloud** - Untuk infrastruktur API

---

## 🎯 Roadmap Future

- [ ] Support untuk bahasa lain (English, Arabic, etc)
- [ ] Real-time sentiment analysis
- [ ] Export results ke PDF/CSV
- [ ] Database integration untuk history analisis
- [ ] Advanced NLP dengan transformer models
- [ ] Mobile app version
- [ ] Docker containerization

---

## 📝 Changelog

### v1.0.0 - 2026-04-18
- ✅ Initial release
- ✅ YouTube scraping functionality
- ✅ Excel file upload
- ✅ K-Means clustering
- ✅ TF-IDF vectorization
- ✅ Sentiment analysis dengan Indonesian keywords
- ✅ Data visualization dengan Chart.js
- ✅ Vercel deployment
- ✅ Responsive design

---

**Happy Analyzing! 🎉**
