# Transaction Anomaly Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://transaction-anomaly-dashboard-areswijaya.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat)

Proyek belajar saya untuk berlatih alur data science dari awal sampai tampil di web: membersihkan data, mengelompokkan nasabah, menandai transaksi yang tidak biasa, lalu menyajikannya di dashboard interaktif.

> **English summary:** A learning project on 2,516 **synthetic** bank transactions. K-Means groups customers into two behavioral segments, Isolation Forest flags unusual transactions, and a simple rule flags repeated login attempts. All results are delivered in a Streamlit dashboard. The data has no real fraud labels, so flagged transactions are candidates for review, not confirmed fraud.

## Yang Saya Kerjakan

1. **Membersihkan data**: membuang duplikat, mengisi nilai kosong (median untuk angka, modus untuk teks).
2. **Segmentasi dengan K-Means** (k = 2, Silhouette Score 0.2634) pada 5 fitur perilaku:
   - **Nasabah Muda Aktif** (34.30%): usia rata-rata sekitar 27 tahun, saldo lebih kecil, porsi transaksi terhadap saldo lebih besar.
   - **Nasabah Mapan Konservatif** (65.70%): usia rata-rata sekitar 54 tahun, saldo lebih besar, porsi transaksi lebih kecil.
3. **Penanda anomali dengan Isolation Forest** pada 6 fitur (5 fitur perilaku + nominal per detik). Model diminta menandai sekitar 5% transaksi yang paling tidak biasa, lalu tiap transaksi diberi skor 0-100.
4. **Aturan risiko login**: `LoginAttempts >= 3` menyaring 93 transaksi (3.70%). Ini aturan sederhana, bukan hasil model.
5. **Dashboard Streamlit** dengan 5 tab (`Ringkasan`, `Segmen`, `Anomali & Login`, `Coba Transaksi`, `Catatan`), filter profesi/kanal/tipe/kota, grafik, tabel yang bisa diunduh, dan form untuk mencoba satu transaksi.

## Temuan Utama

Beberapa pola yang terlihat pada 2,516 transaksi hasil pembersihan:

- **Nominal besar dan rasio saldo tinggi:** rata-rata nominal transaksi anomali 1,019.27 USD (median 1,043.66 USD), sedangkan transaksi normal 258.76 USD (median 199.38 USD). Rata-rata `Balance_Ratio` pada kelompok anomali 1.35 (median 0.74) dibanding 0.14 pada transaksi normal; 58.7% transaksi anomali (74 dari 126) memiliki `Balance_Ratio > 0.5`, dibanding 6.3% (150 dari 2,390) pada transaksi normal.
- **Kecepatan nominal transaksi:** rata-rata `Amount_Per_Second` pada kelompok anomali 19.33 USD/detik dibanding 3.24 USD/detik pada transaksi normal.
- **Sebaran antarsegmen:** tingkat anomali di segmen `Nasabah Muda Aktif` mencapai 9.62% (83 dari 863 transaksi), lebih tinggi dibanding `Nasabah Mapan Konservatif` sebesar 2.60% (43 dari 1,653 transaksi).
- **Irisan dengan aturan login kecil:** hanya 3 transaksi (0.12%) yang sekaligus ditandai Isolation Forest dan memiliki `LoginAttempts >= 3`. Dua mekanisme ini menyaring kandidat yang sebagian besar berbeda.

Catatan: `Balance_Ratio` dan `Amount_Per_Second` dipakai sebagai fitur input Isolation Forest pada data sintetis, jadi angka di atas menggambarkan karakter outlier model, bukan pola fraud asli.

### Contoh Urutan Tinjauan

Jika daftar kandidat di tab `Anomali & Login` dipakai untuk pemeriksaan manual, urutannya bisa dibuat seperti ini:

1. **Prioritas 1 (irisan keduanya, 3 transaksi):** cek transaksi yang sekaligus anomali perilaku dan punya `LoginAttempts >= 3`.
2. **Prioritas 2 (risiko login saja, 90 transaksi):** fokus ke verifikasi akses dan upaya masuk berulang.
3. **Prioritas 3 (anomali perilaku saja, 123 transaksi):** urutkan dari `Anomaly_Score` tertinggi, lalu periksa nominal besar dibanding saldo atau durasi singkat.

Ini hanya contoh alur kerja untuk demo analisis, bukan prosedur operasional produksi.

## Apa yang Saya Pelajari

- Nilai Silhouette yang rendah (0.2634) berarti segmen saling tumpang tindih. Saya melaporkannya apa adanya, tidak dipoles.
- Tanpa label fraud asli, hasil model tidak bisa dihitung akurasinya. "Anomali" di sini hanya berarti "tidak biasa".
- Jumlah anomali ditentukan oleh pengaturan `contamination` yang saya pilih. Saya mengujinya di 0.03, 0.05, dan 0.07 untuk melihat seberapa banyak hasilnya berubah.
- Fitur yang saling menumpuk sebaiknya dipisah. `LoginAttempts` tidak dipakai di K-Means maupun Isolation Forest, supaya aturan login berdiri sendiri.

## Keterbatasan

- Data sintetis (dataset latihan Dicoding, adaptasi Kaggle *Bank Transaction Dataset for Fraud Detection*), transaksi kota-kota Amerika Serikat dalam USD.
- Tidak ada label fraud asli, jadi precision dan recall tidak bisa dihitung.
- Dua segmen pada dasarnya memisahkan nasabah muda bersaldo kecil dan nasabah lebih tua bersaldo besar.

## Struktur Folder

```text
transaction-anomaly-dashboard/
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw.xlsx
│   └── data_clustering_clean.csv
├── models/
│   ├── scaler_clustering.joblib
│   ├── kmeans_model.joblib
│   ├── anomaly_model.joblib
│   └── k_evaluation.csv
├── notebooks/
│   └── clustering.ipynb
├── tests/
│   └── test_pipeline.py
├── app.py
├── requirements.txt
├── LICENSE
├── .gitattributes
├── .gitignore
└── README.md
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Dashboard terbuka di `http://localhost:8501`.

Semua file di `data/` (selain `raw.xlsx`) dan `models/` dihasilkan oleh `notebooks/clustering.ipynb`. Jalankan notebook dari atas ke bawah (*Run All*) untuk membuatnya ulang. Jika ingin menjalankan ulang notebook, pasang paket tambahannya terlebih dulu:

```bash
pip install matplotlib seaborn openpyxl jupyter
```

Pengujian sederhana:

```bash
python -m unittest discover -s tests -v
```

## Kontak

Ilfa Antaries Yusuf Wijaya
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
