# Dashboard Segmentasi Transaksi & Indikator Risiko Login

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://transaction-anomaly-dashboard-areswijaya.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67%2B-FF4B4B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75?style=flat)

Aplikasi web berbasis Streamlit untuk melihat hasil segmentasi perilaku transaksi nasabah memakai K-Means (`k = 2`) dan memantau transaksi berisiko login lewat aturan eksplisit (`LoginAttempts >= 3`).

## Tentang Proyek

Proyek ini membagi 2.516 data transaksi nasabah menjadi dua segmen perilaku menggunakan **K-Means Clustering** (`k = 2`, Silhouette Score `0.2634`) pada 5 fitur numerik perilaku (`TransactionAmount`, `CustomerAge`, `TransactionDuration`, `AccountBalance`, `Balance_Ratio`), tanpa menyertakan `LoginAttempts` di dalam clustering.

Dua segmen perilaku yang terbentuk:
1. **Cluster 0 — Nasabah Muda Aktif (`863` transaksi / `34,30%`)**: Usia rata-rata `27,30` tahun (dominan `Student`), saldo lebih rendah (`$1.584,53`), nominal transaksi rata-rata `$347,07`, dan rasio transaksi terhadap saldo lebih tinggi (`Balance_Ratio` `0,47`). Terdapat `33` transaksi berisiko login (`LoginAttempts >= 3`).
2. **Cluster 1 — Nasabah Mapan Konservatif (`1.653` transaksi / `65,70%`)**: Usia rata-rata `53,71` tahun (dominan `Retired`), saldo tinggi (`$6.955,81`), nominal transaksi rata-rata `$270,63`, dan `Balance_Ratio` rendah (`0,05`). Terdapat `60` transaksi berisiko login (`LoginAttempts >= 3`).

Indikator risiko login ditangani terpisah melalui aturan eksplisit **`LoginAttempts >= 3`**, yang menyaring **`93` transaksi (`3,70%`)** dengan 3–5 kali percobaan login. Pemetaan transaksi baru pada dashboard dilakukan langsung memakai `StandardScaler` + `KMeans` tanpa model klasifikasi perantara.

## Sumber Dataset

File mentah `data/raw.xlsx` (`2.537` baris sebelum pembersihan duplikat menjadi `2.516` baris) berasal dari **dataset latihan Dicoding** yang merupakan adaptasi dari **Kaggle *Bank Transaction Dataset for Fraud Detection*** — berisi data **sintetis** transaksi perbankan di kota-kota Amerika Serikat dalam satuan USD, **tanpa label *fraud* asli (*ground truth*)**.

## Keterbatasan Metodologi

- **Pemisahan cluster masih lemah:** `Silhouette Score` tertinggi berada pada `k = 2` dengan nilai `0.2634` (rentang uji `k = 2..8`), menunjukkan bahwa fitur perilaku numerik pada dataset ini saling tumpang tindih.
- **Tidak ada label *fraud* asli:** Kemampuan mendeteksi *fraud* nyata tidak dapat dibuktikan ataupun diukur dengan metrik klasifikasi.
- **Indikator risiko login berbasis aturan:** Ambang batas `LoginAttempts >= 3` adalah aturan eksplisit sederhana, bukan hasil prediksi *machine learning*.

## Fitur Dashboard

- **Filter Sidebar**: Menyaring data berdasarkan profesi, kanal transaksi, tipe transaksi, dan kota, lengkap dengan tombol reset.
- **Ringkasan**: Menampilkan KPI utama (termasuk jumlah `Risiko Login (>= 3x)`), distribusi jumlah transaksi per segmen, komposisi kategori, dan tabel profil rata-rata.
- **Visualisasi Cluster**: Scatter plot antarfitur yang sumbu X dan Y-nya bisa dipilih, radar chart profil 5 fitur perilaku, serta rekomendasi bisnis untuk 2 segmen.
- **Risiko Login (Aturan)**: Memantau distribusi `LoginAttempts` per segmen, sebaran transaksi dengan `LoginAttempts >= 3`, serta tombol unduh CSV (`risiko_login_tinggi.csv`).
- **Simulasi & Cek Risiko**: Form interaktif untuk memetakan transaksi baru langsung ke `StandardScaler` + `KMeans`, memeriksa status aturan `LoginAttempts >= 3`, dan melihat jarak Euclidean ke setiap centroid.
- **Evaluasi & Keterbatasan**: Menampilkan grafik Elbow Method & Silhouette Score (`k = 2..8`), tabel centroid skala asli, serta catatan sumber data dan keterbatasan metodologi.

## Struktur Folder

```text
transaction-anomaly-dashboard/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── data_clustering_clean.csv
│   └── raw.xlsx
├── models/
│   ├── kmeans_model.joblib
│   └── scaler_clustering.joblib
├── notebooks/
│   └── clustering.ipynb
├── tests/
│   └── test_pipeline.py
├── app.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Cara Menjalankan di Komputer Lokal

Pastikan Python sudah terpasang, lalu jalankan perintah berikut dari dalam folder proyek:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Setelah dijalankan, dashboard bisa dibuka di browser melalui `http://localhost:8501`.

Untuk menjalankan verifikasi otomatis:

```bash
python -m unittest discover -s tests -v
```

## Kontak

**Ilfa Antaries Yusuf Wijaya**
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
