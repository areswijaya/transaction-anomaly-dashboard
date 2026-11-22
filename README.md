# Transaction Anomaly Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://transaction-anomaly-dashboard-areswijaya.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67%2B-FF4B4B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75?style=flat)

Aplikasi web berbasis Streamlit untuk segmentasi perilaku transaksi nasabah memakai K-Means dan deteksi anomali multivariat memakai Isolation Forest, Local Outlier Factor, dan PCA Reconstruction Error.

## Tentang Proyek

Proyek ini menganalisis 2,516 data transaksi nasabah melalui dua lapisan unsupervised machine learning dan satu guardrail aturan eksplisit:

1. Segmentasi perilaku (K-Means, k = 2, Silhouette Score 0.2634) pada 5 fitur (`TransactionAmount`, `CustomerAge`, `TransactionDuration`, `AccountBalance`, `Balance_Ratio`):
   - `0 - Nasabah Muda Aktif` (863 transaksi / 34.30%): rata-rata usia 27.30 tahun, saldo $1,584.53, transaksi $347.07, `Balance_Ratio` 0.47.
   - `1 - Nasabah Mapan Konservatif` (1,653 transaksi / 65.70%): rata-rata usia 53.71 tahun, saldo $6,955.81, transaksi $270.63, `Balance_Ratio` 0.05.
2. Deteksi anomali multivariat (ensemble 3 model) pada 8 fitur, termasuk jeda transaksi `Days_Since_Prev_Tx` dan kecepatan nominal `Amount_Per_Second`:
   - Isolation Forest (`contamination=0.05`)
   - Local Outlier Factor (`n_neighbors=20`, `contamination=0.05`, `novelty=True`)
   - PCA Reconstruction Error (6 komponen dengan 90.11% varians, ambang persentil ke-95)
   - Ketiga model digabung menjadi skor anomali kontinu (0-100) dan konsensus `Anomaly_Votes` (0-3). Sebanyak 70 transaksi (2.78%) termasuk `Anomali Konsensus (>= 2 model)`, dan 37 transaksi ditandai oleh ketiga model sekaligus dengan rata-rata nominal $1,026.27 dan `Balance_Ratio` 2.46.
3. `Risiko Login (LoginAttempts >= 3)`: menyaring 93 transaksi (3.70%) dengan 3-5 upaya login sebagai aturan terpisah.

## Sumber Dataset

File mentah `data/raw.xlsx` (2,537 baris sebelum pembersihan duplikat menjadi 2,516 baris) berasal dari dataset latihan Dicoding yang mengadaptasi Kaggle Bank Transaction Dataset for Fraud Detection. Dataset ini berisi data sintetis transaksi bank di kota-kota Amerika Serikat dalam satuan USD dan tidak memiliki label fraud asli (ground truth).

## Keterbatasan Metodologi

- Tidak ada label fraud asli: seluruh model bersifat unsupervised, sehingga precision dan recall terhadap kejadian fraud nyata tidak dapat dihitung.
- Pemisahan K-Means lemah: Silhouette Score tertinggi berada pada k = 2 sebesar 0.2634, sehingga K-Means hanya dipakai sebagai konteks segmen nasabah, bukan detektor anomali.

## Fitur Dashboard

- **Filter Sidebar**: menyaring data berdasarkan profesi, kanal transaksi, tipe transaksi, dan kota, lengkap dengan tombol reset.
- **Ringkasan**: menampilkan total transaksi, rata-rata nominal, `Anomali Konsensus (>= 2 model)`, `Risiko Login (LoginAttempts >= 3)`, distribusi segmen, komposisi kategori, dan tabel profil.
- **Visualisasi Cluster**: menampilkan scatter plot antarfitur (termasuk skor anomali dan kecepatan transaksi), radar chart profil segmen, serta rekomendasi segmen.
- **Deteksi Anomali**: menyaring `Anomali Konsensus (>= 2 model)`, konsensus 3 model, flag >= 1 model, atau `Risiko Login (LoginAttempts >= 3)`, lalu memplot distribusi skor 0-100 dan menyediakan ekspor `transaksi_anomali.csv`.
- **Simulasi & Skor Anomali**: menghitung segmen K-Means, skor anomali tiap model, konsensus model, dan status risiko login dari input transaksi.
- **Evaluasi & Keterbatasan**: menampilkan matriks kesepakatan antar-model anomali, profil rata-rata per tingkat konsensus (0-3 model), evaluasi Elbow/Silhouette K-Means (k = 2..8), serta sumber data dan keterbatasan.

## Struktur Folder

```text
transaction-anomaly-dashboard/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── data_clustering_clean.csv
│   └── raw.xlsx
├── models/
│   ├── anomaly_bundle.joblib
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

Ilfa Antaries Yusuf Wijaya
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
