# Dashboard Segmentasi & Deteksi Anomali Transaksi

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://your-app-url.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67%2B-FF4B4B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75?style=flat)

Aplikasi web berbasis Streamlit untuk melihat hasil segmentasi transaksi nasabah dan memprediksi kategori transaksi baru secara langsung.

## Tentang Proyek

Proyek ini membagi 2.516 data transaksi nasabah ke dalam tiga segmen menggunakan K-Means Clustering (`k = 3`, Silhouette Score `0.2657`), lalu melatih model Random Forest (akurasi uji `97.82%`, recall Cluster 2 `100%`) untuk memprediksi segmen dari transaksi baru.

Tiga segmen utama yang terbentuk:
1. **Cluster 0 (Nasabah Mapan Konservatif, 63.43%)**: Usia rata-rata 53,67 tahun dengan saldo tinggi ($6.933,26) dan aktivitas login normal.
2. **Cluster 1 (Nasabah Muda Aktif, 32.87%)**: Usia rata-rata 27,25 tahun dengan saldo lebih rendah ($1.578,34) dan rasio transaksi terhadap saldo lebih tinggi (`Balance_Ratio` 0,48).
3. **Cluster 2 (Transaksi Berisiko, 3.70%)**: Menangkap seluruh transaksi dengan anomali upaya login berulang (3 sampai 5 kali) dan durasi transaksi paling lama.

## Fitur Dashboard

- **Filter Sidebar**: Menyaring data berdasarkan profesi, kanal transaksi, tipe transaksi, dan kota, lengkap dengan tombol reset.
- **Ringkasan**: Menampilkan metrik utama (KPI), distribusi jumlah transaksi per segmen, komposisi per kategori, dan tabel profil rata-rata.
- **Visualisasi Cluster**: Scatter plot antarfitur yang sumbu X dan Y-nya bisa diubah, radar chart profil segmen, serta rekomendasi bisnis.
- **Transaksi Berisiko**: Memantau distribusi upaya login, sebaran teratas Cluster 2, dan tombol unduh data transaksi berisiko dalam format CSV.
- **Prediksi**: Form interaktif berbasis Random Forest untuk menguji transaksi baru lengkap dengan skenario contoh dan tingkat kepercayaan model.
- **Evaluasi Model**: Menampilkan metrik uji, confusion matrix, 10 fitur paling berpengaruh, serta grafik pemilihan jumlah cluster.

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
│   ├── rf_classifier_pipeline.joblib
│   └── scaler_clustering.joblib
├── notebooks/
│   ├── clustering.ipynb
│   └── classification.ipynb
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

## Kontak

**Ilfa Antaries Yusuf Wijaya**
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
