# Transaction Anomaly Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://transaction-anomaly-dashboard-areswijaya.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67%2B-FF4B4B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?style=flat)
![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75?style=flat)

> **English summary:** A Streamlit dashboard that combines K-Means behavioral segmentation with a three-model unsupervised anomaly ensemble (Isolation Forest, Local Outlier Factor, PCA reconstruction error) and a separate rule-based login-risk flag, built on 2,516 **synthetic** bank transactions. There is no ground-truth fraud label, so flagged transactions are *candidates for review*, not confirmed fraud. Limitations are documented below.

Aplikasi web berbasis Streamlit untuk segmentasi perilaku transaksi nasabah memakai K-Means dan deteksi anomali multivariat memakai Isolation Forest, Local Outlier Factor, dan PCA Reconstruction Error.

## Tentang Proyek

Proyek ini menganalisis 2,516 data transaksi nasabah melalui dua lapisan unsupervised machine learning dan satu aturan eksplisit:

1. Segmentasi perilaku (K-Means, k = 2, Silhouette Score 0.2634) pada 5 fitur (`TransactionAmount`, `CustomerAge`, `TransactionDuration`, `AccountBalance`, `Balance_Ratio`):
   - `0 - Nasabah Muda Aktif` (863 transaksi / 34.30%): rata-rata usia 27.30 tahun, saldo $1,584.53, transaksi $347.07, `Balance_Ratio` 0.47.
   - `1 - Nasabah Mapan Konservatif` (1,653 transaksi / 65.70%): rata-rata usia 53.71 tahun, saldo $6,955.81, transaksi $270.63, `Balance_Ratio` 0.05.
2. Deteksi anomali multivariat (ensemble 3 model) pada 8 fitur, termasuk jeda transaksi `Days_Since_Prev_Tx` dan kecepatan nominal `Amount_Per_Second`:
   - Isolation Forest (`contamination=0.05`)
   - Local Outlier Factor (`n_neighbors=20`, `contamination=0.05`, `novelty=True`)
   - PCA Reconstruction Error (komponen utama pada 90% varians, ambang persentil ke-95)
   - Ketiga model digabung menjadi skor anomali kontinu (0-100) dan konsensus `Anomaly_Votes` (0-3). Transaksi yang ditandai minimal 2 model disebut `Anomali Konsensus (>= 2 model)`. Jumlahnya mengikuti nilai `contamination` yang dipilih, dan stabilitasnya terhadap nilai tersebut diuji di notebook (bagian 7.1). Angka terkini tampil di tab **Ringkasan** dan **Evaluasi & Keterbatasan** dashboard.
3. `Risiko Login (LoginAttempts >= 3)`: menyaring 93 transaksi (3.70%) dengan 3-5 upaya login sebagai aturan terpisah, bukan hasil prediksi model.

## Sumber Dataset

File mentah `data/raw.xlsx` (2,537 baris sebelum pembersihan duplikat menjadi 2,516 baris) berasal dari dataset latihan Dicoding yang mengadaptasi Kaggle Bank Transaction Dataset for Fraud Detection. Dataset ini berisi data sintetis transaksi bank di kota-kota Amerika Serikat dalam satuan USD dan tidak memiliki label fraud asli (ground truth).

## Keterbatasan Metodologi

- **Tidak ada label fraud asli.** Seluruh model bersifat unsupervised, sehingga precision dan recall terhadap kejadian fraud nyata tidak dapat dihitung. Konsensus antar-model hanya menunjukkan kesepakatan, bukan kebenaran.
- **Jumlah anomali mengikuti pilihan `contamination`.** Nilai 0.05 ditetapkan sendiri, bukan temuan dari data. Notebook menguji 0.03, 0.05, dan 0.07 lalu mengukur kemiripan daftar konsensus (Jaccard index).
- **Pemisahan K-Means lemah.** Silhouette Score tertinggi berada pada k = 2 sebesar 0.2634, dan dua segmen pada dasarnya memisahkan profil usia dan saldo. K-Means dipakai sebagai konteks segmen nasabah, bukan detektor anomali.
- **`Days_Since_Prev_Tx` perlu dibaca hati-hati.** Pada data sintetis ini `TransactionDate` hanya mencakup rentang sempit, sehingga fitur tersebut praktis mencerminkan `PreviousTransactionDate`, bukan perilaku jeda yang bermakna. Notebook (bagian 7.2) mencetak rentang tanggal dan korelasinya.
- **`LoginAttempts` ikut menjadi fitur anomali**, sehingga `Risiko Login` dan skor anomali saling tumpang tindih. Notebook (bagian 7.2) menampilkan tabel silangnya.

## Fitur Dashboard

- **Filter Sidebar**: menyaring data berdasarkan profesi, kanal transaksi, tipe transaksi, dan kota, lengkap dengan tombol reset.
- **Ringkasan**: menampilkan total transaksi, rata-rata nominal, `Anomali Konsensus (>= 2 model)`, `Risiko Login (LoginAttempts >= 3)`, distribusi segmen, komposisi kategori, dan tabel profil.
- **Visualisasi Cluster**: menampilkan scatter plot antarfitur (termasuk skor anomali dan kecepatan transaksi), radar chart profil segmen, serta rekomendasi segmen.
- **Deteksi Anomali**: menyaring `Anomali Konsensus (>= 2 model)`, konsensus 3 model, flag >= 1 model, atau `Risiko Login (LoginAttempts >= 3)`, lalu memplot distribusi skor 0-100 dan menyediakan ekspor `transaksi_anomali.csv`.
- **Simulasi & Skor Anomali**: menghitung segmen K-Means, skor anomali tiap model, konsensus model, dan status risiko login dari input transaksi.
- **Evaluasi & Keterbatasan**: menampilkan matriks kesepakatan antar-model anomali, profil rata-rata per tingkat konsensus (0-3 model), evaluasi Elbow/Silhouette K-Means (k = 2..8, dibaca dari `models/k_evaluation.csv`), serta sumber data dan keterbatasan.

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
│   ├── k_evaluation.csv
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

## Mereproduksi Hasil dari Notebook

Seluruh artefak yang dibaca dashboard dihasilkan oleh `notebooks/clustering.ipynb`. Jalankan notebook dari atas ke bawah (*Run All*) dari folder yang berisi `data/raw.xlsx` dan `models/`. Sel terakhir menulis ulang:

- `data/data_clustering_clean.csv` (data berlabel: segmen, flag model, skor, votes)
- `models/scaler_clustering.joblib` dan `models/kmeans_model.joblib`
- `models/anomaly_bundle.joblib` (scaler, ketiga detektor, ambang PCA, batas skor)
- `models/k_evaluation.csv` (hasil Elbow/Silhouette k = 2..8)

Sel penutup juga mencetak ringkasan angka yang bisa dicocokkan dengan tab Evaluasi di dashboard.

## Pengujian

```bash
python -m unittest discover -s tests -v
```

## Kontak

Ilfa Antaries Yusuf Wijaya
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
