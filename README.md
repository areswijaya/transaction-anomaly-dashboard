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
├── .gitignore
└── README.md
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Dashboard terbuka di `http://localhost:8501`.

Semua file di `data/` (selain `raw.xlsx`) dan `models/` dihasilkan oleh `notebooks/clustering.ipynb`. Jalankan notebook dari atas ke bawah (*Run All*) untuk membuatnya ulang.

Pengujian sederhana:

```bash
python -m unittest discover -s tests -v
```

## Kontak

Ilfa Antaries Yusuf Wijaya
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
