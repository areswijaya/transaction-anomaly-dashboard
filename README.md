# Transaction Anomaly Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://transaction-anomaly-dashboard-areswijaya.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat)

Proyek portofolio data science end-to-end pada 2.516 transaksi bank sintetis (dari 2.537 baris mentah): pembersihan data, segmentasi nasabah, deteksi anomali, uji stabilitas model, dan dashboard interaktif Streamlit.

> **English summary:** End-to-end data science learning project on 2,516 synthetic US bank transactions. K-Means (k = 2) separates customers primarily by age and account balance, Isolation Forest (contamination = 0.05) flags unusual transaction behavior, and a separate rule checks repeated login attempts (LoginAttempts >= 3). Because the dataset has no ground-truth fraud labels, all flags serve as review candidates rather than confirmed fraud, and stability/ablation/injection checks are documented in the notebook.

## Metode

1. **Pembersihan data**: hapus 21 baris duplikat dari 2.537 baris awal menjadi 2.516 transaksi, buang kolom identitas/IP/tanggal, dan isi nilai kosong (sekitar 1% per kolom) dengan median untuk numerik serta modus untuk teks.
2. **Segmentasi K-Means** (`k = 2`, Silhouette Score `0.2634`) pada 5 fitur perilaku terstandarisasi (`TransactionAmount`, `CustomerAge`, `TransactionDuration`, `AccountBalance`, `Balance_Ratio`):
   - **Nasabah Muda Saldo Kecil** (863 transaksi, 34.30%): rata-rata usia 27.30 tahun, saldo 1,584.53 USD, nominal 347.07 USD, `Balance_Ratio` 0.47.
   - **Nasabah Lebih Tua Saldo Besar** (1,653 transaksi, 65.70%): rata-rata usia 53.71 tahun, saldo 6,955.81 USD, nominal 270.63 USD, `Balance_Ratio` 0.05.
3. **Deteksi anomali Isolation Forest** (`n_estimators = 200`, `contamination = 0.05`) pada 6 fitur (5 fitur perilaku ditambah `Amount_Per_Second`). Model menandai 126 transaksi (5.01%) dan memberi skor persentil 0-100 terhadap data latih (ambang anomali pada persentil 94.99; dipilih menggantikan skala min-max yang terdistorsi satu outlier ekstrem).
4. **Aturan risiko login**: `LoginAttempts >= 3` menyaring 93 transaksi (3.70%). Fitur `LoginAttempts` dipisahkan dari K-Means dan Isolation Forest agar sinyal akses akun tidak bercampur dengan pola transaksi.
5. **Dashboard Streamlit (`app.py`)**: 5 tab (`Ringkasan`, `Segmen`, `Anomali & Login`, `Coba Transaksi`, `Catatan`), filter profesi/kanal/tipe/kota, ekspor CSV, dan simulasi satu transaksi dengan peringatan ekstrapolasi di luar rentang data latih.

## Temuan Utama

Fitur penjelas ikut melatih model pada data tanpa label fraud, sehingga angka berikut menggambarkan karakter outlier model, bukan pola fraud dunia nyata.

### Karakter outlier model

- **Nominal, rasio saldo, dan kecepatan tinggi**: rata-rata nominal anomali 1,019.27 USD (median 1,043.66 USD) dibanding 258.76 USD (median 199.38 USD) pada transaksi normal. Rata-rata `Balance_Ratio` 1.35 dibanding 0.14 (58.7% atau 74 dari 126 anomali memiliki `Balance_Ratio > 0.5`, dibanding 6.3% atau 150 dari 2,390 transaksi normal). Rata-rata `Amount_Per_Second` 19.33 dibanding 3.24 USD/detik.
- **Kombinasi fitur, bukan satu variabel saja**: korelasi peringkat Spearman skor mentah tertinggi dengan nominal (`0.42`), `Amount_Per_Second` (`0.38`), dan `Balance_Ratio` (`0.31`), sementara usia, saldo, durasi, dan upaya login masing-masing di bawah `0.02`.

### Efek definisi rasio antar-segmen

Tingkat anomali segmen Muda mencapai 9.62% (83/863) dibanding Lebih Tua 2.60% (43/1,653). Namun di dalam kelompok `Balance_Ratio` yang sama, selisih tersebut hilang: pada kuintil rasio tertinggi, proporsi anomali 17.7% (Muda) vs 18.8% (Lebih Tua), dan pada tiga kuintil terendah segmen Muda bernilai 0.0%. Perbedaan antar-segmen muncul karena saldo kecil memperbesar `Balance_Ratio`, bukan karena perilaku segmen yang berbeda.

### Independensi aturan login

Irisan antara Isolation Forest dan `LoginAttempts >= 3` hanya 3 transaksi (0.12%), mendekati ekspektasi acak jika independen (4.7 transaksi). Sebaran `LoginAttempts` juga menunjukkan pola generator data sintetis: 2,396 transaksi bernilai 1, 27 bernilai 2, serta tepat 31 transaksi masing-masing untuk nilai 3, 4, dan 5.

### Uji stabilitas dan pembanding

| Uji | Hasil |
|---|---|
| `contamination` 0.03 / 0.05 / 0.07 | 76 / 126 / 177 anomali; himpunan bersarang penuh (Jaccard 0.60 dan 0.71 murni rasio jumlah, menunjukkan jumlah anomali adalah pilihan parameter) |
| 10 seed acak | Peringkat skor stabil (Spearman rata-rata 0.98); himpunan 126 anomali konsisten (Jaccard rata-rata 0.86, rentang 0.79 sampai 0.90) |
| Tanpa `Amount_Per_Second` atau tanpa `Balance_Ratio` | Jaccard 0.62 (96 dari 126 transaksi tetap sama; 30 transaksi atau 23.8% berubah) |
| Hanya nominal + saldo | Jaccard 0.33 |
| 126 nilai tertinggi satu fitur | Jaccard 0.44 (nominal), 0.32 (`Amount_Per_Second`), 0.29 (`Balance_Ratio`); model menangkap interaksi multifitur |
| Anomali suntikan (2% data = 50 transaksi, nominal x5) | Recall pada 5% teratas: Isolation Forest 54% (27/50), aturan nominal terbesar 56% (28/50), aturan rasio terbesar 16% (8/50); pada skenario lonjakan nominal tunggal, model tidak mengungguli aturan sederhana |

### Kekokohan segmentasi K-Means

Mengulang K-Means (`k = 2`) dengan transformasi `log1p` (ARI `0.76`, Silhouette `0.34`), tanpa `Balance_Ratio` (ARI `0.78`, Silhouette `0.25`), serta hanya usia + saldo (ARI `0.77`, Silhouette `0.44`) menghasilkan partisi serupa. Nilai Silhouette untuk `k = 2` sampai `8` berkisar `0.23` hingga `0.26`, menandakan batas cluster saling tumpang tindih dan didominasi usia serta saldo.

### Contoh urutan tinjauan kandidat

Untuk demo triase di tab `Anomali & Login` (bukan prosedur operasional produksi):

1. **Prioritas 1 (3 transaksi)**: irisan anomali Isolation Forest dan `LoginAttempts >= 3`.
2. **Prioritas 2 (90 transaksi)**: risiko login saja (`LoginAttempts >= 3`), fokus pada verifikasi akses akun.
3. **Prioritas 3 (123 transaksi)**: anomali perilaku saja, urutkan dari skor persentil tertinggi lalu periksa rasio terhadap saldo dan kecepatan transaksi.

## Keterbatasan

- **Data sintetis tanpa label fraud asli**: berasal dari dataset latihan Dicoding (adaptasi Kaggle *Bank Transaction Dataset for Fraud Detection*, transaksi kota AS dalam USD), sehingga precision dan recall fraud nyata tidak dapat diukur.
- **Segmen tumpang tindih**: Silhouette Score `0.2634` menunjukkan pemisahan lemah; rekomendasi aksi per segmen di dashboard bersifat hipotesis yang belum diuji.
- **Ambang ditentukan parameter**: proporsi 5% anomali mengikuti `contamination = 0.05`, bukan ambang alami dari data.
- **Evaluasi suntikan terbatas**: recall 54% hanya menguji satu skenario sintetis (lonjakan nominal 5x), bukan variasi pola fraud nyata.

## Struktur Proyek

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

Aplikasi terbuka di `http://localhost:8501`.

Seluruh file di `data/` (kecuali `raw.xlsx`) dan `models/` dihasilkan oleh `notebooks/clustering.ipynb` saat dijalankan dari atas ke bawah (*Run All*). Notebook disimpan tanpa output cell agar lolos pemeriksaan test kebersihan repositori. Untuk menjalankan ulang notebook dan suite pengujian:

```bash
pip install matplotlib seaborn openpyxl jupyter
python -m unittest discover -s tests -v
```

## Kontak

Ilfa Antaries Yusuf Wijaya
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
