# Transaction Anomaly Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=flat)](https://transaction-anomaly-dashboard-areswijaya.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat)

Proyek belajar saya untuk berlatih alur data science dari awal sampai tampil di web: membersihkan data, mengelompokkan nasabah, menandai transaksi yang tidak biasa, menguji seberapa kokoh hasilnya, lalu menyajikannya di dashboard interaktif.

> **English summary:** A learning project on 2,516 **synthetic** bank transactions. K-Means groups customers into two segments (which mostly separate young/low-balance from older/high-balance customers), Isolation Forest flags unusual transactions, and a simple rule flags repeated login attempts. Stability checks (seeds, feature ablation, simple baselines, injected anomalies) are in the notebook. The data has no real fraud labels, so flagged transactions are candidates for review, not confirmed fraud.

## Yang Saya Kerjakan

1. **Membersihkan data**: membuang duplikat, mengisi nilai kosong (median untuk angka, modus untuk teks).
2. **Segmentasi dengan K-Means** (k = 2, Silhouette Score 0.2634) pada 5 fitur perilaku. Nama segmen hanya memuat yang terukur, yaitu usia dan saldo:
   - **Nasabah Muda Saldo Kecil** (34.30%): usia rata-rata sekitar 27 tahun, saldo rata-rata 1,584.53 USD, rasio transaksi terhadap saldo 0.47.
   - **Nasabah Lebih Tua Saldo Besar** (65.70%): usia rata-rata sekitar 54 tahun, saldo rata-rata 6,955.81 USD, rasio 0.05.
3. **Penanda anomali dengan Isolation Forest** pada 6 fitur (5 fitur perilaku + nominal per detik). Model diminta menandai sekitar 5% transaksi yang paling tidak biasa. Tiap transaksi diberi **skor persentil 0-100**: skor 97 berarti lebih tidak biasa daripada 97% data latih, dan ambang anomali jatuh di sekitar persentil 95.
4. **Aturan risiko login**: `LoginAttempts >= 3` menyaring 93 transaksi (3.70%). Ini aturan sederhana, bukan hasil model.
5. **Uji kekokohan** (notebook bagian 4.1 dan 5.1 sampai 5.4): ulang model dengan seed dan fitur berbeda, bandingkan dengan aturan satu fitur, dan suntikkan anomali buatan.
6. **Dashboard Streamlit** dengan 5 tab (`Ringkasan`, `Segmen`, `Anomali & Login`, `Coba Transaksi`, `Catatan`), filter profesi/kanal/tipe/kota, grafik, tabel yang bisa diunduh, dan form untuk mencoba satu transaksi (dengan peringatan bila input di luar rentang data latih).

## Temuan dan Batasannya

Semua angka dari 2,516 transaksi hasil pembersihan. Fitur yang melatih model ikut dipakai menjelaskan hasilnya, jadi bagian ini menggambarkan **karakter outlier model**, bukan pola fraud asli.

### Apa yang ditandai model

- **Nominal besar, rasio saldo tinggi, kecepatan tinggi:** rata-rata nominal transaksi anomali 1,019.27 USD (median 1,043.66 USD) dibanding 258.76 USD (median 199.38 USD) pada transaksi normal. Rata-rata `Balance_Ratio` 1.35 dibanding 0.14, dan 58.7% transaksi anomali (74 dari 126) punya `Balance_Ratio > 0.5`, dibanding 6.3% (150 dari 2,390) pada transaksi normal. Rata-rata `Amount_Per_Second` 19.33 dibanding 3.24 USD/detik.
- **Bukan sekadar "nominal besar":** korelasi peringkat (Spearman) skor dengan nominal 0.42, dengan nominal per detik 0.38, dan dengan `Balance_Ratio` 0.31. Skor tidak berkorelasi dengan usia, saldo, durasi, dan login (di bawah 0.02).

### Selisih antar-segmen adalah efek definisi rasio

Tingkat anomali segmen Muda 9.62% (83 dari 863) vs segmen Lebih Tua 2.60% (43 dari 1,653). Namun di dalam kelompok `Balance_Ratio` yang sama, selisih itu hilang: pada 20% rasio tertinggi, anomali 17.7% (Muda) vs 18.8% (Lebih Tua), dan pada 60% rasio terendah segmen Muda hampir tidak punya anomali. Jadi selisihnya muncul karena saldo kecil membuat rasio besar, bukan karena perilaku segmen.

### Login dan anomali model hampir tidak berhubungan

Hanya 3 transaksi (0.12%) yang sekaligus ditandai Isolation Forest dan punya `LoginAttempts >= 3`. Kalau keduanya independen, irisan yang diharapkan 4.7, jadi dua mekanisme ini menyaring kandidat yang praktis berbeda. Sebaran `LoginAttempts` juga terlihat seperti hasil generator data: 2,396 transaksi bernilai 1, 27 bernilai 2, lalu tepat 31 transaksi masing-masing untuk nilai 3, 4, dan 5.

### Seberapa stabil hasilnya

| Uji | Hasil |
|---|---|
| `contamination` 0.03 / 0.05 / 0.07 | 76 / 126 / 177 anomali. Himpunannya selalu bersarang (Jaccard 0.60 dan 0.71 hanya rasio jumlah), jadi ini menunjukkan jumlah anomali adalah **pilihan**, bukan kekokohan model |
| 10 seed berbeda | Peringkat skor stabil (Spearman rata-rata 0.98), daftar 126 transaksi berubah sedikit (Jaccard rata-rata 0.86, rentang 0.79 sampai 0.90) |
| Tanpa `Amount_Per_Second` atau tanpa `Balance_Ratio` | Jaccard 0.62: sekitar 40% daftar bergantung pada fitur turunan |
| Hanya nominal + saldo | Jaccard 0.33 |
| 126 nilai terbesar satu fitur | Jaccard 0.44 (nominal), 0.32 (nominal per detik), 0.29 (`Balance_Ratio`): model menangkap kombinasi fitur, bukan satu fitur |
| Anomali suntikan (2% transaksi, nominal x5) | Recall 54% (27 dari 50) untuk Isolation Forest, 56% untuk aturan "nominal terbesar", 16% untuk aturan "rasio terbesar". Pada skenario ini model **tidak lebih baik** dari aturan nominal sederhana |

### Seberapa kokoh segmentasinya

K-Means dengan `log1p` menghasilkan pembagian yang mirip (ARI 0.76), begitu juga tanpa `Balance_Ratio` (ARI 0.78) dan hanya dengan usia + saldo (ARI 0.77, Silhouette 0.44). Artinya segmentasi ini pada dasarnya adalah usia dan saldo, dan `Balance_Ratio` (yang dihitung dari nominal dan saldo) tidak menambah banyak. Silhouette untuk k = 2 sampai 8 berkisar 0.23 sampai 0.26, jadi tidak ada struktur cluster yang tegas.

### Contoh Urutan Tinjauan

Jika daftar kandidat di tab `Anomali & Login` dipakai untuk pemeriksaan manual, urutannya bisa dibuat seperti ini:

1. **Prioritas 1 (irisan keduanya, 3 transaksi):** cek transaksi yang sekaligus anomali perilaku dan punya `LoginAttempts >= 3`.
2. **Prioritas 2 (risiko login saja, 90 transaksi):** fokus ke verifikasi akses dan upaya masuk berulang.
3. **Prioritas 3 (anomali perilaku saja, 123 transaksi):** urutkan dari skor tertinggi, lalu periksa nominal besar dibanding saldo atau durasi singkat.

Ini hanya contoh alur kerja untuk demo analisis, bukan prosedur operasional produksi.

## Apa yang Saya Pelajari

- Nilai Silhouette yang rendah (0.2634) berarti segmen saling tumpang tindih. Saya melaporkannya apa adanya, dan nama segmen sengaja hanya memuat usia dan saldo karena hanya itu yang membedakan segmen.
- Tanpa label fraud asli, hasil model tidak bisa dihitung akurasinya. "Anomali" di sini hanya berarti "tidak biasa". Satu-satunya angka kinerja yang bisa saya dapat adalah dari anomali suntikan, dan itu hanya satu skenario sintetis.
- Mengubah `contamination` tidak menguji kekokohan model, karena hanya menggeser ambang pada skor yang sama. Uji yang bermakna adalah seed, fitur, dan pembanding sederhana.
- Skor min-max 0-100 bergantung pada satu transaksi paling ekstrem (ambangnya jatuh di sekitar 51 padahal hanya 5% yang ditandai). Skor persentil lebih mudah dibaca.
- Selisih antar-kelompok perlu dicek dengan mengendalikan fitur yang dipakai membentuk skor, sebelum dibaca sebagai perilaku.
- Fitur yang saling menumpuk sebaiknya dipisah. `LoginAttempts` tidak dipakai di K-Means maupun Isolation Forest, supaya aturan login berdiri sendiri.

## Keterbatasan

- Data sintetis (dataset latihan Dicoding, adaptasi Kaggle *Bank Transaction Dataset for Fraud Detection*), transaksi kota-kota Amerika Serikat dalam USD.
- Tidak ada label fraud asli, jadi precision dan recall tidak bisa dihitung. Recall anomali suntikan hanya gambaran kasar.
- Dua segmen pada dasarnya memisahkan nasabah muda bersaldo kecil dan nasabah lebih tua bersaldo besar. Ide tindakan per segmen di dashboard adalah hipotesis yang belum diuji.
- Jumlah anomali mengikuti pilihan `contamination = 0.05`, bukan temuan dari data.

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

Semua file di `data/` (selain `raw.xlsx`) dan `models/` dihasilkan oleh `notebooks/clustering.ipynb`. Jalankan notebook dari atas ke bawah (*Run All*) untuk membuatnya ulang. Notebook di repo disimpan **tanpa output** (dicek oleh test), jadi bersihkan output sebelum commit. Jika ingin menjalankan ulang notebook, pasang paket tambahannya terlebih dulu:

```bash
pip install matplotlib seaborn openpyxl jupyter
```

Pengujian sederhana (menjalankan test pada file hasil notebook, jadi jalankan notebook dulu):

```bash
python -m unittest discover -s tests -v
```

## Kontak

Ilfa Antaries Yusuf Wijaya
- Email: contact.areswijaya@gmail.com
- LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
- GitHub: [github.com/areswijaya](https://github.com/areswijaya)
