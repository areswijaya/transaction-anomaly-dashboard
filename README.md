# Segmentasi & Deteksi Anomali Transaksi Nasabah

Dashboard analitik interaktif berbasis **Streamlit** untuk mengeksplorasi hasil segmentasi transaksi nasabah (**K-Means Clustering**) dan memprediksi segmen transaksi baru secara langsung (**Random Forest Classifier**).

## Ringkasan Proyek
* **Dataset**: `2,516` transaksi nasabah bersih (setelah deduplikasi dan imputasi) dengan fitur tambahan `Balance_Ratio = TransactionAmount / AccountBalance`.
* **Clustering (`K-Means`, `k = 3`, Silhouette Score `0.2657`)**:
  * **Cluster 0 — Nasabah Mapan Konservatif (`63.43%`)**: Usia rata-rata `53.67` thn, saldo tertinggi (`$6,933.26`), upaya login normal (`1.01`).
  * **Cluster 1 — Nasabah Muda Aktif (`32.87%`)**: Usia rata-rata `27.25` thn, saldo rendah (`$1,578.34`), transaksi tertinggi (`$347.13`), `Balance_Ratio` `0.48`.
  * **Cluster 2 — Transaksi Berisiko (`3.70%`)**: Mengisolasi seluruh transaksi dengan anomali upaya login berulang (`3-5` kali, rata-rata `4.00`).
* **Klasifikasi (`Random Forest`)**: Test Accuracy `97.82%`, CV Macro F1 `97.97%`, dan Recall Cluster 2 `100%`.

## Struktur Folder
```text
├── .streamlit/
│   └── config.toml                 # Konfigurasi tema dashboard
├── data/
│   ├── data_clustering_clean.csv   # Dataset bersih untuk dashboard
│   └── raw.xlsx                    # Dataset mentah awal
├── models/
│   ├── rf_classifier_pipeline.joblib  # Pipeline Random Forest untuk prediksi
│   ├── kmeans_model.joblib            # Model K-Means Clustering
│   └── scaler_clustering.joblib       # StandardScaler fitur clustering
├── notebooks/
│   ├── clustering.ipynb            # Eksplorasi & pemodelan K-Means
│   └── classification.ipynb        # Pelatihan & evaluasi Random Forest
├── app.py                          # Aplikasi utama Streamlit
├── requirements.txt                # Dependensi Python untuk deployment
└── .gitignore                      # Pengecualian file lokal & arsip
```

## Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Pembuat
**Ilfa Antaries Yusuf Wijaya**
* Email: `contact.areswijaya@gmail.com`
* LinkedIn: [linkedin.com/in/areswijaya](https://www.linkedin.com/in/areswijaya)
* GitHub: [github.com/areswijaya](https://github.com/areswijaya)
