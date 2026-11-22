from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LOGIN_RISK_THRESHOLD = 3

CLUSTERING_FEATURES = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "AccountBalance",
    "Balance_Ratio",
]

ANOMALY_FEATURES = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "Balance_Ratio",
    "Days_Since_Prev_Tx",
    "Amount_Per_Second",
]

CLUSTER_LABELS = {
    0: "0 - Nasabah Muda Aktif",
    1: "1 - Nasabah Mapan Konservatif",
}

CLUSTER_COLORS = {
    "0 - Nasabah Muda Aktif": "#B45309",
    "1 - Nasabah Mapan Konservatif": "#1D4ED8",
}

CLUSTER_GRADIENTS = {
    "blue": ["#DBEAFE", "#60A5FA", "#1D4ED8"],
    "red": ["#FEE2E2", "#F87171", "#B91C1C"],
}

CLUSTER_RECOMMENDATIONS = {
    0: (
        "Profil transaksi aktif dengan porsi penarikan tinggi terhadap saldo. "
        "Pertimbangkan fitur auto-debit, batas anggaran, atau promo cashback."
    ),
    1: (
        "Profil saldo tinggi dan transaksi konservatif. "
        "Pertimbangkan produk deposito, obligasi, atau reksa dana risiko rendah."
    ),
}

K_EVALUATION_FALLBACK = pd.DataFrame({
    "k": [2, 3, 4, 5, 6, 7, 8],
    "Inertia": [9813.20, 8156.22, 7027.95, 6061.49, 5265.85, 4759.72, 4368.32],
    "Silhouette Score": [0.2634, 0.2339, 0.2342, 0.2380, 0.2531, 0.2567, 0.2328],
})



@st.cache_data
def load_k_evaluation():
    """Baca hasil Elbow/Silhouette dari notebook (models/k_evaluation.csv); fallback ke nilai statis."""
    path = BASE_DIR / "models" / "k_evaluation.csv"
    try:
        evaluation = pd.read_csv(path)
        required = {"k", "Inertia", "Silhouette Score"}
        if required.issubset(evaluation.columns) and not evaluation.empty:
            return evaluation[["k", "Inertia", "Silhouette Score"]]
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError):
        pass
    return K_EVALUATION_FALLBACK


PRESET_SCENARIOS = {
    "Manual / Kustom": {
        "TransactionAmount": 250.0,
        "AccountBalance": 5000.0,
        "TransactionDuration": 110,
        "CustomerAge": 35,
        "LoginAttempts": 1,
        "Days_Since_Prev_Tx": 485.0,
    },
    "Contoh 1: Nasabah Muda Aktif (Normal)": {
        "TransactionAmount": 350.0,
        "AccountBalance": 1200.0,
        "TransactionDuration": 120,
        "CustomerAge": 25,
        "LoginAttempts": 1,
        "Days_Since_Prev_Tx": 460.0,
    },
    "Contoh 2: Nasabah Mapan Konservatif (Normal)": {
        "TransactionAmount": 250.0,
        "AccountBalance": 7500.0,
        "TransactionDuration": 115,
        "CustomerAge": 56,
        "LoginAttempts": 1,
        "Days_Since_Prev_Tx": 490.0,
    },
    "Contoh 3: Anomali Multivariat (Nominal & Rasio Ekstrem)": {
        "TransactionAmount": 1450.0,
        "AccountBalance": 800.0,
        "TransactionDuration": 35,
        "CustomerAge": 40,
        "LoginAttempts": 4,
        "Days_Since_Prev_Tx": 320.0,
    },
}


def _minmax_scale(val, bounds):
    lo, hi = bounds
    if hi <= lo:
        return 0.0
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0)) * 100.0


@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / "data" / "data_clustering_clean.csv")
    data["Segment_Name"] = data["Cluster"].map(CLUSTER_LABELS)
    data["High_Login_Risk"] = data["LoginAttempts"] >= LOGIN_RISK_THRESHOLD
    return data


@st.cache_resource
def load_models():
    scaler_km = joblib.load(BASE_DIR / "models" / "scaler_clustering.joblib")
    kmeans = joblib.load(BASE_DIR / "models" / "kmeans_model.joblib")
    anomaly_bundle = joblib.load(BASE_DIR / "models" / "anomaly_bundle.joblib")
    return scaler_km, kmeans, anomaly_bundle


@st.cache_data
def get_feature_bounds(_df):
    return _df[CLUSTERING_FEATURES].min(), _df[CLUSTERING_FEATURES].max()


def predict_transaction(
    tx_amount,
    acc_balance,
    tx_duration,
    cust_age,
    login_attempts,
    days_since_prev_tx,
    scaler_km,
    kmeans,
    anomaly_bundle,
):
    """Map one transaction to a K-Means segment, 3-model anomaly score (0-100), and login rule."""
    balance_ratio = float(tx_amount / acc_balance) if acc_balance > 0 else 0.0
    amount_per_second = float(tx_amount / max(tx_duration, 1))

    km_df = pd.DataFrame([{
        "TransactionAmount": float(tx_amount),
        "CustomerAge": float(cust_age),
        "TransactionDuration": float(tx_duration),
        "AccountBalance": float(acc_balance),
        "Balance_Ratio": balance_ratio,
    }])[CLUSTERING_FEATURES]

    km_scaled = scaler_km.transform(km_df)
    # K-Means was fitted with column names; wrapping avoids sklearn feature-name warnings.
    km_scaled_df = pd.DataFrame(km_scaled, columns=CLUSTERING_FEATURES)
    cluster = int(kmeans.predict(km_scaled_df)[0])
    centroid_distances = {
        CLUSTER_LABELS[i]: float(np.linalg.norm(km_scaled[0] - center))
        for i, center in enumerate(kmeans.cluster_centers_)
    }

    anom_df = pd.DataFrame([{
        "TransactionAmount": float(tx_amount),
        "CustomerAge": float(cust_age),
        "TransactionDuration": float(tx_duration),
        "LoginAttempts": float(login_attempts),
        "AccountBalance": float(acc_balance),
        "Balance_Ratio": balance_ratio,
        "Days_Since_Prev_Tx": float(days_since_prev_tx),
        "Amount_Per_Second": amount_per_second,
    }])[anomaly_bundle["features"]]

    anom_scaled = anomaly_bundle["scaler"].transform(anom_df)

    iso_flag = bool(anomaly_bundle["iso"].predict(anom_scaled)[0] == -1)
    lof_flag = bool(anomaly_bundle["lof"].predict(anom_scaled)[0] == -1)
    pca_rec = anomaly_bundle["pca"].inverse_transform(
        anomaly_bundle["pca"].transform(anom_scaled)
    )
    pca_raw = float(np.mean((anom_scaled - pca_rec) ** 2))
    pca_flag = bool(pca_raw >= anomaly_bundle["pca_threshold"])

    iso_raw = float(-anomaly_bundle["iso"].decision_function(anom_scaled)[0])
    lof_raw = float(-anomaly_bundle["lof"].decision_function(anom_scaled)[0])

    bounds = anomaly_bundle["score_bounds"]
    model_scores = {
        "Isolation Forest": round(_minmax_scale(iso_raw, bounds["iso"]), 2),
        "Local Outlier Factor": round(_minmax_scale(lof_raw, bounds["lof"]), 2),
        "PCA Reconstruction": round(_minmax_scale(pca_raw, bounds["pca"]), 2),
    }
    anomaly_score = round(sum(model_scores.values()) / 3.0, 2)
    anomaly_votes = int(iso_flag) + int(lof_flag) + int(pca_flag)

    return {
        "cluster": cluster,
        "segment_name": CLUSTER_LABELS[cluster],
        "is_high_login_risk": bool(login_attempts >= LOGIN_RISK_THRESHOLD),
        "balance_ratio": balance_ratio,
        "amount_per_second": amount_per_second,
        "centroid_distances": centroid_distances,
        "anomaly_score": anomaly_score,
        "anomaly_votes": anomaly_votes,
        "model_flags": {
            "Isolation Forest": iso_flag,
            "Local Outlier Factor": lof_flag,
            "PCA Reconstruction": pca_flag,
        },
        "model_scores": model_scores,
    }


def reset_filters(all_occupations, all_channels, all_types):
    st.session_state["filter_occupation"] = all_occupations
    st.session_state["filter_channel"] = all_channels
    st.session_state["filter_type"] = all_types
    st.session_state["filter_location"] = []


def render_sidebar(df):
    all_occupations = sorted(df["CustomerOccupation"].unique())
    all_channels = sorted(df["Channel"].unique())
    all_types = sorted(df["TransactionType"].unique())
    all_locations = sorted(df["Location"].unique())

    st.session_state.setdefault("filter_occupation", all_occupations)
    st.session_state.setdefault("filter_channel", all_channels)
    st.session_state.setdefault("filter_type", all_types)
    st.session_state.setdefault("filter_location", [])

    st.sidebar.header("Filter Data")
    st.sidebar.caption(
        "Berlaku pada tab Ringkasan, Visualisasi Cluster, dan Deteksi Anomali."
    )

    selected_occupations = st.sidebar.multiselect(
        "Profesi", options=all_occupations, key="filter_occupation"
    )
    selected_channels = st.sidebar.multiselect(
        "Kanal", options=all_channels, key="filter_channel"
    )
    selected_types = st.sidebar.multiselect(
        "Tipe Transaksi", options=all_types, key="filter_type"
    )
    selected_locations = st.sidebar.multiselect(
        "Kota (kosong = semua)", options=all_locations, key="filter_location"
    )

    st.sidebar.button(
        "Reset Filter",
        on_click=reset_filters,
        args=(all_occupations, all_channels, all_types),
        key="reset_filters_btn",
        width="stretch",
    )

    filtered_df = df[
        df["CustomerOccupation"].isin(selected_occupations)
        & df["Channel"].isin(selected_channels)
        & df["TransactionType"].isin(selected_types)
    ]
    if selected_locations:
        filtered_df = filtered_df[filtered_df["Location"].isin(selected_locations)]

    st.sidebar.markdown(
        f'<div class="filter-status-box">Menampilkan <b>{len(filtered_df):,}</b> '
        f'dari <b>{len(df):,}</b> transaksi</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
<div class="author-card">
<div class="author-label">Dibuat oleh</div>
<div class="author-name">Ilfa Antaries Yusuf Wijaya</div>
<div class="author-links">
<a href="mailto:contact.areswijaya@gmail.com" target="_blank" rel="noopener noreferrer">Email</a>
<a href="https://www.linkedin.com/in/areswijaya" target="_blank" rel="noopener noreferrer">LinkedIn</a>
<a href="https://github.com/areswijaya" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    return filtered_df


def build_summary_table(filtered_df):
    total_tx = len(filtered_df)
    summary = (
        filtered_df.groupby("Segment_Name")
        .agg(
            Jumlah_Transaksi=("Cluster", "count"),
            Rata_Transaksi_USD=("TransactionAmount", "mean"),
            Rata_Saldo_USD=("AccountBalance", "mean"),
            Rata_Usia=("CustomerAge", "mean"),
            Rata_Balance_Ratio=("Balance_Ratio", "mean"),
            Rata_Skor_Anomali=("Anomaly_Score", "mean"),
        )
        .reindex(list(CLUSTER_COLORS.keys()), fill_value=0)
        .rename_axis("Segment_Name")
        .reset_index()
    )
    summary["Persentase (%)"] = (
        (summary["Jumlah_Transaksi"] / total_tx) * 100 if total_tx > 0 else 0.0
    )
    return summary


def render_tab_summary(filtered_df):
    st.subheader("Ringkasan Utama")

    total_tx = len(filtered_df)
    avg_amount = filtered_df["TransactionAmount"].mean()
    consensus_tx = int((filtered_df["Anomaly_Votes"] >= 2).sum())
    consensus_pct = (consensus_tx / total_tx) * 100 if total_tx > 0 else 0.0
    risk_tx = int(filtered_df["High_Login_Risk"].sum())
    risk_pct = (risk_tx / total_tx) * 100 if total_tx > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.metric("Total Transaksi", f"{total_tx:,}")
    with col2:
        with st.container(border=True):
            st.metric("Rata-Rata Transaksi", f"${avg_amount:,.2f}")
    with col3:
        with st.container(border=True):
            st.metric("Anomali Konsensus (>= 2 model)", f"{consensus_tx:,} ({consensus_pct:.2f}%)")
    with col4:
        with st.container(border=True):
            st.metric("Risiko Login (LoginAttempts >= 3)", f"{risk_tx:,} ({risk_pct:.2f}%)")

    summary_table = build_summary_table(filtered_df)
    chart_col1, chart_col2 = st.columns([1.05, 1.35])

    with chart_col1:
        with st.container(border=True):
            st.markdown("**Jumlah transaksi per segmen perilaku**")
            fig_dist = px.bar(
                summary_table,
                x="Segment_Name",
                y="Jumlah_Transaksi",
                color="Segment_Name",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                text="Jumlah_Transaksi",
                template="plotly_white",
            )
            fig_dist.update_traces(textposition="outside", marker_cornerradius=4)
            fig_dist.update_layout(
                showlegend=False,
                yaxis_title="Jumlah Transaksi",
                xaxis_title="",
                margin=dict(t=25, b=20, l=20, r=20),
                height=370,
            )
            st.plotly_chart(fig_dist, width="stretch")

    with chart_col2:
        with st.container(border=True):
            top_col1, top_col2 = st.columns([1.5, 1])
            with top_col1:
                st.markdown("**Proporsi segmen pada tiap kategori (%)**")
            with top_col2:
                breakdown_by = st.selectbox(
                    "Kategori",
                    options=["CustomerOccupation", "Channel", "TransactionType"],
                    key="summary_breakdown_by",
                    label_visibility="collapsed",
                )

            breakdown = (
                filtered_df.groupby([breakdown_by, "Segment_Name"]).size().reset_index(name="Jumlah")
            )
            breakdown["Persentase (%)"] = (
                breakdown["Jumlah"] / breakdown.groupby(breakdown_by)["Jumlah"].transform("sum") * 100
            )
            fig_breakdown = px.bar(
                breakdown,
                x=breakdown_by,
                y="Persentase (%)",
                color="Segment_Name",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                template="plotly_white",
            )
            fig_breakdown.update_traces(marker_cornerradius=4)
            fig_breakdown.update_layout(
                legend_title_text="Segmen",
                yaxis_title="Persentase (%)",
                xaxis_title=breakdown_by,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=20, r=20),
                height=330,
            )
            st.plotly_chart(fig_breakdown, width="stretch")

    summary_display = summary_table.rename(columns={
        "Segment_Name": "Segmen",
        "Jumlah_Transaksi": "Jumlah",
        "Persentase (%)": "Persentase (%)",
        "Rata_Transaksi_USD": "Transaksi ($)",
        "Rata_Saldo_USD": "Saldo ($)",
        "Rata_Usia": "Usia (Thn)",
        "Rata_Balance_Ratio": "Balance Ratio",
        "Rata_Skor_Anomali": "Skor Anomali",
    })

    with st.container(border=True):
        st.markdown("**Profil rata-rata per segmen**")
        st.dataframe(summary_display.round(2), width="stretch", hide_index=True)


def render_tab_cluster(filtered_df, df):
    st.subheader("Visualisasi & Profil Cluster (K-Means, k = 2)")

    vis_col1, vis_col2 = st.columns([1.35, 1])

    with vis_col1:
        with st.container(border=True):
            st.markdown("**Sebaran antarfitur transaksi**")
            plot_features = CLUSTERING_FEATURES + [
                "Amount_Per_Second",
                "Days_Since_Prev_Tx",
                "Anomaly_Score",
            ]
            axis_col1, axis_col2 = st.columns(2)
            x_axis = axis_col1.selectbox(
                "Sumbu X",
                options=plot_features,
                index=plot_features.index("AccountBalance"),
                key="cluster_x_axis",
            )
            y_axis = axis_col2.selectbox(
                "Sumbu Y",
                options=plot_features,
                index=plot_features.index("TransactionAmount"),
                key="cluster_y_axis",
            )

            scatter_df = filtered_df.sort_values("Anomaly_Votes")
            fig_scatter = px.scatter(
                scatter_df,
                x=x_axis,
                y=y_axis,
                color="Segment_Name",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                hover_data=ANOMALY_FEATURES + ["Anomaly_Score", "Anomaly_Votes"],
                opacity=0.82,
                template="plotly_white",
            )
            fig_scatter.update_traces(marker=dict(size=7))
            fig_scatter.update_layout(
                legend_title_text="Segmen",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(t=40, b=20, l=20, r=20),
                height=440,
            )
            st.plotly_chart(fig_scatter, width="stretch")

    with vis_col2:
        with st.container(border=True):
            st.markdown("**Perbandingan 5 fitur perilaku per segmen (skala 0-1)**")
            feat_min, feat_max = get_feature_bounds(df)
            segment_means = filtered_df.groupby("Segment_Name")[CLUSTERING_FEATURES].mean()
            segment_norm = (segment_means - feat_min) / (feat_max - feat_min)
            radar_data = segment_norm.reset_index().melt(
                id_vars="Segment_Name", var_name="Fitur", value_name="Nilai Normalisasi"
            )

            fig_radar = px.line_polar(
                radar_data,
                r="Nilai Normalisasi",
                theta="Fitur",
                color="Segment_Name",
                line_close=True,
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                template="plotly_white",
            )
            fig_radar.update_traces(fill="toself", opacity=0.45)
            fig_radar.update_layout(
                legend_title_text="Segmen",
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                margin=dict(t=30, b=60, l=30, r=30),
                height=510,
            )
            st.plotly_chart(fig_radar, width="stretch")

    st.markdown("**Interpretasi Segmen K-Means (Data Keseluruhan)**")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("0 - Nasabah Muda Aktif (34.30%)", expanded=True):
            st.write(
                "Profil: 863 transaksi. Rata-rata usia 27.30 tahun (dominan Student), "
                "saldo $1,584.53, transaksi $347.07, Balance_Ratio 0.47."
            )
            st.write("Tindakan: Tawarkan fitur auto-debit, batas anggaran, atau promo cashback.")
    with exp_col2:
        with st.expander("1 - Nasabah Mapan Konservatif (65.70%)", expanded=True):
            st.write(
                "Profil: 1,653 transaksi. Rata-rata usia 53.71 tahun (dominan Retired), "
                "saldo $6,955.81, transaksi $270.63, Balance_Ratio 0.05."
            )
            st.write("Tindakan: Tawarkan deposito, obligasi, atau reksa dana risiko rendah.")


def render_tab_anomaly(filtered_df):
    st.subheader("Deteksi Anomali Multivariat (Ensemble 3 Model + Risiko Login)")
    st.caption(
        "Menggabungkan Isolation Forest, Local Outlier Factor, dan PCA Reconstruction Error "
        "pada 8 fitur (termasuk Days_Since_Prev_Tx dan Amount_Per_Second), tanpa label fraud asli. "
        "Hasilnya penanda kandidat untuk ditinjau, bukan vonis fraud."
    )

    filter_mode = st.selectbox(
        "Filter Transaksi Anomali",
        options=[
            "Anomali Konsensus (>= 2 model)",
            "Konsensus Penuh (3 model)",
            "Semua Flag (>= 1 model)",
            "Risiko Login (LoginAttempts >= 3)",
        ],
        key="anomaly_filter_mode",
    )

    if filter_mode == "Konsensus Penuh (3 model)":
        anom_df = filtered_df[filtered_df["Anomaly_Votes"] == 3]
    elif filter_mode == "Semua Flag (>= 1 model)":
        anom_df = filtered_df[filtered_df["Anomaly_Votes"] >= 1]
    elif filter_mode == "Risiko Login (LoginAttempts >= 3)":
        anom_df = filtered_df[filtered_df["High_Login_Risk"]]
    else:
        anom_df = filtered_df[filtered_df["Anomaly_Votes"] >= 2]

    anom_df = anom_df.sort_values("Anomaly_Score", ascending=False)

    vis_col1, vis_col2 = st.columns([1.3, 1])
    with vis_col1:
        with st.container(border=True):
            st.markdown("**Sebaran skor anomali gabungan (0-100)**")
            fig_hist = px.histogram(
                filtered_df,
                x="Anomaly_Score",
                color=filtered_df["Anomaly_Votes"].astype(str),
                nbins=30,
                template="plotly_white",
                labels={"color": "Jumlah Model Flag"},
            )
            fig_hist.update_layout(
                yaxis_title="Jumlah Transaksi",
                xaxis_title="Skor Anomali Gabungan (0-100)",
                margin=dict(t=25, b=20, l=20, r=20),
                height=360,
            )
            st.plotly_chart(fig_hist, width="stretch")

    with vis_col2:
        with st.container(border=True):
            st.markdown("**Jumlah temuan anomali per metode**")
            counts_df = pd.DataFrame({
                "Metode": [
                    "Isolation Forest (5%)",
                    "Local Outlier Factor (5%)",
                    "PCA Reconstruction (5%)",
                    "Anomali Konsensus (>= 2 model)",
                    "Risiko Login (>= 3)",
                ],
                "Jumlah": [
                    int(filtered_df["ISO_Flag"].sum()),
                    int(filtered_df["LOF_Flag"].sum()),
                    int(filtered_df["PCA_Flag"].sum()),
                    int((filtered_df["Anomaly_Votes"] >= 2).sum()),
                    int(filtered_df["High_Login_Risk"].sum()),
                ],
            })
            fig_bar = px.bar(
                counts_df,
                x="Metode",
                y="Jumlah",
                text="Jumlah",
                color="Jumlah",
                color_continuous_scale=CLUSTER_GRADIENTS["red"],
                template="plotly_white",
            )
            fig_bar.update_traces(textposition="outside", marker_cornerradius=4)
            fig_bar.update_layout(
                coloraxis_showscale=False,
                yaxis_title="Jumlah Transaksi",
                xaxis_title="",
                margin=dict(t=25, b=20, l=20, r=20),
                height=360,
            )
            st.plotly_chart(fig_bar, width="stretch")

    display_cols = [
        "Anomaly_Score",
        "Anomaly_Votes",
        "High_Login_Risk",
        "Segment_Name",
        "TransactionAmount",
        "AccountBalance",
        "Balance_Ratio",
        "Amount_Per_Second",
        "Days_Since_Prev_Tx",
        "LoginAttempts",
        "CustomerAge",
        "Channel",
        "Location",
    ]
    with st.container(border=True):
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            st.markdown(f"**Daftar Transaksi ({len(anom_df):,} data, filter: {filter_mode})**")
        with hdr_col2:
            if not anom_df.empty:
                st.download_button(
                    label="Unduh CSV",
                    data=anom_df[display_cols].to_csv(index=False).encode("utf-8"),
                    file_name="transaksi_anomali.csv",
                    mime="text/csv",
                    key="download_anomaly_csv",
                    width="stretch",
                )

        if anom_df.empty:
            st.info(
                "Tidak ada transaksi yang cocok pada filter ini. "
                "Ubah opsi filter anomali atau klik 'Reset Filter' di sidebar."
            )
        else:
            st.dataframe(anom_df[display_cols].round(2), width="stretch", hide_index=True)


def render_tab_predict(df, scaler_km, kmeans, anomaly_bundle):
    st.subheader("Simulasi Segmen (K-Means) & Skor Anomali Multivariat")
    st.caption(
        "Menghitung segmen K-Means (5 fitur) dan skor anomali ensemble "
        "(Isolation Forest, LOF, PCA Reconstruction) pada 8 fitur."
    )

    preset_choice = st.selectbox(
        "Skenario Contoh",
        options=list(PRESET_SCENARIOS.keys()),
        key="predict_preset",
    )
    preset_vals = PRESET_SCENARIOS[preset_choice]

    with st.container(border=True):
        with st.form("prediction_form"):
            form_col1, form_col2, form_col3 = st.columns(3)

            with form_col1:
                tx_amount = st.number_input(
                    "Nominal Transaksi ($)",
                    min_value=float(df["TransactionAmount"].min()),
                    max_value=float(df["TransactionAmount"].max()),
                    value=float(preset_vals["TransactionAmount"]),
                    step=10.0,
                    key=f"tx_amount_{preset_choice}",
                )
                acc_balance = st.number_input(
                    "Saldo Akun ($)",
                    min_value=0.0,
                    max_value=float(df["AccountBalance"].max()),
                    value=float(preset_vals["AccountBalance"]),
                    step=100.0,
                    key=f"acc_balance_{preset_choice}",
                )

            with form_col2:
                tx_duration = st.number_input(
                    "Durasi (Detik)",
                    min_value=int(df["TransactionDuration"].min()),
                    max_value=int(df["TransactionDuration"].max()),
                    value=int(preset_vals["TransactionDuration"]),
                    step=5,
                    key=f"tx_duration_{preset_choice}",
                )
                cust_age = st.number_input(
                    "Usia Nasabah (Thn)",
                    min_value=int(df["CustomerAge"].min()),
                    max_value=int(df["CustomerAge"].max()),
                    value=int(preset_vals["CustomerAge"]),
                    step=1,
                    key=f"cust_age_{preset_choice}",
                )

            with form_col3:
                login_attempts = st.number_input(
                    "Upaya Login",
                    min_value=int(df["LoginAttempts"].min()),
                    max_value=int(df["LoginAttempts"].max()),
                    value=int(preset_vals["LoginAttempts"]),
                    step=1,
                    key=f"login_attempts_{preset_choice}",
                )
                days_since_prev_tx = st.number_input(
                    "Jeda Transaksi Sebelumnya (Hari)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(preset_vals["Days_Since_Prev_Tx"]),
                    step=10.0,
                    key=f"days_prev_{preset_choice}",
                )

            submitted = st.form_submit_button("Analisis Segmen & Skor Anomali", width="stretch")

    if submitted:
        try:
            result = predict_transaction(
                tx_amount=tx_amount,
                acc_balance=acc_balance,
                tx_duration=tx_duration,
                cust_age=cust_age,
                login_attempts=login_attempts,
                days_since_prev_tx=days_since_prev_tx,
                scaler_km=scaler_km,
                kmeans=kmeans,
                anomaly_bundle=anomaly_bundle,
            )
        except Exception as err:
            st.error(f"Simulasi gagal dijalankan: {err}. Periksa nilai input lalu coba lagi.")
            return

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            with st.container(border=True):
                st.metric("Segmen K-Means", result["segment_name"])
        with res_col2:
            with st.container(border=True):
                st.metric("Skor Anomali (0-100)", f"{result['anomaly_score']:.2f}")
        with res_col3:
            with st.container(border=True):
                st.metric("Konsensus Model", f"{result['anomaly_votes']} / 3")
        with res_col4:
            with st.container(border=True):
                risk_label = "Tinggi (>= 3x)" if result["is_high_login_risk"] else "Normal (< 3x)"
                st.metric("Risiko Login", risk_label)

        if result["anomaly_votes"] >= 2:
            st.error(
                f"Anomali Konsensus ({result['anomaly_votes']}/3 model mendeteksi outlier). "
                "Lakukan verifikasi tambahan sebelum memproses transaksi."
            )
        elif result["anomaly_votes"] == 1 or result["is_high_login_risk"]:
            st.warning(
                "Sinyal Peringatan Parsial (1 model anomali atau aturan LoginAttempts >= 3 aktif). "
                "Pantau aktivitas akun."
            )
        else:
            st.success("Transaksi berada dalam rentang perilaku normal di seluruh model.")

        st.info(f"Rekomendasi segmen: {CLUSTER_RECOMMENDATIONS[result['cluster']]}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            with st.container(border=True):
                st.markdown("**Skor anomali tiap model (0-100)**")
                score_df = pd.DataFrame({
                    "Model": list(result["model_scores"].keys()),
                    "Skor": list(result["model_scores"].values()),
                    "Status": [
                        "Anomali" if result["model_flags"][m] else "Normal"
                        for m in result["model_scores"]
                    ],
                })
                fig_scores = px.bar(
                    score_df,
                    x="Skor",
                    y="Model",
                    orientation="h",
                    color="Status",
                    color_discrete_map={"Anomali": "#B91C1C", "Normal": "#1D4ED8"},
                    text=score_df["Skor"].map("{:.1f}".format),
                    template="plotly_white",
                )
                fig_scores.update_traces(marker_cornerradius=4)
                fig_scores.update_layout(
                    xaxis_range=[0, 108],
                    yaxis_title="",
                    margin=dict(t=15, b=15, l=15, r=20),
                    height=230,
                )
                st.plotly_chart(fig_scores, width="stretch")

        with chart_col2:
            with st.container(border=True):
                st.markdown("**Jarak ke centroid K-Means (skala standar)**")
                dist_df = pd.DataFrame({
                    "Segmen": list(result["centroid_distances"].keys()),
                    "Jarak": list(result["centroid_distances"].values()),
                })
                max_dist = max(dist_df["Jarak"].max(), 1.0)
                fig_dist = px.bar(
                    dist_df,
                    x="Jarak",
                    y="Segmen",
                    orientation="h",
                    color="Segmen",
                    color_discrete_map=CLUSTER_COLORS,
                    text=dist_df["Jarak"].map("{:.3f}".format),
                    template="plotly_white",
                )
                fig_dist.update_traces(marker_cornerradius=4)
                fig_dist.update_layout(
                    showlegend=False,
                    xaxis_range=[0, max_dist * 1.2],
                    yaxis_title="",
                    margin=dict(t=15, b=15, l=15, r=20),
                    height=230,
                )
                st.plotly_chart(fig_dist, width="stretch")


def render_tab_eval(df, scaler_km, kmeans, anomaly_bundle):
    st.subheader("Evaluasi Model Unsupervised & Keterbatasan")
    st.caption("Evaluasi konsensus 3 detektor anomali pada 8 fitur serta seleksi cluster K-Means.")

    k_evaluation = load_k_evaluation()
    silhouette_k2 = float(k_evaluation.loc[k_evaluation["k"] == 2, "Silhouette Score"].iloc[0])
    contamination = anomaly_bundle.get("contamination", 0.05)
    consensus_count = int((df["Anomaly_Votes"] >= 2).sum())
    met1, met2, met3, met4 = st.columns(4)
    with met1:
        with st.container(border=True):
            st.metric("K-Means (k=2) Silhouette", f"{silhouette_k2:.4f}")
    with met2:
        with st.container(border=True):
            st.metric(
                "PCA Komponen (90% Var)",
                f"{anomaly_bundle['pca_n_components']} ({anomaly_bundle['pca_explained_variance']*100:.2f}%)",
            )
    with met3:
        with st.container(border=True):
            st.metric(
                "Anomali Konsensus (>= 2 model)",
                f"{consensus_count} ({consensus_count/len(df)*100:.2f}%)",
            )
    with met4:
        with st.container(border=True):
            st.metric("Fitur Deteksi Anomali", str(len(ANOMALY_FEATURES)))

    eval_col1, eval_col2 = st.columns([1, 1.25])
    with eval_col1:
        with st.container(border=True):
            st.markdown("**Kesepakatan antar-model (jumlah transaksi yang sama)**")
            flags = df[["ISO_Flag", "LOF_Flag", "PCA_Flag"]].astype(int)
            flags.columns = ["Isolation Forest", "LOF", "PCA Error"]
            overlap = flags.T @ flags
            fig_ov = px.imshow(
                overlap,
                text_auto=True,
                color_continuous_scale=CLUSTER_GRADIENTS["blue"],
                template="plotly_white",
            )
            fig_ov.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=340)
            st.plotly_chart(fig_ov, width="stretch")

    with eval_col2:
        with st.container(border=True):
            st.markdown("**Rata-rata fitur per tingkat konsensus (0-3 model)**")
            vote_prof = (
                df.groupby("Anomaly_Votes")[[
                    "TransactionAmount",
                    "AccountBalance",
                    "Balance_Ratio",
                    "Amount_Per_Second",
                    "LoginAttempts",
                ]]
                .mean()
                .round(2)
                .reset_index()
            )
            st.dataframe(vote_prof, width="stretch", hide_index=True)

    with st.expander("Evaluasi Jumlah Cluster K-Means (k = 2..8)"):
        k_col1, k_col2 = st.columns(2)
        fig_elbow = px.line(
            k_evaluation, x="k", y="Inertia", markers=True,
            title="Inertia per k (Elbow Method)", template="plotly_white",
        )
        fig_elbow.update_traces(line_color="#1D4ED8")
        fig_elbow.add_vline(x=2, line_dash="dash", line_color="#B91C1C")
        k_col1.plotly_chart(fig_elbow, width="stretch")

        fig_sil = px.line(
            k_evaluation, x="k", y="Silhouette Score", markers=True,
            title="Silhouette Score per k", template="plotly_white",
        )
        fig_sil.update_traces(line_color="#1D4ED8")
        fig_sil.add_vline(x=2, line_dash="dash", line_color="#B91C1C")
        k_col2.plotly_chart(fig_sil, width="stretch")

        centroids_orig = pd.DataFrame(
            scaler_km.inverse_transform(kmeans.cluster_centers_),
            columns=CLUSTERING_FEATURES,
        )
        centroids_orig.insert(0, "Segmen", [CLUSTER_LABELS[i] for i in range(kmeans.n_clusters)])
        st.dataframe(centroids_orig.round(2), width="stretch", hide_index=True)

    st.warning(
        "Keterbatasan Metodologi & Sumber Data:\n"
        "- Tanpa label fraud asli (ground truth): seluruh model bersifat unsupervised, "
        "sehingga precision dan recall terhadap fraud nyata tidak dapat dihitung. "
        "Konsensus antar-model hanya menunjukkan kesepakatan, bukan kebenaran.\n"
        f"- Pemisahan K-Means lemah: Silhouette Score = {silhouette_k2:.4f} (k = 2) menandakan batas segmen "
        "perilaku masih tumpang tindih; deteksi anomali dipisahkan memakai ensemble 3 detektor multivariat.\n"
        f"- Jumlah anomali mengikuti pilihan contamination = {contamination:.2f}, bukan temuan data. "
        "Uji sensitivitas ada di notebook (bagian 7.1).\n"
        "- Days_Since_Prev_Tx: pada data sintetis ini TransactionDate hanya mencakup rentang sempit, "
        "sehingga fitur tersebut praktis mencerminkan PreviousTransactionDate, bukan perilaku jeda yang bermakna.\n"
        "- LoginAttempts ikut menjadi fitur anomali, sehingga Risiko Login dan skor anomali saling tumpang tindih.\n"
        "- Sumber data: dataset latihan Dicoding (adaptasi Kaggle Bank Transaction Dataset for Fraud Detection, "
        "berisi data sintetis kota-kota AS dalam satuan USD)."
    )


def render_custom_styles_and_banner():
    st.markdown(
        """
<style>
.block-container {
    padding-top: 1.75rem;
}
a[href], button, summary, select {
    cursor: pointer;
}
a:focus-visible, button:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid #1D4ED8;
    outline-offset: 2px;
}
[data-testid="stMetricValue"], [data-testid="stDataFrame"] {
    font-variant-numeric: tabular-nums;
}
.page-header {
    background-color: #0F172A;
    border-bottom: 3px solid #B45309;
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 1.25rem;
}
.page-title {
    font-size: clamp(1.35rem, 2.4vw, 1.8rem);
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #F8FAFC;
    line-height: 1.25;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #CBD5E1;
    margin: 0;
    line-height: 1.45;
}
[data-testid="stSidebar"] {
    background-color: #F1F5F9;
    border-right: 1px solid #64748B;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #64748B;
    border-radius: 6px;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
    border: 1px solid #64748B !important;
    border-radius: 6px !important;
    font-weight: 500;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] svg {
    fill: #1E3A8A !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #64748B;
    border-radius: 6px;
    font-weight: 500;
    min-height: 44px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #DBEAFE;
    border-color: #1D4ED8;
    color: #1E3A8A;
}
.filter-status-box {
    background-color: #FFFFFF;
    border: 1px solid #64748B;
    color: #0F172A;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 0.875rem;
    margin-top: 8px;
}
.author-card {
    background-color: #FFFFFF;
    border: 1px solid #64748B;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 14px;
}
.author-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 4px;
}
.author-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.3;
    margin-bottom: 10px;
}
.author-links {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.author-links a {
    background-color: #FFFFFF;
    color: #1D4ED8 !important;
    border: 1px solid #64748B;
    border-radius: 6px;
    min-height: 44px;
    padding: 0 12px;
    display: inline-flex;
    align-items: center;
    font-size: 0.82rem;
    font-weight: 600;
    text-decoration: none !important;
}
.author-links a:hover {
    background-color: #DBEAFE;
    border-color: #1D4ED8;
    color: #1E3A8A !important;
}
@media (max-width: 768px) {
    .page-header {
        padding: 16px 18px;
    }
    .author-links {
        gap: 6px;
    }
}
@media (prefers-reduced-motion: reduce) {
    * {
        animation: none !important;
        transition: none !important;
    }
}
</style>
<div class="page-header">
<h1 class="page-title">Transaction Anomaly Dashboard</h1>
<p class="page-subtitle">Segmentasi perilaku nasabah (K-Means, k = 2) dan deteksi anomali multivariat (Isolation Forest, LOF, PCA Reconstruction).</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Transaction Anomaly Dashboard",
        layout="wide",
    )

    with st.spinner("Memuat dataset dan model..."):
        try:
            df = load_data()
            scaler_km, kmeans, anomaly_bundle = load_models()
        except (FileNotFoundError, OSError, ValueError, KeyError) as err:
            st.error(
                f"Gagal memuat dataset atau model: {err}. "
                "Pastikan file di folder data/ dan models/ tersedia, lalu muat ulang halaman."
            )
            st.stop()

    render_custom_styles_and_banner()

    filtered_df = render_sidebar(df)
    if filtered_df.empty:
        st.warning(
            "Tidak ada transaksi yang cocok dengan filter aktif. "
            "Klik 'Reset Filter' di sidebar untuk menampilkan semua data."
        )
        return

    tab_summary, tab_cluster, tab_anomaly, tab_predict, tab_eval = st.tabs([
        "Ringkasan",
        "Visualisasi Cluster",
        "Deteksi Anomali",
        "Simulasi & Skor Anomali",
        "Evaluasi & Keterbatasan",
    ])

    with tab_summary:
        render_tab_summary(filtered_df)
    with tab_cluster:
        render_tab_cluster(filtered_df, df)
    with tab_anomaly:
        render_tab_anomaly(filtered_df)
    with tab_predict:
        render_tab_predict(df, scaler_km, kmeans, anomaly_bundle)
    with tab_eval:
        render_tab_eval(df, scaler_km, kmeans, anomaly_bundle)


if __name__ == "__main__":
    main()
