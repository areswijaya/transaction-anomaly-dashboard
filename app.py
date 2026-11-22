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

CLUSTER_LABELS = {
    0: "0 - Nasabah Muda Aktif",
    1: "1 - Nasabah Mapan Konservatif",
}

CLUSTER_COLORS = {
    "0 - Nasabah Muda Aktif": "#D97706",
    "1 - Nasabah Mapan Konservatif": "#2563EB",
}

CLUSTER_GRADIENTS = {
    "blue": ["#93C5FD", "#3B82F6", "#1D4ED8"],
    "red": ["#FCA5A5", "#EF4444", "#B91C1C"],
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

K_EVALUATION = pd.DataFrame({
    "k": [2, 3, 4, 5, 6, 7, 8],
    "Inertia": [9813.20, 8156.22, 7027.95, 6061.49, 5265.85, 4759.72, 4368.32],
    "Silhouette Score": [0.2634, 0.2339, 0.2342, 0.2380, 0.2531, 0.2567, 0.2328],
})

PRESET_SCENARIOS = {
    "Manual / Kustom": {
        "TransactionAmount": 250.0,
        "AccountBalance": 5000.0,
        "TransactionDuration": 110,
        "CustomerAge": 35,
        "LoginAttempts": 1,
    },
    "Contoh 1: Nasabah Muda Aktif (Cluster 0)": {
        "TransactionAmount": 350.0,
        "AccountBalance": 1200.0,
        "TransactionDuration": 120,
        "CustomerAge": 25,
        "LoginAttempts": 1,
    },
    "Contoh 2: Nasabah Mapan Konservatif (Cluster 1)": {
        "TransactionAmount": 250.0,
        "AccountBalance": 7500.0,
        "TransactionDuration": 115,
        "CustomerAge": 56,
        "LoginAttempts": 1,
    },
    "Contoh 3: Risiko Login Tinggi (LoginAttempts >= 3)": {
        "TransactionAmount": 310.0,
        "AccountBalance": 5200.0,
        "TransactionDuration": 130,
        "CustomerAge": 45,
        "LoginAttempts": 4,
    },
}


@st.cache_data
def load_data():
    """Muat dataset hasil clustering dan seragamkan label segmen serta flag risiko login."""
    data = pd.read_csv(BASE_DIR / "data" / "data_clustering_clean.csv")
    data["Segment_Name"] = data["Cluster"].map(CLUSTER_LABELS)
    data["High_Login_Risk"] = data["LoginAttempts"] >= LOGIN_RISK_THRESHOLD
    return data


@st.cache_resource
def load_clustering_models():
    """Muat StandardScaler dan model K-Means clustering."""
    scaler = joblib.load(BASE_DIR / "models" / "scaler_clustering.joblib")
    kmeans = joblib.load(BASE_DIR / "models" / "kmeans_model.joblib")
    return scaler, kmeans


@st.cache_data
def get_feature_bounds(_df):
    """Hitung batas min dan max fitur clustering untuk normalisasi radar chart."""
    return _df[CLUSTERING_FEATURES].min(), _df[CLUSTERING_FEATURES].max()


def predict_transaction(
    tx_amount,
    acc_balance,
    tx_duration,
    cust_age,
    login_attempts,
    scaler,
    kmeans,
):
    """Petakan transaksi ke segmen K-Means dan evaluasi aturan risiko login eksplisit."""
    balance_ratio = float(tx_amount / acc_balance) if acc_balance > 0 else 0.0
    input_df = pd.DataFrame([{
        "TransactionAmount": float(tx_amount),
        "CustomerAge": float(cust_age),
        "TransactionDuration": float(tx_duration),
        "AccountBalance": float(acc_balance),
        "Balance_Ratio": balance_ratio,
    }])[CLUSTERING_FEATURES]

    scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=CLUSTERING_FEATURES)
    cluster = int(kmeans.predict(scaled_df)[0])
    centroid_distances = {
        CLUSTER_LABELS[i]: float(np.linalg.norm(scaled[0] - center))
        for i, center in enumerate(kmeans.cluster_centers_)
    }
    return {
        "cluster": cluster,
        "segment_name": CLUSTER_LABELS[cluster],
        "is_high_login_risk": bool(login_attempts >= LOGIN_RISK_THRESHOLD),
        "balance_ratio": balance_ratio,
        "centroid_distances": centroid_distances,
    }


def reset_filters(all_occupations, all_channels, all_types):
    """Kembalikan seluruh filter sidebar ke kondisi awal."""
    st.session_state["filter_occupation"] = all_occupations
    st.session_state["filter_channel"] = all_channels
    st.session_state["filter_type"] = all_types
    st.session_state["filter_location"] = []


def render_sidebar(df):
    """Tampilkan filter sidebar dan kembalikan DataFrame terfilter."""
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
        "Berlaku pada tab Ringkasan, Visualisasi Cluster, dan Risiko Login (Aturan)."
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
        f'<div class="filter-status-badge">Menampilkan <b>{len(filtered_df):,}</b> '
        f'dari <b>{len(df):,}</b> transaksi</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
<div class="author-card">
<div class="author-label">CREATED BY</div>
<div class="author-name">Ilfa Antaries Yusuf Wijaya</div>
<div class="author-links">
<a href="mailto:contact.areswijaya@gmail.com" target="_blank">Email</a>
<a href="https://www.linkedin.com/in/areswijaya" target="_blank">LinkedIn</a>
<a href="https://github.com/areswijaya" target="_blank">GitHub</a>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    return filtered_df


def build_summary_table(filtered_df):
    """Hitung ringkasan profil per segmen dengan konsistensi 2 segmen."""
    total_tx = len(filtered_df)
    summary = (
        filtered_df.groupby("Segment_Name")
        .agg(
            Jumlah_Transaksi=("Cluster", "count"),
            Rata_Transaksi_USD=("TransactionAmount", "mean"),
            Rata_Saldo_USD=("AccountBalance", "mean"),
            Rata_Usia=("CustomerAge", "mean"),
            Rata_Durasi_Detik=("TransactionDuration", "mean"),
            Rata_LoginAttempts=("LoginAttempts", "mean"),
            Rata_Balance_Ratio=("Balance_Ratio", "mean"),
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
    """Render tab Ringkasan KPI, distribusi segmen, komposisi kategori, dan tabel profil."""
    st.subheader("Ringkasan Utama")

    total_tx = len(filtered_df)
    avg_amount = filtered_df["TransactionAmount"].mean()
    avg_balance = filtered_df["AccountBalance"].mean()
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
            st.metric("Rata-Rata Saldo", f"${avg_balance:,.2f}")
    with col4:
        with st.container(border=True):
            st.metric("Risiko Login (>= 3x)", f"{risk_tx:,} ({risk_pct:.2f}%)")

    summary_table = build_summary_table(filtered_df)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown("**Distribusi Segmen**")
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
            top_col1, top_col2 = st.columns([1.4, 1])
            with top_col1:
                st.markdown("**Komposisi per Kategori (%)**")
            with top_col2:
                breakdown_by = st.selectbox(
                    "Kategori",
                    options=["CustomerOccupation", "Channel", "TransactionType"],
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

    summary_display = summary_table[[
        "Segment_Name",
        "Jumlah_Transaksi",
        "Persentase (%)",
        "Rata_Transaksi_USD",
        "Rata_Saldo_USD",
        "Rata_Usia",
        "Rata_Durasi_Detik",
        "Rata_LoginAttempts",
        "Rata_Balance_Ratio",
    ]].rename(columns={
        "Segment_Name": "Segmen",
        "Jumlah_Transaksi": "Jumlah",
        "Persentase (%)": "Persentase (%)",
        "Rata_Transaksi_USD": "Transaksi ($)",
        "Rata_Saldo_USD": "Saldo ($)",
        "Rata_Usia": "Usia (Thn)",
        "Rata_Durasi_Detik": "Durasi (Dtk)",
        "Rata_LoginAttempts": "Upaya Login",
        "Rata_Balance_Ratio": "Balance Ratio",
    })

    with st.container(border=True):
        st.markdown("**Profil Rata-Rata per Segmen**")
        st.dataframe(summary_display.round(2), width="stretch", hide_index=True)


def render_tab_cluster(filtered_df, df):
    """Render tab Visualisasi Cluster dengan scatter plot interaktif, radar chart, dan profil."""
    st.subheader("Visualisasi & Profil Cluster (5 Fitur Perilaku)")

    vis_col1, vis_col2 = st.columns([1.35, 1])

    with vis_col1:
        with st.container(border=True):
            st.markdown("**Scatter Plot Antarfitur**")
            axis_col1, axis_col2 = st.columns(2)
            x_axis = axis_col1.selectbox(
                "Sumbu X",
                options=CLUSTERING_FEATURES,
                index=CLUSTERING_FEATURES.index("AccountBalance"),
            )
            y_axis = axis_col2.selectbox(
                "Sumbu Y",
                options=CLUSTERING_FEATURES,
                index=CLUSTERING_FEATURES.index("TransactionAmount"),
            )

            scatter_df = filtered_df.sort_values("Cluster", ascending=False)
            fig_scatter = px.scatter(
                scatter_df,
                x=x_axis,
                y=y_axis,
                color="Segment_Name",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                hover_data=CLUSTERING_FEATURES
                + ["LoginAttempts", "CustomerOccupation", "Channel", "TransactionType"],
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
            st.markdown("**Radar Profil Segmen (Normalisasi 0-1)**")
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

    st.info(
        "Catatan pemisahan fitur: `LoginAttempts` sengaja dikeluarkan dari proses K-Means "
        "agar segmen merefleksikan perilaku transaksi, sedangkan `LoginAttempts >= 3` "
        "ditangani terpisah sebagai aturan risiko eksplisit."
    )

    st.markdown("**Interpretasi & Rekomendasi Bisnis (Data Keseluruhan)**")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("0 - Nasabah Muda Aktif (34.30%)", expanded=True):
            st.write(
                "**Profil:** `863` transaksi. Rata-rata usia `27.30` thn (dominan `Student`), "
                "saldo `$1,584.53`, transaksi `$347.07`, `Balance_Ratio` `0.47`."
            )
            st.write(
                "**Tindakan:** Tawarkan fitur auto-debit, batas anggaran, atau promo cashback."
            )
    with exp_col2:
        with st.expander("1 - Nasabah Mapan Konservatif (65.70%)", expanded=True):
            st.write(
                "**Profil:** `1,653` transaksi. Rata-rata usia `53.71` thn (dominan `Retired`), "
                "saldo `$6,955.81`, transaksi `$270.63`, `Balance_Ratio` `0.05`."
            )
            st.write(
                "**Tindakan:** Tawarkan deposito, obligasi, atau reksa dana risiko rendah."
            )


def render_tab_risk(filtered_df):
    """Render tab Risiko Login (Aturan) untuk memantau LoginAttempts >= 3 dan mengunduh CSV."""
    st.subheader("Indikator Risiko Login Berbasis Aturan (LoginAttempts >= 3)")
    st.caption(
        "Filter aturan eksplisit untuk transaksi dengan upaya login berulang (3-5 kali). "
        "Ini bukan deteksi fraud tervalidasi karena dataset sintetis tidak memiliki label fraud asli."
    )

    risk_df = filtered_df[filtered_df["High_Login_Risk"]]
    risk_columns = [
        "Segment_Name",
        "TransactionAmount",
        "AccountBalance",
        "Balance_Ratio",
        "LoginAttempts",
        "TransactionDuration",
        "CustomerAge",
        "CustomerOccupation",
        "Channel",
        "TransactionType",
        "Location",
    ]

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        with st.container(border=True):
            st.markdown("**Distribusi Upaya Login per Segmen**")
            fig_login = px.histogram(
                filtered_df,
                x="LoginAttempts",
                color="Segment_Name",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                barmode="group",
                template="plotly_white",
            )
            fig_login.update_layout(
                legend_title_text="Segmen",
                yaxis_title="Jumlah Transaksi",
                xaxis=dict(dtick=1),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(t=40, b=20, l=20, r=20),
                height=370,
            )
            st.plotly_chart(fig_login, width="stretch")

    with risk_col2:
        with st.container(border=True):
            hdr_col1, hdr_col2 = st.columns([1.4, 1])
            with hdr_col1:
                st.markdown("**Sebaran Risiko Login >= 3 (Top 10)**")
            with hdr_col2:
                group_by = st.selectbox(
                    "Kelompokkan",
                    options=["Segment_Name", "Channel", "Location", "CustomerOccupation"],
                    label_visibility="collapsed",
                )

            if risk_df.empty:
                st.info("Tidak ada transaksi dengan LoginAttempts >= 3 pada filter aktif.")
            else:
                risk_counts = (
                    risk_df.groupby(group_by)
                    .size()
                    .reset_index(name="Jumlah")
                    .sort_values("Jumlah", ascending=False)
                    .head(10)
                )
                fig_risk = px.bar(
                    risk_counts,
                    x=group_by,
                    y="Jumlah",
                    text="Jumlah",
                    color="Jumlah",
                    color_continuous_scale=CLUSTER_GRADIENTS["red"],
                    template="plotly_white",
                )
                fig_risk.update_traces(textposition="outside", marker_cornerradius=4)
                fig_risk.update_layout(
                    coloraxis_showscale=False,
                    yaxis_title="Jumlah Transaksi",
                    xaxis_title=group_by,
                    margin=dict(t=25, b=20, l=20, r=20),
                    height=330,
                )
                st.plotly_chart(fig_risk, width="stretch")

    with st.container(border=True):
        tbl_col1, tbl_col2 = st.columns([3, 1])
        with tbl_col1:
            st.markdown(f"**Daftar Transaksi Risiko Login >= 3 ({len(risk_df):,} data)**")
        with tbl_col2:
            if not risk_df.empty:
                st.download_button(
                    label="Unduh CSV",
                    data=risk_df[risk_columns].to_csv(index=False).encode("utf-8"),
                    file_name="risiko_login_tinggi.csv",
                    mime="text/csv",
                    width="stretch",
                )

        if risk_df.empty:
            st.info("Tidak ada transaksi dengan LoginAttempts >= 3 pada filter aktif.")
        else:
            st.dataframe(risk_df[risk_columns].round(2), width="stretch", hide_index=True)


def render_tab_predict(df, scaler, kmeans):
    """Render tab Simulasi & Cek Risiko langsung memakai K-Means dan aturan login."""
    st.subheader("Simulasi Segmen (K-Means) & Cek Risiko Login")
    st.caption(
        "Menghitung segmen perilaku langsung dengan StandardScaler + K-Means (`k = 2`), "
        "lalu mengecek aturan `LoginAttempts >= 3` secara terpisah."
    )

    preset_choice = st.selectbox(
        "Skenario Contoh",
        options=list(PRESET_SCENARIOS.keys()),
        key="predict_preset",
    )
    preset_vals = PRESET_SCENARIOS[preset_choice]

    with st.container(border=True):
        with st.form("prediction_form"):
            form_col1, form_col2 = st.columns(2)

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
                tx_duration = st.number_input(
                    "Durasi (Detik)",
                    min_value=int(df["TransactionDuration"].min()),
                    max_value=int(df["TransactionDuration"].max()),
                    value=int(preset_vals["TransactionDuration"]),
                    step=5,
                    key=f"tx_duration_{preset_choice}",
                )

            with form_col2:
                cust_age = st.number_input(
                    "Usia Nasabah (Thn)",
                    min_value=int(df["CustomerAge"].min()),
                    max_value=int(df["CustomerAge"].max()),
                    value=int(preset_vals["CustomerAge"]),
                    step=1,
                    key=f"cust_age_{preset_choice}",
                )
                login_attempts = st.number_input(
                    "Upaya Login",
                    min_value=int(df["LoginAttempts"].min()),
                    max_value=int(df["LoginAttempts"].max()),
                    value=int(preset_vals["LoginAttempts"]),
                    step=1,
                    key=f"login_attempts_{preset_choice}",
                )

            submitted = st.form_submit_button("Hitung Segmen & Cek Risiko", width="stretch")

    if submitted:
        try:
            result = predict_transaction(
                tx_amount=tx_amount,
                acc_balance=acc_balance,
                tx_duration=tx_duration,
                cust_age=cust_age,
                login_attempts=login_attempts,
                scaler=scaler,
                kmeans=kmeans,
            )
        except Exception as err:
            st.error(f"Simulasi gagal dijalankan. Detail: {err}")
            return

        pred_cluster = result["cluster"]
        pred_label = result["segment_name"]
        is_risk = result["is_high_login_risk"]
        risk_status = "Tinggi (>= 3x)" if is_risk else "Normal (< 3x)"

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            with st.container(border=True):
                st.metric("Segmen K-Means", pred_label)
        with res_col2:
            with st.container(border=True):
                st.metric("Status Risiko Login", risk_status)
        with res_col3:
            with st.container(border=True):
                st.metric("Balance Ratio", f"{result['balance_ratio']:.2f}")

        if is_risk:
            st.warning(
                "Aturan Risiko Login Aktif (`LoginAttempts >= 3`): Wajibkan verifikasi OTP/2FA "
                "dan tinjau aktivitas akses. Catatan: indikator ini berbasis aturan eksplisit, "
                "bukan prediksi fraud tervalidasi."
            )
        else:
            st.success("Status Login Normal (`LoginAttempts < 3`).")

        st.info(f"Rekomendasi segmen: {CLUSTER_RECOMMENDATIONS[pred_cluster]}")

        with st.container(border=True):
            st.markdown("**Jarak Euclidean ke Centroid (Skala Standar — lebih kecil berarti lebih dekat)**")
            dist_df = pd.DataFrame({
                "Segmen": list(result["centroid_distances"].keys()),
                "Jarak ke Centroid": list(result["centroid_distances"].values()),
            })
            max_dist = max(dist_df["Jarak ke Centroid"].max(), 1.0)
            fig_dist = px.bar(
                dist_df,
                x="Jarak ke Centroid",
                y="Segmen",
                orientation="h",
                color="Segmen",
                color_discrete_map=CLUSTER_COLORS,
                text=dist_df["Jarak ke Centroid"].map("{:.3f}".format),
                template="plotly_white",
            )
            fig_dist.update_traces(marker_cornerradius=4)
            fig_dist.update_layout(
                showlegend=False,
                xaxis_range=[0, max_dist * 1.2],
                yaxis_title="",
                margin=dict(t=15, b=15, l=15, r=20),
                height=220,
            )
            st.plotly_chart(fig_dist, width="stretch")


def render_tab_eval(scaler, kmeans):
    """Render tab Evaluasi & Keterbatasan untuk pemilihan K-Means dan transparansi metodologi."""
    st.subheader("Evaluasi K-Means & Keterbatasan Metodologi")
    st.caption(
        "Pemilihan jumlah cluster dievaluasi pada 2,516 transaksi menggunakan 5 fitur perilaku "
        "(tanpa `LoginAttempts`)."
    )

    k2_row = K_EVALUATION.loc[K_EVALUATION["k"] == 2].iloc[0]

    met1, met2, met3, met4 = st.columns(4)
    with met1:
        with st.container(border=True):
            st.metric("Jumlah Cluster (k)", "2")
    with met2:
        with st.container(border=True):
            st.metric("Silhouette Score", f"{k2_row['Silhouette Score']:.4f}")
    with met3:
        with st.container(border=True):
            st.metric("Inertia (WCSS)", f"{k2_row['Inertia']:,.2f}")
    with met4:
        with st.container(border=True):
            st.metric("Fitur Clustering", str(len(CLUSTERING_FEATURES)))

    eval_col1, eval_col2 = st.columns(2)

    fig_elbow = px.line(
        K_EVALUATION,
        x="k",
        y="Inertia",
        markers=True,
        title="Elbow Method (Inertia)",
        template="plotly_white",
    )
    fig_elbow.update_traces(
        line_color="#2563EB", fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.10)"
    )
    fig_elbow.add_vline(x=2, line_dash="dash", line_color="#DC2626")
    eval_col1.plotly_chart(fig_elbow, width="stretch")

    fig_sil = px.line(
        K_EVALUATION,
        x="k",
        y="Silhouette Score",
        markers=True,
        title="Silhouette Score per k",
        template="plotly_white",
    )
    fig_sil.update_traces(
        line_color="#2563EB", fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.10)"
    )
    fig_sil.add_vline(x=2, line_dash="dash", line_color="#DC2626")
    eval_col2.plotly_chart(fig_sil, width="stretch")

    st.dataframe(K_EVALUATION, width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown("**Centroid K-Means pada Skala Asli**")
        centroids_orig = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=CLUSTERING_FEATURES,
        )
        centroids_orig.insert(0, "Segmen", [CLUSTER_LABELS[i] for i in range(kmeans.n_clusters)])
        st.dataframe(centroids_orig.round(2), width="stretch", hide_index=True)

    st.warning(
        "**Keterbatasan Metodologi & Sumber Data:**\n"
        "- **Pemisahan cluster lemah:** `Silhouette Score = 0.2634` (tertinggi di `k = 2`, rentang "
        "`k = 2..8`) menunjukkan batas segmen perilaku masih tumpang tindih.\n"
        "- **Tidak ada label fraud asli (*ground truth*):** Kemampuan deteksi fraud tidak dapat diuji atau dibuktikan.\n"
        "- **Indikator berbasis aturan:** `LoginAttempts >= 3` merupakan aturan ambang batas eksplisit, bukan prediksi *machine learning*.\n"
        "- **Sumber data:** Dataset latihan Dicoding (adaptasi Kaggle *Bank Transaction Dataset for Fraud Detection* — data sintetis kota-kota AS dalam satuan USD)."
    )


def render_custom_styles_and_banner():
    """Terapkan styling tema dashboard dan tampilkan hero header banner."""
    st.markdown(
        """
<style>
.block-container {
    padding-top: 2rem;
}
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #2563EB 100%);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
    color: #FFFFFF;
}
.hero-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.14);
    color: #93C5FD;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #FFFFFF;
    line-height: 1.25;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #DBEAFE;
    margin: 0;
}
[data-testid="stSidebar"] {
    background-color: #F1F5F9;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #CBD5E1;
    border-radius: 8px;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #DBEAFE !important;
    color: #1E40AF !important;
    border: 1px solid #93C5FD !important;
    border-radius: 6px !important;
    font-weight: 500;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] svg {
    fill: #1E40AF !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FFFFFF;
    color: #334155;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    font-weight: 500;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #EFF6FF;
    border-color: #2563EB;
    color: #1D4ED8;
}
.filter-status-badge {
    background-color: #EFF6FF;
    border-left: 3px solid #2563EB;
    color: #1E3A8A;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    margin-top: 8px;
}
.author-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 14px;
}
.author-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-bottom: 4px;
}
.author-name {
    font-size: 0.9rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.3;
    margin-bottom: 10px;
}
.author-links {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.author-links a {
    background-color: #EFF6FF;
    color: #1E40AF !important;
    border: 1px solid #BFDBFE;
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 0.76rem;
    font-weight: 600;
    text-decoration: none !important;
}
.author-links a:hover {
    background-color: #DBEAFE;
    border-color: #2563EB;
}
[data-testid="stExpanderDetails"] {
    min-height: 135px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
</style>
<div class="hero-banner">
<div class="hero-badge">TRANSACTION SEGMENTATION &amp; LOGIN RISK RULES</div>
<h1 class="hero-title">Segmentasi Transaksi &amp; Indikator Risiko Login</h1>
<p class="hero-subtitle">Segmentasi perilaku nasabah (K-Means, k = 2) dan filter risiko login berbasis aturan (LoginAttempts &gt;= 3).</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Konfigurasi halaman dan jalankan alur utama dashboard."""
    st.set_page_config(
        page_title="Dashboard Segmentasi Transaksi & Indikator Risiko Login",
        layout="wide",
    )

    df = load_data()
    scaler, kmeans = load_clustering_models()

    render_custom_styles_and_banner()

    filtered_df = render_sidebar(df)
    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok. Pilih minimal satu opsi pada setiap filter.")
        return

    tab_summary, tab_cluster, tab_risk, tab_predict, tab_eval = st.tabs([
        "Ringkasan",
        "Visualisasi Cluster",
        "Risiko Login (Aturan)",
        "Simulasi & Cek Risiko",
        "Evaluasi & Keterbatasan",
    ])

    with tab_summary:
        render_tab_summary(filtered_df)
    with tab_cluster:
        render_tab_cluster(filtered_df, df)
    with tab_risk:
        render_tab_risk(filtered_df)
    with tab_predict:
        render_tab_predict(df, scaler, kmeans)
    with tab_eval:
        render_tab_eval(scaler, kmeans)


if __name__ == "__main__":
    main()
