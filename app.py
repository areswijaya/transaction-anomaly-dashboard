from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

CLUSTER_COLORS = {
    "0 - Nasabah Mapan Konservatif": "#2563EB",
    "1 - Nasabah Muda Aktif": "#D97706",
    "2 - Transaksi Berisiko": "#DC2626",
}

CLUSTER_GRADIENTS = {
    "blue": ["#93C5FD", "#3B82F6", "#1D4ED8"],
    "matrix_blue": ["#EFF6FF", "#60A5FA", "#1D4ED8"],
    "red": ["#FCA5A5", "#EF4444", "#B91C1C"],
}

CLUSTERING_FEATURES = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "Balance_Ratio",
]

CLUSTER_LABELS = {
    0: "0 - Nasabah Mapan Konservatif",
    1: "1 - Nasabah Muda Aktif",
    2: "2 - Transaksi Berisiko",
}

CLUSTER_RECOMMENDATIONS = {
    0: "Transaksi normal. Rekomendasikan deposito, obligasi, atau reksa dana risiko rendah.",
    1: "Transaksi normal (penarikan aktif). Rekomendasikan auto-debit, fitur budget, atau cashback.",
    2: "ANOMALI LOGIN: Wajibkan verifikasi OTP/2FA dan kirim peringatan keamanan ke nasabah.",
}

K_EVALUATION = pd.DataFrame({
    "k": [2, 3, 4, 5, 6, 7, 8],
    "Inertia": [12329.12, 10169.29, 8589.45, 7503.40, 6536.45, 5778.12, 5276.25],
    "Silhouette Score": [0.2445, 0.2657, 0.2352, 0.2346, 0.2394, 0.2521, 0.2553],
})

PRESET_SCENARIOS = {
    "Manual / Kustom": {
        "TransactionAmount": 250.0,
        "AccountBalance": 5000.0,
        "TransactionDuration": 110,
        "CustomerAge": 35,
        "LoginAttempts": 1,
        "CustomerOccupation": "Engineer",
        "Channel": "Branch",
        "TransactionType": "Debit",
    },
    "Contoh 1: Nasabah Mapan Konservatif (Cluster 0)": {
        "TransactionAmount": 250.0,
        "AccountBalance": 8500.0,
        "TransactionDuration": 110,
        "CustomerAge": 58,
        "LoginAttempts": 1,
        "CustomerOccupation": "Retired",
        "Channel": "Branch",
        "TransactionType": "Debit",
    },
    "Contoh 2: Nasabah Muda Aktif (Cluster 1)": {
        "TransactionAmount": 420.0,
        "AccountBalance": 950.0,
        "TransactionDuration": 95,
        "CustomerAge": 22,
        "LoginAttempts": 1,
        "CustomerOccupation": "Student",
        "Channel": "Online",
        "TransactionType": "Debit",
    },
    "Contoh 3: Transaksi Berisiko (Cluster 2)": {
        "TransactionAmount": 310.0,
        "AccountBalance": 5200.0,
        "TransactionDuration": 140,
        "CustomerAge": 45,
        "LoginAttempts": 4,
        "CustomerOccupation": "Doctor",
        "Channel": "Online",
        "TransactionType": "Debit",
    },
}


@st.cache_data
def load_data():
    """Muat dataset hasil clustering dan seragamkan label segmen tampilan."""
    data = pd.read_csv(BASE_DIR / "data" / "data_clustering_clean.csv")
    data["Segment_Name"] = data["Cluster"].map(CLUSTER_LABELS)
    return data


@st.cache_resource
def load_model():
    """Muat pipeline model klasifikasi Random Forest."""
    return joblib.load(BASE_DIR / "models" / "rf_classifier_pipeline.joblib")


@st.cache_data
def get_feature_bounds(_df):
    """Hitung batas min dan max fitur clustering untuk normalisasi radar chart."""
    return _df[CLUSTERING_FEATURES].min(), _df[CLUSTERING_FEATURES].max()


@st.cache_data
def evaluate_model(_df, _model):
    """Hitung metrik pada data uji (20% stratified split, random_state=42)."""
    features = _df.drop(columns=["Cluster", "Segment_Name", "Location"])
    target = _df["Cluster"]
    _, x_test, _, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    y_pred = _model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    return report, matrix


@st.cache_data
def get_feature_importance(_model):
    """Ambil feature importance Random Forest dengan nama fitur bersih."""
    raw_names = _model.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
    importances = _model.named_steps["classifier"].feature_importances_
    return (
        pd.DataFrame({"Fitur": clean_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


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
    st.sidebar.caption("Berlaku pada tab Ringkasan, Cluster, dan Risiko.")

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
    """Hitung ringkasan profil per segmen dengan konsistensi 3 segmen."""
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
    summary["Persentase (%)"] = (summary["Jumlah_Transaksi"] / total_tx) * 100
    return summary


def render_tab_summary(filtered_df):
    """Render tab Ringkasan KPI, distribusi segmen, komposisi kategori, dan tabel profil."""
    st.subheader("Ringkasan Utama")

    total_tx = len(filtered_df)
    avg_amount = filtered_df["TransactionAmount"].mean()
    avg_balance = filtered_df["AccountBalance"].mean()
    risk_tx = int((filtered_df["Cluster"] == 2).sum())
    risk_pct = (risk_tx / total_tx) * 100

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
            st.metric("Transaksi Berisiko (Cluster 2)", f"{risk_tx:,} ({risk_pct:.2f}%)")

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
    st.subheader("Visualisasi & Profil Cluster")

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

            # Urutkan Cluster agar titik Cluster 2 (merah) berada paling atas
            scatter_df = filtered_df.sort_values("Cluster")
            fig_scatter = px.scatter(
                scatter_df,
                x=x_axis,
                y=y_axis,
                color="Segment_Name",
                color_discrete_map=CLUSTER_COLORS,
                category_orders={"Segment_Name": list(CLUSTER_COLORS.keys())},
                hover_data=CLUSTERING_FEATURES + ["CustomerOccupation", "Channel", "TransactionType"],
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

    st.markdown("**Interpretasi & Rekomendasi Bisnis (Data Keseluruhan)**")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        with st.expander("0 - Nasabah Mapan Konservatif (63.43%)", expanded=True):
            st.write(
                "**Profil:** Usia `53.67` thn (dominan `Retired`), saldo `$6,933.26`, "
                "transaksi `$272.13`, `Balance_Ratio` `0.05`, login `1.01`."
            )
            st.write("**Tindakan:** Tawarkan deposito, obligasi, atau reksa dana risiko rendah.")
    with exp_col2:
        with st.expander("1 - Nasabah Muda Aktif (32.87%)", expanded=True):
            st.write(
                "**Profil:** Usia `27.25` thn (dominan `Student`), saldo `$1,578.34`, "
                "transaksi `$347.13`, `Balance_Ratio` `0.48`, login `1.01`."
            )
            st.write("**Tindakan:** Tawarkan auto-debit, fitur budget, atau promo cashback.")
    with exp_col3:
        with st.expander("2 - Transaksi Berisiko (3.70%)", expanded=True):
            st.write(
                "**Profil:** Anomali upaya login `3-5` kali (`4.00`), durasi `130.77` dtk, "
                "saldo `$5,318.53`, dominan `Doctor` & `Online`."
            )
            st.write("**Tindakan:** Wajibkan verifikasi OTP/2FA, notifikasi instan, dan audit.")


def render_tab_risk(filtered_df):
    """Render tab Transaksi Berisiko untuk memantau Cluster 2 dan mengunduh CSV."""
    st.subheader("Analisis Transaksi Berisiko (Cluster 2)")
    st.caption("Segmen ini mengisolasi transaksi dengan anomali upaya login berulang (3-5 kali).")

    risk_df = filtered_df[filtered_df["Cluster"] == 2]
    risk_columns = [
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
                st.markdown("**Sebaran Cluster 2 (Top 10)**")
            with hdr_col2:
                group_by = st.selectbox(
                    "Kelompokkan",
                    options=["Channel", "Location", "CustomerOccupation"],
                    label_visibility="collapsed",
                )

            if risk_df.empty:
                st.info("Tidak ada transaksi berisiko pada filter aktif.")
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
            st.markdown(f"**Daftar Transaksi Berisiko ({len(risk_df):,} data)**")
        with tbl_col2:
            if not risk_df.empty:
                st.download_button(
                    label="Unduh CSV",
                    data=risk_df[risk_columns].to_csv(index=False).encode("utf-8"),
                    file_name="transaksi_berisiko.csv",
                    mime="text/csv",
                    width="stretch",
                )

        if risk_df.empty:
            st.info("Tidak ada transaksi berisiko (Cluster 2) pada filter aktif.")
        else:
            st.dataframe(risk_df[risk_columns].round(2), width="stretch", hide_index=True)


def render_tab_predict(df, model):
    """Render tab Prediksi Transaksi Baru menggunakan pipeline Random Forest."""
    st.subheader("Prediksi Transaksi Baru (Random Forest)")
    st.caption("Pilih skenario contoh atau ubah parameter untuk menguji prediksi segmen.")

    preset_choice = st.selectbox(
        "Skenario Contoh",
        options=list(PRESET_SCENARIOS.keys()),
        key="predict_preset",
    )
    preset_vals = PRESET_SCENARIOS[preset_choice]

    occ_options = sorted(df["CustomerOccupation"].unique())
    ch_options = sorted(df["Channel"].unique())
    type_options = sorted(df["TransactionType"].unique())

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
                    min_value=float(df["AccountBalance"].min()),
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
                cust_occupation = st.selectbox(
                    "Profesi",
                    options=occ_options,
                    index=occ_options.index(preset_vals["CustomerOccupation"]),
                    key=f"cust_occupation_{preset_choice}",
                )

            with form_col3:
                tx_channel = st.selectbox(
                    "Kanal",
                    options=ch_options,
                    index=ch_options.index(preset_vals["Channel"]),
                    key=f"tx_channel_{preset_choice}",
                )
                tx_type = st.selectbox(
                    "Tipe Transaksi",
                    options=type_options,
                    index=type_options.index(preset_vals["TransactionType"]),
                    key=f"tx_type_{preset_choice}",
                )

            submitted = st.form_submit_button("Prediksi Segmen", width="stretch")

    if submitted:
        balance_ratio = tx_amount / acc_balance if acc_balance > 0 else 0.0
        input_df = pd.DataFrame([{
            "TransactionAmount": tx_amount,
            "TransactionType": tx_type,
            "Channel": tx_channel,
            "CustomerAge": cust_age,
            "CustomerOccupation": cust_occupation,
            "TransactionDuration": tx_duration,
            "LoginAttempts": login_attempts,
            "AccountBalance": acc_balance,
            "Balance_Ratio": balance_ratio,
        }])

        try:
            pred_cluster = int(model.predict(input_df)[0])
            proba_values = model.predict_proba(input_df)[0]
            class_list = [int(c) for c in model.classes_]
        except Exception as err:
            st.error(f"Prediksi gagal dijalankan. Detail: {err}")
            return

        pred_proba = float(proba_values[class_list.index(pred_cluster)]) * 100
        pred_label = CLUSTER_LABELS[pred_cluster]

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            with st.container(border=True):
                st.metric("Segmen Prediksi", pred_label)
        with res_col2:
            with st.container(border=True):
                st.metric("Confidence", f"{pred_proba:.2f}%")
        with res_col3:
            with st.container(border=True):
                st.metric("Balance Ratio", f"{balance_ratio:.2f}")

        if pred_cluster == 2:
            st.error(CLUSTER_RECOMMENDATIONS[pred_cluster])
        else:
            st.success(CLUSTER_RECOMMENDATIONS[pred_cluster])

        with st.container(border=True):
            st.markdown("**Probabilitas per Segmen**")
            proba_df = pd.DataFrame({
                "Segmen": [CLUSTER_LABELS[c] for c in class_list],
                "Probabilitas (%)": [p * 100 for p in proba_values],
            })
            fig_proba = px.bar(
                proba_df,
                x="Probabilitas (%)",
                y="Segmen",
                orientation="h",
                color="Segmen",
                color_discrete_map=CLUSTER_COLORS,
                text=proba_df["Probabilitas (%)"].map("{:.1f}%".format),
                template="plotly_white",
            )
            fig_proba.update_traces(marker_cornerradius=4)
            fig_proba.update_layout(
                showlegend=False,
                xaxis_range=[0, 105],
                yaxis_title="",
                margin=dict(t=15, b=15, l=15, r=20),
                height=240,
            )
            st.plotly_chart(fig_proba, width="stretch")


def render_tab_eval(df, model):
    """Render tab Evaluasi Model untuk Random Forest dan seleksi K-Means."""
    st.subheader("Evaluasi Model")
    st.caption("Random Forest dievaluasi pada 20% data uji (504 transaksi, stratified split).")

    try:
        report, matrix = evaluate_model(df, model)
    except Exception as err:
        st.error(f"Evaluasi model gagal dihitung. Detail: {err}")
    else:
        met1, met2, met3, met4 = st.columns(4)
        with met1:
            with st.container(border=True):
                st.metric("Akurasi Uji", f"{report['accuracy'] * 100:.2f}%")
        with met2:
            with st.container(border=True):
                st.metric("Precision (Macro)", f"{report['macro avg']['precision'] * 100:.2f}%")
        with met3:
            with st.container(border=True):
                st.metric("Recall (Macro)", f"{report['macro avg']['recall'] * 100:.2f}%")
        with met4:
            with st.container(border=True):
                st.metric("F1-Score (Macro)", f"{report['macro avg']['f1-score'] * 100:.2f}%")

        eval_vis_col1, eval_vis_col2 = st.columns(2)

        with eval_vis_col1:
            with st.container(border=True):
                st.markdown("**Confusion Matrix**")
                fig_cm = px.imshow(
                    matrix,
                    x=[f"Cluster {i}" for i in range(3)],
                    y=[f"Cluster {i}" for i in range(3)],
                    text_auto=True,
                    color_continuous_scale=CLUSTER_GRADIENTS["matrix_blue"],
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    template="plotly_white",
                )
                fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=360)
                st.plotly_chart(fig_cm, width="stretch")

        with eval_vis_col2:
            with st.container(border=True):
                st.markdown("**10 Fitur Paling Berpengaruh**")
                try:
                    feat_imp = get_feature_importance(model).head(10).iloc[::-1]
                except Exception as err:
                    st.info(f"Data pengaruh fitur tidak tersedia. Detail: {err}")
                else:
                    fig_imp = px.bar(
                        feat_imp,
                        x="Importance",
                        y="Fitur",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale=CLUSTER_GRADIENTS["blue"],
                        template="plotly_white",
                    )
                    fig_imp.update_traces(marker_cornerradius=4)
                    fig_imp.update_layout(
                        coloraxis_showscale=False,
                        xaxis_title="Tingkat Pengaruh",
                        yaxis_title="",
                        margin=dict(t=20, b=20, l=20, r=20),
                        height=360,
                    )
                    st.plotly_chart(fig_imp, width="stretch")

        with st.container(border=True):
            st.markdown("**Metrik per Segmen**")
            per_class = pd.DataFrame({
                "Segmen": [CLUSTER_LABELS[i] for i in range(3)],
                "Precision": [report[str(i)]["precision"] for i in range(3)],
                "Recall": [report[str(i)]["recall"] for i in range(3)],
                "F1-Score": [report[str(i)]["f1-score"] for i in range(3)],
                "Data Uji": [int(report[str(i)]["support"]) for i in range(3)],
            })
            st.dataframe(per_class.round(4), width="stretch", hide_index=True)

    with st.expander("Pemilihan Jumlah Cluster (K-Means)"):
        st.write("Dipilih `k = 3` karena menghasilkan **Silhouette Score tertinggi (`0.2657`)**.")
        eval_col1, eval_col2 = st.columns(2)

        fig_elbow = px.line(
            K_EVALUATION, x="k", y="Inertia", markers=True,
            title="Elbow Method (Inertia)", template="plotly_white",
        )
        fig_elbow.update_traces(
            line_color="#2563EB", fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.10)"
        )
        fig_elbow.add_vline(x=3, line_dash="dash", line_color="#DC2626")
        eval_col1.plotly_chart(fig_elbow, width="stretch")

        fig_sil = px.line(
            K_EVALUATION, x="k", y="Silhouette Score", markers=True,
            title="Silhouette Score per k", template="plotly_white",
        )
        fig_sil.update_traces(
            line_color="#2563EB", fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.10)"
        )
        fig_sil.add_vline(x=3, line_dash="dash", line_color="#DC2626")
        eval_col2.plotly_chart(fig_sil, width="stretch")
        st.dataframe(K_EVALUATION, width="stretch", hide_index=True)


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
    min-height: 145px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
</style>
<div class="hero-banner">
<div class="hero-badge">FINTECH ANALYTICS &amp; ANOMALY DETECTION</div>
<h1 class="hero-title">Segmentasi &amp; Deteksi Anomali Transaksi</h1>
<p class="hero-subtitle">Analisis segmen nasabah (K-Means) dan prediksi transaksi baru (Random Forest).</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Konfigurasi halaman dan jalankan alur utama dashboard."""
    st.set_page_config(
        page_title="Dashboard Segmentasi & Deteksi Anomali Transaksi",
        layout="wide",
    )

    df = load_data()
    model = load_model()

    render_custom_styles_and_banner()

    filtered_df = render_sidebar(df)
    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok. Pilih minimal satu opsi pada setiap filter.")
        return

    tab_summary, tab_cluster, tab_risk, tab_predict, tab_eval = st.tabs([
        "Ringkasan",
        "Visualisasi Cluster",
        "Transaksi Berisiko",
        "Prediksi",
        "Evaluasi Model",
    ])

    with tab_summary:
        render_tab_summary(filtered_df)
    with tab_cluster:
        render_tab_cluster(filtered_df, df)
    with tab_risk:
        render_tab_risk(filtered_df)
    with tab_predict:
        render_tab_predict(df, model)
    with tab_eval:
        render_tab_eval(df, model)


if __name__ == "__main__":
    main()
