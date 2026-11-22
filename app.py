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

SEGMENT_COLORS = {
    "Nasabah Muda Aktif": "#B45309",
    "Nasabah Mapan Konservatif": "#1D4ED8",
}

SEGMENT_ACTIONS = {
    "Nasabah Muda Aktif": "Tawarkan fitur auto-debit, batas anggaran, atau promo cashback.",
    "Nasabah Mapan Konservatif": "Tawarkan deposito, obligasi, atau reksa dana risiko rendah.",
}

ANOMALY_COLORS = {
    True: "#B91C1C",
    False: "#64748B",
}

CLUSTER_GRADIENTS_BLUE = ["#DBEAFE", "#60A5FA", "#1D4ED8"]

PRESET_SCENARIOS = {
    "Isi sendiri": {"amount": 250.0, "balance": 5000.0, "duration": 110, "age": 35, "login": 1},
    "Nasabah muda, normal": {"amount": 350.0, "balance": 1200.0, "duration": 120, "age": 25, "login": 1},
    "Nasabah mapan, normal": {"amount": 250.0, "balance": 7500.0, "duration": 115, "age": 56, "login": 1},
    "Nominal besar, saldo kecil": {"amount": 1450.0, "balance": 800.0, "duration": 35, "age": 40, "login": 1},
    "Login berulang": {"amount": 310.0, "balance": 5200.0, "duration": 140, "age": 45, "login": 4},
}


def _style_chart(fig, height=360, show_legend=True):
    fig.update_layout(
        font=dict(color="#0F172A", size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=show_legend,
        height=height,
    )
    fig.update_xaxes(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        tickfont=dict(color="#0F172A", size=11),
        title_font=dict(color="#334155", size=12),
    )
    fig.update_yaxes(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        tickfont=dict(color="#0F172A", size=11),
        title_font=dict(color="#334155", size=12),
    )
    return fig


@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / "data" / "data_clustering_clean.csv")
    data["High_Login_Risk"] = data["LoginAttempts"] >= LOGIN_RISK_THRESHOLD
    return data


@st.cache_resource
def load_models():
    scaler_km = joblib.load(BASE_DIR / "models" / "scaler_clustering.joblib")
    kmeans = joblib.load(BASE_DIR / "models" / "kmeans_model.joblib")
    anomaly = joblib.load(BASE_DIR / "models" / "anomaly_model.joblib")
    return scaler_km, kmeans, anomaly


@st.cache_data
def load_k_evaluation():
    path = BASE_DIR / "models" / "k_evaluation.csv"
    try:
        evaluation = pd.read_csv(path)
        required = {"k", "Inertia", "Silhouette Score"}
        if required.issubset(evaluation.columns) and not evaluation.empty:
            return evaluation[["k", "Inertia", "Silhouette Score"]]
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        pass
    return None


@st.cache_data
def get_feature_bounds(_df):
    return _df[CLUSTERING_FEATURES].min(), _df[CLUSTERING_FEATURES].max()


def get_segment_map(df):
    """Peta nomor cluster ke nama segmen, diambil dari data hasil notebook."""
    return df.groupby("Cluster")["Segment_Name"].first().to_dict()


def predict_transaction(
    amount,
    balance,
    duration,
    age,
    login_attempts,
    scaler_km,
    kmeans,
    anomaly,
    segment_map,
):
    """Petakan satu transaksi ke segmen K-Means, skor anomali 0-100, dan aturan login."""
    balance_ratio = float(amount / balance) if balance > 0 else 0.0
    amount_per_second = float(amount / max(duration, 1))

    km_row = pd.DataFrame([{
        "TransactionAmount": float(amount),
        "CustomerAge": float(age),
        "TransactionDuration": float(duration),
        "AccountBalance": float(balance),
        "Balance_Ratio": balance_ratio,
    }])[CLUSTERING_FEATURES]
    km_scaled = pd.DataFrame(scaler_km.transform(km_row), columns=CLUSTERING_FEATURES)
    cluster = int(kmeans.predict(km_scaled)[0])

    an_row = pd.DataFrame([{
        "TransactionAmount": float(amount),
        "CustomerAge": float(age),
        "TransactionDuration": float(duration),
        "AccountBalance": float(balance),
        "Balance_Ratio": balance_ratio,
        "Amount_Per_Second": amount_per_second,
    }])[anomaly["features"]]
    an_scaled = anomaly["scaler"].transform(an_row)

    raw = float(-anomaly["model"].decision_function(an_scaled)[0])
    lo, hi = anomaly["score_bounds"]
    score = 0.0 if hi <= lo else float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0)) * 100.0
    # Isolation Forest menandai anomali saat -decision_function(x) > 0; konversi 0.0 ke skala 0-100.
    thresh = 0.0 if hi <= lo else float(np.clip((0.0 - lo) / (hi - lo), 0.0, 1.0)) * 100.0
    is_anomaly = bool(anomaly["model"].predict(an_scaled)[0] == -1)

    return {
        "segment": segment_map.get(cluster, f"Cluster {cluster}"),
        "balance_ratio": balance_ratio,
        "amount_per_second": amount_per_second,
        "anomaly_score": round(score, 2),
        "threshold_score": round(thresh, 2),
        "is_anomaly": is_anomaly,
        "is_high_login_risk": bool(login_attempts >= LOGIN_RISK_THRESHOLD),
    }


def render_styles_and_banner(df):
    total_tx = len(df)
    anom_tx = int(df["Anomaly_Flag"].sum())
    anom_pct = (anom_tx / total_tx) * 100 if total_tx > 0 else 0.0
    risk_tx = int(df["High_Login_Risk"].sum())
    risk_pct = (risk_tx / total_tx) * 100 if total_tx > 0 else 0.0

    st.markdown(
        f"""
<style>
.block-container {{
    padding-top: 3.75rem !important;
}}
a[href], button, summary, select, [role="tab"] {{
    cursor: pointer;
}}
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, [tabindex]:focus-visible {{
    outline: 2px solid #1D4ED8;
    outline-offset: 2px;
}}
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .meta-value {{
    font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricLabel"] p {{
    color: #475569 !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
    color: #0F172A !important;
    font-weight: 700 !important;
}}
[class*="st-key-card_"] {{
    background-color: #FFFFFF;
    border-color: #CBD5E1 !important;
    border-radius: 8px;
}}
.page-header {{
    background-color: #0F172A;
    border-bottom: 3px solid #B45309;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 1.15rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}}
.header-copy {{
    flex: 1 1 360px;
}}
.page-header .page-title {{
    font-size: clamp(1.4rem, 2.5vw, 1.9rem) !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    padding: 0 !important;
    color: #F8FAFC !important;
    line-height: 1.2 !important;
}}
.page-header .page-subtitle {{
    font-size: 0.94rem !important;
    color: #E2E8F0 !important;
    margin: 0 !important;
    line-height: 1.45 !important;
}}
.header-meta {{
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}}
.meta-item {{
    background-color: #1E293B;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 8px 14px;
    min-width: 130px;
}}
.meta-value {{
    font-size: 1rem;
    font-weight: 700;
    color: #F8FAFC !important;
    line-height: 1.25;
}}
.meta-label {{
    font-size: 0.76rem;
    font-weight: 500;
    color: #CBD5E1 !important;
    margin-top: 2px;
}}
[data-testid="stTabs"] [role="tablist"] {{
    background-color: #FFFFFF;
    border: 1px solid #64748B;
    border-radius: 8px;
    padding: 4px;
    gap: 6px;
    flex-wrap: wrap;
}}
[data-testid="stTabs"] [role="tab"] {{
    min-height: 44px;
    padding: 8px 16px;
    border-radius: 6px;
    color: #334155 !important;
    font-weight: 600 !important;
    background-color: transparent;
}}
[data-testid="stTabs"] [role="tab"] p {{
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    color: inherit !important;
}}
[data-testid="stTabs"] [role="tab"]:hover {{
    background-color: #F1F5F9;
    color: #0F172A !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    background-color: #0F172A !important;
    color: #F8FAFC !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{
    display: none !important;
}}
.segment-card-head {{
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 2px;
}}
.segment-card-sub {{
    font-size: 0.86rem;
    color: #475569;
    margin-bottom: 10px;
}}
[data-testid="stSidebar"] {{
    background-color: #F1F5F9;
    border-right: 1px solid #CBD5E1;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background-color: #FFFFFF;
    border-color: #64748B;
    border-radius: 6px;
}}
[data-testid="stSidebar"] span[data-baseweb="tag"] {{
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
    border: 1px solid #64748B !important;
    border-radius: 6px !important;
    font-weight: 500;
}}
[data-testid="stSidebar"] span[data-baseweb="tag"] svg {{
    fill: #1E3A8A !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    background-color: #FFFFFF;
    color: #0F172A;
    border: 1px solid #64748B;
    border-radius: 6px;
    font-weight: 600;
    min-height: 44px;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #DBEAFE;
    border-color: #1D4ED8;
    color: #1E3A8A;
}}
.filter-status-box {{
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    color: #0F172A;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 0.875rem;
    margin-top: 8px;
}}
.author-card {{
    background-color: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 14px;
}}
.author-label {{
    font-size: 0.8rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 4px;
}}
.author-name {{
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.3;
    margin-bottom: 10px;
}}
.author-links {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}}
.author-links a {{
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
}}
.author-links a:hover {{
    background-color: #DBEAFE;
    border-color: #1D4ED8;
    color: #1E3A8A !important;
}}
@media (max-width: 768px) {{
    .page-header {{
        padding: 16px 18px;
    }}
    .meta-item {{
        flex: 1 1 120px;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        padding: 8px 12px;
    }}
    .author-links {{
        gap: 6px;
    }}
}}
@media (prefers-reduced-motion: reduce) {{
    * {{
        animation: none !important;
        transition: none !important;
    }}
}}
</style>
<div class="page-header">
<div class="header-copy">
<div class="page-title" role="heading" aria-level="1">Transaction Anomaly Dashboard</div>
<p class="page-subtitle">Segmentasi perilaku nasabah (K-Means), penanda transaksi tidak biasa (Isolation Forest), dan aturan risiko login. Data sintetis, proyek belajar.</p>
</div>
<div class="header-meta">
<div class="meta-item">
<div class="meta-value">{total_tx:,}</div>
<div class="meta-label">Total Transaksi</div>
</div>
<div class="meta-item">
<div class="meta-value">{anom_tx:,} ({anom_pct:.2f}%)</div>
<div class="meta-label">Ditandai Anomali</div>
</div>
<div class="meta-item">
<div class="meta-value">{risk_tx:,} ({risk_pct:.2f}%)</div>
<div class="meta-label">Risiko Login (&gt;= 3)</div>
</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def reset_filters(occupations, channels, types):
    st.session_state["filter_occupation"] = occupations
    st.session_state["filter_channel"] = channels
    st.session_state["filter_type"] = types
    st.session_state["filter_location"] = []


def render_sidebar(df):
    occupations = sorted(df["CustomerOccupation"].unique())
    channels = sorted(df["Channel"].unique())
    types = sorted(df["TransactionType"].unique())
    locations = sorted(df["Location"].unique())

    st.session_state.setdefault("filter_occupation", occupations)
    st.session_state.setdefault("filter_channel", channels)
    st.session_state.setdefault("filter_type", types)
    st.session_state.setdefault("filter_location", [])

    st.sidebar.header("Filter Data")
    st.sidebar.caption("Berlaku pada tab Ringkasan, Segmen, dan Anomali & Login.")
    sel_occ = st.sidebar.multiselect("Profesi", options=occupations, key="filter_occupation")
    sel_chan = st.sidebar.multiselect("Kanal", options=channels, key="filter_channel")
    sel_type = st.sidebar.multiselect("Tipe Transaksi", options=types, key="filter_type")
    sel_loc = st.sidebar.multiselect("Kota (kosong = semua)", options=locations, key="filter_location")

    st.sidebar.button(
        "Reset Filter",
        on_click=reset_filters,
        args=(occupations, channels, types),
        key="reset_filters_btn",
        width="stretch",
    )

    filtered = df[
        df["CustomerOccupation"].isin(sel_occ)
        & df["Channel"].isin(sel_chan)
        & df["TransactionType"].isin(sel_type)
    ]
    if sel_loc:
        filtered = filtered[filtered["Location"].isin(sel_loc)]

    st.sidebar.markdown(
        f'<div class="filter-status-box">Menampilkan <b>{len(filtered):,}</b> '
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
    return filtered


def render_tab_summary(filtered):
    st.subheader("Ringkasan")
    total_tx = len(filtered)
    avg_amt = float(filtered["TransactionAmount"].mean()) if total_tx > 0 else 0.0
    n_anom = int(filtered["Anomaly_Flag"].sum())
    anom_pct = (n_anom / total_tx) * 100 if total_tx > 0 else 0.0
    n_risk = int(filtered["High_Login_Risk"].sum())
    risk_pct = (n_risk / total_tx) * 100 if total_tx > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True, key="card_sum_total"):
            st.metric("Total Transaksi", f"{total_tx:,}")
    with c2:
        with st.container(border=True, key="card_sum_avg"):
            st.metric("Rata-rata Nominal", f"${avg_amt:,.2f}")
    with c3:
        with st.container(border=True, key="card_sum_anom"):
            st.metric("Ditandai Anomali", f"{n_anom:,} ({anom_pct:.2f}%)")
    with c4:
        with st.container(border=True, key="card_sum_risk"):
            st.metric("Risiko Login (>= 3x)", f"{n_risk:,} ({risk_pct:.2f}%)")

    left, right = st.columns([1, 1.4])
    with left:
        with st.container(border=True, key="card_sum_dist"):
            st.markdown("**Jumlah transaksi per segmen**")
            counts = (
                filtered["Segment_Name"]
                .value_counts()
                .reindex(list(SEGMENT_COLORS.keys()), fill_value=0)
                .rename_axis("Segmen")
                .reset_index(name="Jumlah")
            )
            fig = px.bar(
                counts,
                x="Segmen",
                y="Jumlah",
                color="Segmen",
                text="Jumlah",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segmen": list(SEGMENT_COLORS.keys())},
                template="plotly_white",
            )
            fig.update_traces(textposition="outside", marker_cornerradius=4, cliponaxis=False)
            fig.update_layout(
                yaxis_title="Jumlah Transaksi",
                xaxis_title="",
                margin=dict(t=25, b=20, l=20, r=20),
            )
            _style_chart(fig, height=340, show_legend=False)
            st.plotly_chart(fig, width="stretch")

    with right:
        with st.container(border=True, key="card_sum_cat"):
            top_l, top_r = st.columns([1.5, 1])
            with top_l:
                st.markdown("**Proporsi segmen pada tiap kategori (%)**")
            with top_r:
                breakdown_by = st.selectbox(
                    "Kategori",
                    options=["CustomerOccupation", "Channel", "TransactionType"],
                    key="summary_breakdown_by",
                    label_visibility="collapsed",
                )

            breakdown = (
                filtered.groupby([breakdown_by, "Segment_Name"]).size().reset_index(name="Jumlah")
            )
            breakdown["Persentase (%)"] = (
                breakdown["Jumlah"] / breakdown.groupby(breakdown_by)["Jumlah"].transform("sum") * 100
            )
            fig_cat = px.bar(
                breakdown,
                x=breakdown_by,
                y="Persentase (%)",
                color="Segment_Name",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segment_Name": list(SEGMENT_COLORS.keys())},
                template="plotly_white",
            )
            fig_cat.update_traces(marker_cornerradius=4)
            fig_cat.update_layout(
                legend_title_text="Segmen",
                yaxis_title="Persentase (%)",
                xaxis_title=breakdown_by,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=20, r=20),
            )
            _style_chart(fig_cat, height=300, show_legend=True)
            st.plotly_chart(fig_cat, width="stretch")

    with st.container(border=True, key="card_sum_table"):
        st.markdown("**Rata-rata per segmen**")
        profile = (
            filtered.groupby("Segment_Name")[CLUSTERING_FEATURES + ["LoginAttempts"]]
            .mean()
            .round(2)
            .reset_index()
            .rename(columns={"Segment_Name": "Segmen"})
        )
        st.dataframe(profile, width="stretch", hide_index=True)


def render_tab_segments(filtered, df):
    st.subheader("Segmen Nasabah (K-Means)")
    vis_l, vis_r = st.columns([1.15, 1])

    with vis_l:
        with st.container(border=True, key="card_seg_scatter"):
            st.markdown("**Sebaran antarfitur transaksi**")
            options = CLUSTERING_FEATURES + ["Amount_Per_Second", "Anomaly_Score"]
            col_x, col_y = st.columns(2)
            x_axis = col_x.selectbox(
                "Sumbu X", options, index=options.index("AccountBalance"), key="seg_x_axis"
            )
            y_axis = col_y.selectbox(
                "Sumbu Y", options, index=options.index("TransactionAmount"), key="seg_y_axis"
            )

            fig = px.scatter(
                filtered,
                x=x_axis,
                y=y_axis,
                color="Segment_Name",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segment_Name": list(SEGMENT_COLORS.keys())},
                opacity=0.82,
                template="plotly_white",
            )
            fig.update_traces(marker=dict(size=7.5, line=dict(width=0.5, color="#FFFFFF")))
            fig.update_layout(
                legend_title_text="Segmen",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(t=40, b=20, l=20, r=20),
            )
            _style_chart(fig, height=440, show_legend=True)
            st.plotly_chart(fig, width="stretch")

    with vis_r:
        with st.container(border=True, key="card_seg_radar"):
            st.markdown("**Perbandingan 5 fitur perilaku per segmen (skala 0-1)**")
            feat_min, feat_max = get_feature_bounds(df)
            seg_means = filtered.groupby("Segment_Name")[CLUSTERING_FEATURES].mean()
            seg_norm = (seg_means - feat_min) / (feat_max - feat_min)
            radar_data = seg_norm.reset_index().melt(
                id_vars="Segment_Name", var_name="Fitur", value_name="Nilai Normalisasi"
            )

            fig_radar = px.line_polar(
                radar_data,
                r="Nilai Normalisasi",
                theta="Fitur",
                color="Segment_Name",
                line_close=True,
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segment_Name": list(SEGMENT_COLORS.keys())},
                template="plotly_white",
            )
            fig_radar.update_traces(fill="toself", opacity=0.26, line=dict(width=2.5))
            fig_radar.update_layout(
                legend_title_text="Segmen",
                legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
                margin=dict(t=40, b=70, l=65, r=65),
                polar=dict(
                    domain=dict(x=[0.16, 0.84], y=[0.16, 0.86]),
                    bgcolor="#FFFFFF",
                    radialaxis=dict(
                        gridcolor="#CBD5E1",
                        linecolor="#CBD5E1",
                        tickfont=dict(size=11, color="#0F172A"),
                    ),
                    angularaxis=dict(
                        gridcolor="#CBD5E1",
                        linecolor="#CBD5E1",
                        tickfont=dict(size=12, color="#0F172A"),
                    ),
                ),
            )
            _style_chart(fig_radar, height=490, show_legend=True)
            st.plotly_chart(fig_radar, width="stretch")

    st.markdown("**Profil & Rekomendasi Segmen (data keseluruhan)**")
    total_all = len(df)
    seg_cols = st.columns(2)
    for idx, (col_box, name) in enumerate(zip(seg_cols, ["Nasabah Muda Aktif", "Nasabah Mapan Konservatif"])):
        seg = df[df["Segment_Name"] == name]
        if seg.empty:
            continue
        top_job = seg["CustomerOccupation"].mode().iloc[0]
        pct = (len(seg) / total_all) * 100 if total_all > 0 else 0.0
        color = SEGMENT_COLORS[name]
        with col_box:
            with st.container(border=True, key=f"card_seg_{idx}"):
                st.markdown(
                    f'<div class="segment-card-head" style="color: {color};">{name}</div>'
                    f'<div class="segment-card-sub">{len(seg):,} transaksi ({pct:.2f}%), dominan {top_job}</div>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Usia Rata-rata", f"{seg['CustomerAge'].mean():.2f} thn")
                m2.metric("Saldo Rata-rata", f"${seg['AccountBalance'].mean():,.2f}")
                m3, m4 = st.columns(2)
                m3.metric("Transaksi Rata-rata", f"${seg['TransactionAmount'].mean():,.2f}")
                m4.metric("Balance Ratio", f"{seg['Balance_Ratio'].mean():.2f}")
                st.info(f"Ide tindakan: {SEGMENT_ACTIONS.get(name, '-')}")


def render_tab_anomaly(filtered):
    st.subheader("Anomali & Risiko Login")
    st.caption(
        "Anomali = ditandai Isolation Forest. Risiko login = aturan LoginAttempts >= 3. "
        "Keduanya kandidat untuk ditinjau, bukan bukti fraud."
    )
    mode = st.selectbox(
        "Tampilkan",
        ["Anomali (Isolation Forest)", "Risiko Login (LoginAttempts >= 3)", "Keduanya sekaligus"],
        key="anomaly_mode",
    )
    if mode.startswith("Anomali"):
        view = filtered[filtered["Anomaly_Flag"]]
    elif mode.startswith("Risiko"):
        view = filtered[filtered["High_Login_Risk"]]
    else:
        view = filtered[filtered["Anomaly_Flag"] & filtered["High_Login_Risk"]]

    total_filtered = len(filtered)
    view_pct = (len(view) / total_filtered) * 100 if total_filtered > 0 else 0.0
    max_score = float(view["Anomaly_Score"].max()) if not view.empty else 0.0
    mean_amt = float(view["TransactionAmount"].mean()) if not view.empty else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        with st.container(border=True, key="card_anom_count"):
            st.metric("Transaksi Terfilter", f"{len(view):,} ({view_pct:.2f}%)")
    with k2:
        with st.container(border=True, key="card_anom_max"):
            st.metric("Skor Anomali Tertinggi", f"{max_score:.2f}")
    with k3:
        with st.container(border=True, key="card_anom_avg"):
            st.metric("Rata-rata Nominal Terfilter", f"${mean_amt:,.2f}")

    c_l, c_r = st.columns([1.4, 1])
    with c_l:
        with st.container(border=True, key="card_anom_hist"):
            st.markdown("**Sebaran skor anomali (0-100)**")
            fig_hist = px.histogram(
                filtered,
                x="Anomaly_Score",
                color="Anomaly_Flag",
                nbins=40,
                template="plotly_white",
                color_discrete_map=ANOMALY_COLORS,
                labels={"Anomaly_Score": "Skor anomali (0-100)", "Anomaly_Flag": "Ditandai anomali"},
            )
            fig_hist.update_layout(
                yaxis_title="Jumlah Transaksi",
                margin=dict(t=25, b=20, l=20, r=20),
            )
            _style_chart(fig_hist, height=340, show_legend=True)
            st.plotly_chart(fig_hist, width="stretch")

    with c_r:
        with st.container(border=True, key="card_anom_heat"):
            st.markdown("**Anomali vs aturan login (jumlah transaksi)**")
            ct = (
                pd.crosstab(filtered["High_Login_Risk"], filtered["Anomaly_Flag"])
                .reindex(index=[False, True], columns=[False, True], fill_value=0)
            )
            ct.index = ["Normal (< 3)", "Risiko (>= 3)"]
            ct.columns = ["Normal", "Anomali"]
            fig_heat = px.imshow(
                ct,
                text_auto=True,
                color_continuous_scale=CLUSTER_GRADIENTS_BLUE,
                labels=dict(x="Ditandai anomali", y="Risiko login", color="Jumlah"),
                template="plotly_white",
            )
            fig_heat.update_layout(margin=dict(t=25, b=20, l=20, r=20))
            _style_chart(fig_heat, height=340, show_legend=False)
            st.plotly_chart(fig_heat, width="stretch")

    columns = [
        "Segment_Name", "TransactionAmount", "AccountBalance", "Balance_Ratio",
        "TransactionDuration", "CustomerAge", "LoginAttempts", "Anomaly_Score",
    ]
    with st.container(border=True, key="card_anom_table"):
        hdr_l, hdr_r = st.columns([3, 1])
        with hdr_l:
            st.markdown(f"**Daftar transaksi ({len(view):,} data)**")
        with hdr_r:
            if not view.empty:
                sorted_view = view.sort_values("Anomaly_Score", ascending=False)
                st.download_button(
                    "Unduh daftar (CSV)",
                    data=sorted_view[columns].round(2).to_csv(index=False).encode("utf-8"),
                    file_name="transaksi_ditandai.csv",
                    mime="text/csv",
                    key="download_flagged_csv",
                    width="stretch",
                )

        if view.empty:
            st.info(
                "Tidak ada transaksi yang cocok dengan filter ini. "
                "Ubah opsi 'Tampilkan' atau klik 'Reset Filter' di sidebar."
            )
        else:
            sorted_view = view.sort_values("Anomaly_Score", ascending=False)
            st.dataframe(sorted_view[columns].round(2), width="stretch", hide_index=True)


def render_tab_predict(df, scaler_km, kmeans, anomaly):
    st.subheader("Coba Satu Transaksi")
    st.caption("Masukkan data transaksi, lalu lihat segmennya, skor anomalinya, dan status aturan login.")

    preset_name = st.selectbox("Contoh skenario", list(PRESET_SCENARIOS.keys()), key="predict_preset")
    preset = PRESET_SCENARIOS[preset_name]
    segment_map = get_segment_map(df)

    with st.container(border=True, key="card_pred_form"):
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            amount = c1.number_input(
                "Nominal transaksi (USD)", 0.0, 100000.0, float(preset["amount"]), key=f"amount_{preset_name}"
            )
            balance = c1.number_input(
                "Saldo akun (USD)", 0.0, 1000000.0, float(preset["balance"]), key=f"balance_{preset_name}"
            )
            duration = c2.number_input(
                "Durasi transaksi (detik)", 1, 1000, int(preset["duration"]), key=f"duration_{preset_name}"
            )
            age = c2.number_input(
                "Usia nasabah", 17, 100, int(preset["age"]), key=f"age_{preset_name}"
            )
            login = c3.number_input(
                "Jumlah upaya login", 1, 20, int(preset["login"]), key=f"login_{preset_name}"
            )
            submitted = st.form_submit_button("Hitung segmen & skor", width="stretch")

    if not submitted:
        return

    result = predict_transaction(
        amount, balance, duration, age, login, scaler_km, kmeans, anomaly, segment_map
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True, key="card_pred_seg"):
            st.metric("Segmen", result["segment"])
    with m2:
        with st.container(border=True, key="card_pred_score"):
            st.metric("Skor anomali (0-100)", f"{result['anomaly_score']:.1f}")
    with m3:
        with st.container(border=True, key="card_pred_if"):
            st.metric("Isolation Forest", "Anomali" if result["is_anomaly"] else "Tidak anomali")
    with m4:
        with st.container(border=True, key="card_pred_login"):
            st.metric("Aturan login", "Risiko" if result["is_high_login_risk"] else "Normal")

    st.caption(
        f"Balance_Ratio {result['balance_ratio']:.2f} | "
        f"Nominal per detik ${result['amount_per_second']:.2f}"
    )
    if result["is_anomaly"] or result["is_high_login_risk"]:
        st.warning("Transaksi ini layak ditinjau. Ini penanda dari model dan aturan sederhana, bukan bukti fraud.")
    else:
        st.success("Tidak ada penanda pada transaksi ini.")

    with st.container(border=True, key="card_pred_bar"):
        st.markdown("**Posisi skor terhadap ambang anomali**")
        bar_df = pd.DataFrame({
            "Metrik": ["Skor Transaksi"],
            "Skor": [result["anomaly_score"]],
        })
        bar_color = "#B91C1C" if result["is_anomaly"] else "#1D4ED8"
        fig_pos = px.bar(
            bar_df,
            x="Skor",
            y="Metrik",
            orientation="h",
            text=bar_df["Skor"].map("{:.1f}".format),
            template="plotly_white",
        )
        fig_pos.update_traces(marker_color=bar_color, marker_cornerradius=4, textposition="outside", cliponaxis=False)
        fig_pos.add_vline(
            x=result["threshold_score"],
            line_dash="dash",
            line_color="#0F172A",
            annotation_text=f"Ambang ~{result['threshold_score']:.1f}",
            annotation_position="top right",
        )
        fig_pos.update_layout(
            xaxis_range=[0, 108],
            xaxis_title="Skor anomali (0-100)",
            yaxis_title="",
            margin=dict(t=30, b=20, l=15, r=20),
        )
        _style_chart(fig_pos, height=170, show_legend=False)
        st.plotly_chart(fig_pos, width="stretch")

    st.info(f"Ide tindakan untuk segmen ini: {SEGMENT_ACTIONS.get(result['segment'], '-')}")


def render_tab_notes(anomaly):
    st.subheader("Catatan Metode & Keterbatasan")
    k_eval = load_k_evaluation()
    sil_text = "-"
    if k_eval is not None and (k_eval["k"] == 2).any():
        sil_text = f"{float(k_eval.loc[k_eval['k'] == 2, 'Silhouette Score'].iloc[0]):.4f}"

    with st.container(border=True, key="card_notes_body"):
        st.markdown(
            f"""
**Yang dilakukan**
- **K-Means (k = 2)** pada 5 fitur perilaku untuk mengelompokkan nasabah.
- **Isolation Forest** pada 6 fitur (5 fitur perilaku + nominal per detik) untuk menandai transaksi yang tidak biasa.
- **Aturan sederhana** `LoginAttempts >= {LOGIN_RISK_THRESHOLD}` untuk upaya login berulang. Ini aturan, bukan hasil model.

**Keterbatasan**
- **Data sintetis dan tanpa label fraud asli.** Precision dan recall tidak bisa dihitung. Anomali hanya berarti "tidak biasa", bukan "fraud".
- **Segmen saling tumpang tindih.** Silhouette Score k = 2 hanya {sil_text}. Dua segmen pada dasarnya memisahkan nasabah muda bersaldo kecil dan nasabah lebih tua bersaldo besar.
- **Jumlah anomali mengikuti pilihan `contamination = {anomaly.get('contamination', 0.05):.2f}`**, bukan temuan dari data. Uji sensitivitasnya ada di notebook (bagian 5.1).
- **Sumber data:** dataset latihan Dicoding (adaptasi Kaggle *Bank Transaction Dataset for Fraud Detection*), transaksi sintetis kota-kota AS dalam USD.
"""
        )

    if k_eval is not None:
        with st.expander("Hasil Elbow & Silhouette (k = 2 sampai 8)"):
            col_a, col_b = st.columns(2)
            fig_a = px.line(k_eval, x="k", y="Inertia", markers=True, template="plotly_white", title="Inertia")
            fig_a.update_traces(line_color="#1D4ED8")
            fig_a.add_vline(x=2, line_dash="dash", line_color="#B91C1C")
            _style_chart(fig_a, height=320, show_legend=False)
            col_a.plotly_chart(fig_a, width="stretch")

            fig_b = px.line(
                k_eval, x="k", y="Silhouette Score", markers=True, template="plotly_white", title="Silhouette Score"
            )
            fig_b.update_traces(line_color="#1D4ED8")
            fig_b.add_vline(x=2, line_dash="dash", line_color="#B91C1C")
            _style_chart(fig_b, height=320, show_legend=False)
            col_b.plotly_chart(fig_b, width="stretch")

            st.dataframe(k_eval.round(4), width="stretch", hide_index=True)


def main():
    st.set_page_config(page_title="Transaction Anomaly Dashboard", layout="wide")

    with st.spinner("Memuat data dan model..."):
        try:
            df = load_data()
            scaler_km, kmeans, anomaly = load_models()
        except (FileNotFoundError, OSError, ValueError, KeyError) as err:
            st.error(
                f"Gagal memuat data atau model: {err}. "
                "Pastikan folder data/ dan models/ berisi hasil dari notebook."
            )
            st.stop()

    render_styles_and_banner(df)
    filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("Tidak ada transaksi yang cocok dengan filter. Klik 'Reset Filter' di sidebar.")
        return

    tab_sum, tab_seg, tab_anom, tab_pred, tab_notes = st.tabs(
        ["Ringkasan", "Segmen", "Anomali & Login", "Coba Transaksi", "Catatan"]
    )
    with tab_sum:
        render_tab_summary(filtered)
    with tab_seg:
        render_tab_segments(filtered, df)
    with tab_anom:
        render_tab_anomaly(filtered)
    with tab_pred:
        render_tab_predict(df, scaler_km, kmeans, anomaly)
    with tab_notes:
        render_tab_notes(anomaly)


if __name__ == "__main__":
    main()
