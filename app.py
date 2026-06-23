import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Northcard Capital Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ─────────────────────────────────────────────────────────────
C = {
    "navy":   "#2c3e50",
    "orange": "#e67e22",
    "green":  "#27ae60",
    "blue":   "#2980b9",
    "amber":  "#e67e22",
    "red":    "#e74c3c",
    "light":  "#f0f4f8",
    "mid":    "#dce3ed",
}

TIER_COLORS = {"High": C["green"], "Medium": C["blue"], "Low": C["red"]}
PILLAR_COLORS = {
    "score_fdi":     "#e67e22",
    "score_banking": "#2980b9",
    "score_manuf":   "#27ae60",
    "score_digital": "#8e44ad",
    "score_composite": "#2c3e50",
}
PILLAR_LABELS = {
    "score_fdi":      "FDI",
    "score_banking":  "Banking",
    "score_manuf":    "Manufacturing",
    "score_digital":  "Digital Economy",
    "score_composite":"Composite",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #ffffff; border-radius: 10px; padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 10px;
    border: 1px solid #e0e6ef;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #e67e22; }
.metric-label { font-size: 0.85rem; color: #6b7c93; margin-top: 2px; }
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #2c3e50;
    border-left: 4px solid #e67e22; padding-left: 12px; margin: 20px 0 12px 0;
}
.rec-card { border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; border-left: 5px solid #27ae60; background: #f0faf4; }
.tag { display:inline-block; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_master():
    return pd.read_csv("master_wide.csv")

@st.cache_data
def load_forecasts():
    return pd.read_csv("task5_forecasts.csv")

@st.cache_data
def load_intervals():
    return pd.read_csv("task5_intervals.csv")

@st.cache_data
def load_kmeans():
    return pd.read_csv("cluster_labels.csv")

@st.cache_data
def load_hier():
    return pd.read_csv("cluster_assignments.csv")

@st.cache_data
def load_shap():
    return pd.read_csv("task4_shap_values.csv")

@st.cache_data
def load_importance():
    return pd.read_csv("task4_importance_table.csv")

@st.cache_data
def load_scenarios():
    return pd.read_csv("task7_scenario_comparison.csv")

@st.cache_data
def load_recommendations():
    return pd.read_csv("task6_recommendation.csv")

@st.cache_data
def load_knn():
    try:
        return pd.read_csv("knn_investment_results.csv")
    except FileNotFoundError:
        return None

# ── v2.0 data loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_v2_forecasts():
    try:
        return pd.read_csv("t9_forecast_composite_v2.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_v2_scenarios():
    try:
        return pd.read_csv("t9_scenario_comparison_v2.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_v2_recommendations():
    try:
        return pd.read_csv("t9_recommendations_v2.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_v2_risk_rankings():
    try:
        return pd.read_csv("t11_risk_adjusted_rankings.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_v2_policy_factors():
    try:
        return pd.read_csv("t8_impact_factors.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_v2_robustness():
    try:
        return pd.read_csv("t4_robustness_verdict.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def get_v2_tier_thresholds(year=2027, scenario="A_Standard"):
    """Compute v2.0 tier thresholds from full Scenario A pool (all years, all 194 countries).
    Uses p33/p67 quantiles of the 776-point distribution for stable, balanced splits.
    Falls back to v1.0 thresholds if data unavailable."""
    sc = load_v2_scenarios()
    if sc is not None:
        base = sc[sc["Scenario"]=="A_Standard"]["Composite_v2"].dropna()
        if len(base) >= 100:
            low_mid  = round(float(base.quantile(0.333)), 3)
            mid_high = round(float(base.quantile(0.667)), 3)
            return low_mid, mid_high
    return 0.290, 0.362

def assign_tier_v2(score, year=2027, scenario="A_Standard"):
    low_mid, mid_high = get_v2_tier_thresholds(year, scenario)
    if score >= mid_high: return "High"
    elif score >= low_mid: return "Medium"
    else: return "Low"

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Northcard Capital")
    st.markdown("**Where Should Global Businesses Invest Next?**")
    st.markdown("---")

    version = st.radio(
        "Framework Version",
        ["v1.0 — Original", "v2.0 — Recalibrated"],
        index=0,
    )
    V2 = (version == "v2.0 — Recalibrated")

    if V2:
        st.markdown("""
        <div style="background:#f0faf4;border-left:3px solid #27ae60;border-radius:6px;padding:10px;font-size:0.78rem;color:#2c3e50;">
        <b>v2.0 improvements:</b><br>
        • Weights optimised via Spearman vs UNCTAD<br>
        • LLM policy extraction (194 countries)<br>
        • T5→T8 DiD-blended factors<br>
        • Risk-adjusted rankings (T11)<br>
        • Rank stability ρ = 0.979
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f8f9fa;border-left:3px solid #2980b9;border-radius:6px;padding:10px;font-size:0.78rem;color:#2c3e50;">
        <b>v1.0 framework:</b><br>
        • Weights: FDI 30% · Banking 25%<br>
        • Manufacturing 25% · Digital 20%<br>
        • Task 9 manual policy factors<br>
        • Prophet baseline forecasts
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 Country Explorer",
        "🤖 ML Models",
        "📈 Forecasting",
        "🌐 Policy Scenarios",
        "🏆 Investment Recommendations",
    ])
    st.markdown("---")
    st.markdown("**Capstone Team**")
    st.caption("Anant · Lakshmi Priya · Yash · Supriya\nPratham · Sumukh · Chinni Sumanth\nVenkat · Harish")
    st.caption("Mentor: Binish Thomas")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🌍 Where Should Global Businesses Invest Next?")
    st.markdown("**Northcard Capital Analytics | Data Science Capstone**")

    if V2:
        st.markdown("""
        <div style="background:#f0faf4;border:1px solid #27ae60;border-radius:8px;padding:12px 18px;margin-bottom:8px;">
        <b style="color:#27ae60;">v2.0 — Recalibrated Framework</b> &nbsp;|&nbsp;
        Weights optimised via constrained Spearman maximisation against UNCTAD FDI stock rankings.
        Policy factors sourced from Cerebras LLM extraction across 194 countries, blended with historical DiD analogues.
        Rank correlation v1.0 vs v2.0: <b>ρ = 0.979</b> — investment recommendations stable.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    master = load_master()
    forecasts = load_forecasts()

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">194</div><div class="metric-label">Countries Analysed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">2000–23</div><div class="metric-label">Historical Period</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">4,656</div><div class="metric-label">Dataset Rows</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">2024–27</div><div class="metric-label">Forecast Horizon</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-card"><div class="metric-value">776</div><div class="metric-label">Prophet Models Run</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Pillar weights — switch based on version
    st.markdown('<div class="section-header">4-Pillar Investment Framework</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    if V2:
        pillars = [
            ("💰 FDI",           "9.5%",  C["orange"], "Optimised — SHAP: 16.9%"),
            ("🏦 Banking",        "18.8%", C["blue"],   "Optimised — SHAP: 21.5%"),
            ("🏭 Manufacturing",  "34.3%", C["green"],  "Optimised — SHAP: 24.2%"),
            ("💻 Digital Economy","37.4%", "#8e44ad",   "Optimised — SHAP: 37.4%"),
        ]
        st.caption("v2.0 weights optimised via constrained Spearman maximisation vs UNCTAD FDI stock rankings (T3)")
    else:
        pillars = [
            ("💰 FDI",           "30%", C["orange"], "Foreign Direct Investment attractiveness"),
            ("🏦 Banking",        "25%", C["blue"],   "Banking sector strength & depth"),
            ("🏭 Manufacturing",  "25%", C["green"],  "Manufacturing capacity & output"),
            ("💻 Digital Economy","20%", "#8e44ad",   "Digital infrastructure & readiness"),
        ]

    for col, (name, weight, color, desc) in zip([col1,col2,col3,col4], pillars):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <div class="metric-value" style="color:{color};">{weight}</div>
                <div style="font-weight:700;margin-top:4px;">{name}</div>
                <div class="metric-label">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Global composite map — 2023
    st.markdown('<div class="section-header">Global Investment Composite Score — 2023</div>', unsafe_allow_html=True)
    map_df = master[master["TIME_PERIOD"] == 2023][["REF_AREA","Country","score_composite","investment_tier"]].copy()
    fig = px.choropleth(
        map_df, locations="REF_AREA", locationmode="ISO-3",
        color="score_composite", hover_name="Country",
        hover_data={"investment_tier": True, "score_composite": ":.3f", "REF_AREA": False},
        color_continuous_scale=[[0,"#e74c3c"],[0.5,"#f39c12"],[1,"#27ae60"]],
        range_color=[0.15, 0.60],
        labels={"score_composite": "Composite Score"},
    )
    fig.update_layout(
        height=440, margin=dict(l=0,r=0,t=10,b=0),
        coloraxis_colorbar=dict(title="Score", thickness=14),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#ccc"),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pipeline
    st.markdown("---")
    st.markdown('<div class="section-header">Analytical Pipeline</div>', unsafe_allow_html=True)
    if V2:
        steps = [
            ("📥 Data Collection",    "World Bank, UNCTAD — 4 sources, 59 raw indicators"),
            ("🧹 Data Cleaning",       "194 countries, 0 nulls, all scores normalised [0,1]"),
            ("⚖️ Weight Optimisation", "Constrained Spearman vs UNCTAD (T3) + SHAP validation (T1)"),
            ("🤖 ML Models",           "KNN · KMeans · Hierarchical · XGBoost+SHAP"),
            ("📈 Forecasting",         "Prophet — 194 × 4 pillars, v2.0 weights (T9)"),
            ("🌐 Policy Extraction",   "Cerebras LLM → T5→T6→T7→T8 DiD blend, 162 countries"),
            ("🏆 Risk-Adj Rankings",   "Score ÷ 5yr volatility — T11 risk-adjusted final rank"),
        ]
    else:
        steps = [
            ("📥 Data Collection", "World Bank, UNCTAD — 4 sources, 59 raw indicators"),
            ("🧹 Data Cleaning", "194 countries, 0 nulls, all scores normalised [0,1]"),
            ("📊 EDA", "Pillar distributions, tier analysis, correlation heatmaps"),
            ("🤖 ML Models", "KNN · KMeans · Hierarchical · XGBoost+SHAP"),
            ("📈 Forecasting", "Prophet — 194 countries × 4 pillars, 2024–2027"),
            ("🌐 Policy Scenarios", "3 scenarios (Baseline / Shock / Recovery) for 3 economies"),
            ("🏆 Recommendations", "Invest Now / Wait / Avoid per country"),
        ]
    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:{C['light']};border-radius:8px;padding:12px;text-align:center;height:110px;">
                <div style="font-size:1.3rem;">{title.split()[0]}</div>
                <div style="font-weight:700;font-size:0.8rem;color:{C['navy']};margin-top:4px;">{' '.join(title.split()[1:])}</div>
                <div style="font-size:0.72rem;color:#6b7c93;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COUNTRY EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Country Explorer":
    st.markdown("# 🔍 Country Explorer")
    st.markdown("Explore historical pillar scores and composite trends for any country.")
    st.markdown("---")

    master = load_master()
    forecasts = load_forecasts()
    intervals = load_intervals()

    countries = sorted(master["Country"].unique())
    col1, col2 = st.columns([2,1])
    with col1:
        country = st.selectbox("Select Country", countries, index=countries.index("India"), key="explorer_country")
    with col2:
        show_forecast = st.checkbox("Include 2024–2027 Forecast", value=True)

    hist = master[master["Country"] == country].sort_values("TIME_PERIOD")
    latest = hist.iloc[-1]

    # Metrics row
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Composite", f"{latest['score_composite']:.3f}", "score_composite"),
        ("FDI",       f"{latest['score_fdi']:.3f}",       "score_fdi"),
        ("Banking",   f"{latest['score_banking']:.3f}",   "score_banking"),
        ("Manuf.",    f"{latest['score_manuf']:.3f}",     "score_manuf"),
        ("Digital",   f"{latest['score_digital']:.3f}",   "score_digital"),
    ]
    for col, (label, val, key) in zip([col1,col2,col3,col4,col5], metrics):
        with col:
            color = PILLAR_COLORS[key]
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
                <div class="metric-value" style="color:{color};font-size:1.6rem;">{val}</div>
                <div class="metric-label">{label} (2023)</div>
            </div>""", unsafe_allow_html=True)

    tier = latest["investment_tier"]
    tier_color = TIER_COLORS.get(tier, C["navy"])
    st.markdown(f'<span class="tag" style="background:{tier_color};color:white;font-size:1rem;">Investment Tier: {tier}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Line chart — all pillars
    st.markdown('<div class="section-header">Pillar Score Trends</div>', unsafe_allow_html=True)
    pillars_to_plot = ["score_composite","score_fdi","score_banking","score_manuf","score_digital"]

    fig = go.Figure()
    for p in pillars_to_plot:
        fig.add_trace(go.Scatter(
            x=hist["TIME_PERIOD"], y=hist[p],
            name=PILLAR_LABELS[p], line=dict(color=PILLAR_COLORS[p], width=2.5),
            mode="lines+markers", marker=dict(size=4),
        ))

    if show_forecast:
        fc = forecasts[forecasts["Country"] == country].sort_values("Year")
        intv = intervals[intervals["Country"] == country]
        if not fc.empty:
            # Bridge: prepend last historical point so forecast connects
            last_hist_year = int(hist["TIME_PERIOD"].max())
            last_hist_val = hist[hist["TIME_PERIOD"] == last_hist_year]["score_composite"].values[0]
            fc_years = [last_hist_year] + list(fc["Year"])
            fc_vals  = [last_hist_val]  + list(fc["score_composite"])
            # Composite forecast
            fig.add_trace(go.Scatter(
                x=fc_years, y=fc_vals,
                name="Composite (Forecast)", line=dict(color=PILLAR_COLORS["score_composite"], width=2.5, dash="dash"),
                mode="lines+markers", marker=dict(size=5, symbol="diamond"),
            ))
            # Confidence band for composite
            comp_intv = intv[intv["Pillar"] == "score_composite"] if "score_composite" in intv["Pillar"].values else None
            if comp_intv is not None and not comp_intv.empty:
                fig.add_trace(go.Scatter(
                    x=list(comp_intv["Year"]) + list(comp_intv["Year"])[::-1],
                    y=list(comp_intv["yhat_upper"]) + list(comp_intv["yhat_lower"])[::-1],
                    fill="toself", fillcolor="rgba(27,58,91,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="CI",
                ))
            # Divider line
            fig.add_vline(x=last_hist_year + 0.5, line_dash="dot", line_color="#aaa", annotation_text="Forecast →", annotation_position="top")

    fig.update_layout(
        height=420, legend=dict(orientation="h", y=-0.2),
        xaxis=dict(title="Year", range=[2000, 2028], tickformat='d'),
        yaxis=dict(title="Score (0-1)", range=[0,1]),
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        margin=dict(l=40,r=40,t=20,b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart — latest year
    st.markdown('<div class="section-header">Pillar Profile — 2023</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        cats = ["FDI","Banking","Manufacturing","Digital Economy"]
        vals = [latest["score_fdi"], latest["score_banking"], latest["score_manuf"], latest["score_digital"]]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(27,58,91,0.15)",
            line=dict(color=C["navy"], width=2),
            marker=dict(size=6, color=C["orange"]),
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            height=320, margin=dict(l=40,r=40,t=30,b=30),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        # Year-on-year composite change
        hist_sorted = hist.sort_values("TIME_PERIOD").copy()
        hist_sorted["yoy"] = hist_sorted["score_composite"].diff()
        fig_bar = px.bar(
            hist_sorted.dropna(subset=["yoy"]), x="TIME_PERIOD", y="yoy",
            color="yoy", color_continuous_scale=["#e74c3c","#f39c12","#27ae60"],
            labels={"TIME_PERIOD":"Year","yoy":"YoY Change"},
            title="Year-on-Year Composite Change",
        )
        fig_bar.update_layout(height=320, paper_bgcolor="white", coloraxis_showscale=False,
                               margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown("# 🤖 ML Models")
    st.markdown("Four models applied to classify, cluster, and explain investment attractiveness.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["KNN Classifier", "K-Means Clustering", "Hierarchical Clustering", "XGBoost + SHAP"])

    # ── KNN ──────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### K-Nearest Neighbors — Investment Tier Classifier")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value" style="color:#27ae60;">~97%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Classes: High / Medium / Low</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">distance</div><div class="metric-label">Weighting (handles imbalance)</div></div>', unsafe_allow_html=True)

        master = load_master()
        knn_df = load_knn()

        st.markdown("---")

        # Tier distribution over time
        st.markdown('<div class="section-header">Investment Tier Distribution Over Time (2000–2023)</div>', unsafe_allow_html=True)
        tier_time = master.groupby(["TIME_PERIOD","investment_tier"]).size().reset_index(name="count")
        fig = px.line(
            tier_time, x="TIME_PERIOD", y="count", color="investment_tier",
            color_discrete_map=TIER_COLORS,
            labels={"TIME_PERIOD":"Year","count":"Countries","investment_tier":"Tier"},
            markers=True,
        )
        fig.update_layout(height=360, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                          legend=dict(orientation="h",y=-0.2), margin=dict(l=40,r=20,t=10,b=60))
        st.plotly_chart(fig, use_container_width=True)

        # World map — 2023 tiers
        st.markdown('<div class="section-header">Global Investment Tier Map — 2023 (KNN)</div>', unsafe_allow_html=True)
        map_df = master[master["TIME_PERIOD"]==2023][["REF_AREA","Country","investment_tier","score_composite"]].copy()
        tier_num = {"High":1,"Medium":2,"Low":3}
        map_df["tier_num"] = map_df["investment_tier"].map(tier_num)
        fig_map = px.choropleth(
            map_df, locations="REF_AREA", locationmode="ISO-3",
            color="tier_num", hover_name="Country",
            hover_data={"investment_tier":True,"score_composite":":.3f","tier_num":False,"REF_AREA":False},
            color_continuous_scale=[[0,C["green"]],[0.5,C["blue"]],[1,C["red"]]],
            range_color=[1,3],
        )
        fig_map.update_layout(
            height=400, margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_colorbar=dict(tickvals=[1,2,3], ticktext=["High","Medium","Low"], title="Tier"),
            geo=dict(showframe=False), paper_bgcolor="white",
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Pillar boxplots by tier
        st.markdown('<div class="section-header">Pillar Score Distribution by Tier</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for idx, (pillar, label) in enumerate([("score_fdi","FDI"),("score_banking","Banking"),
                                                ("score_manuf","Manufacturing"),("score_digital","Digital")]):
            with [col1,col2][idx%2]:
                fig_box = px.box(
                    master[master["TIME_PERIOD"]==2023], x="investment_tier", y=pillar,
                    color="investment_tier", color_discrete_map=TIER_COLORS,
                    category_orders={"investment_tier":["High","Medium","Low"]},
                    labels={"investment_tier":"Tier", pillar:f"{label} Score"},
                    title=label,
                )
                fig_box.update_layout(height=280, showlegend=False, paper_bgcolor="white",
                                      plot_bgcolor="#f8f9fa", margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_box, use_container_width=True)

    # ── K-MEANS ───────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### K-Means Clustering — 4-Tier Country Grouping")
        kmeans = load_kmeans()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Clusters (K=4)</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">193</div><div class="metric-label">Countries Clustered</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">2000–23</div><div class="metric-label">Avg scores used</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        KMEANS_COLORS = {
            "Tier A – High Attractiveness": C["green"],
            "Tier B – Moderate-High": C["blue"],
            "Tier C – Moderate-Low": C["amber"],
            "Tier D – Low Attractiveness": C["red"],
        }

        # Cluster sizes
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown('<div class="section-header">Cluster Sizes</div>', unsafe_allow_html=True)
            counts = kmeans["cluster_name"].value_counts().reindex(list(KMEANS_COLORS.keys()))
            fig_bar = px.bar(
                x=counts.values, y=counts.index, orientation="h",
                color=counts.index, color_discrete_map=KMEANS_COLORS,
                labels={"x":"Countries","y":""},
                text=counts.values,
            )
            fig_bar.update_layout(height=300, showlegend=False, paper_bgcolor="white",
                                  plot_bgcolor="#f8f9fa", margin=dict(l=20,r=20,t=10,b=20))
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Cluster Profiles — Mean Pillar Scores</div>', unsafe_allow_html=True)
            profile = kmeans.groupby("cluster_name")[["score_fdi","score_banking","score_manuf","score_digital"]].mean()
            profile = profile.reindex(list(KMEANS_COLORS.keys()))
            fig_prof = go.Figure()
            for pillar in ["score_fdi","score_banking","score_manuf","score_digital"]:
                fig_prof.add_trace(go.Bar(
                    name=PILLAR_LABELS[pillar], x=list(KMEANS_COLORS.keys()),
                    y=profile[pillar].values, marker_color=PILLAR_COLORS[pillar],
                ))
            fig_prof.update_layout(
                barmode="group", height=300, paper_bgcolor="white",
                plot_bgcolor="#f8f9fa", legend=dict(orientation="h", y=-0.3),
                xaxis_tickfont_size=11, margin=dict(l=20,r=20,t=10,b=80),
                yaxis=dict(range=[0,0.8], title="Score"),
            )
            st.plotly_chart(fig_prof, use_container_width=True)

        # FDI vs Digital scatter
        st.markdown('<div class="section-header">FDI vs Digital Economy Score (by Cluster)</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            kmeans, x="score_fdi", y="score_digital", color="cluster_name",
            color_discrete_map=KMEANS_COLORS, hover_name="Country",
            labels={"score_fdi":"FDI Score","score_digital":"Digital Score","cluster_name":"Cluster"},
        )
        fig_sc.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                              legend=dict(orientation="h", y=-0.2), margin=dict(l=40,r=20,t=10,b=80))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── HIERARCHICAL ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Hierarchical Clustering — 2023 Snapshot")
        hier = load_hier()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">Tiers (Ward linkage)</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">0.623</div><div class="metric-label">Cophenetic Correlation</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">194</div><div class="metric-label">Countries (2023)</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        HIER_COLORS = {"Tier 1": C["green"], "Tier 2": C["blue"], "Tier 3": C["amber"], "Tier 4": C["red"]}

        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown('<div class="section-header">Tier Distribution</div>', unsafe_allow_html=True)
            counts = hier["investment_cluster"].value_counts().sort_index()
            fig_h = px.bar(x=counts.index, y=counts.values,
                           color=counts.index, color_discrete_map=HIER_COLORS,
                           labels={"x":"Tier","y":"Countries"},
                           text=counts.values)
            fig_h.update_layout(height=300, showlegend=False, paper_bgcolor="white",
                                 plot_bgcolor="#f8f9fa", margin=dict(l=20,r=20,t=10,b=20))
            fig_h.update_traces(textposition="outside")
            st.plotly_chart(fig_h, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Pillar Profiles by Tier</div>', unsafe_allow_html=True)
            profile_h = hier.groupby("investment_cluster")[["score_fdi","score_banking","score_manuf","score_digital"]].mean()
            fig_hp = go.Figure()
            for pillar in ["score_fdi","score_banking","score_manuf","score_digital"]:
                fig_hp.add_trace(go.Bar(
                    name=PILLAR_LABELS[pillar], x=profile_h.index,
                    y=profile_h[pillar].values, marker_color=PILLAR_COLORS[pillar],
                ))
            fig_hp.update_layout(
                barmode="group", height=300, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                legend=dict(orientation="h", y=-0.3), margin=dict(l=20,r=20,t=10,b=80),
                yaxis=dict(range=[0,0.8], title="Score"),
            )
            st.plotly_chart(fig_hp, use_container_width=True)

        # Country table
        st.markdown('<div class="section-header">Countries by Tier</div>', unsafe_allow_html=True)
        tier_sel = st.selectbox("Select Tier", ["Tier 1","Tier 2","Tier 3","Tier 4"], key="hier_tier")
        tier_countries = hier[hier["investment_cluster"]==tier_sel][
            ["Country","score_composite","score_fdi","score_banking","score_manuf","score_digital"]
        ].sort_values("score_composite", ascending=False).reset_index(drop=True)
        tier_countries.columns = ["Country","Composite","FDI","Banking","Manufacturing","Digital"]
        st.dataframe(tier_countries.style.format({c:"{:.3f}" for c in ["Composite","FDI","Banking","Manufacturing","Digital"]}),
                     use_container_width=True, height=300)

    # ── XGBOOST + SHAP ───────────────────────────────────────────────────────
    with tab4:
        st.markdown("### XGBoost + SHAP — Pillar Weight Validation")
        st.info("Goal: Validate whether SHAP-derived feature importance aligns with the team's assigned pillar weights (FDI 30% | Banking 25% | Manuf 25% | Digital 20%).")

        imp = load_importance()
        shap_df = load_shap()

        pillar_col = "Pillar"
        shap_col = "SHAP_Importance_Pct"
        team_col = "Team_Weight_Pct"

        team_weights = {"FDI": 30, "Banking": 25, "Manufacturing": 25, "Digital": 20}

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)
            fig_imp = px.bar(
                imp, x=shap_col, y=pillar_col, orientation="h",
                color=shap_col, color_continuous_scale=["#dde3ec", C["navy"]],
                labels={shap_col: "SHAP Importance (%)", pillar_col: "Pillar"},
                text=imp[shap_col].apply(lambda x: f"{x:.1f}%"),
            )
            fig_imp.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                                  paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                                  margin=dict(l=20,r=20,t=10,b=20))
            fig_imp.update_traces(textposition="outside")
            st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Team Weights vs SHAP Importance</div>', unsafe_allow_html=True)
            pillars_list = imp[pillar_col].tolist()
            shap_vals = imp[shap_col].tolist()
            team_vals = imp[team_col].tolist()

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name="Team Weights", x=pillars_list, y=team_vals,
                                      marker_color=C["navy"], opacity=0.85))
            fig_comp.add_trace(go.Bar(name="SHAP Importance", x=pillars_list, y=shap_vals,
                                      marker_color=C["orange"], opacity=0.85))
            fig_comp.update_layout(
                barmode="group", height=300, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                legend=dict(orientation="h", y=-0.25), margin=dict(l=20,r=20,t=10,b=60),
                yaxis=dict(title="Weight / Importance (%)"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # SHAP over time for selected country
        st.markdown('<div class="section-header">SHAP Values Over Time — Country View</div>', unsafe_allow_html=True)
        countries = sorted(shap_df["Country"].unique())
        sel_country = st.selectbox("Select Country", countries, index=countries.index("India") if "India" in countries else 0, key="shap_country")
        shap_country = shap_df[shap_df["Country"]==sel_country].sort_values("Year")

        shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]
        fig_shap = go.Figure()
        for sc in shap_cols:
            label = sc.replace("shap_","")
            fig_shap.add_trace(go.Scatter(
                x=shap_country["Year"], y=shap_country[sc],
                name=label, mode="lines+markers",
                line=dict(width=2),
            ))
        fig_shap.add_hline(y=0, line_dash="dash", line_color="#aaa")
        fig_shap.update_layout(
            height=360, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
            xaxis_title="Year", yaxis_title="SHAP Value",
            legend=dict(orientation="h", y=-0.2), margin=dict(l=40,r=20,t=10,b=60),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.markdown("# 📈 Prophet Forecasting — 2024 to 2027")

    if V2:
        st.markdown("v2.0 composite recomputed under **optimised weights** (Digital 37.4%, Manufacturing 34.3%, Banking 18.8%, FDI 9.5%). Baseline Prophet pillar forecasts unchanged.")
        v2_fc = load_v2_forecasts()
    else:
        st.markdown("194 countries × 4 pillars × 4 years. Config: `changepoint_prior_scale=0.05`, `interval_width=0.80`.")

    st.markdown("---")

    master = load_master()
    forecasts = load_forecasts()
    intervals = load_intervals()

    tab1, tab2 = st.tabs(["Country Forecast", "Global Rankings 2027"])

    with tab1:
        countries = sorted(forecasts["Country"].unique())
        col1, col2 = st.columns([2,1])
        with col1:
            country = st.selectbox("Select Country", countries, index=countries.index("India"), key="forecast_country")
        with col2:
            pillar_choice = st.selectbox("Pillar", ["Composite","FDI","Banking","Manufacturing","Digital"], key="forecast_pillar")

        pillar_map = {"Composite":"score_composite","FDI":"score_fdi",
                      "Banking":"score_banking","Manufacturing":"score_manuf","Digital":"score_digital"}
        pillar_key = pillar_map[pillar_choice]

        hist = master[master["Country"]==country].sort_values("TIME_PERIOD")
        fc   = forecasts[forecasts["Country"]==country].sort_values("Year")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["TIME_PERIOD"], y=hist[pillar_key],
            name="Historical", line=dict(color=C["navy"], width=2.5),
            mode="lines+markers", marker=dict(size=5),
        ))

        last_hist_year = int(hist["TIME_PERIOD"].max())
        last_hist_val  = hist[hist["TIME_PERIOD"] == last_hist_year][pillar_key].values[0]
        fc_years = [last_hist_year] + list(fc["Year"])
        fc_vals  = [last_hist_val]  + list(fc[pillar_key])
        fig.add_trace(go.Scatter(
            x=fc_years, y=fc_vals,
            name="Forecast (v1.0)", line=dict(color=C["orange"], width=2.5, dash="dash"),
            mode="lines+markers", marker=dict(size=7, symbol="diamond", color=C["orange"]),
        ))

        # v2.0 composite overlay
        if V2 and pillar_choice == "Composite" and v2_fc is not None:
            fc_v2 = v2_fc[v2_fc["country"]==country].sort_values("year")
            if not fc_v2.empty:
                v2_years = [last_hist_year] + list(fc_v2["year"])
                v2_vals  = [last_hist_val]  + list(fc_v2["composite_v2"])
                fig.add_trace(go.Scatter(
                    x=v2_years, y=v2_vals,
                    name="Composite v2.0 (optimised weights)",
                    line=dict(color="#27ae60", width=2.5, dash="dot"),
                    mode="lines+markers", marker=dict(size=7, symbol="circle", color="#27ae60"),
                ))

        if pillar_key != "score_composite":
            intv = intervals[(intervals["Country"]==country) & (intervals["Pillar"]==pillar_key)]
        else:
            intv = intervals[intervals["Country"]==country].groupby("Year")[["yhat_lower","yhat_upper"]].mean().reset_index()
            intv["Pillar"] = "score_composite"

        if not intv.empty:
            fig.add_trace(go.Scatter(
                x=list(intv["Year"]) + list(intv["Year"])[::-1],
                y=list(intv["yhat_upper"]) + list(intv["yhat_lower"])[::-1],
                fill="toself", fillcolor="rgba(224,123,57,0.15)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="80% CI",
            ))

        fig.add_vline(x=last_hist_year + 0.5, line_dash="dot", line_color="#aaa",
                      annotation_text="Forecast →", annotation_position="top right")
        fig.update_layout(
            height=440, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
            xaxis=dict(title="Year", range=[2000, 2028], dtick=5, tickmode="linear"),
            yaxis=dict(title="Score (0-1)", range=[0, 1]),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40, r=40, t=40, b=60),
            title=f"{country} — {pillar_choice} Score (2000–2027)",
        )
        st.plotly_chart(fig, use_container_width=True)

        fc27 = fc[fc["Year"]==2027].iloc[0] if len(fc[fc["Year"]==2027]) > 0 else None
        if fc27 is not None:
            st.markdown("**2027 Forecast Summary**")
            c1,c2,c3,c4,c5 = st.columns(5)
            for col, (label, key) in zip([c1,c2,c3,c4,c5], [
                ("Composite","score_composite"),("FDI","score_fdi"),
                ("Banking","score_banking"),("Manuf.","score_manuf"),("Digital","score_digital")
            ]):
                with col:
                    color = PILLAR_COLORS[key]
                    val = fc27[key]
                    # Show v2.0 composite alongside v1.0
                    if V2 and key == "score_composite" and v2_fc is not None:
                        fc_v2_27 = v2_fc[(v2_fc["country"]==country) & (v2_fc["year"]==2027)]
                        v2_val = fc_v2_27["composite_v2"].values[0] if not fc_v2_27.empty else None
                        extra = f"<div style='font-size:0.72rem;color:#27ae60;margin-top:2px;'>v2.0: {v2_val:.3f}</div>" if v2_val else ""
                    else:
                        extra = ""
                    st.markdown(f"""<div class="metric-card" style="border-top:3px solid {color};">
                        <div class="metric-value" style="color:{color};font-size:1.4rem;">{val:.3f}</div>
                        <div class="metric-label">{label} 2027</div>{extra}</div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">Top 20 Countries by Composite Score — 2027</div>', unsafe_allow_html=True)

        if V2 and v2_fc is not None:
            # Show v2.0 composite ranking
            fc27_all = v2_fc[v2_fc["year"]==2027].sort_values("composite_v2", ascending=False).head(20)
            fig_top = px.bar(
                fc27_all, x="composite_v2", y="country", orientation="h",
                color="composite_v2",
                color_continuous_scale=[[0,C["blue"]],[1,C["green"]]],
                labels={"composite_v2":"Composite Score 2027 (v2.0)","country":""},
                text=fc27_all["composite_v2"].apply(lambda x: f"{x:.3f}"),
            )
        else:
            fc27_all = forecasts[forecasts["Year"]==2027].sort_values("score_composite", ascending=False).head(20)
            fig_top = px.bar(
                fc27_all, x="score_composite", y="Country", orientation="h",
                color="score_composite",
                color_continuous_scale=[[0,C["blue"]],[1,C["green"]]],
                labels={"score_composite":"Composite Score 2027","Country":""},
                text=fc27_all["score_composite"].apply(lambda x: f"{x:.3f}"),
            )

        fig_top.update_layout(
            height=560, yaxis=dict(autorange="reversed"),
            paper_bgcolor="white", plot_bgcolor="#f8f9fa",
            coloraxis_showscale=False, margin=dict(l=120,r=60,t=20,b=40),
        )
        fig_top.update_traces(textposition="outside")
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown('<div class="section-header">2027 Global Composite Map</div>', unsafe_allow_html=True)
        if V2 and v2_fc is not None:
            map_src = v2_fc[v2_fc["year"]==2027].copy()
            # Need REF_AREA — merge from forecasts
            ref_map = forecasts[forecasts["Year"]==2027][["Country","REF_AREA"]].rename(columns={"Country":"country"})
            map_src = map_src.merge(ref_map, on="country", how="left")
            fig_map27 = px.choropleth(
                map_src, locations="REF_AREA", locationmode="ISO-3",
                color="composite_v2", hover_name="country",
                hover_data={"composite_v2":":.3f","REF_AREA":False},
                color_continuous_scale=[[0,"#e74c3c"],[0.5,"#f39c12"],[1,"#27ae60"]],
                range_color=[0.15, 0.85],
                labels={"composite_v2":"Score (v2.0)"},
            )
        else:
            fig_map27 = px.choropleth(
                forecasts[forecasts["Year"]==2027], locations="REF_AREA", locationmode="ISO-3",
                color="score_composite", hover_name="Country",
                hover_data={"score_composite":":.3f","REF_AREA":False},
                color_continuous_scale=[[0,"#e74c3c"],[0.5,"#f39c12"],[1,"#27ae60"]],
                range_color=[0.15, 0.60],
                labels={"score_composite":"Score"},
            )
        fig_map27.update_layout(
            height=400, margin=dict(l=0,r=0,t=10,b=0),
            geo=dict(showframe=False), paper_bgcolor="white",
        )
        st.plotly_chart(fig_map27, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — POLICY SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Policy Scenarios":
    st.markdown("# 🌐 Policy Scenarios — A / B / C")

    if V2:
        st.markdown("Three scenarios under **v2.0 optimised weights**. Policy factors from T8 (LLM + DiD blend, 162 countries). India / USA / Viet Nam use manually-verified Task 9 factors.")
        scenarios = load_v2_scenarios()
        if scenarios is None:
            st.error("t9_scenario_comparison_v2.csv not found — upload to repo root.")
            st.stop()
        # v2.0 column names use Composite_v2
        _comp_col = "Composite_v2"
    else:
        st.markdown("Three scenarios modelled for India, United States, and Viet Nam.")
        scenarios = load_scenarios()
        _comp_col = "Composite"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card" style="border-top:4px solid {C['green']};">
            <b>Scenario A — Baseline</b><br>
            <span style="font-size:0.85rem;color:#6b7c93;">Prophet forecast, no policy overlay</span>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="border-top:4px solid {C['red']};">
            <b>Scenario B — Policy Adjusted</b><br>
            <span style="font-size:0.85rem;color:#6b7c93;">Pillar scores adjusted by {"T8 blended" if V2 else "Task 9"} factors</span>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="border-top:4px solid {C['orange']};">
            <b>Scenario C — Policy Normalisation</b><br>
            <span style="font-size:0.85rem;color:#6b7c93;">Negative adjustments fade 2026–27, positive policies held</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    SCEN_COLORS = {
        "A_Standard":       C["green"],
        "B_PolicyAdjusted": C["red"],
        "C_ShockFade":      C["orange"],
    }
    SCEN_LABELS = {
        "A_Standard":       "A — Baseline",
        "B_PolicyAdjusted": "B — Policy Adjusted",
        "C_ShockFade":      "C — Recovery",
    }

    tab_focus, tab_all = st.tabs(["🔎 Focus Countries (A/B/C)", "🌍 All 194 Countries (A/B/C)"])

    with tab_all:
        st.markdown("**All 194 countries — A / B / C scenarios under v2.0 weights**")
        _t1, _t2 = get_v2_tier_thresholds(2027, "A_Standard")
        st.caption(f"Tier thresholds recalibrated to v2.0 distribution (equal-width bins, 2027 Scenario A): Low <{_t1} · Medium {_t1}–{_t2} · High ≥{_t2}. Thresholds update per selected scenario/year. 162 countries have T8 policy factors; 32 data-poor countries have B = C = A.")
        v2_sc_all = load_v2_scenarios()
        if v2_sc_all is not None:
            col_sc, col_yr, col_f, col_t = st.columns([2,1,3,1])
            with col_sc:
                sel_scen_all = st.selectbox(
                    "Scenario",
                    ["A_Standard","B_PolicyAdjusted","C_ShockFade"],
                    format_func=lambda x: {"A_Standard":"A — Baseline",
                                           "B_PolicyAdjusted":"B — Policy Adjusted",
                                           "C_ShockFade":"C — Recovery"}[x],
                    key="all_scen"
                )
            with col_yr:
                sel_yr_all = st.selectbox("Year", [2024,2025,2026,2027], index=3, key="all_yr")
            with col_f:
                search_all = st.text_input("Search country", placeholder="Type to filter...", key="all_search")
            with col_t:
                tier_filter_all = st.selectbox("Tier", ["All","High","Medium","Low"], key="all_tier")

            sc_slice = v2_sc_all[
                (v2_sc_all["Scenario"]==sel_scen_all) &
                (v2_sc_all["Year"]==sel_yr_all)
            ].copy().sort_values("Composite_v2", ascending=False).reset_index(drop=True)

            # Recompute tier from live distribution — v1.0 thresholds are wrong for v2.0 scores
            sc_slice["Tier_v2"] = sc_slice["Composite_v2"].apply(
                lambda s: assign_tier_v2(s, sel_yr_all, sel_scen_all)
            )
            sc_slice.insert(0, "Rank", range(1, len(sc_slice)+1))

            if search_all:
                sc_slice = sc_slice[sc_slice["Country"].str.contains(search_all, case=False, na=False)]
            if tier_filter_all != "All":
                sc_slice = sc_slice[sc_slice["Tier_v2"]==tier_filter_all]

            st.dataframe(
                sc_slice[["Rank","Country","Composite_v2","Composite_v1","FDI","Banking","Manufacturing","Digital","Tier_v2","Factors_source"]].rename(columns={
                    "Composite_v2":"Composite v2.0","Composite_v1":"Composite v1.0",
                    "Tier_v2":"Tier","Factors_source":"Factors"
                }).style
                    .format({"Composite v2.0":"{:.3f}","Composite v1.0":"{:.3f}",
                             "FDI":"{:.3f}","Banking":"{:.3f}","Manufacturing":"{:.3f}","Digital":"{:.3f}"})
                    .background_gradient(subset=["Composite v2.0"], cmap="Greens"),
                use_container_width=True, height=540,
            )

            # Tier pie — always on full unfiltered slice for that scenario/year
            sc_full = v2_sc_all[
                (v2_sc_all["Scenario"]==sel_scen_all) &
                (v2_sc_all["Year"]==sel_yr_all)
            ].copy()
            sc_full["Tier_v2"] = sc_full["Composite_v2"].apply(
                lambda s: assign_tier_v2(s, sel_yr_all, sel_scen_all)
            )
            tier_counts = sc_full["Tier_v2"].value_counts().reset_index()
            tier_counts.columns = ["Tier","Count"]
            fig_pie = px.pie(
                tier_counts, names="Tier", values="Count",
                color="Tier", color_discrete_map={"High":C["green"],"Medium":C["amber"],"Low":C["red"]},
                title=f"Tier Distribution — {sel_yr_all} · {sel_scen_all.replace('_',' ')} (194 countries, v2.0)"
            )
            fig_pie.update_layout(height=300, paper_bgcolor="white", margin=dict(t=40,b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Upload t9_scenario_comparison_v2.csv to the repo root to enable this view.")

    with tab_focus:
        countries = scenarios["Country"].unique().tolist()
        col1, col2 = st.columns([2,1])
        with col1:
            sel_country = st.selectbox("Select Country", countries, key="focus_country")
        with col2:
            sel_pillar = st.selectbox("Pillar", ["Composite","FDI","Banking","Manufacturing","Digital"], key="focus_pillar")

        pillar_col_map = {"Composite": _comp_col, "FDI":"FDI","Banking":"Banking",
                          "Manufacturing":"Manufacturing","Digital":"Digital"}
        pcol = pillar_col_map[sel_pillar]

        sc_country = scenarios[scenarios["Country"]==sel_country]

        fig = go.Figure()
        for scen, color in SCEN_COLORS.items():
            sc_data = sc_country[sc_country["Scenario"]==scen].sort_values("Year")
            if pcol not in sc_data.columns:
                continue
            fig.add_trace(go.Scatter(
                x=sc_data["Year"], y=sc_data[pcol],
                name=SCEN_LABELS[scen], line=dict(color=color, width=2.5),
                mode="lines+markers", marker=dict(size=8),
            ))

        # CI bands (v2.0 only)
        if V2 and "CI_Lower_95" in sc_country.columns and sel_pillar == "Composite":
            sc_b = sc_country[sc_country["Scenario"]=="B_PolicyAdjusted"].sort_values("Year")
            if not sc_b.empty:
                fig.add_trace(go.Scatter(
                    x=list(sc_b["Year"]) + list(sc_b["Year"])[::-1],
                    y=list(sc_b["CI_Upper_95"]) + list(sc_b["CI_Lower_95"])[::-1],
                    fill="toself", fillcolor="rgba(231,76,60,0.1)",
                    line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="95% CI (B)",
                ))

        fig.update_layout(
            height=420, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
            xaxis=dict(title="Year", tickformat='d'),
            yaxis_title="Score (0–1)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=40,r=20,t=20,b=60),
            title=f"{sel_country} — {sel_pillar} Score: Scenario Comparison (2024–2027)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pillar breakdown by scenario — 2027
        st.markdown('<div class="section-header">All Pillars by Scenario — 2027</div>', unsafe_allow_html=True)
        sc27 = sc_country[sc_country["Year"]==2027]
        pillars_sc = ["FDI","Banking","Manufacturing","Digital", _comp_col]
        fig_sc = go.Figure()
        for scen, color in SCEN_COLORS.items():
            row = sc27[sc27["Scenario"]==scen]
            if not row.empty:
                vals = []
                for p in pillars_sc:
                    if p in row.columns:
                        vals.append(row[p].values[0])
                    else:
                        vals.append(0)
                labels = ["FDI","Banking","Manufacturing","Digital","Composite"]
                fig_sc.add_trace(go.Bar(name=SCEN_LABELS[scen], x=labels, y=vals,
                                        marker_color=color, opacity=0.85))
        fig_sc.update_layout(
            barmode="group", height=340, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
            legend=dict(orientation="h", y=-0.25), margin=dict(l=20,r=20,t=10,b=70),
            yaxis=dict(title="Score (0–1)"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Policy context
        POLICY_CONTEXT = {
            "India": {
                "shock": "RBI rate tightening (6.5% repo), global FDI headwinds in 2024",
                "recovery": "RBI easing, PLI schemes entering full production, hyperscaler data centre investments ($7B+)",
            },
            "United States": {
                "shock": "Post-SVB banking regulatory tightening (Basel III Endgame), tariff uncertainty",
                "recovery": "IRA + CHIPS Act manufacturing surge, AI investment maintaining >60% global share",
            },
            "Viet Nam": {
                "shock": "Global supply chain re-routing uncertainty, currency pressure",
                "recovery": "China+1 strategy beneficiary, Samsung/Intel FDI inflows, digital infrastructure buildout",
            },
        }

        # v2.0: also show T8 policy factors
        if V2:
            pf = load_v2_policy_factors()
            if pf is not None:
                name_variants = {"United States": ["United States","USA"], "Viet Nam": ["Viet Nam","Vietnam"]}
                variants = name_variants.get(sel_country, [sel_country])
                country_factors = pf[pf["country"].isin(variants)][
                    ["pillar","direction","blended_magnitude","confidence","policy_name","years_applicable"]
                ].sort_values(["pillar","direction"])
                if not country_factors.empty:
                    st.markdown("---")
                    st.markdown('<div class="section-header">T8 Policy Factors — Blended Adjustments</div>', unsafe_allow_html=True)
                    st.dataframe(
                        country_factors.rename(columns={
                            "pillar":"Pillar","direction":"Direction",
                            "blended_magnitude":"Magnitude %","confidence":"Confidence",
                            "policy_name":"Policy","years_applicable":"Years"
                        }).style.format({"Magnitude %":"{:+.1f}%"}),
                        use_container_width=True, height=280,
                    )

        ctx = POLICY_CONTEXT.get(sel_country, {})
        if ctx:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""<div style="background:#fff5f5;border-left:4px solid {C['red']};border-radius:8px;padding:14px;">
                    <b>Scenario B — Policy Adjusted Driver</b><br>
                    <span style="font-size:0.88rem;">{ctx.get('shock','')}</span>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div style="background:#f0faf4;border-left:4px solid {C['green']};border-radius:8px;padding:14px;">
                    <b>Scenario C — Policy Normalisation Driver</b><br>
                    <span style="font-size:0.88rem;">{ctx.get('recovery','')}</span>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INVESTMENT RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Investment Recommendations":
    st.markdown("# 🏆 Investment Recommendations")

    if V2:
        st.markdown("Final investment verdict based on **v2.0 recalibrated composite** — optimised weights + T8 policy overlay + T11 risk-adjusted rankings.")
        rec = load_v2_recommendations()
        scenarios = load_v2_scenarios()
        risk = load_v2_risk_rankings()
        forecasts_src = load_v2_forecasts()
        _comp_col = "composite_v2"
        _country_col = "country"
    else:
        st.markdown("Final investment verdict based on Scenario C (Post-Recovery) 2027 composite trajectory.")
        rec = load_recommendations()
        scenarios = load_scenarios()
        risk = None
        forecasts_src = load_forecasts()
        _comp_col = "score_composite"
        _country_col = "Country"

    forecasts = load_forecasts()
    st.markdown("---")

    # ── Recommendation cards ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Country Verdicts — 2027</div>', unsafe_allow_html=True)
    for _, row in rec.iterrows():
        country      = row.get("Country", row.get("country",""))
        recommendation = row.get("Recommendation", row.get("recommendation",""))
        rationale    = row.get("Rationale", row.get("rationale",""))
        tier         = row.get("Tier_2027", row.get("tier_2027",""))
        score_2024   = float(row.get("Score_2024_ScenC", row.get("score_2024", 0)) or 0)
        score_2027   = float(row.get("Score_2027_ScenC", row.get("score_2027", 0)) or 0)
        _trend_raw   = row.get("Trend_2024_2027", row.get("trend", 0))
        if isinstance(_trend_raw, str):
            trend = 1.0 if _trend_raw.strip().lower() == "rising" else -1.0
        else:
            try:    trend = float(_trend_raw or 0)
            except: trend = 0.0

        rec_color = {"Invest Now": C["green"], "Wait": C["amber"], "Avoid": C["red"]}.get(recommendation, C["navy"])

        # Risk rank badge for v2.0
        risk_badge = ""
        if V2 and risk is not None:
            r_row = risk[risk["country"]==country]
            if not r_row.empty:
                rr = int(r_row["rank_risk_adj"].values[0])
                rc = int(r_row["rank_composite"].values[0])
                arrow = "▲" if rr < rc else ("▼" if rr > rc else "–")
                risk_badge = f'<span style="background:#f0faf4;color:#27ae60;border-radius:20px;padding:4px 12px;font-size:0.82rem;border:1px solid #27ae60;">Risk-adj rank: #{rr} {arrow}</span>'

        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px;
                    box-shadow:0 2px 10px rgba(0,0,0,0.08);border-left:6px solid {rec_color};">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;flex-wrap:wrap;">
                <div style="font-size:1.4rem;font-weight:700;color:{C['navy']};">{country}</div>
                <span style="background:{rec_color};color:white;border-radius:20px;padding:4px 14px;font-weight:700;font-size:0.9rem;">
                    {recommendation}
                </span>
                <span style="background:{C['light']};color:{C['navy']};border-radius:20px;padding:4px 12px;font-size:0.85rem;">
                    Tier: {tier}
                </span>
                {risk_badge}
            </div>
            <div style="display:flex;gap:24px;margin-bottom:10px;">
                <div><span style="color:#6b7c93;font-size:0.82rem;">2024 Score</span><br>
                     <span style="font-weight:700;font-size:1.1rem;">{score_2024:.4f}</span></div>
                <div><span style="color:#6b7c93;font-size:0.82rem;">2027 Score</span><br>
                     <span style="font-weight:700;font-size:1.1rem;color:{rec_color};">{score_2027:.4f}</span></div>
                <div><span style="color:#6b7c93;font-size:0.82rem;">Trend (2024–27)</span><br>
                     <span style="font-weight:700;font-size:1.1rem;color:{C['green'] if trend>0 else C['red']};">
                     {"▲" if trend > 0 else "▼"} {abs(trend):.4f}</span></div>
            </div>
            <div style="font-size:0.88rem;color:#555;">{rationale}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── v2.0: Risk-adjusted ranking table (top 20) ────────────────────────────
    if V2 and risk is not None:
        st.markdown('<div class="section-header">Risk-Adjusted Rankings — Top 20 (2027, v2.0)</div>', unsafe_allow_html=True)
        st.caption("Risk-adjusted score = composite_v2 ÷ 5yr rolling volatility (std). Countries with stable, high scores rank best.")
        top20_risk = risk.sort_values("rank_risk_adj").head(20)[
            ["country","composite_v2","vol_std","risk_adj_score","rank_composite","rank_risk_adj"]
        ].reset_index(drop=True)
        top20_risk.index += 1
        top20_risk.columns = ["Country","Composite v2.0","Volatility (5yr std)","Risk-Adj Score","Composite Rank","Risk-Adj Rank"]
        top20_risk["Rank Δ"] = top20_risk["Composite Rank"] - top20_risk["Risk-Adj Rank"]
        st.dataframe(
            top20_risk.style
                .format({"Composite v2.0":"{:.3f}","Volatility (5yr std)":"{:.4f}","Risk-Adj Score":"{:.2f}"})
                .background_gradient(subset=["Composite v2.0"], cmap="Greens")
                .background_gradient(subset=["Risk-Adj Score"], cmap="Blues"),
            use_container_width=True, height=700,
        )
        st.markdown("---")

    # ── Scenario C trajectory ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Scenario C Composite Trajectories (2024–2027)</div>', unsafe_allow_html=True)
    if V2 and scenarios is not None:
        sc_c = scenarios[scenarios["Scenario"]=="C_ShockFade"].sort_values(["Country","Year"])
        fig_traj = px.line(
            sc_c, x="Year", y="Composite_v2", color="Country",
            markers=True,
            color_discrete_sequence=[C["green"], C["blue"], C["orange"]],
            labels={"Composite_v2":"Composite Score (v2.0)","Year":"Year"},
        )
    else:
        sc_c = load_scenarios()
        sc_c = sc_c[sc_c["Scenario"]=="C_ShockFade"].sort_values(["Country","Year"])
        fig_traj = px.line(
            sc_c, x="Year", y="Composite", color="Country",
            markers=True, line_dash="Country",
            color_discrete_sequence=[C["green"], C["blue"], C["orange"]],
            labels={"Composite":"Composite Score","Year":"Year"},
        )

    fig_traj.update_layout(
        height=380, paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        xaxis=dict(tickformat='d'),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(range=[0.3, 0.85] if V2 else [0.3, 0.7]),
        margin=dict(l=40,r=20,t=20,b=60),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ── Global top 10 ─────────────────────────────────────────────────────────
    if V2 and forecasts_src is not None:
        st.markdown('<div class="section-header">Global Top 10 — Composite Score 2027 (v2.0, All 194 Countries)</div>', unsafe_allow_html=True)
        top10 = forecasts_src[forecasts_src["year"]==2027].sort_values("composite_v2", ascending=False).head(10)
        # Merge REF_AREA from v1 forecasts for display
        ref_map = forecasts[forecasts["Year"]==2027][["Country","REF_AREA"]].rename(columns={"Country":"country"})
        top10 = top10.merge(ref_map, on="country", how="left")
        top10_display = top10[["country","composite_v2","composite_v1"]].copy()
        top10_display.columns = ["Country","Composite v2.0","Composite v1.0"]
        top10_display["Δ v2–v1"] = top10_display["Composite v2.0"] - top10_display["Composite v1.0"]
        top10_display = top10_display.reset_index(drop=True)
        top10_display.index += 1
        st.dataframe(
            top10_display.style
                .format({c:"{:.3f}" for c in ["Composite v2.0","Composite v1.0","Δ v2–v1"]})
                .background_gradient(subset=["Composite v2.0"], cmap="Greens"),
            use_container_width=True, height=380,
        )
    else:
        st.markdown('<div class="section-header">Global Top 10 — Composite Score 2027 (All 194 Countries)</div>', unsafe_allow_html=True)
        top10 = forecasts[forecasts["Year"]==2027].sort_values("score_composite", ascending=False).head(10)
        top10_display = top10[["Country","score_composite","score_fdi","score_banking","score_manuf","score_digital"]].copy()
        top10_display.columns = ["Country","Composite","FDI","Banking","Manufacturing","Digital"]
        top10_display = top10_display.reset_index(drop=True)
        top10_display.index += 1
        st.dataframe(
            top10_display.style.format({c:"{:.3f}" for c in ["Composite","FDI","Banking","Manufacturing","Digital"]})
                               .background_gradient(subset=["Composite"], cmap="Greens"),
            use_container_width=True, height=380,
        )

    # ── Key insights ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    if V2:
        insights = [
            ("🏆 United States #1 risk-adjusted",   "Highest risk-adj score — strong composite AND lowest volatility (0.022 std). Consistent performer across all weight sets."),
            ("🇳🇱 Netherlands #1 composite (v2.0)", "Composite 0.810 in 2027 — balanced pillar strength amplified by Digital upweighting. No single-pillar dependency."),
            ("📈 India — Digital-led rise",           "Composite 0.594 by 2027 under v2.0. Manufacturing strength + accelerating digital. FDI remains the drag."),
            ("⚖️ Rank stability ρ = 0.979",          "Spearman correlation between v1.0 and v2.0 composite rankings is 0.979 — investment recommendations unchanged by recalibration."),
            ("💻 Digital Economy is the key differentiator", "SHAP confirms 37.4% importance — now reflected in optimised weights. Countries with weak digital lose rank under v2.0."),
        ]
    else:
        insights = [
            ("🏆 Netherlands #1 in 2027",                f"Composite 0.576 — balanced strength across all 4 pillars simultaneously. No single-pillar dependency."),
            ("💻 USA strong but digital-concentrated",    "Composite 0.531 — digital score near ceiling. Growth limited by FDI volatility."),
            ("📈 India — Digital-led rise",               "Composite 0.444 by 2027. Manufacturing strength + accelerating digital. FDI remains the drag."),
            ("📉 UK — FDI-driven decline",                "2023: 0.426 → 2027: 0.354. Post-Brexit FDI volatility extrapolated by Prophet. No policy overlay."),
            ("🌐 Digital Economy is the key differentiator", "SHAP confirms 72.7% of composite variance explained by digital score. Highest separation across tiers."),
        ]
    for title, body in insights:
        st.markdown(f"""
        <div style="background:{C['light']};border-radius:8px;padding:14px;margin-bottom:8px;border-left:4px solid {C['orange']};">
            <div style="font-weight:700;color:{C['navy']};">{title}</div>
            <div style="font-size:0.88rem;color:#555;margin-top:4px;">{body}</div>
        </div>""", unsafe_allow_html=True)
