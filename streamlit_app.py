import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Import Dependency Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(48, 43, 99, 0.35);
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-banner p {
    color: rgba(255,255,255,0.7);
    font-size: 0.95rem;
    margin: 0;
}

.metric-container {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card.green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
}
.metric-card .metric-label {
    color: rgba(255,255,255,0.85);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.metric-card .metric-value {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 800;
}
.metric-card .metric-unit {
    color: rgba(255,255,255,0.7);
    font-size: 0.85rem;
    font-weight: 400;
    margin-left: 4px;
}

.algo-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: #fff;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.algo-badge.xgb {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    color: #333;
}
.algo-badge.ridge {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    color: #333;
}

.context-tag {
    text-align: center;
    color: #888;
    font-size: 0.82rem;
    margin-top: 0.5rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] label {
    color: #e0e0e0 !important;
}

.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding: 1rem 0;
    border-top: 1px solid #eee;
}

.stat-row {
    display: flex;
    gap: 0.75rem;
    margin: 0.8rem 0;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 120px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid rgba(102, 126, 234, 0.25);
}
.stat-card .stat-num {
    color: #667eea;
    font-size: 1.6rem;
    font-weight: 800;
}
.stat-card .stat-label {
    color: rgba(255,255,255,0.55);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}

.perf-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}
.perf-table th {
    background: linear-gradient(135deg, #302b63, #24243e);
    color: #fff;
    padding: 0.8rem 1rem;
    text-align: left;
    font-weight: 600;
}
.perf-table td {
    padding: 0.7rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    color: #ddd;
}
.perf-table tr:hover td {
    background: rgba(102, 126, 234, 0.08);
}
.perf-table .best {
    color: #38ef7d;
    font-weight: 700;
}

.future-grid {
    display: flex;
    gap: 0.75rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.future-card {
    flex: 1;
    min-width: 100px;
    border-radius: 12px;
    padding: 1.2rem 0.8rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.future-card.rf {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border-color: rgba(102, 126, 234, 0.3);
}
.future-card.xgb {
    background: linear-gradient(135deg, #fa709a22, #fee14022);
    border-color: rgba(250, 112, 154, 0.3);
}
.future-card.ridge {
    background: linear-gradient(135deg, #43e97b22, #38f9d722);
    border-color: rgba(67, 233, 123, 0.3);
}
.future-card .fy-year {
    color: rgba(255,255,255,0.5);
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.future-card .fy-val {
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 0.3rem;
}
.future-card.rf .fy-val { color: #667eea; }
.future-card.xgb .fy-val { color: #fa709a; }
.future-card.ridge .fy-val { color: #43e97b; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Load Models, Data & Metrics
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'Area_enc', 'Item_enc', 'Year',
    'Import_Lag1', 'Import_Lag2', 'Import_Lag3',
    'Dep_Lag1', 'Dep_Lag2',
    'Import_RollingMean3',
    'FoodSupply_Lag1',
]

FUTURE_YEARS = list(range(2014, 2019))   # 5-year for predictor
FORECAST_3YR = list(range(2014, 2017))   # 3-year for dashboard


@st.cache_resource
def load_models():
    m = {}
    m['rf_import']      = joblib.load("model/rf_model_import.pkl")
    m['rf_dep']         = joblib.load("model/rf_model_dependency.pkl")
    m['xgb_import']     = joblib.load("model/xgb_model_import.pkl")
    m['xgb_dep']        = joblib.load("model/xgb_model_dependency.pkl")
    m['ridge_import']   = joblib.load("model/ridge_model_import.pkl")
    m['ridge_dep']      = joblib.load("model/ridge_model_dependency.pkl")
    m['area_encoder']   = joblib.load("model/area_encoder.pkl")
    m['item_encoder']   = joblib.load("model/item_encoder.pkl")
    return m

@st.cache_data
def load_data():
    return pd.read_csv("dataset/processed_data.csv")

@st.cache_data
def load_metrics():
    with open("model/metrics.json", "r") as f:
        return json.load(f)


models  = load_models()
df      = load_data()
metrics = load_metrics()

area_encoder = models['area_encoder']
item_encoder = models['item_encoder']
areas = sorted(area_encoder.classes_.tolist())
items = sorted(item_encoder.classes_.tolist())


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction helpers — compute lag features on the fly
# ══════════════════════════════════════════════════════════════════════════════
def _build_lookups(area, item):
    """Build dicts of historical values for a given (area, item) pair."""
    hist = df[(df['Area'] == area) & (df['Item'] == item)].sort_values('Year')
    return (
        dict(zip(hist['Year'], hist['Import_Quantity'])),
        dict(zip(hist['Year'], hist['Dependency_Pct'])),
        dict(zip(hist['Year'], hist['Food_Supply'])),
        hist,
    )


def predict_years(model_imp, model_dep, area, item, years):
    """
    Predict Import Quantity and Dependency % for one or more years.
    Computes lag features from historical data.
    For multi-year forecasts, feeds predictions back as lags (iterative).
    Returns list of (imp_val, dep_val) tuples.
    """
    area_enc = int(area_encoder.transform([area])[0])
    item_enc = int(item_encoder.transform([item])[0])
    imp_lk, dep_lk, fs_lk, _ = _build_lookups(area, item)

    results = []
    for yr in years:
        imp1 = imp_lk.get(yr - 1, 0.0)
        imp2 = imp_lk.get(yr - 2, 0.0)
        imp3 = imp_lk.get(yr - 3, 0.0)
        dep1 = dep_lk.get(yr - 1, 0.0)
        dep2 = dep_lk.get(yr - 2, 0.0)
        roll3 = np.mean([imp1, imp2, imp3])
        fs1   = fs_lk.get(yr - 1, fs_lk[max(fs_lk)] if fs_lk else 0.0)

        feat = pd.DataFrame([[
            area_enc, item_enc, yr,
            imp1, imp2, imp3,
            dep1, dep2,
            roll3, fs1,
        ]], columns=FEATURE_COLS)

        imp_val = round(float(model_imp.predict(feat)[0]), 2)
        dep_val = round(float(model_dep.predict(feat)[0]), 2)

        # Store predictions so next iteration can use them as lags
        imp_lk[yr] = imp_val
        dep_lk[yr] = dep_val
        fs_lk[yr]  = fs1          # carry forward last known food supply

        results.append((imp_val, dep_val))

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar — Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 Navigation")
    page = st.radio("Go to", ["🔮 Predictor", "📊 Dashboard"],
                    label_visibility="collapsed")
    st.markdown("---")


# ██████████████████████████████████████████████████████████████████████████████
#  PAGE 1 — PREDICTOR
# ██████████████████████████████████████████████████████████████████████████████
if page == "🔮 Predictor":

    with st.sidebar:
        st.markdown("## ⚙️ Prediction Settings")
        algorithm = st.selectbox("🤖 Algorithm",
                                 ["Random Forest", "XGBoost", "Ridge Regression"],
                                 help="Choose the ML algorithm for prediction")
        area = st.selectbox("🌍 Country / Area", [""] + areas,
                            format_func=lambda x: "-- Select Country --" if x == "" else x)
        item = st.selectbox("🍎 Food Item", [""] + items,
                            format_func=lambda x: "-- Select Food Item --" if x == "" else x)
        year = st.number_input("📅 Year (1961–2018)",
                               min_value=1961, max_value=2018, value=2015, step=1)
        st.markdown("---")
        predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

    st.markdown("""
    <div class="hero-banner">
        <h1>🌾 Food Import Dependency Predictor</h1>
        <p>Powered by RandomForest, XGBoost & Ridge Regression · FAO data (1961–2013) · Predicts up to 2018</p>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        if not area or not item:
            st.error("⚠️ Please select both a Country/Area and a Food Item.")
        else:
            try:
                if algorithm == "Random Forest":
                    m_imp, m_dep, algo_key = models['rf_import'], models['rf_dep'], "rf"
                elif algorithm == "XGBoost":
                    m_imp, m_dep, algo_key = models['xgb_import'], models['xgb_dep'], "xgb"
                else:
                    m_imp, m_dep, algo_key = models['ridge_import'], models['ridge_dep'], "ridge"

                # Single-year prediction
                [(imp_val, dep_val)] = predict_years(m_imp, m_dep, area, item, [year])

                badge = f"algo-badge {algo_key}"
                st.markdown(
                    f'<div style="text-align:center"><span class="{badge}">{algorithm}</span></div>',
                    unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Import Quantity</div>
                        <div class="metric-value">{imp_val:,.2f}<span class="metric-unit">tonnes</span></div>
                    </div>
                    <div class="metric-card green">
                        <div class="metric-label">Import Dependency</div>
                        <div class="metric-value">{dep_val:,.2f}<span class="metric-unit">%</span></div>
                    </div>
                </div>
                <div class="context-tag">{area} · {item} · {year}</div>
                """, unsafe_allow_html=True)

                # ── Charts ────────────────────────────────────────────────
                _, _, _, hist_df = _build_lookups(area, item)
                if len(hist_df) > 0:
                    hist_years  = sorted(hist_df['Year'].tolist())[-10:]
                    hist_import = [round(hist_df.loc[hist_df['Year'] == y, 'Import_Quantity'].values[0], 2)
                                   for y in hist_years if y in hist_df['Year'].values]
                    hist_dep    = [round(hist_df.loc[hist_df['Year'] == y, 'Dependency_Pct'].values[0], 2)
                                   for y in hist_years if y in hist_df['Year'].values]

                    future_preds = predict_years(m_imp, m_dep, area, item, FUTURE_YEARS)
                    future_import = [p[0] for p in future_preds]
                    future_dep    = [p[1] for p in future_preds]

                    st.markdown("### 📈 Import Quantity Trend")
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Scatter(
                        x=hist_years, y=hist_import, mode='lines+markers', name='Historical',
                        line=dict(color='#667eea', width=3), marker=dict(size=6),
                        fill='tozeroy', fillcolor='rgba(102,126,234,0.1)'))
                    fig_imp.add_trace(go.Scatter(
                        x=FUTURE_YEARS, y=future_import, mode='lines+markers', name='Forecast',
                        line=dict(color='#f5576c', width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond'),
                        fill='tozeroy', fillcolor='rgba(245,87,108,0.08)'))
                    fig_imp.update_layout(
                        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.02)', xaxis_title='Year',
                        yaxis_title='Import Quantity (tonnes)',
                        legend=dict(orientation='h', y=-0.15),
                        margin=dict(l=40, r=20, t=20, b=60), height=380)
                    st.plotly_chart(fig_imp, key="pred_imp_chart")

                    st.markdown("### 📊 Import Dependency % Trend")
                    fig_dep = go.Figure()
                    fig_dep.add_trace(go.Scatter(
                        x=hist_years, y=hist_dep, mode='lines+markers', name='Historical',
                        line=dict(color='#11998e', width=3), marker=dict(size=6),
                        fill='tozeroy', fillcolor='rgba(17,153,142,0.1)'))
                    fig_dep.add_trace(go.Scatter(
                        x=FUTURE_YEARS, y=future_dep, mode='lines+markers', name='Forecast',
                        line=dict(color='#fa709a', width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond'),
                        fill='tozeroy', fillcolor='rgba(250,112,154,0.08)'))
                    fig_dep.update_layout(
                        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.02)', xaxis_title='Year',
                        yaxis_title='Dependency %',
                        legend=dict(orientation='h', y=-0.15),
                        margin=dict(l=40, r=20, t=20, b=60), height=380)
                    st.plotly_chart(fig_dep, key="pred_dep_chart")
                else:
                    st.info("ℹ️ No historical data for this Country+Item combination.")

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")


# ██████████████████████████████████████████████████████████████████████████████
#  PAGE 2 — DASHBOARD
# ██████████████████████████████████████████████████████████████████████████████
elif page == "📊 Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <h1>📊 Model Dashboard & Analytics</h1>
        <p>Evaluation Metrics · Feature Importance · Future 3-Year Forecast · Dataset Overview</p>
    </div>
    """, unsafe_allow_html=True)

    ds   = metrics['dataset_info']
    mods = metrics['models']
    hp   = metrics['hyperparameters']

    # ══════════════════════════════════════════════════════════════════════════
    #  Section 1 — Dataset Overview
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📋 Dataset Overview")
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-num">{ds['total_rows']:,}</div><div class="stat-label">Total Rows</div></div>
        <div class="stat-card"><div class="stat-num">{ds['training_rows']:,}</div><div class="stat-label">After Lag Eng.</div></div>
        <div class="stat-card"><div class="stat-num">{ds['train_rows']:,}</div><div class="stat-label">Train Set</div></div>
        <div class="stat-card"><div class="stat-num">{ds['test_rows']:,}</div><div class="stat-label">Test Set</div></div>
        <div class="stat-card"><div class="stat-num">{ds['n_countries']}</div><div class="stat-label">Countries</div></div>
        <div class="stat-card"><div class="stat-num">{ds['n_items']}</div><div class="stat-label">Food Items</div></div>
        <div class="stat-card"><div class="stat-num">{ds['year_min']}–{ds['year_max']}</div><div class="stat-label">Year Range</div></div>
        <div class="stat-card"><div class="stat-num">{ds['n_features']}</div><div class="stat-label">Features</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Features ({ds['n_features']}):** `{', '.join(ds['features'])}`  |  "
                f"**Split:** 80 / 20  |  "
                f"**Lag rows dropped:** {ds['lag_rows_dropped']:,}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  Section 2 — Hyperparameters
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🔧 Hyperparameters")
    hp1, hp2, hp3 = st.columns(3)
    with hp1:
        st.markdown("#### 🌲 Random Forest")
        for k, v in hp['random_forest'].items():
            st.markdown(f"- **{k}**: `{v}`")
    with hp2:
        st.markdown("#### 🚀 XGBoost")
        for k, v in hp['xgboost'].items():
            st.markdown(f"- **{k}**: `{v}`")
    with hp3:
        st.markdown("#### 📐 Ridge Regression")
        for k, v in hp['ridge'].items():
            st.markdown(f"- **{k}**: `{v}`")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  Section 3 — Evaluation Metrics
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📏 Evaluation Metrics")

    model_names = {
        'rf_import':    'RandomForest – Import Qty',
        'rf_dep':       'RandomForest – Dependency %',
        'xgb_import':   'XGBoost – Import Qty',
        'xgb_dep':      'XGBoost – Dependency %',
        'ridge_import': 'Ridge – Import Qty',
        'ridge_dep':    'Ridge – Dependency %',
    }

    # Only include models that exist in the metrics file
    available_models = {k: v for k, v in model_names.items() if k in mods}

    r2_vals   = {k: mods[k]['r2_test'] for k in available_models}
    mae_vals  = {k: mods[k]['mae'] for k in available_models}
    rmse_vals = {k: mods[k]['rmse'] for k in available_models}
    mape_vals = {k: mods[k]['mape'] for k in available_models}
    best_r2   = max(r2_vals.values())
    best_mae  = min(mae_vals.values())
    best_rmse = min(rmse_vals.values())
    best_mape = min(mape_vals.values())

    # Best model badges
    best_info = metrics.get('best_model', {})
    if best_info:
        best_imp_key = best_info.get('import_quantity', '')
        best_dep_key = best_info.get('dependency_pct', '')
        best_imp_name = available_models.get(best_imp_key, 'N/A')
        best_dep_name = available_models.get(best_dep_key, 'N/A')
        st.markdown(f"**🏆 Best for Import Qty:** `{best_imp_name}`  |  "
                    f"**🏆 Best for Dependency %:** `{best_dep_name}`")

    table_rows = ""
    for key, name in available_models.items():
        r2_c   = ' class="best"' if mods[key]['r2_test'] == best_r2 else ''
        mae_c  = ' class="best"' if mods[key]['mae'] == best_mae else ''
        rmse_c = ' class="best"' if mods[key]['rmse'] == best_rmse else ''
        mape_c = ' class="best"' if mods[key]['mape'] == best_mape else ''
        table_rows += f"""
        <tr>
            <td>{name}</td>
            <td{r2_c}>{mods[key]['r2_train']:.6f}</td>
            <td{r2_c}>{mods[key]['r2_test']:.6f}</td>
            <td{mae_c}>{mods[key]['mae']:,.4f}</td>
            <td{rmse_c}>{mods[key]['rmse']:,.4f}</td>
            <td{mape_c}>{mods[key]['mape']:.2f}%</td>
            <td>{mods[key]['train_time_sec']}s</td>
        </tr>"""

    st.markdown(f"""
    <table class="perf-table">
        <thead><tr>
            <th>Model</th><th>R² (Train)</th><th>R² (Test)</th>
            <th>MAE</th><th>RMSE</th><th>MAPE</th><th>Time</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    <p style="color:#38ef7d; font-size:0.78rem; margin-top:0.3rem;">
        🟢 Green = best value in column
    </p>
    """, unsafe_allow_html=True)

    # R² bar chart
    st.markdown("### R² Score Comparison")
    r2_df = pd.DataFrame({
        'Model': list(available_models.values()),
        'R² (Train)': [mods[k]['r2_train'] for k in available_models],
        'R² (Test)':  [mods[k]['r2_test'] for k in available_models],
    })
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(
        name='R² Train', x=r2_df['Model'], y=r2_df['R² (Train)'],
        marker_color='#667eea', text=r2_df['R² (Train)'].round(4), textposition='outside'))
    fig_r2.add_trace(go.Bar(
        name='R² Test', x=r2_df['Model'], y=r2_df['R² (Test)'],
        marker_color='#38ef7d', text=r2_df['R² (Test)'].round(4), textposition='outside'))
    fig_r2.update_layout(
        barmode='group', template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.02)',
        yaxis_title='R² Score', height=400,
        legend=dict(orientation='h', y=-0.15),
        margin=dict(l=40, r=20, t=20, b=80))
    st.plotly_chart(fig_r2, key="r2_bar")

    # RMSE / MAE bar chart
    st.markdown("### RMSE & MAE Comparison")
    err_df = pd.DataFrame({
        'Model': list(available_models.values()),
        'MAE':  [mods[k]['mae'] for k in available_models],
        'RMSE': [mods[k]['rmse'] for k in available_models],
    })
    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(name='MAE', x=err_df['Model'], y=err_df['MAE'], marker_color='#fa709a'))
    fig_err.add_trace(go.Bar(name='RMSE', x=err_df['Model'], y=err_df['RMSE'], marker_color='#fee140'))
    fig_err.update_layout(
        barmode='group', template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.02)',
        yaxis_title='Error', height=400,
        legend=dict(orientation='h', y=-0.15),
        margin=dict(l=40, r=20, t=20, b=80))
    st.plotly_chart(fig_err, key="err_bar")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  Section 4 — Feature Importance
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🎯 Feature Importance / Coefficients")

    fi_pairs = [
        ('rf_import', 'RF – Import Qty', '#667eea'),
        ('rf_dep', 'RF – Dependency %', '#764ba2'),
        ('xgb_import', 'XGB – Import Qty', '#fa709a'),
        ('xgb_dep', 'XGB – Dependency %', '#fee140'),
        ('ridge_import', 'Ridge – Import Qty', '#43e97b'),
        ('ridge_dep', 'Ridge – Dependency %', '#38f9d7'),
    ]
    # Filter to available models
    fi_pairs = [(k, t, c) for k, t, c in fi_pairs if k in mods]

    fi_cols = st.columns(2)
    for idx, (key, title, color) in enumerate(fi_pairs):
        fi = mods[key]['feature_importance']
        # Sort by importance descending
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        with fi_cols[idx % 2]:
            fig_fi = go.Figure(go.Bar(
                x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
                orientation='h', marker_color=color,
                text=[f"{v:.4f}" for v in fi_sorted.values()],
                textposition='outside'))
            fig_fi.update_layout(
                title=dict(text=title, font=dict(size=14, color='#ddd')),
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.02)',
                xaxis_title='Importance', height=320,
                margin=dict(l=140, r=50, t=40, b=40),
                yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_fi, key=f"fi_{key}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  Section 5 — Future 3-Year Dependency Forecast
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🔮 Future 3-Year Dependency Forecast (2014–2016)")
    st.markdown("Select a country and food item to compare **RandomForest vs XGBoost vs Ridge** predictions side-by-side.")

    fc1, fc2 = st.columns(2)
    with fc1:
        dash_area = st.selectbox("🌍 Country / Area", [""] + areas,
                                  format_func=lambda x: "-- Select --" if x == "" else x,
                                  key="dash_area")
    with fc2:
        dash_item = st.selectbox("🍎 Food Item", [""] + items,
                                  format_func=lambda x: "-- Select --" if x == "" else x,
                                  key="dash_item")

    if dash_area and dash_item:
        try:
            # Predict with all three algorithms (iterative, using lag features)
            rf_preds    = predict_years(models['rf_import'], models['rf_dep'],
                                        dash_area, dash_item, FORECAST_3YR)
            xgb_preds   = predict_years(models['xgb_import'], models['xgb_dep'],
                                        dash_area, dash_item, FORECAST_3YR)
            ridge_preds = predict_years(models['ridge_import'], models['ridge_dep'],
                                        dash_area, dash_item, FORECAST_3YR)

            rf_imp    = [p[0] for p in rf_preds]
            rf_dep    = [p[1] for p in rf_preds]
            xgb_imp   = [p[0] for p in xgb_preds]
            xgb_dep   = [p[1] for p in xgb_preds]
            ridge_imp = [p[0] for p in ridge_preds]
            ridge_dep = [p[1] for p in ridge_preds]

            # ── Forecast cards ────────────────────────────────────────────
            st.markdown("#### Import Dependency % — RF vs XGBoost vs Ridge")
            cards = '<div class="future-grid">'
            for i, fy in enumerate(FORECAST_3YR):
                cards += f"""
                <div class="future-card rf">
                    <div class="fy-year">RF {fy}</div>
                    <div class="fy-val">{rf_dep[i]:.2f}%</div>
                </div>
                <div class="future-card xgb">
                    <div class="fy-year">XGB {fy}</div>
                    <div class="fy-val">{xgb_dep[i]:.2f}%</div>
                </div>
                <div class="future-card ridge">
                    <div class="fy-year">Ridge {fy}</div>
                    <div class="fy-val">{ridge_dep[i]:.2f}%</div>
                </div>"""
            cards += '</div>'
            st.markdown(cards, unsafe_allow_html=True)

            # ── Charts with historical + forecast ─────────────────────────
            _, _, _, hist_df = _build_lookups(dash_area, dash_item)
            if len(hist_df) > 0:
                hist_years = sorted(hist_df['Year'].tolist())[-10:]
                hist_dep_vals = [round(hist_df.loc[hist_df['Year'] == y, 'Dependency_Pct'].values[0], 2)
                                 for y in hist_years if y in hist_df['Year'].values]
                hist_imp_vals = [round(hist_df.loc[hist_df['Year'] == y, 'Import_Quantity'].values[0], 2)
                                 for y in hist_years if y in hist_df['Year'].values]

                # Dependency chart
                st.markdown("#### 📈 Dependency % — Historical + 3-Year Forecast")
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=hist_years, y=hist_dep_vals, mode='lines+markers', name='Historical',
                    line=dict(color='#11998e', width=3), marker=dict(size=6),
                    fill='tozeroy', fillcolor='rgba(17,153,142,0.1)'))
                fig_fc.add_trace(go.Scatter(
                    x=FORECAST_3YR, y=rf_dep, mode='lines+markers', name='RF Forecast',
                    line=dict(color='#667eea', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')))
                fig_fc.add_trace(go.Scatter(
                    x=FORECAST_3YR, y=xgb_dep, mode='lines+markers', name='XGB Forecast',
                    line=dict(color='#fa709a', width=3, dash='dot'),
                    marker=dict(size=8, symbol='star')))
                fig_fc.add_trace(go.Scatter(
                    x=FORECAST_3YR, y=ridge_dep, mode='lines+markers', name='Ridge Forecast',
                    line=dict(color='#43e97b', width=3, dash='dashdot'),
                    marker=dict(size=8, symbol='cross')))
                fig_fc.update_layout(
                    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.02)', xaxis_title='Year',
                    yaxis_title='Dependency %',
                    legend=dict(orientation='h', y=-0.15),
                    margin=dict(l=40, r=20, t=20, b=60), height=400)
                st.plotly_chart(fig_fc, key="dash_dep_chart")

                # Import chart
                st.markdown("#### 📈 Import Quantity — Historical + 3-Year Forecast")
                fig_fi2 = go.Figure()
                fig_fi2.add_trace(go.Scatter(
                    x=hist_years, y=hist_imp_vals, mode='lines+markers', name='Historical',
                    line=dict(color='#11998e', width=3), marker=dict(size=6),
                    fill='tozeroy', fillcolor='rgba(17,153,142,0.1)'))
                fig_fi2.add_trace(go.Scatter(
                    x=FORECAST_3YR, y=rf_imp, mode='lines+markers', name='RF Forecast',
                    line=dict(color='#667eea', width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')))
                fig_fi2.add_trace(go.Scatter(
                    x=FORECAST_3YR, y=xgb_imp, mode='lines+markers', name='XGB Forecast',
                    line=dict(color='#fa709a', width=3, dash='dot'),
                    marker=dict(size=8, symbol='star')))
                fig_fi2.add_trace(go.Scatter(
                    x=FORECAST_3YR, y=ridge_imp, mode='lines+markers', name='Ridge Forecast',
                    line=dict(color='#43e97b', width=3, dash='dashdot'),
                    marker=dict(size=8, symbol='cross')))
                fig_fi2.update_layout(
                    template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.02)', xaxis_title='Year',
                    yaxis_title='Import Quantity (tonnes)',
                    legend=dict(orientation='h', y=-0.15),
                    margin=dict(l=40, r=20, t=20, b=60), height=400)
                st.plotly_chart(fig_fi2, key="dash_imp_chart")

                # Forecast table
                st.markdown("#### 📋 Forecast Data Table")
                st.dataframe(pd.DataFrame({
                    'Year': FORECAST_3YR,
                    'RF Import Qty': rf_imp,
                    'XGB Import Qty': xgb_imp,
                    'Ridge Import Qty': ridge_imp,
                    'RF Dependency %': rf_dep,
                    'XGB Dependency %': xgb_dep,
                    'Ridge Dependency %': ridge_dep,
                }), hide_index=True, use_container_width=True)
            else:
                st.info("ℹ️ No historical data for this combination.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  Section 6 — Data Distribution
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Data Distribution")
    d1, d2 = st.columns(2)
    with d1:
        fig_h1 = go.Figure(go.Histogram(
            x=df['Import_Quantity'].clip(upper=df['Import_Quantity'].quantile(0.95)),
            nbinsx=50, marker_color='#667eea', opacity=0.8))
        fig_h1.update_layout(
            title=dict(text='Import Quantity (95th pctl)', font=dict(size=13, color='#ddd')),
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.02)',
            xaxis_title='Import Quantity', yaxis_title='Count',
            height=320, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_h1, key="dist_imp")
    with d2:
        fig_h2 = go.Figure(go.Histogram(
            x=df['Dependency_Pct'], nbinsx=50,
            marker_color='#38ef7d', opacity=0.8))
        fig_h2.update_layout(
            title=dict(text='Dependency % Distribution', font=dict(size=13, color='#ddd')),
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.02)',
            xaxis_title='Dependency %', yaxis_title='Count',
            height=320, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_h2, key="dist_dep")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with Streamlit · Models: RandomForest, XGBoost & Ridge Regression · Data: FAO (1961–2013)
</div>
""", unsafe_allow_html=True)
