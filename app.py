"""
AquaAdapt — Streamlit Frontend
================================
AI-Based Adaptive Irrigation Using Soil Response Feedback

A professional dashboard that demonstrates the complete
Stage 1 → Test Dose → Stage 2 adaptive irrigation pipeline.
"""

import os
import sys
import json
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.stage1_model import predict_stage1, get_metadata as get_stage1_meta
from src.stage2_model import predict_stage2, get_metadata as get_stage2_meta
from src.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaAdapt — Adaptive Irrigation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a premium, polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Hide Streamlit defaults ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---------- Main background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0e0 !important;
}

/* ---------- Glassmorphism Card ---------- */
.glass-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,198,255,0.10);
}

/* ---------- Metric Card ---------- */
.metric-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: scale(1.03);
}
.metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #8ec5fc;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-unit {
    font-size: 0.85rem;
    color: #aaa;
    margin-left: 4px;
}

/* ---------- Stage Badge ---------- */
.stage-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.stage-1-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.stage-2-badge {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

/* ---------- Flow Arrow ---------- */
.flow-arrow {
    text-align: center;
    font-size: 1.6rem;
    color: rgba(255,255,255,0.35);
    margin: 8px 0;
    animation: pulse-arrow 2s infinite;
}
@keyframes pulse-arrow {
    0%, 100% { opacity: 0.35; }
    50% { opacity: 0.8; }
}

/* ---------- Section title ---------- */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #8ec5fc;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
}

/* ---------- Hero header ---------- */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #00c6ff 50%, #72f088 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.55);
    font-weight: 400;
    margin-bottom: 28px;
}

/* ---------- Result highlight ---------- */
.result-highlight {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%);
    border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
}
.result-highlight-adaptive {
    background: linear-gradient(135deg, rgba(114,240,136,0.12) 0%, rgba(0,198,255,0.12) 100%);
    border-left: 4px solid #72f088;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 8px 0;
}
.result-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #8ec5fc;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.result-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}

/* ---------- Pipeline flow step ---------- */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    margin-bottom: 6px;
    transition: background 0.2s ease;
}
.pipeline-step:hover {
    background: rgba(255,255,255,0.08);
}
.pipeline-icon {
    font-size: 1.3rem;
    min-width: 32px;
    text-align: center;
}
.pipeline-text {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.8);
    font-weight: 400;
}

/* ---------- Divider ---------- */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    margin: 24px 0;
}

/* ---------- Button overrides ---------- */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;
}

/* ---------- Number input ---------- */
[data-testid="stNumberInput"] label {
    color: #c0c0c0 !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] label {
    color: #c0c0c0 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def render_metric(label: str, value: str, unit: str = ""):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_flow_arrow():
    """Render an animated downward arrow."""
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR — System Flow & Model Info
# ===========================================================================
with st.sidebar:
    st.markdown("## 🌱 AquaAdapt")
    st.markdown("##### System Workflow")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    steps = [
        ("🌱", "Initial Soil Conditions"),
        ("🤖", "Stage 1 — CatBoost Prediction"),
        ("💧", "Initial Water Requirement"),
        ("🧪", "Small Test Dose Applied"),
        ("📊", "Observe Soil Response"),
        ("🧠", "Stage 2 — Adaptive Prediction"),
        ("💧", "Final Water Requirement"),
    ]
    for icon, text in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <span class="pipeline-icon">{icon}</span>
            <span class="pipeline-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ---- Model info ----
    st.markdown("##### 📋 Model Information")

    try:
        s1_meta = get_stage1_meta()
        s1_eval = s1_meta.get("evaluation", {})
        s1_info = s1_meta.get("dataset_info", {})

        st.markdown(f"""
        <div class="glass-card" style="padding:16px;">
            <div class="section-title">Stage 1 — Initial Model</div>
            <p style="color:#aaa; font-size:0.85rem; margin:4px 0;">
                <b>Algorithm:</b> CatBoost Regression<br>
                <b>Target:</b> Water Required (L)<br>
                <b>Dataset:</b> {s1_info.get('crop', 'Tomato')} — {s1_info.get('soil', 'Sandy clay loam')}<br>
                <b>Records:</b> {s1_info.get('total_records', 3000)}<br>
                <b>Train / Test:</b> {s1_info.get('train_records', '—')} / {s1_info.get('test_records', '—')}
            </p>
            <hr style="border-color: rgba(255,255,255,0.08); margin:10px 0;">
            <p style="color:#8ec5fc; font-size:0.78rem; font-weight:600; letter-spacing:0.8px; text-transform:uppercase;">
                Evaluation Metrics
            </p>
            <p style="color:#ccc; font-size:0.85rem; margin:4px 0;">
                MAE&nbsp;&nbsp;: <b>{s1_eval.get('mae', '—')}</b><br>
                MSE&nbsp;&nbsp;: <b>{s1_eval.get('mse', '—')}</b><br>
                RMSE : <b>{s1_eval.get('rmse', '—')}</b><br>
                R²&nbsp;&nbsp;&nbsp;: <b>{s1_eval.get('r2', '—')}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Stage 1 model not trained yet. Run `python 'Stage 1 Prediction Model/train.py'` first.")

    st.markdown("""
    <div class="glass-card" style="padding:16px;">
        <div class="section-title">Stage 2 — Adaptive Model</div>
        <p style="color:#aaa; font-size:0.85rem; margin:4px 0;">
            <b>Algorithm:</b> CatBoost Regression<br>
            <b>Purpose:</b> Adaptive correction after test dose<br>
            <b>Extra inputs:</b> Post-test-dose soil response
        </p>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# MAIN CONTENT
# ===========================================================================

# ---- Hero header ----
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <div class="hero-title">AquaAdapt</div>
    <div class="hero-subtitle">AI-Based Adaptive Irrigation Using Soil Response Feedback</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ===========================================================================
# SECTION 1 — INITIAL FIELD CONDITIONS
# ===========================================================================
st.markdown("""
<div class="section-title">🌱 &nbsp;Step 1 — Initial Field Conditions</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        soil_type = st.selectbox(
            "Soil Type", ["Sandy clay loam"],
            help="Currently trained on Sandy clay loam"
        )
    with col2:
        crop_type = st.selectbox(
            "Crop Type", ["Tomato"],
            help="Currently trained on Tomato"
        )
    with col3:
        crop_stage = st.selectbox(
            "Crop Stage",
            ["Seedling", "Vegetative", "Flowering", "Maturity"],
            index=2,
            help="Growth stage of the crop"
        )

    st.markdown("")
    col4, col5, col6 = st.columns(3)

    with col4:
        soil_moisture_before = st.number_input(
            "Soil Moisture (%)", min_value=0.0, max_value=100.0,
            value=28.0, step=0.5, format="%.1f"
        )
    with col5:
        temperature_before = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=60.0,
            value=32.0, step=0.5, format="%.1f"
        )
    with col6:
        humidity_before = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0,
            value=55.0, step=0.5, format="%.1f"
        )

    col7, col8, col9 = st.columns(3)

    with col7:
        rainfall_forecast_before = st.number_input(
            "Rainfall Forecast (mm)", min_value=0.0, max_value=500.0,
            value=2.0, step=0.5, format="%.1f"
        )
    with col8:
        wind_speed_before = st.number_input(
            "Wind Speed (km/h)", min_value=0.0, max_value=200.0,
            value=8.0, step=0.5, format="%.1f"
        )
    with col9:
        solar_radiation_before = st.number_input(
            "Solar Radiation (W/m²)", min_value=0.0, max_value=2000.0,
            value=650.0, step=10.0, format="%.0f"
        )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ===========================================================================
# SECTION 2 — POST-TEST-DOSE OBSERVATIONS (for Stage 2)
# ===========================================================================
st.markdown("""
<div class="section-title">🧪 &nbsp;Step 2 — Soil Response After Test Dose</div>
<p style="color:rgba(255,255,255,0.45); font-size:0.88rem; margin-top:-8px; margin-bottom:16px;">
    Enter the observed field readings <strong>after</strong> the test dose was applied.
</p>
""", unsafe_allow_html=True)

with st.container():
    col_td, col_mg = st.columns(2)

    with col_td:
        test_dose_applied = st.number_input(
            "Test Dose Applied (L)", min_value=0.0, max_value=100.0,
            value=25.0, step=1.0, format="%.1f",
            help="Small controlled water dose applied to the field"
        )
    with col_mg:
        moisture_gain = st.number_input(
            "Moisture Gain (%)", min_value=-20.0, max_value=50.0,
            value=3.5, step=0.1, format="%.2f",
            help="Change in soil moisture (after − before)"
        )

    st.markdown("")
    col10, col11, col12 = st.columns(3)

    with col10:
        soil_moisture_after = st.number_input(
            "Soil Moisture After (%)", min_value=0.0, max_value=100.0,
            value=31.5, step=0.5, format="%.1f"
        )
    with col11:
        temperature_after = st.number_input(
            "Temperature After (°C)", min_value=-10.0, max_value=60.0,
            value=31.0, step=0.5, format="%.1f"
        )
    with col12:
        humidity_after = st.number_input(
            "Humidity After (%)", min_value=0.0, max_value=100.0,
            value=57.0, step=0.5, format="%.1f"
        )

    col13, col14, col15 = st.columns(3)

    with col13:
        rainfall_forecast_after = st.number_input(
            "Rainfall Forecast After (mm)", min_value=0.0, max_value=500.0,
            value=2.5, step=0.5, format="%.1f"
        )
    with col14:
        wind_speed_after = st.number_input(
            "Wind Speed After (km/h)", min_value=0.0, max_value=200.0,
            value=7.5, step=0.5, format="%.1f"
        )
    with col15:
        solar_radiation_after = st.number_input(
            "Solar Radiation After (W/m²)", min_value=0.0, max_value=2000.0,
            value=640.0, step=10.0, format="%.0f"
        )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ===========================================================================
# RUN BUTTON
# ===========================================================================
col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
with col_btn_c:
    run_clicked = st.button("🚀  Run AquaAdapt", use_container_width=True)

# ===========================================================================
# RESULTS
# ===========================================================================
if run_clicked:
    # ---- Validation ----
    errors = []
    if soil_moisture_before < 0:
        errors.append("Soil Moisture cannot be negative.")
    if rainfall_forecast_before < 0:
        errors.append("Rainfall Forecast cannot be negative.")
    if test_dose_applied <= 0:
        errors.append("Test Dose Applied must be greater than zero.")

    if errors:
        for err in errors:
            st.error(f"❌ {err}")
    else:
        try:
            with st.spinner("Running AquaAdapt pipeline..."):
                result = run_pipeline(
                    soil_type=soil_type,
                    crop_type=crop_type,
                    crop_stage=crop_stage,
                    soil_moisture_before=soil_moisture_before,
                    temperature_before=temperature_before,
                    humidity_before=humidity_before,
                    rainfall_forecast_before=rainfall_forecast_before,
                    wind_speed_before=wind_speed_before,
                    solar_radiation_before=solar_radiation_before,
                    test_dose_applied=test_dose_applied,
                    soil_moisture_after=soil_moisture_after,
                    temperature_after=temperature_after,
                    humidity_after=humidity_after,
                    rainfall_forecast_after=rainfall_forecast_after,
                    wind_speed_after=wind_speed_after,
                    solar_radiation_after=solar_radiation_after,
                    moisture_gain=moisture_gain,
                )

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ---- Results header ----
            st.markdown("""
            <div style="text-align:center; margin-bottom:20px;">
                <div class="section-title" style="font-size:1.4rem;">
                    🎯 &nbsp;AquaAdapt Prediction Results
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ---- Step-by-step result cards ----

            # Stage 1 result
            st.markdown(f"""
            <div class="glass-card">
                <span class="stage-badge stage-1-badge">Stage 1 — Initial Prediction</span>
                <div class="result-highlight" style="margin-top:14px;">
                    <div class="result-label">Initial Water Requirement</div>
                    <div class="result-value">💧 {result['initial_water_required']:.2f} <span style="font-size:1rem;color:#aaa;">Litres</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            render_flow_arrow()

            # Test dose
            st.markdown(f"""
            <div class="glass-card">
                <span class="stage-badge" style="background:linear-gradient(135deg,#f6d365 0%,#fda085 100%);color:#333;">
                    🧪 Test Dose
                </span>
                <div style="margin-top:14px; display:flex; gap:20px; flex-wrap:wrap;">
                    <div>
                        <div class="metric-label">Dose Applied</div>
                        <div style="font-size:1.4rem;font-weight:700;color:white;">{result['test_dose_applied']:.1f} L</div>
                    </div>
                    <div>
                        <div class="metric-label">Moisture Gain</div>
                        <div style="font-size:1.4rem;font-weight:700;color:white;">{result['moisture_gain']:.2f} %</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            render_flow_arrow()

            # Soil Response Summary
            st.markdown(f"""
            <div class="glass-card">
                <span class="stage-badge" style="background:linear-gradient(135deg,#a18cd1 0%,#fbc2eb 100%);color:#333;">
                    📊 Soil Response
                </span>
                <div style="margin-top:14px; display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:10px;">
                    <div class="metric-card">
                        <div class="metric-label">Moisture After</div>
                        <div class="metric-value" style="font-size:1.2rem;">{soil_moisture_after:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Temp After</div>
                        <div class="metric-value" style="font-size:1.2rem;">{temperature_after:.1f}°C</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Humidity After</div>
                        <div class="metric-value" style="font-size:1.2rem;">{humidity_after:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Rainfall After</div>
                        <div class="metric-value" style="font-size:1.2rem;">{rainfall_forecast_after:.1f}mm</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Wind After</div>
                        <div class="metric-value" style="font-size:1.2rem;">{wind_speed_after:.1f}km/h</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Solar After</div>
                        <div class="metric-value" style="font-size:1.2rem;">{solar_radiation_after:.0f}W/m²</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            render_flow_arrow()

            # Stage 2 result
            st.markdown(f"""
            <div class="glass-card">
                <span class="stage-badge stage-2-badge">Stage 2 — Adaptive Prediction</span>
                <div class="result-highlight-adaptive" style="margin-top:14px;">
                    <div class="result-label">Final Adaptive Water Requirement</div>
                    <div class="result-value">💧 {result['final_water_required']:.2f} <span style="font-size:1rem;color:#aaa;">Litres</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ---- Summary metrics ----
            st.markdown("""
            <div class="section-title" style="text-align:center;">📊 &nbsp;Summary</div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric("Initial (Stage 1)", f"{result['initial_water_required']:.2f}", "L")
            with m2:
                render_metric("Final (Stage 2)", f"{result['final_water_required']:.2f}", "L")
            with m3:
                adj_sign = "−" if result['adjustment_litres'] >= 0 else "+"
                render_metric("Adjustment", f"{adj_sign} {abs(result['adjustment_litres']):.2f}", "L")
            with m4:
                pct_sign = "−" if result['adjustment_percent'] >= 0 else "+"
                render_metric("Adjustment %", f"{pct_sign} {abs(result['adjustment_percent']):.1f}", "%")

        except FileNotFoundError as e:
            st.error(f"❌ Model not found: {e}")
            st.info("💡 Please train the Stage 1 model first:\n```\npython 'Stage 1 Prediction Model/train.py'\n```")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)


# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:16px 0; color:rgba(255,255,255,0.25); font-size:0.78rem;">
    AquaAdapt &nbsp;•&nbsp; AI-Based Adaptive Irrigation &nbsp;•&nbsp; CatBoost Regression
</div>
""", unsafe_allow_html=True)
