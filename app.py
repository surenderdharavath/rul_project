import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Turbofan Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
    }
    .status-healthy { color: #27ae60; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }
    .status-critical { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("⚙️ Turbofan Fleet Predictive Maintenance")
st.markdown("""
**Deep Learning-based Remaining Useful Life (RUL) Prognostics System**  
*Benchmarked on NASA C-MAPSS Turbofan Engine Fleet (FD001)*
""")

# Sidebar Navigation
st.sidebar.header("Industrial Control Panel")
menu = st.sidebar.radio(
    "Navigation",
    ["Fleet Overview & RUL Countdown", "Engine Diagnostic Inspector", "Model Performance & Benchmarks", "Overfitting Diagnosis (Before & After)"]
)

@st.cache_data
def get_prediction_data():
    pred_path = os.path.join(RESULTS_DIR, "engine_predictions.csv")
    metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
    
    if os.path.exists(pred_path) and os.path.exists(metrics_path):
        df = pd.read_csv(pred_path)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return df, metrics
    return None, None

@st.cache_data
def get_raw_test_data():
    test_path = os.path.join(DATA_DIR, "test_FD001.txt")
    if os.path.exists(test_path):
        from src.data_loader import ALL_COLS
        df = pd.read_csv(test_path, sep=r"\s+", header=None).iloc[:, :26]
        df.columns = ALL_COLS
        return df
    return None

df_preds, metrics = get_prediction_data()
raw_test_df = get_raw_test_data()

if df_preds is None:
    st.warning("⚠️ Predictions not found. Please run the training and evaluation pipeline first using:")
    st.code("python src/train.py\npython src/evaluate.py", language="bash")
    st.stop()

# Helper status function
def get_health_status(rul):
    if rul < 25:
        return "CRITICAL (Maintenance Immediate)", "#e74c3c", "🚨"
    elif rul < 55:
        return "WARNING (Service Scheduled)", "#f39c12", "⚠️"
    else:
        return "OPTIMAL (Normal Operation)", "#27ae60", "✅"

# PAGE 1: Fleet Overview
if menu == "Fleet Overview & RUL Countdown":
    st.subheader("🌐 Fleet Health & Prognostics Overview (100 Turbofan Engines)")

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    total_engines = len(df_preds)
    critical_count = (df_preds["pred_rul"] < 25).sum()
    warning_count = ((df_preds["pred_rul"] >= 25) & (df_preds["pred_rul"] < 55)).sum()
    healthy_count = (df_preds["pred_rul"] >= 55).sum()

    c1.metric("Total Turbofans", f"{total_engines} Units")
    c2.metric("Critical State (<25 cycles)", f"{critical_count} Units", delta="-High Risk" if critical_count > 0 else "0", delta_color="inverse")
    c3.metric("Warning State (25-55 cycles)", f"{warning_count} Units")
    c4.metric("Healthy State (>55 cycles)", f"{healthy_count} Units")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.write("#### Fleet RUL Degradation Heatmap")
        st.markdown("Engines approaching end-of-life require targeted maintenance scheduling.")
        sorted_fleet = df_preds.sort_values(by="pred_rul").reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        colors = ["#e74c3c" if r < 25 else "#f39c12" if r < 55 else "#27ae60" for r in sorted_fleet["pred_rul"]]
        ax.bar(range(len(sorted_fleet)), sorted_fleet["pred_rul"], color=colors, width=0.8)
        ax.axhline(25, color="#e74c3c", linestyle="--", label="Critical Threshold (25 cycles)")
        ax.axhline(55, color="#f39c12", linestyle=":", label="Warning Threshold (55 cycles)")
        ax.set_xlabel("Engines (Ranked by Urgency)")
        ax.set_ylabel("Predicted RUL (Cycles)")
        ax.set_title("Turbofan Remaining Useful Life Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_right:
        st.write("#### Top 5 Most Urgent Maintenance Actions")
        urgent_engines = df_preds.sort_values(by="pred_rul").head(5)
        for _, row in urgent_engines.iterrows():
            status_text, color, icon = get_health_status(row["pred_rul"])
            st.markdown(f"""
            <div style="border-left: 4px solid {color}; padding: 8px 12px; margin-bottom: 8px; background: #fafafa; border-radius: 4px;">
                <strong>{icon} Engine Unit #{int(row['unit_id'])}</strong><br/>
                <span style="font-size: 0.9em;">Predicted RUL: <b>{row['pred_rul']:.1f} cycles</b> (Ground Truth: {row['true_rul']:.0f} cycles)</span><br/>
                <span style="color: {color}; font-size: 0.85em;">Status: {status_text}</span>
            </div>
            """, unsafe_allow_html=True)

# PAGE 2: Engine Diagnostic Inspector
elif menu == "Engine Diagnostic Inspector":
    st.subheader("🔍 Engine Telemetry & Diagnostic Inspector")
    
    selected_unit = st.sidebar.selectbox("Select Engine Unit ID", sorted(df_preds["unit_id"].unique()))
    engine_row = df_preds[df_preds["unit_id"] == selected_unit].iloc[0]
    pred_rul = engine_row["pred_rul"]
    true_rul = engine_row["true_rul"]
    error = engine_row["error"]
    status_text, color, icon = get_health_status(pred_rul)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selected Engine", f"Unit #{int(selected_unit)}")
    k2.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
    k3.metric("True RUL", f"{true_rul:.0f} cycles")
    k4.metric("Prognostic Error", f"{error:+.1f} cycles")

    st.markdown(f"""
    <div style="background-color: {color}15; border: 1px solid {color}; border-radius: 6px; padding: 12px; margin: 10px 0;">
        <h4 style="margin: 0; color: {color};">{icon} Condition Assessment: {status_text}</h4>
    </div>
    """, unsafe_allow_html=True)

    if raw_test_df is not None:
        unit_history = raw_test_df[raw_test_df["unit_id"] == selected_unit].copy()
        st.write(f"#### Multi-Sensor Telemetry Run-to-Date (Cycles 1 to {len(unit_history)})")
        
        sensor_display_options = {
            "s_2": "Sensor 2: Total Temp at LPC Outlet (°R)",
            "s_3": "Sensor 3: Total Temp at HPC Outlet (°R)",
            "s_4": "Sensor 4: Total Temp at LPT Outlet (°R)",
            "s_7": "Sensor 7: Total Pressure at HPC Outlet (psia)",
            "s_11": "Sensor 11: Static Pressure at HPC Outlet (psia)",
            "s_12": "Sensor 12: Ratio of Fuel Flow to Ps30 (pps/psia)"
        }
        selected_sensors = st.multiselect(
            "Select Sensors to Graph",
            list(sensor_display_options.keys()),
            default=["s_2", "s_3", "s_4", "s_11"],
            format_func=lambda x: sensor_display_options[x]
        )

        if selected_sensors:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            for s in selected_sensors:
                ax.plot(unit_history["cycle"], unit_history[s], label=sensor_display_options[s], linewidth=1.8)
            ax.set_xlabel("Operational Cycle")
            ax.set_ylabel("Sensor Reading")
            ax.set_title(f"Degradation Trajectory for Turbofan #{int(selected_unit)}")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# PAGE 3: Model Performance & Benchmarks
elif menu == "Model Performance & Benchmarks":
    st.subheader("📊 NASA C-MAPSS FD001 Model Performance Benchmark")
    
    b1, b2, b3 = st.columns(3)
    b1.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']} cycles")
    b2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']} cycles")
    b3.metric("NASA Penalty Score", f"{metrics['NASA_Score']}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Ground Truth vs LSTM Predicted RUL")
        p1 = os.path.join(RESULTS_DIR, "actual_vs_predicted_rul.png")
        if os.path.exists(p1):
            st.image(p1, caption="100 Test Engines: Ground Truth vs Predicted RUL", use_container_width=True)
    with col2:
        st.write("#### Prognostic Error Residual Distribution")
        p2 = os.path.join(RESULTS_DIR, "rul_error_distribution.png")
        if os.path.exists(p2):
            st.image(p2, caption="Error Residual Distribution (Predicted - True)", use_container_width=True)

# PAGE 4: Overfitting Diagnosis
elif menu == "Overfitting Diagnosis (Before & After)":
    st.subheader("🔬 Overfitting Diagnosis & Engineering Fix")
    st.markdown("""
    As noted in the project methodology, unregularized LSTMs trained on raw linear RUL suffer from severe overfitting 
    because sensors do not display degradation during early healthy cycles.
    
    **The Three-Part Fix Applied:**
    1. **Piecewise Linear RUL Target Clipping ($RUL_{max} = 125$):** Prevents the network from attempting to learn non-existent early wear patterns.
    2. **Network Downsizing & Dropout Regularization:** Compact 2-layer LSTM (64-32 units) with dropout (0.2).
    3. **Early Stopping with Learning Rate Scheduling:** Prevents validation divergence.
    """)
    
    comp_plot = os.path.join(RESULTS_DIR, "overfitting_comparison.png")
    if os.path.exists(comp_plot):
        st.image(comp_plot, caption="Before & After Training Curves Proving Overfitting Resolution", use_container_width=True)
    else:
        st.info("Run `python src/overfitting_experiment.py` to generate the side-by-side comparison curves.")
