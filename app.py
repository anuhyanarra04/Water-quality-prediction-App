import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import time
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import joblib

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# ==================================================
# PROFESSIONAL WATER QUALITY PREDICTION DASHBOARD
# Major Project – Enhanced & Feature-Rich Version
# ==================================================

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Water Quality Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Global Styling
# ----------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        text-align: center;
    }
    .status-safe { color:#0f7b42; font-weight:700; }
    .status-good { color:#8a6d00; font-weight:700; }
    .status-unsafe { color:#b42318; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# Custom Attention Layer (Required for Model Loading)
# ==================================================
class TemporalAttention(Layer):
    def build(self, input_shape):
        self.w = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.w, axes=[2, 0]) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * x, axis=1)
        return context

# ==================================================
# Load Model & Scaler (Cached)
# ==================================================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model("model/trained_wqi_model.h5", custom_objects={"TemporalAttention": TemporalAttention})
        scaler = joblib.load("model/scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, model_error = load_model_and_scaler()
model_loaded = model is not None

# ==================================================
# Dataset Handling
# ==================================================
@st.cache_data
def load_default_dataset():
    try:
        return pd.read_csv("data/testing.csv")
    except Exception:
        return None


def get_active_dataset():
    if "uploaded_df" in st.session_state:
        return st.session_state.uploaded_df.copy()
    return load_default_dataset()

# ==================================================
# Prediction Utilities
# ==================================================
FEATURES = ['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)']


def predict_wqi(df):
    scaled = scaler.transform(df[FEATURES])
    scaled = scaled.reshape(1, scaled.shape[1], 1)
    pred = model.predict(scaled, verbose=0)[0][0]
    return float(pred)


def classify_wqi(wqi):
    if wqi < 0.3:
        return "Unsafe"
    elif wqi < 0.7:
        return "Good"
    return "Safe"

# ==================================================
# Professional PDF Report
# ==================================================

def generate_pdf(input_df, wqi, status):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Water Quality Prediction Report</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now()}", styles['Normal']))
    elements.append(Spacer(1, 12))

    table_data = [["Parameter", "Value"]] + [[c, str(input_df[c].values[0])] for c in FEATURES]
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph(f"<b>Predicted WQI:</b> {wqi:.4f}", styles['Normal']))
    elements.append(Paragraph(f"<b>Status:</b> {status}", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================================================
# MAIN APPLICATION
# ==================================================

def main():
    st.title("💧 Water Quality Prediction & Monitoring System")
    st.caption("AI‑based CNN‑BiLSTM‑Attention model for real‑time and batch water quality analysis")

    # Sidebar
    st.sidebar.header("System Panel")
    if model_loaded:
        st.sidebar.success("Model Loaded Successfully")
    else:
        st.sidebar.error(model_error)

    tabs = st.tabs([
        "📂 Data Management",
        "📊 Advanced Analytics",
        "🔮 Prediction",
        "⏱️ Live Monitoring",
        "📄 Reports"
    ])

    # ---------------- DATA MANAGEMENT ----------------
    with tabs[0]:
        st.subheader("Dataset Management")
        uploaded = st.file_uploader("Upload Water Quality CSV", type="csv")
        if uploaded:
            st.session_state.uploaded_df = pd.read_csv(uploaded)
            st.success("Dataset uploaded successfully")

        ds = get_active_dataset()
        if ds is not None:
            st.dataframe(ds.head(100), use_container_width=True)

    # ---------------- ADVANCED ANALYTICS ----------------
    with tabs[1]:
        st.subheader("Exploratory Data Analysis")
        ds = get_active_dataset()
        if ds is not None:
            numeric_cols = ds.select_dtypes(include=np.number).columns
            fig = px.imshow(ds[numeric_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- PREDICTION ----------------
    with tabs[2]:
        st.subheader("Single Sample Prediction")
        cols = st.columns(3)
        user_values = {}
        for i, f in enumerate(FEATURES):
            with cols[i % 3]:
                user_values[f] = st.number_input(f, value=0.0)

        input_df = pd.DataFrame([user_values])

        if st.button("Run Prediction") and model_loaded:
            wqi = predict_wqi(input_df)
            status = classify_wqi(wqi)

            st.metric("Predicted WQI", f"{wqi:.4f}")
            st.markdown(f"### Status: **{status}**")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wqi,
                gauge={'axis': {'range': [0, 1]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)

    # ---------------- LIVE MONITORING ----------------
    with tabs[3]:
        st.subheader("Real‑Time Monitoring")
        ds = get_active_dataset()
        if ds is not None and model_loaded:
            st.info("Simulated real‑time prediction from dataset")
            for i, row in ds.iterrows():
                wqi = predict_wqi(row.to_frame().T)
                status = classify_wqi(wqi)
                st.write(f"Record {i+1} → WQI: {wqi:.3f} ({status})")
                time.sleep(0.5)

    # ---------------- REPORTS ----------------
    with tabs[4]:
        st.subheader("Generate Professional Report")
        if model_loaded:
            sample_df = pd.DataFrame([{f: 0 for f in FEATURES}])
            wqi = predict_wqi(sample_df)
            status = classify_wqi(wqi)
            pdf = generate_pdf(sample_df, wqi, status)
            st.download_button("Download Sample Report", pdf, file_name="WQI_Report.pdf")


if __name__ == "__main__":
    main()
