import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import time
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ----------------------
# Improved Streamlit App
# ----------------------

# Page config + small theme tweaks
st.set_page_config(page_title="Water Quality Prediction", layout="wide")
st.markdown(
    """
    <style>
    .status-badge { padding: 6px 10px; border-radius: 8px; font-weight: 600; display:inline-block; }
    .status-safe { background:#e7f7ee; color:#0f7b42; border:1px solid #bfead3; }
    .status-good { background:#fff8e6; color:#8a6d00; border:1px solid #ffe6a1; }
    .status-unsafe { background:#fde8e8; color:#b42318; border:1px solid #f3b4b4; }
    .small-muted { color: #6b7280; font-size:12px }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Custom Attention Layer
# ----------------------
class TemporalAttention(Layer):
    def build(self, input_shape):
        self.w = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        score = tf.nn.tanh(tf.tensordot(x, self.w, axes=[2, 0]) + self.b)
        alpha = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(alpha * x, axis=1)
        return context

# ----------------------
# Load model & scaler
# ----------------------
@st.cache_resource
def load_model_and_scaler(model_path: str = "model/trained_wqi_model.h5", scaler_path: str = "model/scaler.pkl"):
    """Load model and scaler once and cache the resource. Returns (model, scaler, error_message)"""
    try:
        _model = load_model(model_path, custom_objects={"TemporalAttention": TemporalAttention})
        _scaler = joblib.load(scaler_path)
        return _model, _scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_model_and_scaler()
model_loaded = model is not None and scaler is not None

# ----------------------
# Data loading helpers
# ----------------------
@st.cache_data
def load_default_dataset(path: str = "data/testing.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_active_dataset():
    if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
        return st.session_state.uploaded_df.copy()
    return load_default_dataset()

# ----------------------
# Input controls
# ----------------------
DEFAULTS = {
    'NITRATE(PPM)': 23.0,
    'PH': 7.0,
    'AMMONIA(mg/l)': 1.0,
    'TEMP': 30.0,
    'DO': 8.0,
    'TURBIDITY': 50.0,
    'MANGANESE(mg/l)': 1.0,
}


def user_input_widget(preset: str = None) -> pd.DataFrame:
    st.write("### Enter water quality parameters")
    col1, col2, col3 = st.columns(3)

    def slider_or_default(label, min_val, max_val, step, default):
        return st.slider(label, min_val, max_val, default, step)

    with col1:
        nitrate = slider_or_default("Nitrate (PPM)", 0.0, 140.0, 0.1, DEFAULTS['NITRATE(PPM)'])
        ph = slider_or_default("pH", 0.0, 10.0, 0.1, DEFAULTS['PH'])
        ammonia = slider_or_default("Ammonia (mg/L)", 0.0, 2.0, 0.1, DEFAULTS['AMMONIA(mg/l)'])
    with col2:
        temp = slider_or_default("Temperature (°C)", 14.0, 42.0, 0.1, DEFAULTS['TEMP'])
        do = slider_or_default("Dissolved Oxygen (mg/L)", 6.5, 12.0, 0.1, DEFAULTS['DO'])
    with col3:
        turbidity = slider_or_default("Turbidity", 0.0, 400.0, 0.1, DEFAULTS['TURBIDITY'])
        manganese = slider_or_default("Manganese (mg/L)", 0.0, 10.0, 0.1, DEFAULTS['MANGANESE(mg/l)'])

    df = pd.DataFrame([[nitrate, ph, ammonia, temp, do, turbidity, manganese]],
                      columns=['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)'])
    return df

# ----------------------
# Prediction helpers
# ----------------------

def predict_wqi(input_data: pd.DataFrame) -> float:
    # scaler expects 2D array (n_samples, n_features)
    arr = input_data.values.astype(float)
    arr_scaled = scaler.transform(arr)  # shape (1, features)
    # model expects (batch, timesteps, features) = (1, features, 1)
    arr_scaled = arr_scaled.reshape(1, arr_scaled.shape[1], 1)
    pred = model.predict(arr_scaled, verbose=0)
    return float(pred.ravel()[0])


def categorize_wqi(predicted_wqi: float):
    if predicted_wqi < 0.3:
        return "Unsafe", "WQI < 0.3"
    elif predicted_wqi < 0.7:
        return "Good", "0.3 ≤ WQI < 0.7"
    else:
        return "Safe", "WQI ≥ 0.7"


def status_badge_html(status: str) -> str:
    if status == "Safe":
        return '<span class="status-badge status-safe">🟢 Safe</span>'
    if status == "Good":
        return '<span class="status-badge status-good">🟡 Good</span>'
    return '<span class="status-badge status-unsafe">🔴 Unsafe</span>'

# ----------------------
# PDF generation
# ----------------------

def generate_pdf_report(input_data, predicted_wqi, status, title="Water Quality Report"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2.0, height - 50, title)

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Model: CNN-BiLSTM-Attention")
    c.drawString(50, height - 95, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 120, "Input Parameters:")
    c.setFont("Helvetica", 10)
    y = height - 140
    for column, value in zip(input_data.columns, input_data.values[0]):
        c.drawString(70, y, f"{column}: {value}")
        y -= 14

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Prediction Result:")
    c.setFont("Helvetica", 10)
    c.drawString(70, y - 28, f"Predicted WQI: {predicted_wqi:.4f}")
    c.drawString(70, y - 46, f"Water Quality Status: {status}")

    # Simple suggestions when unsafe
    if status == "Unsafe":
        y -= 70
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Possible issues and suggestions:")
        y -= 18
        suggestions = [
            "Reduce agricultural runoff and improve wastewater treatment.",
            "Adjust pH using appropriate treatment (acid/alkali).",
            "Increase aeration to improve dissolved oxygen.",
            "Install better filtration to reduce turbidity and remove heavy metals.",
        ]
        c.setFont("Helvetica", 10)
        for s in suggestions:
            c.drawString(70, y, f"- {s}")
            y -= 14

    c.save()
    buffer.seek(0)
    return buffer

# ----------------------
# Visualization
# ----------------------

def plot_corr_heatmap(df: pd.DataFrame, columns: list):
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(columns, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig, use_container_width=True)

# ----------------------
# Live monitoring (improved UX)
# ----------------------

def live_monitoring(dataset: pd.DataFrame):
    st.subheader("Live Monitoring")

    required = ['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)']
    for r in required:
        if r not in dataset.columns:
            st.error(f"Missing required column: {r}")
            return

    # control buttons
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Monitoring"):
            st.session_state.monitoring = True
    with col2:
        if st.button("Stop Monitoring"):
            st.session_state.monitoring = False

    if not model_loaded:
        st.error(f"Model/scaler not loaded: {load_error}")
        return

    placeholder = st.empty()
    progress = st.progress(0)

    results = []
    total = len(dataset)

    # iterate but allow user to stop
    for i, row in dataset.iterrows():
        if not st.session_state.monitoring:
            break

        input_row = row[required].to_frame().T
        try:
            pred = predict_wqi(input_row)
        except Exception as e:
            st.error(f"Prediction failed on row {i}: {e}")
            break

        status, _ = categorize_wqi(pred)
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results.append({**row[required].to_dict(), 'Predicted_WQI': pred, 'Status': status, 'Timestamp': dt})

        # show small live dashboard
        with placeholder.container():
            st.write(f"### Record {i+1} / {total}")
            colA, colB = st.columns([2, 1])
            with colA:
                st.dataframe(pd.DataFrame([results[-1]]).set_index(pd.Index([dt]))[required + ['Predicted_WQI', 'Status']])
            with colB:
                st.markdown(status_badge_html(status), unsafe_allow_html=True)
                st.write(f"**WQI:** {pred:.4f}")

        progress.progress(min(100, int(((i+1)/total)*100)))

        # append to CSV
        out_df = pd.DataFrame([results[-1]])
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/Predicted.csv'
        header = not os.path.exists(csv_path)
        out_df.to_csv(csv_path, mode='a', index=False, header=header)

        # small delay to simulate live feed
        time.sleep(1)

    st.success("Monitoring stopped or completed.")
    if results:
        st.download_button("Download Monitoring Results (CSV)", data=pd.DataFrame(results).to_csv(index=False), file_name="monitoring_results.csv", mime='text/csv')

# ----------------------
# App layout
# ----------------------

def main():
    st.title("💧 Water Quality Prediction Dashboard — Improved")

    # Sidebar
    st.sidebar.header("Settings & Model")
    st.sidebar.write("Model status:")
    if model_loaded:
        st.sidebar.success("Model and scaler loaded")
    else:
        st.sidebar.error(f"Failed to load: {load_error}")

    st.sidebar.write("\n")
    st.sidebar.info("Upload a CSV with columns: NITRATE(PPM), PH, AMMONIA(mg/l), TEMP, DO, TURBIDITY, MANGANESE(mg/l)")

    tabs = st.tabs(["Data", "Visualization", "Predict", "Live Monitor"]) 

    # DATA TAB
    with tabs[0]:
        st.subheader("Dataset")
        uploaded = st.file_uploader("Upload a CSV to use instead of default dataset", type=["csv"]) 
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.uploaded_df = df
                st.success("Uploaded dataset loaded into session")
            except Exception as e:
                st.error(f"Unable to read uploaded CSV: {e}")

        ds = get_active_dataset()
        if ds is not None:
            st.write("### Preview (first 100 rows)")
            st.dataframe(ds.head(100), use_container_width=True)
        else:
            st.warning("No dataset available. Place a file at data/testing.csv or upload one.")

    # VIS TAB
    with tabs[1]:
        st.subheader("Data Visualization")
        ds = get_active_dataset()
        if ds is None:
            st.error("No dataset available to visualize.")
        else:
            numeric_cols = ds.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("### Correlation heatmap (numeric cols)")
                plot_corr_heatmap(ds, numeric_cols)
            else:
                st.warning("No numeric columns to visualize.")

    # PREDICT TAB
    with tabs[2]:
        st.subheader("Make a Prediction")
        user_df = user_input_widget()
        st.write("#### Current Input")
        st.dataframe(user_df)

        if st.button("Predict WQI"):
            if not model_loaded:
                st.error(f"Model not loaded: {load_error}")
            else:
                with st.spinner("Predicting..."):
                    try:
                        pred = predict_wqi(user_df)
                        status, desc = categorize_wqi(pred)
                        if status == 'Unsafe':
                            st.error(f"Predicted WQI: {pred:.4f} → {status} ({desc})")
                        elif status == 'Good':
                            st.warning(f"Predicted WQI: {pred:.4f} → {status} ({desc})")
                        else:
                            st.success(f"Predicted WQI: {pred:.4f} → {status} ({desc})")

                        st.markdown(status_badge_html(status), unsafe_allow_html=True)

                        pdf = generate_pdf_report(user_df, pred, status, title="Single Prediction Report")
                        name = f"prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        st.download_button("Download PDF Report", data=pdf, file_name=name, mime='application/pdf')

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    # LIVE TAB
    with tabs[3]:
        ds = get_active_dataset()
        if ds is None:
            st.error("No dataset available for live monitoring.")
        else:
            live_monitoring(ds)

if __name__ == '__main__':
    main()
