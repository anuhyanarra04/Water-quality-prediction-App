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


# --------------------------------------------------
# Page Config & Styles
# --------------------------------------------------
st.set_page_config(page_title="Water Quality Prediction", layout="wide")
st.markdown(
    """
    <style>
    .status-badge { padding: 6px 10px; border-radius: 8px; font-weight: 600; display:inline-block; }
    .status-safe { background:#e7f7ee; color:#0f7b42; border:1px solid #bfead3; }
    .status-good { background:#fff8e6; color:#8a6d00; border:1px solid #ffe6a1; }
    .status-unsafe { background:#fde8e8; color:#b42318; border:1px solid #f3b4b4; }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Custom Attention Layer (must be present to load model)
# --------------------------------------------------
class TemporalAttention(Layer):
    def build(self, input_shape):
        self.w = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.w, axes=[2, 0]) + self.b)
        alpha = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(alpha * x, axis=1)
        return context


# --------------------------------------------------
# Load model & scaler
# --------------------------------------------------
try:
    model = load_model(
        'model/trained_wqi_model.h5',
        custom_objects={'TemporalAttention': TemporalAttention}
    )
    scaler = joblib.load('model/scaler.pkl')
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = f"Error: Unable to connect to the server. Details: {str(e)}"


# --------------------------------------------------
# Data loading (default + upload support)
# --------------------------------------------------
@st.cache_data
def load_default_dataset():
    try:
        dataset = pd.read_csv('data/testing.csv')
        return dataset
    except Exception:
        return None


def get_active_dataset():
    """
    Returns the currently active dataset:
    - If user uploaded a CSV in this session, use that.
    - Else fall back to default 'data/testing.csv'
    """
    if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
        return st.session_state.uploaded_df.copy()
    return load_default_dataset()


# --------------------------------------------------
# User input controls
# --------------------------------------------------
def get_user_input():
    st.write("### Enter values for the water quality parameters:")
    nitrate = st.slider("Nitrate (PPM) [0-140]", 0.0, 140.0, 23.0, 0.1)
    ph = st.slider("pH [0-10]", 0.0, 10.0, 7.0, 0.1)
    ammonia = st.slider("Ammonia (mg/L) [0-2]", 0.0, 2.0, 1.0, 0.1)
    temp = st.slider("Temperature (°C) [14-42]", 14.0, 42.0, 30.0, 0.1)
    do = st.slider("Dissolved Oxygen (mg/L) [6.5-12]", 6.5, 12.0, 8.0, 0.1)
    turbidity = st.slider("Turbidity [0-400]", 0.0, 400.0, 50.0, 0.1)
    manganese = st.slider("Manganese (mg/L) [0.5-3.6]", 0.5, 3.6, 1.0, 0.1)

    user_data = pd.DataFrame([[nitrate, ph, ammonia, temp, do, turbidity, manganese]],
                             columns=['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP',
                                      'DO', 'TURBIDITY', 'MANGANESE(mg/l)'])
    return user_data


# --------------------------------------------------
# Prediction helpers
# --------------------------------------------------
def predict_wqi(input_data: pd.DataFrame) -> float:
    input_data_normalized = scaler.transform(input_data)
    input_data_normalized = input_data_normalized.reshape(1, input_data_normalized.shape[1], 1)
    predicted_wqi = model.predict(input_data_normalized, verbose=0)[0][0]
    return float(predicted_wqi)


def categorize_wqi(predicted_wqi: float):
    if predicted_wqi < 0.3:
        return "Unsafe", "WQI < 0.3"
    elif 0.3 <= predicted_wqi < 0.7:
        return "Good", "0.3 ≤ WQI < 0.7"
    else:
        return "Safe", "WQI ≥ 0.7"


def status_badge_html(status: str) -> str:
    if status == "Safe":
        return '<span class="status-badge status-safe">🟢 Safe</span>'
    if status == "Good":
        return '<span class="status-badge status-good">🟡 Good</span>'
    return '<span class="status-badge status-unsafe">🔴 Unsafe</span>'


# --------------------------------------------------
# PDF report generation
# --------------------------------------------------
def generate_pdf_report(input_data, predicted_wqi, status, title="Water Quality Report"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(165, height - 50, title)

    # Model info
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 75, "Model: CNN-BiLSTM-Attention")

    # Input Parameters
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Input Parameters:")
    c.setFont("Helvetica", 10)
    y = height - 120
    for column, value in zip(input_data.columns, input_data.values[0]):
        c.drawString(70, y, f"{column}: {value}")
        y -= 18

    # Prediction
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Prediction Result:")
    c.setFont("Helvetica", 10)
    c.drawString(70, y - 28, f"Predicted WQI: {predicted_wqi:.4f}")
    c.drawString(70, y - 46, f"Water Quality Status: {status}")

    # Unsafe: parameters & suggestions
    if status == "Unsafe":
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y - 70, "Causing Parameters and Solutions:")
        y -= 90
        thresholds = {
            "NITRATE(PPM)": 50.0,
            "PH": (6.5, 8.5),
            "AMMONIA(mg/l)": 0.5,
            "TEMP": 35.0,
            "DO": 6.5,
            "TURBIDITY": 5.0,
            "MANGANESE(mg/l)": 0.1
        }
        issues = {}
        for column in input_data.columns:
            value = input_data[column].values[0]
            if isinstance(thresholds[column], tuple):
                if not (thresholds[column][0] <= value <= thresholds[column][1]):
                    issues[column] = value
            else:
                if value > thresholds[column]:
                    issues[column] = value

        solutions = {
            "NITRATE(PPM)": "Reduce agricultural runoff and improve wastewater treatment.",
            "PH": "Add alkaline or acidic substances to adjust pH levels.",
            "AMMONIA(mg/l)": "Improve wastewater treatment or reduce agricultural runoff.",
            "TEMP": "Introduce cooling systems to reduce water temperature.",
            "DO": "Increase aeration or reduce organic pollution.",
            "TURBIDITY": "Install filtration systems or reduce erosion.",
            "MANGANESE(mg/l)": "Improve filtration and water treatment processes."
        }

        for param, value in issues.items():
            c.drawString(70, y, f"{param}: {value}")
            c.drawString(70, y - 16, f"Solution: {solutions.get(param, 'No solution available.')}")
            y -= 34

    c.save()
    buffer.seek(0)
    return buffer


# --------------------------------------------------
# Visualization helpers
# --------------------------------------------------
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


# --------------------------------------------------
# Live Monitoring
# --------------------------------------------------
def live_monitoring(dataset: pd.DataFrame, model_loaded: bool):
    st.subheader("Live Monitoring with Visualization and Report Download")

    if dataset is None or dataset.empty:
        st.error("Dataset is empty. Please provide valid data.")
        return

    if not model_loaded:
        st.error(error_message)
        return

    required_features = ['NITRATE(PPM)', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'DO', 'TURBIDITY', 'MANGANESE(mg/l)']
    for rf in required_features:
        if rf not in dataset.columns:
            st.error(f"Missing required feature: {rf}")
            return

    dataset = dataset[required_features].copy()

    placeholder_table = st.empty()
    placeholder_chart = st.empty()
    placeholder_status = st.empty()
    placeholder_report = st.empty()

    live_data_records = pd.DataFrame(columns=required_features + ['Predicted_WQI', 'Status'])

    csv_file_path = 'data/Predicted.csv'
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(csv_file_path):
        live_data_records.to_csv(csv_file_path, index=False, mode='w')
    else:
        live_data_records.to_csv(csv_file_path, index=False, mode='a', header=False)

    delay_seconds = st.slider("Update interval (seconds)", 1, 10, 3)

    for index, row in dataset.iterrows():
        live_data = row.to_frame().T
        predicted_wqi = predict_wqi(live_data)

        if predicted_wqi < 0.3:
            status = "Unsafe"
        elif 0.3 <= predicted_wqi < 0.7:
            status = "Good"
        else:
            status = "Safe"

        live_data['Predicted_WQI'] = predicted_wqi
        live_data['Status'] = status
        live_data_records = pd.concat([live_data_records, live_data], ignore_index=True)

        with placeholder_table.container():
            st.write("#### Current Water Quality Data")
            st.dataframe(live_data)

        with placeholder_status.container():
            st.write("#### Water Quality Status")
            st.markdown(status_badge_html(status), unsafe_allow_html=True)
            st.write(f"**Predicted WQI:** {predicted_wqi:.4f}")

        with placeholder_chart.container():
            st.write("#### Live Predicted WQI Trend")
            st.line_chart(live_data_records[['Predicted_WQI']])

        with placeholder_report.container():
            st.write("#### Download Prediction Report")
            if st.button(f"Generate Report for Record {index + 1}"):
                pdf = generate_pdf_report(
                    live_data[required_features],
                    predicted_wqi,
                    status,
                    f"Record {index + 1}: Real-time Monitoring Report"
                )
                current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                file_name = f"live_monitoring_report_{current_time}.pdf"

                st.download_button(
                    label="Download PDF Report",
                    data=pdf,
                    file_name=file_name,
                    mime="application/pdf"
                )

        time.sleep(delay_seconds)


# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():
    st.title("💧 Water Quality Prediction Dashboard")

    # Tabs UI
    tab_data, tab_viz, tab_predict, tab_live = st.tabs(
        ["📂 Data", "📈 Visualization", "🔮 Predict", "⏱️ Live Monitor"]
    )

    # ---------------- Data Tab ----------------
    with tab_data:
        st.subheader("Dataset")
        uploaded = st.file_uploader("Upload a CSV to use instead of default dataset", type="csv")
        if uploaded is not None:
            try:
                st.session_state.uploaded_df = pd.read_csv(uploaded)
                st.success("Uploaded dataset loaded.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        ds = get_active_dataset()
        if ds is not None:
            st.write("### Preview (first 100 rows)")
            st.dataframe(ds.head(100), use_container_width=True)
            st.caption("Using uploaded dataset" if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None else "Using default dataset (data/testing.csv)")
        else:
            st.error("Could not load any dataset.")

    # ---------------- Visualization Tab ----------------
    with tab_viz:
        st.subheader("Data Visualization")
        ds = get_active_dataset()
        if ds is not None:
            if 'Date' in ds.columns:
                ds = ds.copy()
                ds['Date'] = pd.to_datetime(ds['Date'], errors='coerce')
                ds = ds.dropna(subset=['Date'])
                ds['Month'] = ds['Date'].dt.month

                col1, col2 = st.columns(2)
                with col1:
                    selected_month = st.selectbox(
                        "Select Month",
                        range(1, 13),
                        format_func=lambda x: pd.to_datetime(f"2024-{x:02d}-01").strftime('%B')
                    )
                with col2:
                    # Exclude non-numeric & helper cols
                    candidate_cols = [c for c in ds.columns if c not in ['Station', 'Month']]
                    if 'Date' in candidate_cols:
                        candidate_cols.remove('Date')
                    attributes = st.multiselect(
                        "Select Attributes to Visualize",
                        candidate_cols,
                        default=[c for c in ['NITRATE(PPM)', 'PH', 'TEMP'] if c in candidate_cols]
                    )

                month_data = ds[ds['Month'] == selected_month]
                if not month_data.empty and attributes:
                    st.line_chart(month_data.set_index('Date')[attributes])
                else:
                    st.warning("No data for this month or no attributes selected.")

                # Correlation heatmap (numeric only)
                numeric_cols = ds.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.write("### Correlation Heatmap")
                    plot_corr_heatmap(ds, numeric_cols)
            else:
                st.error("No 'Date' column in dataset to visualize time series.")
        else:
            st.error("No dataset available.")

    # ---------------- Predict Tab ----------------
    with tab_predict:
        st.subheader("Make a Prediction")
        user_input = get_user_input()
        if st.button("Predict WQI"):
            if model_loaded:
                with st.spinner("Running CNN-BiLSTM-Attention model..."):
                    predicted_wqi = predict_wqi(user_input)
                    status, range_desc = categorize_wqi(predicted_wqi)

                if status == "Unsafe":
                    st.error(f"Predicted WQI: {predicted_wqi:.4f} → {status} ({range_desc})")
                elif status == "Good":
                    st.warning(f"Predicted WQI: {predicted_wqi:.4f} → {status} ({range_desc})")
                else:
                    st.success(f"Predicted WQI: {predicted_wqi:.4f} → {status} ({range_desc})")

                st.markdown(status_badge_html(status), unsafe_allow_html=True)

                st.write("#### Download Prediction Report")
                pdf = generate_pdf_report(
                    user_input,
                    predicted_wqi,
                    status,
                    "Single Prediction Report"
                )
                current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                file_name = f"prediction_report_{current_time}.pdf"
                st.download_button(
                    label="Download PDF Report",
                    data=pdf,
                    file_name=file_name,
                    mime="application/pdf"
                )
            else:
                st.error(error_message)

    # ---------------- Live Monitor Tab ----------------
    with tab_live:
        ds = get_active_dataset()
        if ds is not None:
            live_monitoring(ds, model_loaded)
        else:
            st.error("No dataset available for live monitoring.")


if __name__ == "__main__":
    main()
