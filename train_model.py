import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SeparableConv1D, MaxPooling1D, Dropout,
    Bidirectional, LSTM, Dense, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


# -----------------------------
# Simple Self-Attention Layer
# -----------------------------
class TemporalAttention(Layer):
    """
    Computes attention weights over timesteps for a 3D input (batch, timesteps, features)
    and returns a context vector (batch, features).
    """
    def build(self, input_shape):
        # Weight to compute attention score per timestep
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
        # x: (batch, timesteps, features)
        # score: (batch, timesteps, 1)
        score = tf.nn.tanh(tf.tensordot(x, self.w, axes=[2, 0]) + self.b)
        # attention weights over timesteps
        alpha = tf.nn.softmax(score, axis=1)
        # weighted sum -> context: (batch, features)
        context = tf.reduce_sum(alpha * x, axis=1)
        return context


# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Use only numeric columns (exclude 'Station', 'Date', etc.)
    numeric_data = data.select_dtypes(include=[np.number])

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data),
                                columns=numeric_data.columns)

    # Normalize
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed),
                               columns=data_imputed.columns)

    # Synthetic WQI target (same as your current logic)
    data_scaled['WQI'] = (
        0.3  * data_scaled['NITRATE(PPM)'] +
        0.2  * data_scaled['PH'] +
        0.15 * data_scaled['AMMONIA(mg/l)'] +
        0.1  * data_scaled['TEMP'] +
        0.1  * data_scaled['DO'] +
        0.1  * data_scaled['TURBIDITY'] +
        0.05 * data_scaled['MANGANESE(mg/l)']
    )

    # Features/target
    X = data_scaled.drop(columns=['WQI']).values
    y = data_scaled['WQI'].values

    # Shape for CNN-BiLSTM: (samples, timesteps, features)
    # Here we treat each original feature as a "timestep" with 1 feature channel.
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, scaler


# -----------------------------
# Model: CNN + BiLSTM + Attention
# -----------------------------
def build_model(input_shape):
    model = Sequential([
        # CNN feature extractor
        SeparableConv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Sequence model
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),

        # Attention over time steps
        TemporalAttention(),

        # Regressor head
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')  # regression output
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=MeanSquaredError(),
                  metrics=['mae'])
    return model


# -----------------------------
# Train & Save
# -----------------------------
def train_and_save_model(
    file_path='data/training.csv',
    model_save_path='model/trained_wqi_model.h5',
    scaler_save_path='model/scaler.pkl'
):
    X, y, scaler = load_and_preprocess_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(input_shape=(X.shape[1], 1))

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,              # a bit more for LSTMs
        batch_size=32,
        verbose=1
    )

    # Save model & scaler
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")


if __name__ == "__main__":
    train_and_save_model()
