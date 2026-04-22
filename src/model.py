"""2-layer LSTM for 1-hour-ahead traffic flow forecasting.

Input:  (batch, 96, n_features)  — last 24h at 15-min intervals
Output: (batch, 4)               — flow for next 4 x 15-min slots
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm(
    input_steps: int,
    n_features: int,
    output_steps: int,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_steps, n_features))
    x = layers.LSTM(units, return_sequences=True, dropout=dropout)(inputs)
    x = layers.LSTM(units, dropout=dropout)(x)
    outputs = layers.Dense(output_steps)(x)
    model = models.Model(inputs, outputs, name="traffic_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
