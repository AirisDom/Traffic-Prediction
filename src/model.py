"""Model zoo for traffic-flow forecasting.

Both models share the I/O contract:
  Input:  (batch, input_steps=96, n_features)
  Output: (batch, output_steps=4)
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


@tf.keras.utils.register_keras_serializable(package="traffic")
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = tf.cast(tf.range(max_len)[:, None], tf.float32)
        i = tf.cast(tf.range(d_model)[None, :], tf.float32)
        angle = pos / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        sines = tf.sin(angle[:, 0::2])
        cosines = tf.cos(angle[:, 1::2])
        pe = tf.reshape(
            tf.concat([sines[..., None], cosines[..., None]], axis=-1),
            (max_len, d_model),
        )
        self.pe = tf.constant(pe[None, ...])

    def call(self, x):
        return x + self.pe[:, : tf.shape(x)[1], :]

    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len, "d_model": self.d_model}


def _transformer_block(x, d_model: int, n_heads: int, ff_dim: int, dropout: float):
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads,
                                      dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)


def build_transformer(
    input_steps: int,
    n_features: int,
    output_steps: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_blocks: int = 2,
    ff_dim: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 5e-4,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_steps, n_features))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(input_steps, d_model)(x)
    for _ in range(n_blocks):
        x = _transformer_block(x, d_model, n_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output_steps)(x)
    model = models.Model(inputs, outputs, name="traffic_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build(kind: str, **kwargs) -> tf.keras.Model:
    if kind == "lstm":
        return build_lstm(**kwargs)
    if kind == "transformer":
        return build_transformer(**kwargs)
    raise ValueError(f"unknown model kind: {kind}")
