"""Seasonal-naive baseline: prediction = same slot, one week ago.

If the LSTM can't beat this, the LSTM is wrong.
"""
from __future__ import annotations

import numpy as np

from .preprocess import INPUT_STEPS, OUTPUT_STEPS, STEPS_PER_HOUR, Dataset, inverse_target

STEPS_PER_WEEK = 7 * 24 * STEPS_PER_HOUR  # 672


def seasonal_naive_predict(series_scaled: np.ndarray, target_idx: int) -> np.ndarray:
    """Predict y[t:t+OUTPUT_STEPS] as series[t - one_week : t - one_week + OUTPUT_STEPS].

    Returns predictions aligned with the windows produced by make_windows —
    i.e. one prediction per sliding window.
    """
    total = INPUT_STEPS + OUTPUT_STEPS
    n_windows = len(series_scaled) - total + 1
    preds = np.zeros((n_windows, OUTPUT_STEPS), dtype=np.float32)
    for i in range(n_windows):
        forecast_start = i + INPUT_STEPS
        lookback = forecast_start - STEPS_PER_WEEK
        if lookback < 0:
            preds[i] = series_scaled[forecast_start:forecast_start + OUTPUT_STEPS, target_idx]
        else:
            preds[i] = series_scaled[lookback:lookback + OUTPUT_STEPS, target_idx]
    return preds


def evaluate(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, ds: Dataset) -> dict:
    y_true = inverse_target(y_true_scaled, ds)
    y_pred = inverse_target(y_pred_scaled, ds)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}
