"""Seasonal-naive baseline evaluation.

Predictions are precomputed inside `build_dataset` / `build_multi_dataset` and
stored on the Dataset as `baseline_test_scaled`. This module just evaluates.
"""
from __future__ import annotations

import numpy as np

from .preprocess import Dataset, inverse_target


def evaluate_scaled(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    ds: Dataset,
    site_tags_of_row: np.ndarray | None = None,
) -> dict:
    y_true = inverse_target(y_true_scaled, ds, site_tags_of_row)
    y_pred = inverse_target(y_pred_scaled, ds, site_tags_of_row)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}
