"""Load a saved model + dataset and predict the next hour from the latest window."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .preprocess import inverse_target, load_dataset

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def predict_next_hour(name: str) -> pd.DataFrame:
    ds = load_dataset(name)
    model = tf.keras.models.load_model(MODELS_DIR / f"{name}.keras")

    last_X = ds.X_test[-1:]
    last_site = ds.test_site_of_window[-1:]
    y_pred_scaled = model.predict(last_X, verbose=0)
    y_pred = inverse_target(y_pred_scaled, ds, last_site)

    return pd.DataFrame(
        {
            "step_ahead_15min": np.arange(1, y_pred.shape[1] + 1),
            "site": last_site[0],
            "predicted_flow": y_pred[0],
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    args = p.parse_args()
    print(predict_next_hour(args.name).to_string(index=False))
