"""End-to-end: raw CSV -> dataset -> baseline metrics -> LSTM -> test metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .baseline import evaluate, seasonal_naive_predict
from .model import build_lstm
from .preprocess import INPUT_STEPS, OUTPUT_STEPS, TARGET, build_dataset, save_dataset

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run(raw_csv: str, name: str, epochs: int = 30, batch_size: int = 64):
    print(f"[1/4] Building dataset from {raw_csv}")
    ds = build_dataset(raw_csv)
    save_dataset(ds, name)
    print(f"  features: {ds.feature_cols}")
    print(f"  X_train: {ds.X_train.shape}  X_val: {ds.X_val.shape}  X_test: {ds.X_test.shape}")

    print("[2/4] Seasonal-naive baseline on test split")
    import pandas as pd
    from .preprocess import (add_time_features, chronological_split, clean,
                             load_raw)
    df = add_time_features(clean(load_raw(raw_csv)))[ds.feature_cols].dropna()
    _, _, test_df = chronological_split(df)
    test_scaled = ds.scaler.transform(test_df.values)
    target_idx = ds.feature_cols.index(TARGET)
    naive_pred = seasonal_naive_predict(test_scaled, target_idx)
    baseline_metrics = evaluate(ds.y_test, naive_pred, ds)
    print(f"  baseline: {baseline_metrics}")

    print("[3/4] Training LSTM")
    n_features = ds.X_train.shape[-1]
    model = build_lstm(INPUT_STEPS, n_features, OUTPUT_STEPS)
    model.summary()

    ckpt = MODELS_DIR / f"{name}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_mae"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_mae", save_best_only=True
        ),
    ]
    model.fit(
        ds.X_train, ds.y_train,
        validation_data=(ds.X_val, ds.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    print("[4/4] Evaluating LSTM on test")
    y_pred = model.predict(ds.X_test, verbose=0)
    lstm_metrics = evaluate(ds.y_test, y_pred, ds)
    print(f"  lstm:     {lstm_metrics}")

    uplift_mae = baseline_metrics["mae"] - lstm_metrics["mae"]
    print(f"\n  LSTM beats baseline MAE by: {uplift_mae:.2f} vehicles / 15min")

    results = {
        "name": name,
        "baseline": baseline_metrics,
        "lstm": lstm_metrics,
        "uplift_mae": uplift_mae,
        "feature_cols": ds.feature_cols,
    }
    (MODELS_DIR / f"{name}.metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved model -> {ckpt}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="path to raw CSV from data_loader")
    p.add_argument("--name", required=True, help="run name (used for model + dataset)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    run(args.raw, args.name, args.epochs, args.batch_size)
