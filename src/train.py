"""End-to-end: raw CSV(s) -> dataset -> baseline -> train -> evaluate.

Single-site:
  python -m src.train --raw data/raw/site_5_20240101_20241231.csv \
      --weather data/raw/weather_site5_20240101_20241231.csv \
      --model transformer --name m25_site5_tfm

Multi-site:
  python -m src.train --multi site5=...csv site27=...csv \
      --weather-multi site5=...csv site27=...csv \
      --model transformer --name m25_multi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from .baseline import evaluate_scaled
from .model import build
from .preprocess import (INPUT_STEPS, OUTPUT_STEPS, build_dataset,
                         build_multi_dataset, save_dataset)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_kv_list(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        k, _, v = item.partition("=")
        if not k or not v:
            raise ValueError(f"expected key=value, got {item!r}")
        out[k] = v
    return out


def run(args):
    if args.multi:
        site_csvs = _parse_kv_list(args.multi)
        weather_csvs = _parse_kv_list(args.weather_multi) if args.weather_multi else None
        print(f"[1/4] Building multi-site dataset ({len(site_csvs)} sites)")
        ds = build_multi_dataset(site_csvs, weather_csvs)
    else:
        print(f"[1/4] Building dataset from {args.raw}")
        ds = build_dataset(args.raw, args.weather)

    save_dataset(ds, args.name)
    print(f"  features: {ds.feature_cols}")
    print(f"  sites:    {ds.site_tags}")
    print(f"  X_train {ds.X_train.shape}  X_val {ds.X_val.shape}  X_test {ds.X_test.shape}")

    print("[2/4] Seasonal-naive baseline on test split")
    baseline_metrics = evaluate_scaled(
        ds.y_test, ds.baseline_test_scaled, ds, ds.test_site_of_window
    )
    print(f"  baseline: {baseline_metrics}")

    print(f"[3/4] Training {args.model}")
    n_features = ds.X_train.shape[-1]
    model = build(
        args.model,
        input_steps=INPUT_STEPS,
        n_features=n_features,
        output_steps=OUTPUT_STEPS,
    )
    model.summary()

    ckpt = MODELS_DIR / f"{args.name}.keras"
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    print("[4/4] Evaluating on test")
    y_pred = model.predict(ds.X_test, verbose=0)
    model_metrics = evaluate_scaled(ds.y_test, y_pred, ds, ds.test_site_of_window)
    print(f"  {args.model}: {model_metrics}")

    uplift_mae = baseline_metrics["mae"] - model_metrics["mae"]
    print(f"\n  {args.model} beats baseline MAE by: {uplift_mae:.2f} vehicles / 15min")

    results = {
        "name": args.name,
        "model": args.model,
        "feature_cols": ds.feature_cols,
        "sites": ds.site_tags,
        "baseline": baseline_metrics,
        "model_metrics": model_metrics,
        "uplift_mae": uplift_mae,
    }
    (MODELS_DIR / f"{args.name}.metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved model -> {ckpt}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw", help="single-site raw CSV from data_loader")
    g.add_argument("--multi", nargs="+",
                   help="multi-site as tag=path pairs, e.g. site5=path1.csv site27=path2.csv")
    p.add_argument("--weather", help="single-site weather CSV (optional)")
    p.add_argument("--weather-multi", nargs="+",
                   help="multi-site weather as tag=path pairs")
    p.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    run(args)
