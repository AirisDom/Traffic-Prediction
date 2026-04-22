"""Clean raw WebTRIS CSV -> modeling-ready numpy windows.

Responsibilities:
  1. Load raw, enforce a continuous 15-min DatetimeIndex (gaps become NaN).
  2. Interpolate short gaps, drop long ones.
  3. Add cyclical time features (hour, day-of-week) + weekend flag.
  4. Split chronologically into train/val/test.
  5. Fit a scaler on train only and transform the rest.
  6. Slice into sliding windows: X=(96, n_feat), y=(4,).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STEPS_PER_HOUR = 4
INPUT_STEPS = 24 * STEPS_PER_HOUR   # 96
OUTPUT_STEPS = 1 * STEPS_PER_HOUR   # 4
TARGET = "flow"


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: list[str]
    scaler: StandardScaler
    target_mean: float
    target_std: float


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df = df.reindex(full)
    df.index.name = "timestamp"
    return df


def clean(df: pd.DataFrame, max_gap: int = 4) -> pd.DataFrame:
    """Interpolate runs of up to `max_gap` missing steps; drop longer ones."""
    df = df.copy()
    df["flow"] = df["flow"].interpolate(method="linear", limit=max_gap)
    if "avg_mph" in df.columns:
        df["avg_mph"] = df["avg_mph"].interpolate(method="linear", limit=max_gap)
    df = df.dropna(subset=["flow"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    return df


def chronological_split(
    df: pd.DataFrame, train: float = 0.7, val: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def make_windows(
    arr: np.ndarray, target_idx: int, input_steps: int, output_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    total = input_steps + output_steps
    n = len(arr) - total + 1
    if n <= 0:
        return np.empty((0, input_steps, arr.shape[1])), np.empty((0, output_steps))
    X = np.stack([arr[i : i + input_steps] for i in range(n)])
    y = np.stack([arr[i + input_steps : i + total, target_idx] for i in range(n)])
    return X.astype(np.float32), y.astype(np.float32)


def build_dataset(
    raw_csv: str | Path,
    feature_cols: list[str] | None = None,
) -> Dataset:
    df = load_raw(raw_csv)
    df = clean(df)
    df = add_time_features(df)

    if feature_cols is None:
        feature_cols = ["flow", "avg_mph", "hour_sin", "hour_cos",
                        "dow_sin", "dow_cos", "is_weekend"]
        feature_cols = [c for c in feature_cols if c in df.columns]

    df = df[feature_cols].dropna()
    train_df, val_df, test_df = chronological_split(df)

    scaler = StandardScaler().fit(train_df.values)
    t_mean = float(train_df[TARGET].mean())
    t_std = float(train_df[TARGET].std())

    train_a = scaler.transform(train_df.values)
    val_a = scaler.transform(val_df.values)
    test_a = scaler.transform(test_df.values)

    target_idx = feature_cols.index(TARGET)
    X_train, y_train = make_windows(train_a, target_idx, INPUT_STEPS, OUTPUT_STEPS)
    X_val, y_val = make_windows(val_a, target_idx, INPUT_STEPS, OUTPUT_STEPS)
    X_test, y_test = make_windows(test_a, target_idx, INPUT_STEPS, OUTPUT_STEPS)

    return Dataset(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_cols, scaler, t_mean, t_std,
    )


def save_dataset(ds: Dataset, name: str) -> Path:
    out = PROCESSED_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(ds, f)
    return out


def load_dataset(name: str) -> Dataset:
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def inverse_target(scaled: np.ndarray, ds: Dataset) -> np.ndarray:
    """Undo StandardScaler for the target column only."""
    idx = ds.feature_cols.index(TARGET)
    mean = ds.scaler.mean_[idx]
    scale = ds.scaler.scale_[idx]
    return scaled * scale + mean
