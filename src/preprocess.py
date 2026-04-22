"""Clean raw WebTRIS CSV -> modeling-ready numpy windows.

Supports:
  - UK bank holiday feature (via `holidays` package)
  - Optional merge of hourly weather (Open-Meteo) forward-filled to 15-min grid
  - Single-site and multi-site dataset construction with per-site scaling
    and one-hot site identity.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .weather import merge_weather

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STEPS_PER_HOUR = 4
INPUT_STEPS = 24 * STEPS_PER_HOUR   # 96
OUTPUT_STEPS = 1 * STEPS_PER_HOUR   # 4
STEPS_PER_WEEK = 7 * 24 * STEPS_PER_HOUR  # 672
TARGET = "flow"

BASE_FEATURES = ["flow", "avg_mph",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "is_weekend", "is_holiday"]
WEATHER_FEATURES = ["temp_c", "precip_mm", "wind_kmh", "cloud_pct"]


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_cols: list[str]
    scalers: dict[str, StandardScaler]           # keyed by site tag ("single" for 1-site)
    site_tags: list[str]
    test_site_of_window: np.ndarray              # site tag for each test window
    baseline_test_scaled: np.ndarray             # precomputed seasonal-naive preds, scaled


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="15min")
    df = df.reindex(full)
    df.index.name = "timestamp"
    return df


def clean(df: pd.DataFrame, max_gap: int = 4) -> pd.DataFrame:
    df = df.copy()
    df["flow"] = df["flow"].interpolate(method="linear", limit=max_gap)
    if "avg_mph" in df.columns:
        df["avg_mph"] = df["avg_mph"].interpolate(method="linear", limit=max_gap)
    df = df.dropna(subset=["flow"])
    return df


def _uk_holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    years = range(index.min().year, index.max().year + 1)
    uk = holidays.UnitedKingdom(years=years, subdiv="ENG")
    return np.array([d.date() in uk for d in index], dtype=float)


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
    df["is_holiday"] = _uk_holiday_mask(idx)
    return df


def _load_weather_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()


def chronological_split(
    df: pd.DataFrame, train: float = 0.7, val: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def make_windows(
    arr: np.ndarray, target_idx: int, input_steps: int = INPUT_STEPS,
    output_steps: int = OUTPUT_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    total = input_steps + output_steps
    n = len(arr) - total + 1
    if n <= 0:
        return np.empty((0, input_steps, arr.shape[1])), np.empty((0, output_steps))
    X = np.stack([arr[i : i + input_steps] for i in range(n)])
    y = np.stack([arr[i + input_steps : i + total, target_idx] for i in range(n)])
    return X.astype(np.float32), y.astype(np.float32)


def _seasonal_naive_scaled(series_scaled: np.ndarray, target_idx: int) -> np.ndarray:
    """Precompute per-window seasonal-naive predictions in scaled space."""
    total = INPUT_STEPS + OUTPUT_STEPS
    n_windows = len(series_scaled) - total + 1
    if n_windows <= 0:
        return np.empty((0, OUTPUT_STEPS), dtype=np.float32)
    preds = np.zeros((n_windows, OUTPUT_STEPS), dtype=np.float32)
    for i in range(n_windows):
        forecast_start = i + INPUT_STEPS
        lookback = forecast_start - STEPS_PER_WEEK
        if lookback < 0:
            src = series_scaled[forecast_start:forecast_start + OUTPUT_STEPS, target_idx]
        else:
            src = series_scaled[lookback:lookback + OUTPUT_STEPS, target_idx]
        preds[i] = src
    return preds


def _prep_site_frame(
    raw_csv: str | Path,
    weather_csv: str | Path | None,
    site_tags: list[str] | None,
    this_tag: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = add_time_features(clean(load_raw(raw_csv)))

    feature_cols = [c for c in BASE_FEATURES if c in df.columns]

    if weather_csv is not None:
        wx = _load_weather_csv(weather_csv)
        df = merge_weather(df, wx)
        feature_cols += [c for c in WEATHER_FEATURES if c in df.columns]

    # Multi-site: append one-hot site identity columns
    if site_tags is not None:
        for tag in site_tags:
            col = f"site_{tag}"
            df[col] = 1.0 if tag == this_tag else 0.0
            feature_cols.append(col)

    df = df[feature_cols].dropna()
    return df, feature_cols


def build_dataset(raw_csv: str | Path, weather_csv: str | Path | None = None) -> Dataset:
    """Single-site dataset."""
    df, feature_cols = _prep_site_frame(raw_csv, weather_csv, None, None)
    train_df, val_df, test_df = chronological_split(df)

    scaler = StandardScaler().fit(train_df.values)
    target_idx = feature_cols.index(TARGET)

    train_a = scaler.transform(train_df.values)
    val_a = scaler.transform(val_df.values)
    test_a = scaler.transform(test_df.values)

    X_train, y_train = make_windows(train_a, target_idx)
    X_val, y_val = make_windows(val_a, target_idx)
    X_test, y_test = make_windows(test_a, target_idx)

    baseline_test = _seasonal_naive_scaled(test_a, target_idx)
    test_site = np.array(["single"] * len(X_test))

    return Dataset(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_cols, {"single": scaler}, ["single"], test_site, baseline_test,
    )


def build_multi_dataset(
    site_csvs: dict[str, str | Path],
    weather_csvs: dict[str, str | Path] | None = None,
) -> Dataset:
    """Multi-site dataset: per-site scaling + one-hot site identity feature."""
    site_tags = sorted(site_csvs.keys())
    scalers: dict[str, StandardScaler] = {}
    feature_cols: list[str] | None = None

    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X, te_y, te_site, te_baseline = [], [], [], []

    for tag in site_tags:
        wx_path = weather_csvs.get(tag) if weather_csvs else None
        df, cols = _prep_site_frame(site_csvs[tag], wx_path, site_tags, tag)
        if feature_cols is None:
            feature_cols = cols
        elif feature_cols != cols:
            raise ValueError(f"Feature mismatch for site {tag}: {cols} vs {feature_cols}")

        train_df, val_df, test_df = chronological_split(df)
        scaler = StandardScaler().fit(train_df.values)
        scalers[tag] = scaler
        target_idx = feature_cols.index(TARGET)

        train_a = scaler.transform(train_df.values)
        val_a = scaler.transform(val_df.values)
        test_a = scaler.transform(test_df.values)

        Xtr, ytr = make_windows(train_a, target_idx)
        Xv, yv = make_windows(val_a, target_idx)
        Xte, yte = make_windows(test_a, target_idx)
        base = _seasonal_naive_scaled(test_a, target_idx)

        tr_X.append(Xtr); tr_y.append(ytr)
        va_X.append(Xv); va_y.append(yv)
        te_X.append(Xte); te_y.append(yte)
        te_baseline.append(base)
        te_site.append(np.array([tag] * len(Xte)))

    X_train = np.concatenate(tr_X)
    y_train = np.concatenate(tr_y)
    # Shuffle training windows across sites for better batching
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    X_val = np.concatenate(va_X)
    y_val = np.concatenate(va_y)
    X_test = np.concatenate(te_X)
    y_test = np.concatenate(te_y)
    baseline_test = np.concatenate(te_baseline)
    test_site = np.concatenate(te_site)

    return Dataset(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_cols, scalers, site_tags, test_site, baseline_test,
    )


def inverse_target(scaled: np.ndarray, ds: Dataset, site_tags_of_row: np.ndarray | None = None) -> np.ndarray:
    """Undo StandardScaler for the target column.

    For multi-site, pass `site_tags_of_row` of length len(scaled). For single-site, omit.
    """
    idx = ds.feature_cols.index(TARGET)

    if len(ds.scalers) == 1:
        sc = next(iter(ds.scalers.values()))
        return scaled * sc.scale_[idx] + sc.mean_[idx]

    if site_tags_of_row is None:
        raise ValueError("Multi-site dataset requires site_tags_of_row for inverse transform")

    out = np.empty_like(scaled, dtype=np.float64)
    for tag, sc in ds.scalers.items():
        mask = site_tags_of_row == tag
        if mask.any():
            out[mask] = scaled[mask] * sc.scale_[idx] + sc.mean_[idx]
    return out


def save_dataset(ds: Dataset, name: str) -> Path:
    out = PROCESSED_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(ds, f)
    return out


def load_dataset(name: str) -> Dataset:
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
