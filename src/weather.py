"""Open-Meteo historical weather client.

Free, no key required. Returns hourly weather for a lat/lon and date range;
we resample to the 15-min flow grid by forward-filling within each hour.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]


def fetch_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    r = requests.get(
        BASE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "Europe/London",
        },
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()["hourly"]
    df = pd.DataFrame(payload)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "time": "timestamp",
            "temperature_2m": "temp_c",
            "precipitation": "precip_mm",
            "wind_speed_10m": "wind_kmh",
            "cloud_cover": "cloud_pct",
        }
    )
    df = df.set_index("timestamp").sort_index()
    return df


def download_weather(lat: float, lon: float, start: date, end: date, tag: str) -> Path:
    df = fetch_weather(lat, lon, start, end)
    out = RAW_DIR / f"weather_{tag}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    df.to_csv(out)
    return out


def merge_weather(flow: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Align hourly weather to the 15-min flow index via forward-fill.

    flow.index: 15-min DatetimeIndex
    weather.index: 1-hour DatetimeIndex
    Returns flow + weather columns on the flow index.
    """
    wx = weather.reindex(flow.index, method="ffill")
    return flow.join(wx, how="left")


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tag", required=True, help="used in output filename")
    args = p.parse_args()

    out = download_weather(
        args.lat, args.lon,
        datetime.strptime(args.start, "%Y-%m-%d").date(),
        datetime.strptime(args.end, "%Y-%m-%d").date(),
        args.tag,
    )
    print(f"saved -> {out}")
