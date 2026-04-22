"""WebTRIS (National Highways) client.

WebTRIS is public — no API key needed. The `daily` report returns flow,
speed, and occupancy already bucketed into 15-min intervals, which is
exactly the granularity the model expects.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://webtris.nationalhighways.co.uk/api/v1.0"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def list_sites() -> pd.DataFrame:
    r = requests.get(f"{BASE_URL}/sites", timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json()["sites"])


def search_sites(query: str) -> pd.DataFrame:
    sites = list_sites()
    mask = sites["Name"].str.contains(query, case=False, na=False)
    return sites.loc[mask, ["Id", "Name", "Status", "Latitude", "Longitude"]]


def _fetch_daily_page(site_id: int, start: date, end: date, page: int) -> dict:
    params = {
        "sites": site_id,
        "start_date": start.strftime("%d%m%Y"),
        "end_date": end.strftime("%d%m%Y"),
        "page": page,
        "page_size": 40000,
    }
    r = requests.get(f"{BASE_URL}/reports/daily", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_daily_report(site_id: int, start: date, end: date) -> pd.DataFrame:
    """Fetch 15-min bucketed flow/speed for a site across a date range.

    Paginates until exhausted. Rows with all-null measurements are dropped.
    """
    rows: list[dict] = []
    page = 1
    while True:
        payload = _fetch_daily_page(site_id, start, end, page)
        batch = payload.get("Rows") or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 40000:
            break
        page += 1
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(
        df["Report Date"].str.slice(0, 10) + " " + df["Time Period Ending"],
        format="%Y-%m-%d %H:%M:%S",
    )
    df = df.rename(
        columns={
            "Avg mph": "avg_mph",
            "Total Volume": "flow",
            "Site Name": "site_name",
        }
    )
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df["avg_mph"] = pd.to_numeric(df["avg_mph"], errors="coerce")
    df = df[["timestamp", "site_name", "flow", "avg_mph"]]
    df = df.dropna(subset=["flow"]).sort_values("timestamp").reset_index(drop=True)
    return df


def download_site_history(
    site_id: int,
    start: date,
    end: date,
    chunk_days: int = 30,
) -> Path:
    """Download a full history in chunks and persist to data/raw/."""
    chunks: list[pd.DataFrame] = []
    cursor = start
    pbar = tqdm(total=(end - start).days, desc=f"site {site_id}")
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        part = fetch_daily_report(site_id, cursor, chunk_end)
        if not part.empty:
            chunks.append(part)
        pbar.update((chunk_end - cursor).days)
        cursor = chunk_end + timedelta(days=1)
    pbar.close()

    if not chunks:
        raise RuntimeError(f"No data returned for site {site_id}")

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp"])
    out = RAW_DIR / f"site_{site_id}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("search", help="Search sites by name substring")
    s1.add_argument("query")

    s2 = sub.add_parser("download", help="Download history for a site")
    s2.add_argument("--site", type=int, required=True)
    s2.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    s2.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")

    args = p.parse_args()
    if args.cmd == "search":
        hits = search_sites(args.query)
        print(hits.to_string(index=False))
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        path = download_site_history(args.site, start, end)
        print(f"saved -> {path}")
