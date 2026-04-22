# Traffic Prediction — TfL / National Highways

A time-series forecasting project: predict traffic flow one hour ahead
(4 × 15-min steps) from the previous 24 hours of flow + calendar features.

## APIs & keys

You don't need to register for anything to run this end-to-end:

| Source | Used here | Key? |
| --- | --- | --- |
| **WebTRIS** (National Highways) — 15-min flow/speed on motorways & A-roads | **yes, primary** | **none** |
| TfL Unified API — road disruptions, status (London only) | optional / later | yes, free |
| Met Office DataPoint — weather | optional / weekend 3 | yes, free |

If later you want to layer in TfL disruption signals, register at
<https://api.tfl.gov.uk/> and drop the key into `.env` as `TFL_API_KEY`.

## Setup (Apple Silicon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env          # only needed if you add TfL later
```

TensorFlow 2.16+ installs the unified Mac wheel from `tensorflow`, and
`tensorflow-metal` gives you GPU acceleration (marginal for an LSTM this
size — CPU is fine).

## Pipeline

```
WebTRIS  ->  data_loader.py  ->  data/raw/*.csv
                               ->  preprocess.py  ->  Dataset (scaled windows)
                                                   ->  baseline.py  (seasonal naive)
                                                   ->  model.py     (2-layer LSTM)
                                                   ->  train.py     (fits + compares)
                                                   ->  predict.py
```

### 1. Find a site

```bash
python -m src.data_loader search "A406"
```

Pick one `Id`. Long-running, "Active" sites with lots of history are best.

### 2. Download history

```bash
python -m src.data_loader download \
  --site 6867 \
  --start 2023-01-01 \
  --end   2023-12-31
```

Writes `data/raw/site_6867_20230101_20231231.csv`.

### 3. Train

```bash
python -m src.train \
  --raw data/raw/site_6867_20230101_20231231.csv \
  --name a406_site6867
```

This:
1. Builds windows + chronological 70/15/15 split
2. Evaluates the **seasonal-naive baseline** (= same slot, one week ago) on the test set
3. Trains a 2-layer LSTM (64 units) with early stopping
4. Prints MAE/RMSE/MAPE for both, and the LSTM's uplift over the baseline

The LSTM's job is to beat the naive baseline. If it doesn't, don't add
features — figure out why. Common causes: too little data, site has a
sensor outage pattern, target isn't stationary.

### 4. Predict

```bash
python -m src.predict --name a406_site6867
```

## Project layout

```
.
├── data/
│   ├── raw/           # WebTRIS CSVs (gitignored)
│   └── processed/     # pickled Dataset objects (gitignored)
├── models/            # .keras checkpoints + metrics.json (gitignored)
├── notebooks/         # exploration
├── src/
│   ├── data_loader.py # WebTRIS client
│   ├── preprocess.py  # clean, window, split, scale
│   ├── baseline.py    # seasonal-naive predictor + metrics
│   ├── model.py       # LSTM definition
│   ├── train.py       # full end-to-end training
│   └── predict.py     # inference
├── requirements.txt
├── .env.example
└── .gitignore
```

## Common pitfalls (already handled)

- **Time zones**: WebTRIS returns local time; we treat it as a naive 15-min grid. If you mix sources, normalize to UTC before merging.
- **Leakage**: split is chronological, scaler is fit on train only.
- **Missing data**: gaps ≤ 1h are interpolated; longer gaps are dropped (check `preprocess.clean`).
- **Normalization**: `StandardScaler` over all features; predictions are inverse-transformed for metrics.
- **Weekends**: included as both cyclical `dow_sin/cos` and a binary `is_weekend` flag.
