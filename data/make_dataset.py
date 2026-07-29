"""
make_dataset.py

Downloads KOSPI200 daily data and builds the shared dataset CSV that all six
model scripts read (xgboost_final.py, randomforest_final.py, transformer_final.py,
pinn_final.py, final_comparison.py).

Run this once before running any model:

    pip install yfinance pandas numpy
    python make_dataset.py

Output: kospi200_dataset.csv

The CSV contains, per trading day:
  - 13 features (5 lagged returns, 3 trailing realized vols, 3 MA ratios,
    high-low range, volume ratio)
  - target_vol : forward 21-day annualized realized volatility (the forecast target)
  - regime     : low / mid / high, by tercile of trailing 21-day realized volatility
  - rvol_21    : trailing 21-day realized vol (used by the physics anchor)
"""

import numpy as np
import pandas as pd
import yfinance as yf

TICKER       = "^KS200"
START        = "2008-01-01"
END          = None            # None = up to today
TRADING_DAYS = 252
H            = 21              # forward window for the target
GAP_DAYS     = 10             # calendar gaps larger than this break rolling windows
QUANTILES    = (1/3, 2/3)     # tercile cutoffs for regimes
OUT_PATH     = "kospi200_dataset.csv"

FEATURES = ([f"ret_lag_{l}" for l in range(1, 6)]
            + [f"rvol_{w}" for w in (5, 10, 21)]
            + [f"ma_ratio_{w}" for w in (5, 10, 20)]
            + ["hl_range", "vol_ratio"])


def download():
    df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(
            "No data returned. Check the ticker ('^KS200') and your internet connection."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    return df


def build(raw):
    df = raw.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    daydiff = df.index.to_series().diff().dt.days.fillna(9999).values
    df.loc[daydiff > GAP_DAYS, "log_ret"] = np.nan

    for l in range(1, 6):
        df[f"ret_lag_{l}"] = df["log_ret"].shift(l)
    for w in (5, 10, 21):
        df[f"rvol_{w}"] = df["log_ret"].rolling(w).std() * np.sqrt(TRADING_DAYS)
    for w in (5, 10, 20):
        df[f"ma_ratio_{w}"] = df["Close"] / df["Close"].rolling(w).mean() - 1
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    vmean = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = np.where(vmean > 0, df["Volume"] / vmean.replace(0, np.nan), 1.0)

    df["target_vol"] = df["log_ret"].shift(-1).rolling(H).std() * np.sqrt(TRADING_DAYS)

    df = df.dropna(subset=FEATURES + ["target_vol", "rvol_21"])

    q_lo, q_hi = df["rvol_21"].quantile(list(QUANTILES))
    df["regime"] = np.where(df["rvol_21"] <= q_lo, "low",
                     np.where(df["rvol_21"] <= q_hi, "mid", "high"))

    keep = ["Open", "High", "Low", "Close", "Volume", "log_ret"] + FEATURES + \
           ["target_vol", "regime"]
    return df[keep], (q_lo, q_hi)


def report(df, q_lo, q_hi):
    print(f"Rows: {len(df)}")
    print(f"Range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Regime cutoffs (annualized vol): low <= {q_lo:.3f} < mid <= {q_hi:.3f} < high")
    print("Regime counts:")
    print(df["regime"].value_counts().reindex(["low", "mid", "high"]).to_string())
    print("\nHigh-regime years (should recover 2008 and 2020 crises):")
    print(df[df["regime"] == "high"].index.year.value_counts().sort_index().to_string())


def main():
    print(f"Downloading {TICKER} ...")
    raw = download()
    df, (q_lo, q_hi) = build(raw)
    report(df, q_lo, q_hi)
    df.to_csv(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}  ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()