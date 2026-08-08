"""
Script 01: Build an 11-dimensional daily-frequency market-state database

Input: raw data under data/raw/ (Deribit DVOL / Binance K-lines / funding rate / metrics)
Output: data/processed/state_db_{symbol}.csv

Each row is an 11-dimensional feature vector for one trading day, used for subsequent retrieval and backtesting.

11-dimension feature design (corresponding to the four subspaces in paper §2.2):
  S_vol (volatility subspace, 4-dim):
    IVP    - implied volatility percentile (rolling quantile of DVOL, observed dynamics)
    VRP    - volatility risk premium (DVOL - HV_20d)
    Slope  - term-structure proxy (deviation of DVOL from its 30-day moving average, reflecting term-structure shifts)
    Skew   - skew proxy (20-day third moment of returns normalized, realized skewness)
  S_mkt (market subspace, 4-dim):
    R_7d   - 7-day return
    R_30d  - 30-day return
    RSI    - relative strength index (14-day)
    HV     - historical volatility (20-day annualized)
  S_mic (microstructure subspace, 3-dim):
    FR     - funding rate (binance funding rate, daily average)
    LS     - long/short ratio (binance metrics count_long_short_ratio)
    ΔOI    - open-interest change rate (day-over-day change of OI)

Note: Since multi-expiry/multi-strike IV data are unavailable, Slope and Skew use the
proxy calculations above based on available data (detailed derivation in paper §2.2 and
data/plan.md). NetFlow (S_flow) is currently missing and is covered by the No-S_flow
ablation experiment in paper §3.6.

Normalization: each feature is mapped via a quantile (1%/99%) min-max and clipped to [-1, +1];
only extreme black-swan events are truncated to the boundary, preserving feature discrimination
and avoiding damage to the Mahalanobis-distance covariance computation.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def load_klines(symbol: str) -> pd.DataFrame:
    """Load Binance spot daily K-lines."""
    df = pd.read_csv(DATA_RAW / "binance" / f"klines_{symbol}_1d.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
    return df


def load_dvol(currency: str) -> pd.DataFrame:
    """Load Deribit DVOL."""
    df = pd.read_csv(DATA_RAW / "deribit" / f"dvol_{currency.lower()}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "dvol"]].sort_values("date").reset_index(drop=True)
    return df


def load_funding(symbol: str) -> pd.DataFrame:
    """Load Binance funding rate (8h, aggregated to daily average)."""
    df = pd.read_csv(DATA_RAW / "binance" / f"funding_rate_{symbol}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    daily = df.groupby("date")["fundingRate"].mean().reset_index()
    return daily.rename(columns={"fundingRate": "fr"})


def load_metrics(symbol: str) -> pd.DataFrame:
    """Load Binance metrics (OI and long/short ratio)."""
    df = pd.read_csv(DATA_RAW / "binance" / f"metrics_{symbol}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute the RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_state_db(currency: str) -> pd.DataFrame:
    """Build the daily-frequency state database for the given asset."""
    symbol = SYMBOL_MAP[currency]
    logger.info(f"Building the {currency} state database...")

    # Load raw data
    klines = load_klines(symbol)
    dvol = load_dvol(currency)
    funding = load_funding(symbol)
    metrics = load_metrics(symbol)

    # ---- Compute market-subspace features from K-lines ----
    close = klines.set_index("date")["close"]
    ret = close.pct_change()

    k_features = pd.DataFrame(index=close.index)
    k_features["r_7d"] = (close / close.shift(7)) - 1
    k_features["r_30d"] = (close / close.shift(30)) - 1
    k_features["rsi"] = compute_rsi(close)
    k_features["hv"] = ret.rolling(20).std() * np.sqrt(365)  # 20-day annualized historical volatility
    # Skew proxy: 20-day third moment of returns (realized skewness), standardized
    k_features["skew_proxy"] = ret.rolling(20).skew()
    k_features.reset_index(inplace=True)

    # ---- Compute volatility-subspace features from DVOL ----
    dvol_df = dvol.set_index("date")["dvol"].sort_index()
    v_features = pd.DataFrame(index=dvol_df.index)
    v_features["dvol"] = dvol_df
    # IVP: rolling 1-year quantile of DVOL (0~1)
    v_features["ivp"] = dvol_df.rolling(365, min_periods=60).rank(pct=True)
    # Align HV with DVOL
    hv_series = k_features.set_index("date")["hv"]
    v_features["dvol"] = dvol_df
    v_features["hv"] = hv_series.reindex(dvol_df.index)
    # VRP: DVOL - HV_20d (volatility risk premium)
    v_features["vrp"] = v_features["dvol"] - v_features["hv"]
    # Slope proxy: deviation of DVOL from its 30-day moving average (term-structure shifts)
    v_features["slope"] = v_features["dvol"] - v_features["dvol"].rolling(30).mean()
    v_features.reset_index(inplace=True)

    # ---- Compute microstructure features from the funding rate ----
    fr_features = funding.set_index("date")["fr"].sort_index().reset_index()

    # ---- Compute ΔOI from metrics ----
    oi_features = metrics[["date", "open_interest", "long_short_ratio"]].copy()
    oi_features["d_oi"] = oi_features["open_interest"].pct_change()  # ΔOI day-over-day change

    # ---- Align all features to a common date axis ----
    # hv is no longer kept in k_features (provided uniformly by v_features to avoid merge column-name conflicts)
    k_feat_out = k_features.copy()
    k_feat_out = k_feat_out.drop(columns=["hv"], errors="ignore")
    state = v_features.merge(k_feat_out, on="date", how="outer")
    state = state.merge(fr_features, on="date", how="outer")
    state = state.merge(oi_features[["date", "d_oi", "long_short_ratio"]], on="date", how="outer")

    # ---- Trim to the DVOL coverage window (2023-10-05 ~ 2026-06-30) ----
    state = state[state["dvol"].notna()].sort_values("date").reset_index(drop=True)

    # ---- Build the 11-dimension feature matrix ----
    features = pd.DataFrame({
        "date": state["date"],
        # S_vol
        "ivp": state["ivp"],
        "vrp": state["vrp"],
        "slope": state["slope"],
        "skew": state["skew_proxy"],
        # S_mkt
        "r_7d": state["r_7d"],
        "r_30d": state["r_30d"],
        "rsi": state["rsi"],
        "hv": state["hv"],
        # S_mic
        "fr": state["fr"],
        "ls": state["long_short_ratio"],
        "d_oi": state["d_oi"],
    })

    # ---- Missing-value handling: forward fill ----
    feature_cols = ["ivp", "vrp", "slope", "skew", "r_7d", "r_30d", "rsi", "hv", "fr", "ls", "d_oi"]
    features = features.ffill()

    # ---- Drop the rolling warm-up period at the start of the window (covers the longest 60-day warm-up of ivp) ----
    # Warm-up features are NaN (insufficient rolling window), ffill cannot backfill the head, so drop them
    features = features[features[feature_cols].notna().all(axis=1)].reset_index(drop=True)

    # ---- Outlier filtering: single-day return jumps >50% are replaced with the previous value ----
    for col in ["r_7d", "r_30d"]:
        mask = features[col].abs() > 0.5
        features.loc[mask, col] = features[col].shift(1)

    # ---- Normalization: quantile min-max mapping to [-1, 1] ----
    # Use the 1%/99% quantiles for min-max, preserving the discrimination of the middle 98% of data;
    # only extreme black-swan events are clipped to the boundary, avoiding the over-flattening caused by z-score+clip.
    for col in feature_cols:
        lo, hi = features[col].quantile(0.01), features[col].quantile(0.99)
        if hi - lo < 1e-12:
            features[col] = 0.0
        else:
            features[col] = (features[col] - lo) / (hi - lo) * 2.0 - 1.0
            features[col] = features[col].clip(-1.0, 1.0)

    logger.info(f"[{currency}] state database: {len(features)} rows ({features.date.min().date()} ~ {features.date.max().date()})")
    logger.info(f"[{currency}] total NaN: {features[feature_cols].isna().sum().sum()}")
    return features


def main():
    parser = argparse.ArgumentParser(description="Build the daily-frequency state database")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    state = build_state_db(args.symbol)
    output_path = DATA_PROCESSED / f"state_db_{args.symbol.lower()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()