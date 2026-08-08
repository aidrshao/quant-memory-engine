"""
Script 00: Full-data automatic downloader

Download all feasible data required by the paper from free public sources into data/raw/.

Data window: 2023-10-05 to 2026-06-30 (determined by the 1000-record limit of the public DVOL API)
Assets: BTC, ETH

Data sources:
  1. Deribit DVOL          -> data/raw/deribit/dvol_{currency}.csv
  2. Binance spot K-lines  -> data/raw/binance/klines_{symbol}_1d.csv   (data.binance.vision mirror, bypasses regional restrictions)
  3. Binance funding rate  -> data/raw/binance/funding_rate_{symbol}.csv (data.binance.vision mirror)

Later steps (require a paid Key or confirmed skip):
  - Historical option-chain snapshots  -> use BS model + DVOL instead (see backtest_engine)
  - NetFlow / long-short ratio / OI history -> skipped for now (covered by ablation No-S_flow / No-S_mic)

Usage:
  python3 scripts/00_download_all.py
"""
import argparse
import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"

# Time window: determined by the 1000-record limit of the DVOL API
WINDOW_START = "2023-10-05"
WINDOW_END = "2026-06-30"

# Binance vision mirror (bypasses regional restrictions)
VISION_BASE = "https://data.binance.vision"

SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def ts_ms(date_str: str, end_of_day: bool = False) -> int:
    """Convert a date string to a millisecond timestamp."""
    fmt = "%Y-%m-%d %H:%M:%S" if end_of_day else "%Y-%m-%d"
    val = f"{date_str} 23:59:59" if end_of_day else date_str
    dt = datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ------------------------------------------------------------
# 1. Deribit DVOL
# ------------------------------------------------------------
def fetch_deribit_dvol(currency: str) -> pd.DataFrame:
    """Fetch Deribit historical DVOL daily data."""
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": ts_ms(WINDOW_START),
        "end_timestamp": ts_ms(WINDOW_END, end_of_day=True),
        "resolution": "1D",
    }
    logger.info(f"[Deribit] Fetching {currency} DVOL...")
    res = requests.get(url, params=params, timeout=60).json()
    data = res.get("result", {}).get("data", [])
    if not data:
        logger.warning(f"[Deribit] {currency} DVOL returned no data: {res}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.normalize()
    df["currency"] = currency
    df = df[["date", "currency", "close"]].rename(columns={"close": "dvol"})
    logger.info(f"[Deribit] {currency} DVOL: {len(df)} rows ({df.date.min().date()} ~ {df.date.max().date()})")
    return df


# ------------------------------------------------------------
# 2. Binance spot K-lines (vision mirror)
# ------------------------------------------------------------
def fetch_binance_klines(symbol: str) -> pd.DataFrame:
    """Download daily K-lines from data.binance.vision (monthly zip)."""
    frames = []
    start = datetime.strptime(WINDOW_START, "%Y-%m-%d")
    end = datetime.strptime(WINDOW_END, "%Y-%m-%d")

    # Iterate month by month, fixing the timestamp unit each month (ms before 2024 / ns from 2025)
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        month_str = f"{y:04d}-{m:02d}"
        url = f"{VISION_BASE}/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{month_str}.zip"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                logger.warning(f"  [Binance] {month_str} download failed: {r.status_code}")
            else:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    name = z.namelist()[0]
                    with z.open(name) as f:
                        df = pd.read_csv(
                            f, header=None,
                            names=["open_time", "open", "high", "low", "close", "volume",
                                   "close_time", "quote_vol", "count",
                                   "taker_buy_vol", "taker_buy_quote", "ignore"],
                        )
                # Determine the unit per month: median>1e15 means microseconds (from 2025), convert to milliseconds (÷1000)
                df["open_time"] = df["open_time"].astype("int64")
                if df["open_time"].median() > 1e15:
                    df["open_time"] = df["open_time"] // 1_000
                df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
                frames.append(df)
                logger.info(f"  [Binance] {symbol} {month_str}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  [Binance] {month_str} error: {e}")

        # Next month
        m += 1
        if m > 12:
            m = 1
            y += 1

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[["date", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    # Trim to the window (date is tz-aware UTC, so compare with tz-aware boundaries)
    start_ts = pd.Timestamp(WINDOW_START, tz="UTC")
    end_ts = pd.Timestamp(WINDOW_END, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].drop_duplicates("date")
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"[Binance] {symbol} K-lines: {len(df)} rows ({df.date.min().date()} ~ {df.date.max().date()})")
    return df


# ------------------------------------------------------------
# 3. Binance funding rate (vision mirror)
# ------------------------------------------------------------
def fetch_binance_funding(symbol: str) -> pd.DataFrame:
    """Download funding rate data from data.binance.vision (monthly zip)."""
    frames = []
    start = datetime.strptime(WINDOW_START, "%Y-%m-%d")
    end = datetime.strptime(WINDOW_END, "%Y-%m-%d")

    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        month_str = f"{y:04d}-{m:02d}"
        url = f"{VISION_BASE}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month_str}.zip"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                logger.warning(f"  [Binance-FR] {month_str} download failed: {r.status_code}")
            else:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    name = z.namelist()[0]
                    with z.open(name) as f:
                        df = pd.read_csv(f)  # includes headers calc_time, funding_interval_hours, last_funding_rate
                # Determine the unit per month: median>1e15 means microseconds (from 2025), convert to milliseconds (÷1000)
                df["calc_time"] = df["calc_time"].astype("int64")
                if df["calc_time"].median() > 1e15:
                    df["calc_time"] = df["calc_time"] // 1_000
                df["date"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.normalize()
                frames.append(df)
                logger.info(f"  [Binance-FR] {symbol} {month_str}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  [Binance-FR] {month_str} error: {e}")

        m += 1
        if m > 12:
            m = 1
            y += 1

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[["date", "last_funding_rate"]].rename(columns={"last_funding_rate": "fundingRate"})
    df["fundingRate"] = df["fundingRate"].astype(float)
    start_ts = pd.Timestamp(WINDOW_START, tz="UTC")
    end_ts = pd.Timestamp(WINDOW_END, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"[Binance-FR] {symbol}: {len(df)} rows ({df.date.min().date()} ~ {df.date.max().date()})")
    return df


# ------------------------------------------------------------
# 4. Binance open interest OI + long/short ratio (metrics, one file per day)
#    Multi-threaded concurrent download: 1000 small zip files, 20 threads, about 15 seconds
# ------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Cache directory: avoid re-downloading
METRICS_CACHE = ROOT / "data" / "raw" / "binance" / "metrics_cache"


def _download_single_day_metrics(day_str: str, symbol: str) -> Optional[dict]:
    """Download and parse one day of metrics, returning {date, daily_oi, daily_lsr} or None."""
    cache_file = METRICS_CACHE / f"{symbol}_{day_str}.csv"
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            return {"date": pd.Timestamp(day_str, tz="UTC"),
                    "daily_oi": float(cached["daily_oi"].iloc[0]),
                    "daily_lsr": float(cached["daily_lsr"].iloc[0])}
        except Exception:
            pass

    url = f"{VISION_BASE}/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{day_str}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        # 5-minute granularity -> daily frequency (take the daily mean)
        daily_oi = df["sum_open_interest"].mean()
        daily_lsr = df["count_long_short_ratio"].mean()
        # Write cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"daily_oi": [daily_oi], "daily_lsr": [daily_lsr]}).to_csv(cache_file, index=False)
        return {"date": pd.Timestamp(day_str, tz="UTC"),
                "daily_oi": daily_oi, "daily_lsr": daily_lsr}
    except Exception as e:
        logger.warning(f"  [Binance-MET] {day_str} error: {e}")
        return None


def fetch_binance_metrics(symbol: str) -> pd.DataFrame:
    """Concurrently download daily metrics (including OI and long/short ratio)."""
    start = datetime.strptime(WINDOW_START, "%Y-%m-%d")
    end = datetime.strptime(WINDOW_END, "%Y-%m-%d")

    # Generate the list of dates
    date_list = []
    d = start
    while d <= end:
        date_list.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    logger.info(f"[Binance-MET] {symbol}: downloading {len(date_list)} days concurrently (20 threads)...")
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for res in executor.map(lambda dt: _download_single_day_metrics(dt, symbol), date_list):
            if res is not None:
                results.append(res)

    if not results:
        return pd.DataFrame()

    agg = pd.DataFrame(results).set_index("date").sort_index()
    agg = agg.reset_index()
    agg = agg.rename(columns={"daily_oi": "open_interest", "daily_lsr": "long_short_ratio"})
    agg = agg.sort_values("date").reset_index(drop=True)
    logger.info(f"[Binance-MET] {symbol}: {len(agg)} days ({agg.date.min().date()} ~ {agg.date.max().date()})")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Full-data downloader")
    parser.add_argument("--currency", choices=["BTC", "ETH"], default=None,
                        help="Download only the specified currency, all by default")
    args = parser.parse_args()

    currencies = ["BTC", "ETH"] if args.currency is None else [args.currency]

    for cur in currencies:
        symbol = SYMBOL_MAP[cur]
        logger.info(f"========== Processing {cur} ==========")

        # Deribit DVOL
        dvol = fetch_deribit_dvol(cur)
        if not dvol.empty:
            out = RAW / "deribit" / f"dvol_{cur.lower()}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            dvol.to_csv(out, index=False)
            logger.info(f"saved: {out}")

        # Binance K-lines
        klines = fetch_binance_klines(symbol)
        if not klines.empty:
            out = RAW / "binance" / f"klines_{symbol}_1d.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            klines.to_csv(out, index=False)
            logger.info(f"saved: {out}")

        # Binance funding rate
        funding = fetch_binance_funding(symbol)
        if not funding.empty:
            out = RAW / "binance" / f"funding_rate_{symbol}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            funding.to_csv(out, index=False)
            logger.info(f"saved: {out}")

        # Binance open interest OI + long/short ratio (metrics)
        metrics = fetch_binance_metrics(symbol)
        if not metrics.empty:
            out = RAW / "binance" / f"metrics_{symbol}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(out, index=False)
            logger.info(f"saved: {out}")

    logger.info("All automatically fetchable data downloaded.")


if __name__ == "__main__":
    main()