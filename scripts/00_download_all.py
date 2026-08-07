"""
脚本 00: 全量数据自动下载器

从免费公开源下载论文所需的全部可行数据，存入 data/raw/。

数据窗口: 2023-10-05 至 2026-06-30（DVOL 公开 API 的 1000 条上限决定）
标的: BTC, ETH

数据源:
  1. Deribit DVOL          -> data/raw/deribit/dvol_{currency}.csv
  2. Binance 现货 K线      -> data/raw/binance/klines_{symbol}_1d.csv   (data.binance.vision 镜像, 绕过区
  3. Binance 资金费率      -> data/raw/binance/funding_rate_{symbol}.csv (data.binance.vision 镜像)

后续 (需付费 Key 或已确认跳过):
  - 历史期权链快照  -> 改用 BS 模型 + DVOL 推算 (见 backtest_engine)
  - NetFlow / 多空比 / OI 历史 -> 暂时跳过 (消融实验 No-S_flow / No-S_mic 覆盖)

用法:
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

# 时间窗口: DVOL API 1000 条上限决定
WINDOW_START = "2023-10-05"
WINDOW_END = "2026-06-30"

# Binance vision 镜像（绕过地区封锁）
VISION_BASE = "https://data.binance.vision"

SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def ts_ms(date_str: str, end_of_day: bool = False) -> int:
    """日期字符串 -> 毫秒时间戳。"""
    fmt = "%Y-%m-%d %H:%M:%S" if end_of_day else "%Y-%m-%d"
    val = f"{date_str} 23:59:59" if end_of_day else date_str
    dt = datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ------------------------------------------------------------
# 1. Deribit DVOL
# ------------------------------------------------------------
def fetch_deribit_dvol(currency: str) -> pd.DataFrame:
    """抓取 Deribit 历史 DVOL 日线。"""
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": ts_ms(WINDOW_START),
        "end_timestamp": ts_ms(WINDOW_END, end_of_day=True),
        "resolution": "1D",
    }
    logger.info(f"[Deribit] 抓取 {currency} DVOL...")
    res = requests.get(url, params=params, timeout=60).json()
    data = res.get("result", {}).get("data", [])
    if not data:
        logger.warning(f"[Deribit] {currency} DVOL 无数据返回: {res}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.normalize()
    df["currency"] = currency
    df = df[["date", "currency", "close"]].rename(columns={"close": "dvol"})
    logger.info(f"[Deribit] {currency} DVOL: {len(df)} 条 ({df.date.min().date()} ~ {df.date.max().date()})")
    return df


# ------------------------------------------------------------
# 2. Binance 现货 K 线 (vision 镜像)
# ------------------------------------------------------------
def fetch_binance_klines(symbol: str) -> pd.DataFrame:
    """从 data.binance.vision 下载日线 K 线 (按月 zip)。"""
    frames = []
    start = datetime.strptime(WINDOW_START, "%Y-%m-%d")
    end = datetime.strptime(WINDOW_END, "%Y-%m-%d")

    # 逐月遍历，逐月修正时间戳单位（2024前毫秒 / 2025起纳秒）
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        month_str = f"{y:04d}-{m:02d}"
        url = f"{VISION_BASE}/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{month_str}.zip"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                logger.warning(f"  [Binance] {month_str} 下载失败: {r.status_code}")
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
                # 逐月判断单位：median>1e15 为微秒(2025起)，转毫秒(÷1000)
                df["open_time"] = df["open_time"].astype("int64")
                if df["open_time"].median() > 1e15:
                    df["open_time"] = df["open_time"] // 1_000
                df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
                frames.append(df)
                logger.info(f"  [Binance] {symbol} {month_str}: {len(df)} 行")
        except Exception as e:
            logger.warning(f"  [Binance] {month_str} 异常: {e}")

        # 下月
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
    # 裁剪到窗口 (date 为 tz-aware UTC, 需用 tz-aware 边界比较)
    start_ts = pd.Timestamp(WINDOW_START, tz="UTC")
    end_ts = pd.Timestamp(WINDOW_END, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].drop_duplicates("date")
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"[Binance] {symbol} K线: {len(df)} 条 ({df.date.min().date()} ~ {df.date.max().date()})")
    return df


# ------------------------------------------------------------
# 3. Binance 资金费率 (vision 镜像)
# ------------------------------------------------------------
def fetch_binance_funding(symbol: str) -> pd.DataFrame:
    """从 data.binance.vision 下载资金费率 (按月 zip)。"""
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
                logger.warning(f"  [Binance-FR] {month_str} 下载失败: {r.status_code}")
            else:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    name = z.namelist()[0]
                    with z.open(name) as f:
                        df = pd.read_csv(f)  # 自带表头 calc_time, funding_interval_hours, last_funding_rate
                # 逐月判断单位：median>1e15 为微秒(2025起)，转毫秒(÷1000)
                df["calc_time"] = df["calc_time"].astype("int64")
                if df["calc_time"].median() > 1e15:
                    df["calc_time"] = df["calc_time"] // 1_000
                df["date"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.normalize()
                frames.append(df)
                logger.info(f"  [Binance-FR] {symbol} {month_str}: {len(df)} 行")
        except Exception as e:
            logger.warning(f"  [Binance-FR] {month_str} 异常: {e}")

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
    logger.info(f"[Binance-FR] {symbol}: {len(df)} 条 ({df.date.min().date()} ~ {df.date.max().date()})")
    return df


# ------------------------------------------------------------
# 4. Binance 未平仓量 OI + 多空比 (metrics, 每日一个文件)
#    多线程并发下载：1000 个小 zip 文件，20 线程，约 15 秒
# ------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# 缓存目录：避免重复下载
METRICS_CACHE = ROOT / "data" / "raw" / "binance" / "metrics_cache"


def _download_single_day_metrics(day_str: str, symbol: str) -> Optional[dict]:
    """下载并解析单日 metrics，返回 {date, daily_oi, daily_lsr} 或 None。"""
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
        # 5 分钟粒度 -> 日频（取日均值）
        daily_oi = df["sum_open_interest"].mean()
        daily_lsr = df["count_long_short_ratio"].mean()
        # 写缓存
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"daily_oi": [daily_oi], "daily_lsr": [daily_lsr]}).to_csv(cache_file, index=False)
        return {"date": pd.Timestamp(day_str, tz="UTC"),
                "daily_oi": daily_oi, "daily_lsr": daily_lsr}
    except Exception as e:
        logger.warning(f"  [Binance-MET] {day_str} 异常: {e}")
        return None


def fetch_binance_metrics(symbol: str) -> pd.DataFrame:
    """多线程并发下载每日 metrics（含 OI 与多空比）。"""
    start = datetime.strptime(WINDOW_START, "%Y-%m-%d")
    end = datetime.strptime(WINDOW_END, "%Y-%m-%d")

    # 生成日期列表
    date_list = []
    d = start
    while d <= end:
        date_list.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    logger.info(f"[Binance-MET] {symbol}: 并发下载 {len(date_list)} 天 (20 线程)...")
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
    logger.info(f"[Binance-MET] {symbol}: {len(agg)} 天 ({agg.date.min().date()} ~ {agg.date.max().date()})")
    return agg


def main():
    parser = argparse.ArgumentParser(description="全量数据下载器")
    parser.add_argument("--currency", choices=["BTC", "ETH"], default=None,
                        help="只下载指定币种，默认全部")
    args = parser.parse_args()

    currencies = ["BTC", "ETH"] if args.currency is None else [args.currency]

    for cur in currencies:
        symbol = SYMBOL_MAP[cur]
        logger.info(f"========== 处理 {cur} ==========")

        # Deribit DVOL
        dvol = fetch_deribit_dvol(cur)
        if not dvol.empty:
            out = RAW / "deribit" / f"dvol_{cur.lower()}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            dvol.to_csv(out, index=False)
            logger.info(f"saved: {out}")

        # Binance K线
        klines = fetch_binance_klines(symbol)
        if not klines.empty:
            out = RAW / "binance" / f"klines_{symbol}_1d.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            klines.to_csv(out, index=False)
            logger.info(f"saved: {out}")

        # Binance 资金费率
        funding = fetch_binance_funding(symbol)
        if not funding.empty:
            out = RAW / "binance" / f"funding_rate_{symbol}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            funding.to_csv(out, index=False)
            logger.info(f"saved: {out}")

        # Binance 未平仓量 OI + 多空比 (metrics)
        metrics = fetch_binance_metrics(symbol)
        if not metrics.empty:
            out = RAW / "binance" / f"metrics_{symbol}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(out, index=False)
            logger.info(f"saved: {out}")

    logger.info("全部可自动获取的数据下载完成。")


if __name__ == "__main__":
    main()