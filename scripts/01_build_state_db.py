"""
脚本 01: 构建11维日频市场状态数据库

输入: data/raw/ 下的原始数据（Deribit DVOL / Binance K线 / 资金费率 / metrics）
输出: data/processed/state_db_{symbol}.csv

每行一个交易日的11维特征向量，用于后续检索与回测。

11 维特征设计（对应论文 §2.2 四子空间）:
  S_vol (波动率子空间, 4维):
    IVP    - 隐含波动率百分位 (DVOL 的滚动分位数, 观测力学)
    VRP    - 波动率风险溢价 (DVOL - HV_20d)
    Slope  - 期限结构代理 (DVOL 相对其 30 日均线的偏离, 反映期限结构抬升/下移)
    Skew   - 偏斜代理 (20日收益率三阶矩 normalized, 已实现偏度)
  S_mkt (市场子空间, 4维):
    R_7d   - 7日收益率
    R_30d  - 30日收益率
    RSI    - 相对强弱指标 (14日)
    HV     - 历史波动率 (20日年化)
  S_mic (微观结构子空间, 3维):
    FR     - 资金费率 (binance funding rate, 日均)
    LS     - 多空比 (binance metrics count_long_short_ratio)
    ΔOI    - 未平仓量变化率 (OI 日环比)

【注】Slope 与 Skew 因无多到期日/多行权价 IV 数据，采用上述基于可得数据的
代理计算（详细推导见论文 §2.2 与 data/plan.md）。NetFlow (S_flow) 暂缺，
由论文 §3.6 的 No-S_flow 消融实验覆盖。

归一化: 各特征经分位数(1%/99%) min-max 映射后 clip 至 [-1, +1]，
仅极端黑天鹅被截断到边界，保留特征区分度，防止破坏马氏距离协方差计算。
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
    """加载 Binance 现货日线 K 线。"""
    df = pd.read_csv(DATA_RAW / "binance" / f"klines_{symbol}_1d.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "close"]].sort_values("date").reset_index(drop=True)
    return df


def load_dvol(currency: str) -> pd.DataFrame:
    """加载 Deribit DVOL。"""
    df = pd.read_csv(DATA_RAW / "deribit" / f"dvol_{currency.lower()}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[["date", "dvol"]].sort_values("date").reset_index(drop=True)
    return df


def load_funding(symbol: str) -> pd.DataFrame:
    """加载 Binance 资金费率（8h, 聚合为日均）。"""
    df = pd.read_csv(DATA_RAW / "binance" / f"funding_rate_{symbol}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    daily = df.groupby("date")["fundingRate"].mean().reset_index()
    return daily.rename(columns={"fundingRate": "fr"})


def load_metrics(symbol: str) -> pd.DataFrame:
    """加载 Binance metrics（OI 与多空比）。"""
    df = pd.read_csv(DATA_RAW / "binance" / f"metrics_{symbol}.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_state_db(currency: str) -> pd.DataFrame:
    """构建指定标的的10维日频状态数据库。"""
    symbol = SYMBOL_MAP[currency]
    logger.info(f"开始构建 {currency} 10维状态数据库...")

    # 加载原始数据
    klines = load_klines(symbol)
    dvol = load_dvol(currency)
    funding = load_funding(symbol)
    metrics = load_metrics(symbol)

    # ---- 基于 K 线计算市场子空间特征 ----
    close = klines.set_index("date")["close"]
    ret = close.pct_change()

    k_features = pd.DataFrame(index=close.index)
    k_features["r_7d"] = (close / close.shift(7)) - 1
    k_features["r_30d"] = (close / close.shift(30)) - 1
    k_features["rsi"] = compute_rsi(close)
    k_features["hv"] = ret.rolling(20).std() * np.sqrt(365)  # 20日年化历史波动率
    # Skew 代理: 20日收益率三阶矩 (已实现偏度), 标准化
    k_features["skew_proxy"] = ret.rolling(20).skew()
    k_features.reset_index(inplace=True)

    # ---- 基于 DVOL 计算波动率子空间特征 ----
    dvol_df = dvol.set_index("date")["dvol"].sort_index()
    v_features = pd.DataFrame(index=dvol_df.index)
    v_features["dvol"] = dvol_df
    # IVP: DVOL 的滚动 1 年分位数 (0~1)
    v_features["ivp"] = dvol_df.rolling(365, min_periods=60).rank(pct=True)
    # HV 与 DVOL 对齐
    hv_series = k_features.set_index("date")["hv"]
    v_features["dvol"] = dvol_df
    v_features["hv"] = hv_series.reindex(dvol_df.index)
    # VRP: DVOL - HV_20d (波动率风险溢价)
    v_features["vrp"] = v_features["dvol"] - v_features["hv"]
    # Slope 代理: DVOL 相对 30 日均线的偏离 (期限结构抬升/下移)
    v_features["slope"] = v_features["dvol"] - v_features["dvol"].rolling(30).mean()
    v_features.reset_index(inplace=True)

    # ---- 基于资金费率计算微观结构特征 ----
    fr_features = funding.set_index("date")["fr"].sort_index().reset_index()

    # ---- 基于 metrics 计算 ΔOI ----
    oi_features = metrics[["date", "open_interest", "long_short_ratio"]].copy()
    oi_features["d_oi"] = oi_features["open_interest"].pct_change()  # ΔOI 日环比

    # ---- 对齐所有特征到同一日期轴 ----
    # k_features 中不再保留 hv（由 v_features 统一提供，避免 merge 列名冲突）
    k_feat_out = k_features.copy()
    k_feat_out = k_feat_out.drop(columns=["hv"], errors="ignore")
    state = v_features.merge(k_feat_out, on="date", how="outer")
    state = state.merge(fr_features, on="date", how="outer")
    state = state.merge(oi_features[["date", "d_oi", "long_short_ratio"]], on="date", how="outer")

    # ---- 裁剪到 DVOL 覆盖窗口 (2023-10-05 ~ 2026-06-30) ----
    state = state[state["dvol"].notna()].sort_values("date").reset_index(drop=True)

    # ---- 构造 11 维特征矩阵 ----
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

    # ---- 缺失值处理: 前值填充 ----
    feature_cols = ["ivp", "vrp", "slope", "skew", "r_7d", "r_30d", "rsi", "hv", "fr", "ls", "d_oi"]
    features = features.ffill()

    # ---- 丢弃窗口开头的滚动预热期（覆盖 ivp 60天最长预热）----
    # 预热期特征为 NaN（rolling 窗口不足），ffill 无法回填开头，直接丢弃
    features = features[features[feature_cols].notna().all(axis=1)].reset_index(drop=True)

    # ---- 异常值过滤: 单日收益跳变 >50% 用前值替换 ----
    for col in ["r_7d", "r_30d"]:
        mask = features[col].abs() > 0.5
        features.loc[mask, col] = features[col].shift(1)

    # ---- 归一化: 分位数 min-max 映射至 [-1, 1] ----
    # 用 1%/99% 分位数做 min-max，保留中间 98% 数据的区分度，
    # 仅极端黑天鹅被 clip 到边界，避免 z-score+clip 导致的过度平坦化。
    for col in feature_cols:
        lo, hi = features[col].quantile(0.01), features[col].quantile(0.99)
        if hi - lo < 1e-12:
            features[col] = 0.0
        else:
            features[col] = (features[col] - lo) / (hi - lo) * 2.0 - 1.0
            features[col] = features[col].clip(-1.0, 1.0)

    logger.info(f"[{currency}] 状态库: {len(features)} 行 ({features.date.min().date()} ~ {features.date.max().date()})")
    logger.info(f"[{currency}] NaN 总数: {features[feature_cols].isna().sum().sum()}")
    return features


def main():
    parser = argparse.ArgumentParser(description="构建10维日频状态数据库")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    state = build_state_db(args.symbol)
    output_path = DATA_PROCESSED / f"state_db_{args.symbol.lower()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(output_path, index=False)
    logger.info(f"已保存: {output_path}")


if __name__ == "__main__":
    main()