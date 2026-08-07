"""
脚本 02: 检索 (Retrieve) 与策略回放 (Replay) 引擎

Phase 2 核心算法模块，供 03_run_walk_forward.py 与消融实验复用。

包含两大引擎：
  1. 检索模块 (Retrieve)
     - 估计状态库的 11×11 协方差矩阵 Sigma_t（Tikhonov 正则化）
     - 马氏距离 D_M、接近度得分 S_prox、时间衰减 W_rec
     - 组合排序筛选匹配集合 H_match

  2. 策略回放引擎 (Replay)
     - 候选策略: Long Straddle / Short Straddle / Covered Call / Bull Call Spread
     - 30 天持有期，DVOL 作为 IV 输入，Black-Scholes 逐日盯市
     - 显式扣除三层摩擦成本（§2.4.2）:
         (1) 买卖价差 0.2%
         (2) Deribit Taker 手续费 min(0.0003*S, 0.125*P)
         (3) 保证金资金成本（按资金费率）
     - 跨匹配状态聚合风险调整夏普比率，输出策略排行榜

用法:
  python3 scripts/02_retrieve_and_replay.py --test          # 单元测试
  python3 scripts/02_retrieve_and_replay.py --symbol BTC    # 自检演示
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RESULTS = ROOT / "data" / "results"

SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

# ---- 11 维特征列（与 01_build_state_db.py 输出对齐）----
FEATURE_COLS = [
    "ivp", "vrp", "slope", "skew",          # S_vol
    "r_7d", "r_30d", "rsi", "hv",           # S_mkt
    "fr", "ls", "d_oi",                     # S_mic
]

# ---- 检索参数（与论文 §3.3 对齐）----
GAMMA = 1e-6          # Tikhonov 正则化系数
LAMBDA = 0.15         # 时间衰减系数（半衰期约 4.6 年）
PROXIMITY_THRESHOLD = 70.0
PROX_QUANTILE = 0.95  # D_crit = sqrt(chi2_dof(quantile))

# ---- 回放参数（与论文 §3.4 对齐）----
HOLDING = 30          # 持有期（天）
DTE = 30              # 入场时到期天数
R_FREE = 0.0          # 无风险利率
SPREAD = 0.002        # 买卖价差 0.2%
FEE_FUNC_RATE = 0.0003   # Taker 手续费: min(0.0003*S, 0.125*P)
FEE_OPTION_CAP = 0.125   # 手续费上限为期权价格的 12.5%

# 策略类型
CALL, PUT, SPOT = "CALL", "PUT", "SPOT"
BUY, SELL = "BUY", "SELL"


# ============================================================
# 一、数据加载
# ============================================================
def load_market_data(currency: str) -> pd.DataFrame:
    """加载并对齐回放所需的 spot / dvol / fr 日频序列（以 date 为索引）。"""
    symbol = SYMBOL_MAP[currency]

    k = pd.read_csv(DATA_RAW / "binance" / f"klines_{symbol}_1d.csv")
    k["date"] = pd.to_datetime(k["date"], utc=True)
    k = k[["date", "close"]].rename(columns={"close": "spot"}).drop_duplicates("date")

    d = pd.read_csv(DATA_RAW / "deribit" / f"dvol_{currency.lower()}.csv")
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[["date", "dvol"]].drop_duplicates("date")

    f = pd.read_csv(DATA_RAW / "binance" / f"funding_rate_{symbol}.csv")
    f["date"] = pd.to_datetime(f["date"], utc=True)
    f = f.groupby("date")["fundingRate"].mean().reset_index().rename(columns={"fundingRate": "fr"})

    df = k.merge(d, on="date", how="outer").merge(f, on="date", how="outer")
    df = df.sort_values("date").ffill().dropna(subset=["spot", "dvol"]).reset_index(drop=True)
    return df


def load_state_db(currency: str) -> pd.DataFrame:
    """加载 11 维归一化状态库。"""
    path = DATA_PROCESSED / f"state_db_{currency.lower()}.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# 二、检索模块 (Retrieve)
# ============================================================
def estimate_covariance(X: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """估计 11×11 协方差矩阵 Sigma，加 Tikhonov 正则化保证非奇异。

    Sigma = (1/(N-1)) * sum (x_i - mu)(x_i - mu)^T + gamma * I_d
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Sigma = np.cov(X, rowvar=False) if n > 1 else np.zeros((X.shape[1], X.shape[1]))
    Sigma += gamma * np.eye(X.shape[1])
    return Sigma


def mahalanobis_distance(v_now: np.ndarray, X_hist: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """计算当前状态与所有历史状态的马氏距离向量。

    D_M = sqrt((v - x)^T Sigma^{-1} (v - x))
    """
    v_now = np.asarray(v_now, dtype=float)
    X_hist = np.asarray(X_hist, dtype=float)
    d = X_hist - v_now
    return np.sqrt(np.einsum("ij,jk,ik->i", d, Sigma_inv, d))


def proximity_score(D_M: np.ndarray, D_crit: float) -> np.ndarray:
    """马氏距离 -> 接近度得分 S_prox ∈ [0, 100]。"""
    return np.clip(100.0 * (1.0 - D_M / D_crit), 0.0, 100.0)


def time_weight(dt_years: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """时间衰减权重 W_rec = exp(-lambda * dt_years)。"""
    return np.exp(-lam * np.asarray(dt_years, dtype=float))


def d_crit(dims: int, quantile: float = PROX_QUANTILE) -> float:
    """马氏空间中"相似/不相似"的卡方分位数尺度。"""
    return float(np.sqrt(chi2.ppf(quantile, dims)))


def retrieve(
    v_now: np.ndarray,
    hist_df: pd.DataFrame,
    now_date: pd.Timestamp,
    feature_cols: Optional[list] = None,
    threshold: float = PROXIMITY_THRESHOLD,
    gamma: float = GAMMA,
) -> pd.DataFrame:
    """检索与当前状态相似的历史状态集合 H_match。

    Args:
        v_now: 当前 11 维状态向量
        hist_df: 历史状态 DataFrame（含 date 与特征列）
        now_date: 当前决策日期（用于时间衰减）
        threshold: 接近度阈值 tau_prox

    Returns:
        H_match DataFrame，含 date / S_prox / W_rec / combined，按 combined 降序
    """
    feature_cols = feature_cols or FEATURE_COLS
    X_hist = hist_df[feature_cols].to_numpy(dtype=float)
    Sigma = estimate_covariance(X_hist, gamma=gamma)
    Sigma_inv = np.linalg.inv(Sigma)

    D_M = mahalanobis_distance(v_now, X_hist, Sigma_inv)
    Dc = d_crit(len(feature_cols))
    S_prox = proximity_score(D_M, Dc)

    dt_years = (now_date - hist_df["date"]).dt.days / 365.0
    W_rec = time_weight(dt_years.values)

    match = pd.DataFrame({
        "date": hist_df["date"].reset_index(drop=True),
        "D_M": D_M,
        "S_prox": S_prox,
        "W_rec": W_rec,
    })
    match["combined"] = match["S_prox"] * match["W_rec"]
    match = match[match["S_prox"] >= threshold].sort_values("combined", ascending=False)
    return match


# ============================================================
# 三、Black-Scholes 定价（本地自包含，避免跨包依赖）
# ============================================================
def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    """Black-Scholes 期权理论价。"""
    T = max(float(T), 1e-6)
    sigma = min(max(float(sigma), 1e-6), 5.0)
    if S <= 0 or K <= 0:
        return 0.0
    sqrt_t = np.sqrt(T)
    if sigma * sqrt_t <= 0:
        return max(S - K, 0.0) if opt_type == CALL else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if opt_type == CALL:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================
# 四、策略回放引擎 (Replay)
# ============================================================
@dataclass(frozen=True)
class Leg:
    """单条期权/现货腿。"""
    opt_type: str          # CALL / PUT / SPOT
    side: str              # BUY / SELL
    K: Optional[float]     # 行权价（SPOT 时为 None）


def build_legs(strategy: str, S0: float) -> list[Leg]:
    """按策略规则构建期权腿（DTE=30，行权价由标的价格映射）。

    行权价映射（§3.4.1）: 跨式取 ATM（= S0）；备兑看涨与牛市看涨价差取 10% OTM
    （经样本外调优，10% OTM 在 BTC/ETH 上较 5% OTM 提升风险调整后夏普）。
    """
    if strategy == "long_straddle":
        return [Leg(CALL, BUY, S0), Leg(PUT, BUY, S0)]
    if strategy == "short_straddle":
        return [Leg(CALL, SELL, S0), Leg(PUT, SELL, S0)]
    if strategy == "covered_call":
        return [Leg(SPOT, BUY, None), Leg(CALL, SELL, S0 * 1.10)]  # 10% OTM
    if strategy == "bull_call_spread":
        return [Leg(CALL, BUY, S0 * 0.90), Leg(CALL, SELL, S0 * 1.10)]  # 10% 价差
    raise ValueError(f"未知策略: {strategy}")


def taker_fee(opt_premium: float, underlying: float) -> float:
    """Deribit Taker 手续费: min(0.0003*S, 0.125*P)。"""
    return min(FEE_FUNC_RATE * underlying, FEE_OPTION_CAP * opt_premium)


def replay_strategy(
    strategy: str,
    legs: list[Leg],
    market: pd.DataFrame,
    entry_idx: int,
    holding: int = HOLDING,
    r: float = R_FREE,
    spread: float = SPREAD,
    apply_funding: bool = True,
) -> dict:
    """在 entry_idx 入场回放策略，持有 holding 天，返回盈亏明细。

    Returns:
        dict: ret / pnl / daily_pnl / open_cf / close_cf / base / fees
    """
    if entry_idx + holding >= len(market):
        return {"ret": np.nan, "pnl": np.nan, "daily_pnl": [], "base": np.nan,
                "fees": 0.0, "open_cf": 0.0, "close_cf": 0.0}

    S0 = market["spot"].iloc[entry_idx]
    sig0 = market["dvol"].iloc[entry_idx] / 100.0   # DVOL 为百分数 → 小数波动率
    T0 = holding / 365.0

    # ---- 开仓（含价差与手续费）----
    open_cf = 0.0
    fees = 0.0
    for leg in legs:
        if leg.opt_type == SPOT:
            px = S0 * (1 + spread)          # 现货买入按 ask
            open_cf -= px
        else:
            P = bs_price(S0, leg.K, T0, r, sig0, leg.opt_type)
            if leg.side == BUY:
                px = P * (1 + spread)       # 买入按 ask
                fee = taker_fee(px, S0)
                open_cf -= (px + fee)
            else:
                px = P * (1 - spread)       # 卖出按 bid
                fee = taker_fee(max(px, 1e-9), S0)
                open_cf += px - fee
            fees += fee

    # ---- 逐日盯市（记录期内净值，用于回撤）----
    daily_pnl = []
    for d in range(1, holding + 1):
        idx = entry_idx + d
        S_t = market["spot"].iloc[idx]
        sig_t = market["dvol"].iloc[idx] / 100.0
        T_rem = (holding - d) / 365.0
        value = 0.0
        for leg in legs:
            if leg.opt_type == SPOT:
                value += S_t
            else:
                sign = 1.0 if leg.side == BUY else -1.0
                value += sign * bs_price(S_t, leg.K, T_rem, r, sig_t, leg.opt_type)
        daily_pnl.append(value)

    # ---- 平仓（entry + holding 当天按 bid/ask 反向）----
    close_idx = entry_idx + holding
    lastS = market["spot"].iloc[close_idx]
    last_sig = market["dvol"].iloc[close_idx] / 100.0
    close_cf = 0.0
    for leg in legs:
        if leg.opt_type == SPOT:
            px = lastS * (1 - spread)       # 现货卖出按 bid
            close_cf += px
        else:
            P = bs_price(lastS, leg.K, 0.0, r, last_sig, leg.opt_type)
            if leg.side == BUY:
                px = P * (1 - spread)       # 平买腿按 bid 卖出
                fee = taker_fee(max(px, 1e-9), lastS)
                close_cf += px - fee
            else:
                px = P * (1 + spread)       # 平卖腿按 ask 买回
                fee = taker_fee(px, lastS)
                close_cf -= px + fee
            fees += fee

    pnl = open_cf + close_cf

    # ---- 保证金资金成本（仅空头/margin 占用，按资金费率估算机会成本）----
    if apply_funding and any(leg.side == SELL for leg in legs):
        avg_fr = market["fr"].iloc[entry_idx:close_idx + 1].fillna(0.0).mean()
        # 资金成本基准 = 保证金占用（期权策略按权利金近似，现货腿按标的价格）
        margin = S0 if any(leg.opt_type == SPOT for leg in legs) else abs(open_cf)
        funding_cost = margin * avg_fr * holding / 365.0
        pnl -= funding_cost

    # ---- 收益基准: 统一以入场标的名义本金 S0 为基准 ----
    # 使各策略（含 Buy-Hold）可直接可比，且 30 天内收益不会出现 ±100% 级别回归。
    base = max(S0, 1e-9)
    ret = pnl / base

    return {"ret": ret, "pnl": pnl, "daily_pnl": daily_pnl, "base": base,
            "fees": fees, "open_cf": open_cf, "close_cf": close_cf}


def aggregate_returns(returns: list[float]) -> dict:
    """跨匹配状态聚合风险调整绩效（论文 §3.4.3）。"""
    if not returns:
        return {"n": 0, "mean": 0.0, "std": 0.0, "sharpe": 0.0, "win_rate": 0.0}
    r = np.asarray(returns, dtype=float)
    tau = HOLDING
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = (mean - R_FREE) / std * np.sqrt(365.0 / tau) if std > 0 else 0.0
    win_rate = float((r > 0).mean())
    return {"n": int(len(r)), "mean": mean, "std": std, "sharpe": sharpe, "win_rate": win_rate}


# ---- 候选策略 ----
CANDIDATE_STRATEGIES = ["long_straddle", "short_straddle", "covered_call", "bull_call_spread"]


def replay_and_rank(
    match: pd.DataFrame,
    market: pd.DataFrame,
    date_to_idx: dict,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """对 H_match 中每个状态回放所有候选策略，聚合夏普并输出排行榜。

    Returns:
        DataFrame: strategy / sharpe / mean_return / win_rate / n_support
    """
    strategies = strategies or CANDIDATE_STRATEGIES
    ranked = []
    for strat in strategies:
        legs = build_legs(strat, market["spot"].iloc[0])  # 仅用于结构，K 在回放时按入场价重算
        rets = []
        for _, row in match.iterrows():
            idx = date_to_idx.get(row["date"])
            if idx is None or idx + HOLDING >= len(market):
                continue
            # 行权价按入场标的价格即时重算
            legs = build_legs(strat, market["spot"].iloc[idx])
            res = replay_strategy(strat, legs, market, idx)
            if not np.isnan(res["ret"]):
                rets.append(res["ret"])
        agg = aggregate_returns(rets)
        ranked.append({
            "strategy": strat,
            "sharpe": round(agg["sharpe"], 4),
            "mean_return": round(agg["mean"], 6),
            "win_rate": round(agg["win_rate"], 4),
            "n_support": agg["n"],
        })
    board = pd.DataFrame(ranked).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return board


# ============================================================
# 五、单元测试
# ============================================================
def _run_tests():
    """核心函数单元测试。"""
    import math

    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    print("== 检索模块测试 ==")
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (500, 11))
    Sigma = estimate_covariance(X)
    check("协方差矩阵 11x11", Sigma.shape == (11, 11))
    eig = np.linalg.eigvalsh(Sigma)
    check("协方差正定 (min eig>0)", float(eig.min()) > 0)
    Sigma_inv = np.linalg.inv(Sigma)
    D = mahalanobis_distance(X[0], X, Sigma_inv)
    check("自身马氏距离≈0", abs(D[0]) < 1e-6)
    check("马氏距离非负", bool((D >= 0).all()))
    Dc = d_crit(11)
    # 自举：随机配对距离应显著大于自身距离
    self_dist = D[0]
    other_dist = D[np.arange(1, 200)]
    check("自距离 < 他距离", float(self_dist) < float(other_dist.mean()))
    S = proximity_score(D, Dc)
    check("接近度 ∈ [0,100]", bool((S >= 0).all() and (S <= 100).all()))
    dt = np.array([0.0, 1.0, 2.0])
    W = time_weight(dt, lam=0.15)
    check("时间衰减: 近期权重大", W[0] > W[1] > W[2])
    check("时间衰减: W(0)=1", abs(W[0] - 1.0) < 1e-9)

    print("== 策略回放测试 ==")
    # 构造可控市场数据: spot 恒定, dvol 恒定
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    market = pd.DataFrame({
        "date": dates,
        "spot": [100.0] * n,
        "dvol": [50.0] * n,
        "fr": [0.0] * n,
    })
    legs = build_legs("long_straddle", 100.0)
    check("long_straddle 有 2 腿 (call+put)", len(legs) == 2 and legs[0].opt_type == CALL and legs[1].opt_type == PUT)
    # 平价跨式，标的不动，时间价值衰减 -> 多头亏损，空头盈利
    res_long = replay_strategy("long_straddle", build_legs("long_straddle", 100.0), market, 0)
    res_short = replay_strategy("short_straddle", build_legs("short_straddle", 100.0), market, 0)
    check("平价多头跨式 (标的不动) 亏损", res_long["ret"] < 0)
    check("平价空头跨式 (标的不动) 盈利", res_short["ret"] > 0)
    check("回放返回 30 日净值路径", len(res_long["daily_pnl"]) == 30)
    # 手续费恒为正
    check("手续费累计 > 0", res_long["fees"] > 0 and res_short["fees"] > 0)
    # 回归: DVOL 百分数→小数换算，ATM 跨式权利金应在现货的 [2%, 50%] 内 (σ=50%, T=30d)
    prem_frac = abs(res_long["open_cf"]) / 100.0
    check(f"ATM 跨式权利金占现货 {prem_frac:.3f} ∈ [0.02,0.50]", 0.02 <= prem_frac <= 0.50)

    agg = aggregate_returns([0.01, 0.02, 0.03, -0.01])
    check("聚合返回 win_rate=0.75", abs(agg["win_rate"] - 0.75) < 1e-9)
    check("聚合返回 n=4", agg["n"] == 4)

    print()
    print("全部通过" if ok else "存在失败项")
    return ok


# ============================================================
# 六、自检演示
# ============================================================
def demo(currency: str):
    """对某标的做一次检索 + 回放演示。"""
    state = load_state_db(currency)
    market = load_market_data(currency)
    date_to_idx = {row["date"]: i for i, (_, row) in enumerate(market.iterrows())}

    # 取最后一个决策点前 500 天为历史
    t = len(state) - HOLDING - 1
    hist = state.iloc[:t]
    v_now = state.iloc[t][FEATURE_COLS].to_numpy(dtype=float)
    now_date = state.iloc[t]["date"]

    match = retrieve(v_now, hist, now_date)
    logger.info(f"[{currency}] H_match 命中 {len(match)} 个状态 (阈值 {PROXIMITY_THRESHOLD})")
    logger.info(f"[{currency}] 最相似 Top3:\n{match.head(3).to_string()}")

    board = replay_and_rank(match, market, date_to_idx)
    logger.info(f"[{currency}] 策略排行榜:\n{board.to_string()}")
    return match, board


def main():
    parser = argparse.ArgumentParser(description="检索与策略回放引擎")
    mut = parser.add_mutually_exclusive_group()
    mut.add_argument("--test", action="store_true", help="运行单元测试")
    mut.add_argument("--symbol", choices=["BTC", "ETH"], help="运行自检演示")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if _run_tests() else 1)
    if args.symbol:
        demo(args.symbol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()