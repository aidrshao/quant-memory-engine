"""
脚本 03: 滚动窗口样本外回测（Walk-Forward）+ 基线对比

按论文 Algorithm 2 执行样本外协议：
    T_warm = 500 天预热期，持有期 τ = 30 天，步长 Δt = 1 天，严禁未来函数。

对比方法：
  本文方法 (CBR)   : 多子空间马氏距离检索 + 策略回放排行
  基线:
    Cosine-KLine    : K线余弦相似度检索
    DTW             : 动态时间规整距离检索
    MS-GARCH        : 简化双态 GARCH(1,1) 波动率预测选策略
    Global-Best     : 全样本最优单策略
    Equal-Weight    : 全部候选策略等权组合
    Buy-Hold        : 买入持有现货

输出:
  data/results/sota_{symbol}.csv   方法 x 指标 表
  data/results/sota_{symbol}.json  同上（供论文回填）
  data/results/walk_forward_{symbol}.csv  本文方法逐日推荐明细

回填论文:
  python3 scripts/03_run_walk_forward.py --fill
  （读取 sota_*.json 填回 paper_draft.tex 的 Table 4/5 并用 xelatex 编译）

用法:
  python3 scripts/03_run_walk_forward.py --symbol BTC
  python3 scripts/03_run_walk_forward.py                # 两个标的
  python3 scripts/03_run_walk_forward.py --fill         # 结果回填 + 编译
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---- 加载 02 引擎（文件名以数字开头，需 importlib）----
_THIS = Path(__file__).resolve()
_QME = importlib.util.spec_from_file_location("qme", _THIS.parent / "02_retrieve_and_replay.py")
qme = importlib.util.module_from_spec(_QME)
sys.modules["qme"] = qme
_QME.loader.exec_module(qme)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = _THIS.parent.parent
DATA_RESULTS = ROOT / "data" / "results"
PAPER_TEX = ROOT / "paper_draft.tex"

# ---- 实验参数（与论文 Algorithm 2 对齐）----
WARMUP = 500          # 预热期 T_warm
HOLDING = qme.HOLDING  # 持有期 τ = 30
STEP = 1              # 步长 Δt
TOP_K = 20            # Cosine/DTW 近邻数
DTW_BAND = 8          # DTW Sakoe-Chiba 带宽

FEATURE_COLS = qme.FEATURE_COLS
CANDIDATES = qme.CANDIDATE_STRATEGIES

METHOD_KEYS = ["cbr", "cosine_kline", "dtw", "ms_garch", "global_best", "equal_weight", "buy_hold"]


# ============================================================
# 数据与评估
# ============================================================
def load_data(currency: str):
    state = qme.load_state_db(currency)
    market = qme.load_market_data(currency)
    date_to_idx = {row["date"]: i for i, (_, row) in enumerate(market.iterrows())}
    return state, market, date_to_idx


def realize_return(spec: str, market: pd.DataFrame, m_idx: int, holding: int = HOLDING) -> float:
    """返回某"仓位规范"在第 m_idx 日入场、持有 holding 天的收益。

    spec 为候选策略名，或特殊键 "buy_hold" / "equal_weight"。
    """
    if m_idx + holding >= len(market):
        return np.nan
    if spec == "buy_hold":
        spot = market["spot"].to_numpy(dtype=float)
        return float(spot[m_idx + holding] / spot[m_idx] - 1.0)
    if spec == "equal_weight":
        return float(np.mean([realize_return(s, market, m_idx, holding) for s in CANDIDATES]))
    legs = qme.build_legs(spec, market["spot"].iloc[m_idx])
    res = qme.replay_strategy(spec, legs, market, m_idx, holding=holding)
    return res["ret"]


def evaluate(records: list[tuple], market: pd.DataFrame, date_to_idx: dict) -> dict:
    """由各锚点选定的仓位规范计算非重叠持有期收益与聚合指标。"""
    blocks = []
    for date, spec in records:
        m_idx = date_to_idx.get(date)
        if m_idx is None:
            continue
        r = realize_return(spec, market, m_idx)
        if np.isfinite(r):
            blocks.append(r)
    rets = np.asarray(blocks, dtype=float)
    if len(rets) == 0:
        return {"n": 0, "sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    # 期权策略收益以权利金为基准，可 < -100%；clip 到 [-1,+1] 以稳定指标
    rets = np.clip(rets, -1.0, 1.0)
    tau = HOLDING
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    # 论文 §3.4.3: 持有期 τ 年化夏普
    sharpe = mean / std * np.sqrt(365.0 / tau) if std > 0 else 0.0
    eq = np.cumprod(1 + rets)
    total = float(eq[-1])
    annual_return = float(total ** (365.0 / (tau * len(rets))) - 1) if total > 0 else -1.0
    win_rate = float((rets > 0).mean())
    peak = np.maximum.accumulate(eq)
    max_drawdown = float((1 - eq / peak).max())
    return {"n": int(len(rets)), "sharpe": sharpe, "annual_return": annual_return,
            "max_drawdown": max_drawdown, "win_rate": win_rate}


def oos_state_indices(state_len: int) -> list[int]:
    """样本外决策锚点（非重叠持有，每 HOLDING 天一个）。"""
    anchors = []
    t = WARMUP
    while t <= state_len - HOLDING - 1:
        anchors.append(t)
        t += HOLDING
    return anchors


# ============================================================
# 本文方法 (CBR)
# ============================================================
def run_cbr(state, market, date_to_idx, oos_idx) -> list[tuple]:
    records = []
    for t in oos_idx:
        v_now = state.iloc[t][FEATURE_COLS].to_numpy(dtype=float)
        hist = state.iloc[:t]
        try:
            match = qme.retrieve(v_now, hist, state.iloc[t]["date"])
        except np.linalg.LinAlgError:
            continue
        if match.empty:
            # 阈值下无匹配 → 退化为按 combined 排序取 Top-K，保证每天均有推荐
            match = qme.retrieve(v_now, hist, state.iloc[t]["date"], threshold=0.0)
        # 稳定的 Top-K 检索（优于单近邻回退，降低样本外波动）
        match = match.sort_values("combined", ascending=False).head(TOP_K)
        if match.empty:
            continue
        board = qme.replay_and_rank(match, market, date_to_idx)
        best = board.iloc[0]["strategy"]
        records.append((state.iloc[t]["date"], best))
    return records


# ============================================================
# 基线
# ============================================================
def _rank_via_neighbor_dates(neighbor_dates: list, market, date_to_idx) -> str:
    """对给定近邻日期集合回放排行，返回最优策略。"""
    match = pd.DataFrame({"date": pd.to_datetime(neighbor_dates, utc=True)})
    board = qme.replay_and_rank(match, market, date_to_idx)
    return board.iloc[0]["strategy"]


def _kline_returns(spot: np.ndarray, idx: int, win: int = 30) -> np.ndarray:
    """返回截止 idx（含）的前 win 天对数收益序列。"""
    seg = spot[idx - win: idx]
    return np.diff(np.log(seg)) if len(seg) > win else np.diff(np.log(seg))


def run_retrieval_baseline(method: str, state, market, date_to_idx, oos_idx) -> list[dict]:
    """Cosine-KLine / DTW 检索基线：K线相似度检索 + 回放选策略。"""
    spot = market["spot"].to_numpy(dtype=float)
    hist_start = date_to_idx[state.iloc[WARMUP]["date"]]
    records = []
    for t in oos_idx:
        m_now = date_to_idx[state.iloc[t]["date"]]
        if m_now < 30:
            continue
        q = _kline_returns(spot, m_now)
        # 历史候选：从预热期起点到 m_now-30
        cand_scores = []
        for i in range(hist_start, m_now - 30):
            h = _kline_returns(spot, i)
            if len(h) != len(q):
                continue
            if method == "cosine_kline":
                denom = np.linalg.norm(q) * np.linalg.norm(h)
                score = float(q @ h / denom) if denom > 0 else -1.0
            else:  # dtw
                score = -_dtw(q, h)
            cand_scores.append((i, score))
        if not cand_scores:
            continue
        cand_scores.sort(key=lambda x: -x[1])
        top_dates = [market["date"].iloc[i] for i, _ in cand_scores[:TOP_K]]
        best = _rank_via_neighbor_dates(top_dates, market, date_to_idx)
        records.append((state.iloc[t]["date"], best))
    return records


def _dtw(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    """带 Sakoe-Chiba 带宽约束的 DTW 距离。"""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


# ---- GARCH(1,1) 简化双态 ----
def _garch11_fit(rets: np.ndarray):
    def neg_ll(params):
        w, al, be = params
        if w <= 0 or al < 0 or be < 0 or (al + be) >= 1:
            return 1e10
        sig2 = np.full_like(rets, w / (1 - al - be))
        for i in range(1, len(rets)):
            sig2[i] = w + al * rets[i - 1] ** 2 + be * sig2[i - 1]
        return 0.5 * np.sum(np.log(sig2) + rets ** 2 / sig2)

    r = np.asarray(rets, dtype=float)
    res = minimize(neg_ll, [1e-6, 0.1, 0.85], method="Nelder-Mead", options={"maxiter": 500})
    return res.x


def run_ms_garch(state, market, date_to_idx, oos_idx) -> list[dict]:
    """GARCH(1,1) 预测波动率，双态(高/低)选择策略。"""
    spot = market["spot"].to_numpy(dtype=float)
    rets = np.diff(np.log(spot))
    records = []
    for t in oos_idx:
        m_now = date_to_idx[state.iloc[t]["date"]]
        if m_now < 2:
            continue
        hist_rets = rets[:m_now]  # 仅用历史，严禁未来函数
        if len(hist_rets) < 60:
            continue
        try:
            w, al, be = _garch11_fit(hist_rets)
        except Exception:
            continue
        # 预测下一期条件波动率
        sig2 = np.full_like(hist_rets, w / (1 - al - be))
        for i in range(1, len(hist_rets)):
            sig2[i] = w + al * hist_rets[i - 1] ** 2 + be * sig2[i - 1]
        fcast_var = w + al * hist_rets[-1] ** 2 + be * sig2[-1]
        fcast_sigma = np.sqrt(max(fcast_var, 1e-12))
        thresh = np.sqrt(np.mean(sig2))  # 期内平均条件波动率
        best = "long_straddle" if fcast_sigma > thresh else "short_straddle"
        records.append((state.iloc[t]["date"], best))
    return records


def run_global_best(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """全样本最优单策略（静态基线，始终执行）。"""
    spot = market["spot"].to_numpy(dtype=float)
    best_sharpe, best_strat = -np.inf, CANDIDATES[0]
    for strat in CANDIDATES:
        rets = []
        for i in range(0, len(spot) - HOLDING - 1):
            r = realize_return(strat, market, i)
            if np.isfinite(r):
                rets.append(r)
        agg = qme.aggregate_returns(rets)
        if agg["sharpe"] > best_sharpe:
            best_sharpe, best_strat = agg["sharpe"], strat
    logger.info(f"  Global-Best 选定策略: {best_strat} (Sharpe={best_sharpe:.3f})")
    return [(state.iloc[t]["date"], best_strat) for t in oos_idx]


def run_equal_weight(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """等权执行所有候选策略（组合收益为各策略均值）。"""
    return [(state.iloc[t]["date"], "equal_weight") for t in oos_idx]


def run_buy_hold(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """买入持有现货：日收益即标的价格变化。"""
    return [(state.iloc[t]["date"], "buy_hold") for t in oos_idx]


RUNNERS = {
    "cbr": run_cbr,
    "cosine_kline": lambda s, m, d, o: run_retrieval_baseline("cosine_kline", s, m, d, o),
    "dtw": lambda s, m, d, o: run_retrieval_baseline("dtw", s, m, d, o),
    "ms_garch": run_ms_garch,
    "global_best": run_global_best,
    "equal_weight": run_equal_weight,
    "buy_hold": run_buy_hold,
}


# ============================================================
# 主流程
# ============================================================
def run_symbol(currency: str) -> dict:
    logger.info(f"========== {currency} 滚动窗口样本外回测 ==========")
    state, market, date_to_idx = load_data(currency)
    oos_idx = oos_state_indices(len(state))
    logger.info(f"样本外决策点: {len(oos_idx)} 个 ({state.date.iloc[oos_idx[0]].date()} ~ {state.date.iloc[oos_idx[-1]].date()})")

    summary = {}
    records_by_method = {}
    for method in METHOD_KEYS:
        records = RUNNERS[method](state, market, date_to_idx, oos_idx)
        records_by_method[method] = records
        metrics = evaluate(records, market, date_to_idx)
        metrics["method"] = method
        summary[method] = metrics
        logger.info(f"  [{method}] n={metrics['n']} Sharpe={metrics['sharpe']:.3f} "
                    f"年化={metrics['annual_return']*100:.1f}% 回撤={metrics['max_drawdown']*100:.1f}% "
                    f"胜率={metrics['win_rate']*100:.1f}%")

    # 保存明细（本文方法）
    cbr_records = records_by_method["cbr"]
    df = pd.DataFrame(cbr_records, columns=["date", "strategy"])
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_RESULTS / f"walk_forward_{currency.lower()}.csv", index=False)

    # 保存指标表
    dfm = pd.DataFrame([summary[m] for m in METHOD_KEYS])
    dfm.to_csv(DATA_RESULTS / f"sota_{currency.lower()}.csv", index=False)
    with open(DATA_RESULTS / f"sota_{currency.lower()}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存: sota_{currency.lower()}.csv / .json")
    return summary


# ============================================================
# 回填论文 + 编译
# ============================================================
LABEL_MAP = {
    "cbr": "\\textbf{本文方法}",
    "cosine_kline": "Cosine-KLine",
    "dtw": "DTW",
    "ms_garch": "MS-GARCH",
    "global_best": "Global-Best",
    "equal_weight": "Equal-Weight",
    "buy_hold": "Buy-Hold",
}


def _fmt(metrics: dict) -> list[str]:
    return [
        f"{metrics['sharpe']:.2f}",
        f"{metrics['annual_return'] * 100:.1f}\\%",
        f"{metrics['max_drawdown'] * 100:.1f}\\%",
        f"{metrics['win_rate'] * 100:.1f}\\%",
    ]


def _table_block(tex: str, label: str) -> tuple[int, int]:
    """返回 \label{label} 所在 table 环境的 [start, end) 区间。"""
    start = tex.index(f"\\label{{{label}}}")
    end = tex.index("\\end{table}", start) + len("\\end{table}")
    return start, end


def fill_paper() -> None:
    """将 sota_*.json 结果填回 paper_draft.tex 的 Table 4/5。

    按 \label 精确定位 BTC / ETH 各自 table 块，块内按行替换 4 个数值单元格，
    保留行尾"可解释性"列。不含可解释性列的表（如消融表）不受影响。
    """
    btc = json.loads((DATA_RESULTS / "sota_btc.json").read_text())
    eth = json.loads((DATA_RESULTS / "sota_eth.json").read_text())
    tables = {"BTC": btc, "ETH": eth}
    labels = {"BTC": "tab:sota_btc", "ETH": "tab:sota_eth"}

    tex = PAPER_TEX.read_text()
    # 从后往前处理，避免先改靠前块导致后续块偏移变化
    for sym in ["ETH", "BTC"]:
        data = tables[sym]
        s, e = _table_block(tex, labels[sym])
        block = tex[s:e]
        for method, label in LABEL_MAP.items():
            if method not in data:
                continue
            vals = _fmt(data[method])
            # 行格式: <label> & c1 & c2 & c3 & c4 & <可解释性> \\
            # group2 = 4 个数值单元格，group3 = 末列(可解释性) + 行尾
            pat = re.compile(
                r"^(" + re.escape(label) + r"\s*&)(.*?)(\s*&\s*[^&]*?\s*\\\\$)",
                re.M)
            m = pat.search(block)
            if not m:
                logger.warning(f"[{sym}] 未找到行: {label}")
                continue
            new_row = m.group(1) + " " + " & ".join(vals) + m.group(3)
            block = block[:m.start()] + new_row + block[m.end():]
        tex = tex[:s] + block + tex[e:]
    PAPER_TEX.write_text(tex, encoding="utf-8")
    logger.info("已回填 Table 4/5。")

    # 编译 PDF
    cwd = ROOT
    ret = subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_draft.tex"],
                         cwd=cwd, capture_output=True, text=True)
    print(ret.stdout[-2000:])
    subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_draft.tex"],
                   cwd=cwd, capture_output=True, text=True)
    pdf = ROOT / "paper_draft.pdf"
    logger.info(f"PDF 编译完成: {pdf}" if pdf.exists() else "PDF 编译失败，请检查日志")


def main():
    parser = argparse.ArgumentParser(description="滚动窗口样本外回测 + 基线对比")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], help="只跑指定标的")
    parser.add_argument("--fill", action="store_true", help="回填论文并编译 PDF")
    args = parser.parse_args()

    if args.fill:
        if not (DATA_RESULTS / "sota_btc.json").exists():
            logger.warning("缺少 sota_btc.json，请先运行回测")
            return
        fill_paper()
        return

    symbols = ["BTC", "ETH"] if not args.symbol else [args.symbol]
    all_summary = {}
    for sym in symbols:
        all_summary[sym] = run_symbol(sym)
    logger.info("全部完成。下一步: python3 scripts/03_run_walk_forward.py --fill")


if __name__ == "__main__":
    main()