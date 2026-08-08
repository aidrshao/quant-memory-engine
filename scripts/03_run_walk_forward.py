"""
Script 03: Rolling-window out-of-sample backtest (Walk-Forward) + baseline comparison

Run the out-of-sample protocol according to paper Algorithm 2:
    T_warm = 500-day warm-up, holding period τ = 30 days, step Δt = 1 day, strictly no look-ahead.

Compared methods:
  Our method (CBR)   : multi-subspace Mahalanobis-distance retrieval + strategy replay ranking
  Baselines:
    Cosine-KLine    : K-line cosine-similarity retrieval
    DTW             : dynamic time warping distance retrieval
    MS-GARCH        : simplified two-state GARCH(1,1) volatility forecast for strategy selection
    Global-Best     : best single strategy over the full sample
    Equal-Weight    : equal-weighted combination of all candidate strategies
    Buy-Hold        : buy-and-hold the spot

Output:
  data/results/sota_{symbol}.csv   method x metric table
  data/results/sota_{symbol}.json  same as above (for filling into the paper)
  data/results/walk_forward_{symbol}.csv   daily recommendations of our method

Fill into the paper:
  python3 scripts/03_run_walk_forward.py --fill
  (reads sota_*.json, fills Table 4/5 of paper_draft.tex, and compiles with xelatex)

Usage:
  python3 scripts/03_run_walk_forward.py --symbol BTC
  python3 scripts/03_run_walk_forward.py                # both assets
  python3 scripts/03_run_walk_forward.py --fill         # fill results + compile
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

# ---- Load the 02 engine (filename starts with a digit, so use importlib) ----
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

# ---- Experiment parameters (aligned with paper Algorithm 2) ----
WARMUP = 500          # warm-up period T_warm
HOLDING = qme.HOLDING  # holding period τ = 30
STEP = 1              # step Δt
TOP_K = 20            # number of Cosine/DTW neighbors
DTW_BAND = 8          # DTW Sakoe-Chiba band

FEATURE_COLS = qme.FEATURE_COLS
CANDIDATES = qme.CANDIDATE_STRATEGIES

METHOD_KEYS = ["cbr", "cosine_kline", "dtw", "ms_garch", "global_best", "equal_weight", "buy_hold"]


# ============================================================
# Data and evaluation
# ============================================================
def load_data(currency: str):
    state = qme.load_state_db(currency)
    market = qme.load_market_data(currency)
    date_to_idx = {row["date"]: i for i, (_, row) in enumerate(market.iterrows())}
    return state, market, date_to_idx


def realize_return(spec: str, market: pd.DataFrame, m_idx: int, holding: int = HOLDING) -> float:
    """Return the return of a "position spec" entered at day m_idx and held for holding days.

    spec is a candidate strategy name, or a special key "buy_hold" / "equal_weight".
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
    """Compute non-overlapping holding-period returns and aggregate metrics from the position specs selected at each anchor."""
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
    # Option-strategy returns are premium-based and can be < -100%; clip to [-1,+1] to stabilize metrics
    rets = np.clip(rets, -1.0, 1.0)
    tau = HOLDING
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    # Paper §3.4.3: annualized Sharpe over the holding period τ
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
    """Out-of-sample decision anchors (non-overlapping holds, one every HOLDING days)."""
    anchors = []
    t = WARMUP
    while t <= state_len - HOLDING - 1:
        anchors.append(t)
        t += HOLDING
    return anchors


# ============================================================
# Our method (CBR)
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
            # No match under the threshold -> fall back to taking Top-K sorted by combined, ensuring a recommendation every day
            match = qme.retrieve(v_now, hist, state.iloc[t]["date"], threshold=0.0)
        # Stable Top-K retrieval (better than a single-neighbor fallback, reduces out-of-sample variance)
        match = match.sort_values("combined", ascending=False).head(TOP_K)
        if match.empty:
            continue
        board = qme.replay_and_rank(match, market, date_to_idx)
        best = board.iloc[0]["strategy"]
        records.append((state.iloc[t]["date"], best))
    return records


# ============================================================
# Baselines
# ============================================================
def _rank_via_neighbor_dates(neighbor_dates: list, market, date_to_idx) -> str:
    """Replay and rank over the given set of neighbor dates, returning the best strategy."""
    match = pd.DataFrame({"date": pd.to_datetime(neighbor_dates, utc=True)})
    board = qme.replay_and_rank(match, market, date_to_idx)
    return board.iloc[0]["strategy"]


def _kline_returns(spot: np.ndarray, idx: int, win: int = 30) -> np.ndarray:
    """Return the log-return series of the win days up to (and including) idx."""
    seg = spot[idx - win: idx]
    return np.diff(np.log(seg)) if len(seg) > win else np.diff(np.log(seg))


def run_retrieval_baseline(method: str, state, market, date_to_idx, oos_idx) -> list[dict]:
    """Cosine-KLine / DTW retrieval baselines: K-line similarity retrieval + replay to select a strategy."""
    spot = market["spot"].to_numpy(dtype=float)
    hist_start = date_to_idx[state.iloc[WARMUP]["date"]]
    records = []
    for t in oos_idx:
        m_now = date_to_idx[state.iloc[t]["date"]]
        if m_now < 30:
            continue
        q = _kline_returns(spot, m_now)
        # Historical candidates: from the warm-up start to m_now-30
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
    """DTW distance with a Sakoe-Chiba band constraint."""
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


# ---- GARCH(1,1) simplified two-state ----
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
    """GARCH(1,1) volatility forecast; select a strategy by two states (high/low)."""
    spot = market["spot"].to_numpy(dtype=float)
    rets = np.diff(np.log(spot))
    records = []
    for t in oos_idx:
        m_now = date_to_idx[state.iloc[t]["date"]]
        if m_now < 2:
            continue
        hist_rets = rets[:m_now]  # use history only, strictly no look-ahead
        if len(hist_rets) < 60:
            continue
        try:
            w, al, be = _garch11_fit(hist_rets)
        except Exception:
            continue
        # Forecast the next-period conditional volatility
        sig2 = np.full_like(hist_rets, w / (1 - al - be))
        for i in range(1, len(hist_rets)):
            sig2[i] = w + al * hist_rets[i - 1] ** 2 + be * sig2[i - 1]
        fcast_var = w + al * hist_rets[-1] ** 2 + be * sig2[-1]
        fcast_sigma = np.sqrt(max(fcast_var, 1e-12))
        thresh = np.sqrt(np.mean(sig2))  # in-period mean conditional volatility
        best = "long_straddle" if fcast_sigma > thresh else "short_straddle"
        records.append((state.iloc[t]["date"], best))
    return records


def run_global_best(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """Best single strategy over the full sample (static baseline, always executed)."""
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
    logger.info(f"  Global-Best selected strategy: {best_strat} (Sharpe={best_sharpe:.3f})")
    return [(state.iloc[t]["date"], best_strat) for t in oos_idx]


def run_equal_weight(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """Execute all candidate strategies with equal weights (portfolio return is the mean of the strategies)."""
    return [(state.iloc[t]["date"], "equal_weight") for t in oos_idx]


def run_buy_hold(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """Buy and hold the spot: the return is simply the change in the underlying price."""
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
# Main flow
# ============================================================
def run_symbol(currency: str) -> dict:
    logger.info(f"========== {currency} rolling-window out-of-sample backtest ==========")
    state, market, date_to_idx = load_data(currency)
    oos_idx = oos_state_indices(len(state))
    logger.info(f"out-of-sample decision points: {len(oos_idx)} ({state.date.iloc[oos_idx[0]].date()} ~ {state.date.iloc[oos_idx[-1]].date()})")

    summary = {}
    records_by_method = {}
    for method in METHOD_KEYS:
        records = RUNNERS[method](state, market, date_to_idx, oos_idx)
        records_by_method[method] = records
        metrics = evaluate(records, market, date_to_idx)
        metrics["method"] = method
        summary[method] = metrics
        logger.info(f"  [{method}] n={metrics['n']} Sharpe={metrics['sharpe']:.3f} "
                    f"annual={metrics['annual_return']*100:.1f}% drawdown={metrics['max_drawdown']*100:.1f}% "
                    f"win_rate={metrics['win_rate']*100:.1f}%")

    # Save the details (our method)
    cbr_records = records_by_method["cbr"]
    df = pd.DataFrame(cbr_records, columns=["date", "strategy"])
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_RESULTS / f"walk_forward_{currency.lower()}.csv", index=False)

    # Save the metrics table
    dfm = pd.DataFrame([summary[m] for m in METHOD_KEYS])
    dfm.to_csv(DATA_RESULTS / f"sota_{currency.lower()}.csv", index=False)
    with open(DATA_RESULTS / f"sota_{currency.lower()}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: sota_{currency.lower()}.csv / .json")
    return summary


# ============================================================
# Fill into the paper + compile
# ============================================================
LABEL_MAP = {
    "cbr": "\\textbf{Our method}",
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
    """Return the [start, end) range of the table environment containing \label{label}."""
    start = tex.index(f"\\label{{{label}}}")
    end = tex.index("\\end{table}", start) + len("\\end{table}")
    return start, end


def fill_paper() -> None:
    """Fill the sota_*.json results back into Table 4/5 of paper_draft.tex.

    Locate each BTC / ETH table block precisely by \label, replace the 4 numeric cells row by row,
    and keep the trailing "interpretability" column. Tables without an interpretability column
    (e.g., ablation tables) are unaffected.
    """
    btc = json.loads((DATA_RESULTS / "sota_btc.json").read_text())
    eth = json.loads((DATA_RESULTS / "sota_eth.json").read_text())
    tables = {"BTC": btc, "ETH": eth}
    labels = {"BTC": "tab:sota_btc", "ETH": "tab:sota_eth"}

    tex = PAPER_TEX.read_text()
    # Process from last to first to avoid shifting later blocks when earlier blocks change
    for sym in ["ETH", "BTC"]:
        data = tables[sym]
        s, e = _table_block(tex, labels[sym])
        block = tex[s:e]
        for method, label in LABEL_MAP.items():
            if method not in data:
                continue
            vals = _fmt(data[method])
            # row format: <label> & c1 & c2 & c3 & c4 & <interpretability> \\
            # group2 = the 4 numeric cells, group3 = the last column (interpretability) + end of row
            pat = re.compile(
                r"^(" + re.escape(label) + r"\s*&)(.*?)(\s*&\s*[^&]*?\s*\\\\$)",
                re.M)
            m = pat.search(block)
            if not m:
                logger.warning(f"[{sym}] row not found: {label}")
                continue
            new_row = m.group(1) + " " + " & ".join(vals) + m.group(3)
            block = block[:m.start()] + new_row + block[m.end():]
        tex = tex[:s] + block + tex[e:]
    PAPER_TEX.write_text(tex, encoding="utf-8")
    logger.info("Filled Table 4/5.")

    # Compile the PDF
    cwd = ROOT
    ret = subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_draft.tex"],
                         cwd=cwd, capture_output=True, text=True)
    print(ret.stdout[-2000:])
    subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_draft.tex"],
                   cwd=cwd, capture_output=True, text=True)
    pdf = ROOT / "paper_draft.pdf"
    logger.info(f"PDF compiled: {pdf}" if pdf.exists() else "PDF compilation failed, please check the log")


def main():
    parser = argparse.ArgumentParser(description="Rolling-window out-of-sample backtest + baseline comparison")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], help="run only the specified asset")
    parser.add_argument("--fill", action="store_true", help="fill into the paper and compile the PDF")
    args = parser.parse_args()

    if args.fill:
        if not (DATA_RESULTS / "sota_btc.json").exists():
            logger.warning("Missing sota_btc.json, please run the backtest first")
            return
        fill_paper()
        return

    symbols = ["BTC", "ETH"] if not args.symbol else [args.symbol]
    all_summary = {}
    for sym in symbols:
        all_summary[sym] = run_symbol(sym)
    logger.info("All done. Next step: python3 scripts/03_run_walk_forward.py --fill")


if __name__ == "__main__":
    main()