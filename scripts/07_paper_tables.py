"""
Script 07: Ablation study + statistical significance tests + complete remaining baselines + fill into the paper

Features:
  1. Ablation study (BTC): Full Model / No-S_mic / No-S_vol / No-Revise / No-Retain
  2. Statistical significance: Diebold-Mariano (Newey-West HAC) + Hansen SPA (bootstrap)
  3. Complete baselines: Time2Vec+k-NN (Time2Vec embedding + k-NN retrieval) and E2E-DL (small MLP)
  4. Fill Table 4/5 (including T2V/E2E rows), Table 6 (ablation), and Table 7 (significance) of paper_draft.tex
  5. Recompile the PDF

Usage:
  python3 scripts/07_paper_tables.py          # all computation + fill + compile
  python3 scripts/07_paper_tables.py --fill   # fill + compile based on existing json only
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
from scipy import stats

# ---- Reuse the 02/03 engines ----
_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parent
_QME = importlib.util.spec_from_file_location("qme", _SCRIPTS / "02_retrieve_and_replay.py")
qme = importlib.util.module_from_spec(_QME)
sys.modules["qme"] = qme
_QME.loader.exec_module(qme)

_WF = importlib.util.spec_from_file_location("wf", _SCRIPTS / "03_run_walk_forward.py")
wf = importlib.util.module_from_spec(_WF)
sys.modules["wf"] = wf
_WF.loader.exec_module(wf)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = _THIS.parent.parent
DATA_RESULTS = ROOT / "data" / "results"
PAPER_TEX = ROOT / "paper_draft.tex"

WARMUP = wf.WARMUP
HOLDING = wf.HOLDING
FEATURE_COLS = qme.FEATURE_COLS
CANDIDATES = qme.CANDIDATE_STRATEGIES

# ---- Methods and paper-table labels ----
METHOD_LABELS = {
    "cbr": "\\textbf{Our method}",
    "cosine_kline": "Cosine-KLine",
    "dtw": "DTW",
    "ms_garch": "MS-GARCH",
    "time2vec_knn": "Time2Vec+k-NN",
    "e2e_dl": "E2E-DL",
    "global_best": "Global-Best",
    "equal_weight": "Equal-Weight",
    "buy_hold": "Buy-Hold",
}
SOTA_METHODS = list(METHOD_LABELS.keys())


# ============================================================
# Data and evaluation
# ============================================================
def load_data(currency):
    return wf.load_data(currency)


def oos_indices(state_len):
    return wf.oos_state_indices(state_len)


def returns_by_date(records, market, date_to_idx) -> dict:
    """{date: realized_return}, keeping only finite values."""
    out = {}
    for date, spec in records:
        m_idx = date_to_idx.get(date)
        if m_idx is None:
            continue
        r = wf.realize_return(spec, market, m_idx)
        if np.isfinite(r):
            out[date] = r
    return out


def metrics_from_returns(rets: np.ndarray) -> dict:
    """Compute paper metrics from non-overlapping holding-period returns."""
    rets = np.clip(np.asarray(rets, float), -1.0, 1.0)
    if len(rets) == 0:
        return {"n": 0, "sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
    tau = HOLDING
    mean, std = float(rets.mean()), float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = mean / std * np.sqrt(365.0 / tau) if std > 0 else 0.0
    eq = np.cumprod(1 + rets)
    total = float(eq[-1])
    annual_return = float(total ** (365.0 / (tau * len(rets))) - 1) if total > 0 else -1.0
    win_rate = float((rets > 0).mean())
    peak = np.maximum.accumulate(eq)
    max_drawdown = float((1 - eq / peak).max())
    return {"n": int(len(rets)), "sharpe": sharpe, "annual_return": annual_return,
            "max_drawdown": max_drawdown, "win_rate": win_rate}


# ============================================================
# CBR engine (configurable ablation)
# ============================================================
def run_cbr_cfg(state, market, date_to_idx, oos_idx, feature_cols=FEATURE_COLS,
                revise=True, retain=True) -> list[tuple]:
    """CBR rolling out-of-sample, supporting ablation switches.

    revise=False: strike mapping uses a fixed reference price (median underlying over the warm-up), no daily adaptation.
    retain=False: the case library is frozen to the initial warm-up window (does not grow with history), i.e., no new cases are retained.
    """
    if not revise:
        s_ref = float(np.median(market["spot"].iloc[:WARMUP]))
    records = []
    for t in oos_idx:
        v_now = state.iloc[t][feature_cols].to_numpy(float)
        hist_lo = 0 if retain else 0
        hist_hi = t if retain else WARMUP  # no_retain: use only the initial library
        hist = state.iloc[hist_lo:hist_hi]
        if len(hist) < 2:
            continue
        try:
            match = qme.retrieve(v_now, hist, state.iloc[t]["date"], feature_cols=feature_cols)
        except np.linalg.LinAlgError:
            continue
        if match.empty:
            match = qme.retrieve(v_now, hist, state.iloc[t]["date"], feature_cols=feature_cols,
                                 threshold=0.0)
        # Stable Top-K retrieval (consistent with the main experiment)
        match = match.sort_values("combined", ascending=False).head(wf.TOP_K)
        if match.empty:
            continue
        board = _board(match, market, date_to_idx, revise, s_ref if not revise else None)
        if board is None or board.empty:
            continue
        records.append((state.iloc[t]["date"], board.iloc[0]["strategy"]))
    return records


def _board(match, market, date_to_idx, revise=True, s_ref=None):
    ranked = []
    for strat in CANDIDATES:
        rets = []
        for _, row in match.iterrows():
            idx = date_to_idx.get(row["date"])
            if idx is None or idx + HOLDING >= len(market):
                continue
            if revise:
                legs = qme.build_legs(strat, market["spot"].iloc[idx])
            else:
                legs = _fixed_legs(strat, s_ref)
            res = qme.replay_strategy(strat, legs, market, idx)
            if not np.isnan(res["ret"]):
                rets.append(res["ret"])
        agg = qme.aggregate_returns(rets)
        ranked.append({"strategy": strat, "sharpe": agg["sharpe"], "n": agg["n"]})
    df = pd.DataFrame(ranked)
    return df[df["n"] > 0].sort_values("sharpe", ascending=False) if not df.empty else None


def _fixed_legs(strategy, s_ref):
    """Strategy legs under a fixed reference price (No-Revise)."""
    if strategy == "long_straddle":
        return [qme.Leg(qme.CALL, qme.BUY, s_ref), qme.Leg(qme.PUT, qme.BUY, s_ref)]
    if strategy == "short_straddle":
        return [qme.Leg(qme.CALL, qme.SELL, s_ref), qme.Leg(qme.PUT, qme.SELL, s_ref)]
    if strategy == "covered_call":
        return [qme.Leg(qme.SPOT, qme.BUY, None), qme.Leg(qme.CALL, qme.SELL, s_ref * 1.10)]
    if strategy == "bull_call_spread":
        return [qme.Leg(qme.CALL, qme.BUY, s_ref * 0.90), qme.Leg(qme.CALL, qme.SELL, s_ref * 1.10)]
    raise ValueError(strategy)


# ============================================================
# Baselines: Time2Vec+k-NN and E2E-DL
# ============================================================
_FREQS = [2.0, 4.0, 8.0, 16.0, 32.0]  # Time2Vec periods


def _t2v_embed(seg: np.ndarray) -> np.ndarray:
    """Apply a Time2Vec-style embedding (linear time + multi-frequency sine) to a 29-length log-return window."""
    n = len(seg)
    t = np.arange(n, dtype=float)
    feats = [seg]
    for P in _FREQS:
        feats.append(np.sin(2 * np.pi * t / P))
        feats.append(np.cos(2 * np.pi * t / P))
    return np.concatenate(feats)


def run_time2vec(state, market, date_to_idx, oos_idx, top_k=wf.TOP_K) -> list[tuple]:
    """Time2Vec embedding + cosine k-NN retrieval, then replay to select a strategy."""
    spot = market["spot"].to_numpy(float)
    win = 30
    hist_start = date_to_idx[state.iloc[WARMUP]["date"]]
    records = []
    for t in oos_idx:
        m_now = date_to_idx[state.iloc[t]["date"]]
        if m_now < win:
            continue
        seg = np.diff(np.log(spot[m_now - win:m_now]))
        q = _t2v_embed(seg)
        cands = []
        for i in range(hist_start, m_now - win):
            hseg = np.diff(np.log(spot[i - win:i]))
            if len(hseg) != len(seg):
                continue
            h = _t2v_embed(hseg)
            denom = np.linalg.norm(q) * np.linalg.norm(h)
            score = float(q @ h / denom) if denom > 0 else -1.0
            cands.append((i, score))
        if not cands:
            continue
        cands.sort(key=lambda x: -x[1])
        dates = [market["date"].iloc[i] for i, _ in cands[:top_k]]
        match = pd.DataFrame({"date": pd.to_datetime(dates, utc=True)})
        board = qme.replay_and_rank(match, market, date_to_idx)
        if board.empty:
            continue
        records.append((state.iloc[t]["date"], board.iloc[0]["strategy"]))
    return records


def _mlp_predict(Xtr, ytr, Xte, hidden=16, epochs=200, lr=0.005, seed=0) -> int:
    """Small 1-hidden-layer MLP (tanh+softmax, gradient descent + weight clipping), returning the class argmax."""
    rng = np.random.default_rng(seed)
    Xtr = np.asarray(Xtr, float)
    n = len(Xtr)
    n_cls = len(CANDIDATES)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    X = (Xtr - mu) / sd
    W1 = rng.normal(0, 0.05, (X.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.05, (hidden, n_cls))
    b2 = np.zeros(n_cls)
    Y = np.eye(n_cls)[ytr]

    def clip_params():
        for P in (W1, W2, b1, b2):
            np.clip(P, -5.0, 5.0, out=P)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(epochs):
            h = np.tanh(np.clip(X @ W1 + b1, -500, 500))
            z = np.clip(h @ W2 + b2, -30, 30)
            z = z - z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            g = (p - Y) / n
            gW2 = h.T @ g; gb2 = g.sum(0)
            gh = (g @ W2.T) * (1 - h * h)
            gW1 = X.T @ gh; gb1 = gh.sum(0)
            for P, G in ((W1, gW1), (b1, gb1), (W2, gW2), (b2, gb2)):
                P -= lr * np.clip(G, -1.0, 1.0)
            clip_params()
        Xte_s = (np.asarray(Xte, float).reshape(1, -1) - mu) / sd
        h = np.tanh(np.clip(Xte_s @ W1 + b1, -500, 500))
        return int(np.argmax(h @ W2 + b2))


def run_e2e_dl(state, market, date_to_idx, oos_idx) -> list[tuple]:
    """End-to-end deep learning baseline: a small MLP learns the state -> best strategy mapping, retrained at each anchor."""
    spot = market["spot"].to_numpy(float)
    records = []
    # Precompute the forward 30-day return of each strategy for every historical state (as training labels)
    n_state = len(state)
    strat_rets = np.full((n_state, len(CANDIDATES)), np.nan)
    for i in range(n_state):
        m_idx = date_to_idx.get(state.iloc[i]["date"])
        if m_idx is None or m_idx + HOLDING >= len(market):
            continue
        for j, strat in enumerate(CANDIDATES):
            legs = qme.build_legs(strat, spot[m_idx])
            res = qme.replay_strategy(strat, legs, market, m_idx)
            if not np.isnan(res["ret"]):
                strat_rets[i, j] = res["ret"]
    for t in oos_idx:
        Xtr, ytr = [], []
        for i in range(0, t):
            if np.all(~np.isnan(strat_rets[i])):
                Xtr.append(state.iloc[i][FEATURE_COLS].to_numpy(float))
                ytr.append(int(np.argmax(strat_rets[i])))
        if len(ytr) < 30:
            continue
        pred = _mlp_predict(np.asarray(Xtr), np.asarray(ytr),
                            state.iloc[t][FEATURE_COLS].to_numpy(float))
        records.append((state.iloc[t]["date"], CANDIDATES[pred]))
    return records


# ============================================================
# Statistical significance tests
# ============================================================
def _nw_hac_var(d: np.ndarray, lag: int = 3) -> float:
    """Newey-West HAC variance estimator."""
    T = len(d)
    d = d - d.mean()
    g0 = np.mean(d * d)
    lam = 2.0 * np.sum([(1 - k / (lag + 1)) * np.mean(d[k:] * d[:-k])
                        for k in range(1, min(lag + 1, T))])
    return (g0 + lam) / T


def dm_test_pvalue(ret_cbr: dict, ret_base: dict) -> float:
    """Diebold-Mariano test p-value (H1: CBR is better than the baseline)."""
    dates = sorted(set(ret_cbr) & set(ret_base))
    if len(dates) < 3:
        return np.nan
    d = np.array([ret_cbr[dt] - ret_base[dt] for dt in dates])  # >0 means CBR is better
    T = len(d)
    var = _nw_hac_var(d)
    if var <= 0:
        return 1.0 if d.mean() <= 0 else 0.0
    stat = d.mean() / np.sqrt(var)  # one-sided: large positive => CBR better
    return float(1.0 - stats.norm.cdf(stat))


def hansen_spa_pvalue(ret_cbr: dict, base_rets: dict, n_boot: int = 2000, seed: int = 0) -> float:
    """Hansen SPA test p-value (null: no baseline beats CBR)."""
    common = sorted(set(ret_cbr).intersection(*[set(b) for b in base_rets.values()]))
    if len(common) < 3:
        return np.nan
    T = len(common)
    # d[b,t] = baseline b - CBR (positive means the baseline is better)
    keys = list(base_rets.keys())
    D = np.zeros((len(keys), T))
    for b, base in enumerate(keys):
        D[b] = np.array([base_rets[base][dt] - ret_cbr[dt] for dt in common])
    mu = D.mean(1)
    sd = D.std(1, ddof=1) + 1e-12
    tib_obs = np.sqrt(T) * mu / sd
    obs = float(tib_obs.max())
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_boot):
        idx = rng.integers(0, T, T)
        Db = D[:, idx]
        mub = Db.mean(1)
        sdb = Db.std(1, ddof=1) + 1e-12
        tib = np.sqrt(T) * (mub - mu) / sdb  # recentering under the null
        if tib.max() >= obs:
            cnt += 1
    return float(cnt / n_boot)


# ============================================================
# Main computation
# ============================================================
def run_all(currency) -> dict:
    logger.info(f"========== {currency} ==========")
    state, market, date_to_idx = load_data(currency)
    oos_idx = oos_indices(len(state))

    records = {}
    for m in ["cbr", "cosine_kline", "dtw", "ms_garch", "global_best", "equal_weight", "buy_hold"]:
        records[m] = wf.RUNNERS[m](state, market, date_to_idx, oos_idx)
    records["time2vec_knn"] = run_time2vec(state, market, date_to_idx, oos_idx)
    records["e2e_dl"] = run_e2e_dl(state, market, date_to_idx, oos_idx)

    summary = {}
    rets_by_method = {}
    for m in SOTA_METHODS:
        rd = returns_by_date(records[m], market, date_to_idx)
        rets_by_method[m] = rd
        met = metrics_from_returns(np.array(list(rd.values())))
        met["method"] = m
        summary[m] = met
        logger.info(f"  [{m}] n={met['n']} Sharpe={met['sharpe']:.3f} "
                    f"annual={met['annual_return']*100:.1f}% drawdown={met['max_drawdown']*100:.1f}% "
                    f"win_rate={met['win_rate']*100:.1f}%")

    return {"summary": summary, "rets_by_method": rets_by_method}


def run_ablation(currency="BTC") -> dict:
    logger.info(f"========== {currency} ablation study ==========")
    state, market, date_to_idx = load_data(currency)
    oos_idx = oos_indices(len(state))
    no_mic = [c for c in FEATURE_COLS if c not in ("fr", "ls", "d_oi")]
    no_vol = [c for c in FEATURE_COLS if c not in ("ivp", "vrp", "slope", "skew")]
    configs = {
        "Full Model": dict(feature_cols=FEATURE_COLS, revise=True, retain=True),
        "No-$\\mathcal{S}_{mic}$": dict(feature_cols=no_mic, revise=True, retain=True),
        "No-$\\mathcal{S}_{vol}$": dict(feature_cols=no_vol, revise=True, retain=True),
        "No-Revise": dict(feature_cols=FEATURE_COLS, revise=False, retain=True),
        "No-Retain": dict(feature_cols=FEATURE_COLS, revise=True, retain=False),
    }
    out = {}
    for name, cfg in configs.items():
        rec = run_cbr_cfg(state, market, date_to_idx, oos_idx, **cfg)
        rd = returns_by_date(rec, market, date_to_idx)
        out[name] = metrics_from_returns(np.array(list(rd.values())))
        logger.info(f"  [{name}] Sharpe={out[name]['sharpe']:.3f} "
                    f"annual={out[name]['annual_return']*100:.1f}% drawdown={out[name]['max_drawdown']*100:.1f}%")
    return out


def run_significance(sym_data: dict) -> dict:
    """sym_data: {sym: {'rets_by_method': {...}}}."""
    out = {}
    for sym, d in sym_data.items():
        rets = d["rets_by_method"]
        cbr = rets["cbr"]
        row = {}
        non_sig_baselines = []
        for m in ["cosine_kline", "dtw", "ms_garch", "time2vec_knn", "e2e_dl", "global_best"]:
            if m not in rets:
                continue
            p = dm_test_pvalue(cbr, rets[m])
            row[m] = p
            if p < 0.05:
                non_sig_baselines.append(m)
        out[sym] = row
    # Hansen SPA: joint test (using BTC)
    btc = sym_data["BTC"]["rets_by_method"]
    spa_p = hansen_spa_pvalue(btc["cbr"], {m: btc[m] for m in btc if m != "cbr"})
    out["_spa_p"] = spa_p
    logger.info(f"Hansen SPA p-value = {spa_p:.3f}")
    return out


# ============================================================
# Fill into the paper
# ============================================================
def _fmt_metrics(m: dict) -> list[str]:
    return [
        f"{m['sharpe']:.2f}",
        f"{m['annual_return'] * 100:.1f}\\%",
        f"{m['max_drawdown'] * 100:.1f}\\%",
        f"{m['win_rate'] * 100:.1f}\\%",
    ]


def _table_block(tex: str, label: str) -> tuple[int, int]:
    start = tex.index(f"\\label{{{label}}}")
    end = tex.index("\\end{table}", start) + len("\\end{table}")
    return start, end


def _replace_cells(block: str, label: str, values: list[str], has_interp: bool) -> str:
    """Replace the data cells of the label row within the block.

    has_interp=True: row format `<label> & c1..c4 & <interp> \\`, replace the 4 numeric cells and keep the last column.
    has_interp=False: row format `<label> & v1..vk \\`, replace the placeholders in order.
    """
    if has_interp:
        pat = re.compile(r"^(" + re.escape(label) + r"\s*&)(.*?)(\s*&\s*[^&]*?\s*\\\\$)", re.M)
        m = pat.search(block)
        if not m:
            return None
        return m.group(1) + " " + " & ".join(values) + m.group(3)
    pat = re.compile(r"^(" + re.escape(label) + r"\s*&)(.*?)\\\\$", re.M)
    m = pat.search(block)
    if not m:
        return None
    # Replace the data cells as a whole (compatible with placeholders and already-filled numbers)
    return m.group(1) + " " + " & ".join(values) + "\\\\"


def fill_tables(sota_summary, ablation, significance) -> None:
    tex = PAPER_TEX.read_text()
    replacements = []

    # ---- Table 4/5 (SOTA) ----
    for sym, label in [("BTC", "tab:sota_btc"), ("ETH", "tab:sota_eth")]:
        s, e = _table_block(tex, label)
        block = tex[s:e]
        for m, lbl in METHOD_LABELS.items():
            if m not in sota_summary[sym]["summary"]:
                continue
            new_row = _replace_cells(block, lbl, _fmt_metrics(sota_summary[sym]["summary"][m]),
                                     has_interp=True)
            if new_row is None:
                logger.warning(f"[{sym}] SOTA row not found: {lbl}")
                continue
            block = re.sub(r"^" + re.escape(lbl) + r".*?\\\\$",
                           lambda _m: new_row, block, count=1, flags=re.M)
        replacements.append((s, e, block))

    # ---- Table 6 (ablation, BTC) ----
    s, e = _table_block(tex, "tab:ablation_btc")
    block = tex[s:e]
    for name, met in ablation.items():
        new_row = _replace_cells(block, name, _fmt_metrics(met), has_interp=False)
        if new_row is None:
            logger.warning(f"Ablation row not found: {name}")
            continue
        block = re.sub(r"^" + re.escape(name) + r".*?\\\\$",
                       lambda _m: new_row, block, count=1, flags=re.M)
    replacements.append((s, e, block))

    # ---- Table 7 (significance) ----
    s, e = _table_block(tex, "tab:significance")
    block = tex[s:e]
    spa_p = significance["_spa_p"]
    for m, lbl in [("cosine_kline", "Cosine-KLine"), ("dtw", "DTW"), ("ms_garch", "MS-GARCH"),
                   ("time2vec_knn", "Time2Vec+k-NN"), ("e2e_dl", "E2E-DL"), ("global_best", "Global-Best")]:
        p_btc = significance["BTC"].get(m, np.nan)
        p_eth = significance["ETH"].get(m, np.nan)
        # The conclusion is based on the DM p-values of both assets
        if (np.isfinite(p_btc) and p_btc < 0.05) or (np.isfinite(p_eth) and p_eth < 0.05):
            concl = "significantly better"
        elif (np.isfinite(p_btc) and p_btc < 0.10) or (np.isfinite(p_eth) and p_eth < 0.10):
            concl = "marginally significant"
        else:
            concl = "no significant difference"
        vals = [_fmt_p(p_btc), _fmt_p(p_eth), concl]
        new_row = _replace_cells(block, lbl, vals, has_interp=False)
        if new_row is None:
            logger.warning(f"Significance row not found: {lbl}")
            continue
        block = re.sub(r"^" + re.escape(lbl) + r".*?\\\\$",
                       lambda _m: new_row, block, count=1, flags=re.M)
    replacements.append((s, e, block))

    # Apply the replacements from last to first
    for s, e, nb in sorted(replacements, key=lambda x: -x[0]):
        tex = tex[:s] + nb + tex[e:]
    PAPER_TEX.write_text(tex, encoding="utf-8")
    logger.info("Filled Table 4/5/6/7.")


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "---"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def compile_pdf() -> None:
    cwd = ROOT
    subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_draft.tex"],
                   cwd=cwd, capture_output=True, text=True)
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "paper_draft.tex"],
                       cwd=cwd, capture_output=True, text=True)
    pdf = ROOT / "paper_draft.pdf"
    logger.info(f"PDF compiled: {pdf}" if pdf.exists() else "PDF compilation failed, please check the log")
    m = re.search(r"Output written on [^\n]+\((\d+) pages", r.stdout)
    if m:
        logger.info(f"PDF pages: {m.group(1)}")


def persist(sota_summary, ablation, significance) -> None:
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    for sym, d in sota_summary.items():
        with open(DATA_RESULTS / f"sota_{sym.lower()}.json", "w") as f:
            json.dump(d["summary"], f, indent=2, ensure_ascii=False)
    with open(DATA_RESULTS / "ablation_btc.json", "w") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)
    with open(DATA_RESULTS / "significance.json", "w") as f:
        json.dump(significance, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Persisted sota_*.json / ablation_btc.json / significance.json")


def main():
    parser = argparse.ArgumentParser(description="Ablation + significance + complete baselines + fill into the paper")
    parser.add_argument("--fill", action="store_true", help="fill and compile based on existing json only")
    args = parser.parse_args()

    if args.fill:
        btc = json.loads((DATA_RESULTS / "sota_btc.json").read_text())
        eth = json.loads((DATA_RESULTS / "sota_eth.json").read_text())
        ablation = json.loads((DATA_RESULTS / "ablation_btc.json").read_text())
        significance = json.loads((DATA_RESULTS / "significance.json").read_text())
        fill_tables({"BTC": {"summary": btc}, "ETH": {"summary": eth}}, ablation, significance)
        compile_pdf()
        return

    sota = {}
    sig_in = {}
    for sym in ["BTC", "ETH"]:
        sota[sym] = run_all(sym)
        sig_in[sym] = sota[sym]
    ablation = run_ablation("BTC")
    significance = run_significance(sig_in)
    persist(sota, ablation, significance)
    fill_tables(sota, ablation, significance)
    compile_pdf()


if __name__ == "__main__":
    main()