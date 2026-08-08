"""
Script 08: Friction-cost stress test (Stress Test)

Purpose:
  Respond to the reviewers' concern about the "BS synthetic option pricing + single bid-ask spread" assumption.
  Re-run the full CBR retrieve-rank-realize pipeline and all baselines under higher bid-ask spreads
  δ_spread = 0.5% / 1.0%, verifying that even under extreme liquidity drought, CBR's Alpha over the
  static baselines (Global-Best / Equal-Weight / Buy-Hold) still holds robustly.

Implementation:
  - Reuse the 02 replay engine and the 03 rolling-window baseline runners;
  - Use functools.partial to dynamically replace the default spread of qme.replay_strategy
    (02's internal replay_and_rank / 03's realize_return both resolve globally at call time,
    so after the replacement all strategy selection and realized returns are recomputed with the new spread);
  - Only write an independent stress_test.json, never overwriting data/results/sota_*.json.

Usage:
  python3 scripts/08_stress_test.py
"""
from __future__ import annotations

import functools
import importlib.util
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ---- Load the 02 engine and the 03 experiments (digit-prefixed filenames, need importlib) ----
def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).parent / file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

walk = _load("walk", "03_run_walk_forward.py")
# 03 builds its own independent qme module via importlib (walk.qme);
# we must patch walk.qme rather than a separately loaded instance, otherwise it has no effect.
qme = walk.qme

ORIG_REPLAY = qme.replay_strategy

# Stress-test spread levels: 0.5% and 1.0% (the baseline is 0.2%)
STRESS_SPREADS = [0.005, 0.010]
SYMBOLS = ["BTC", "ETH"]


def run_symbol_eval(currency: str, spread: float) -> dict:
    """Recompute the aggregate metrics of all methods (CBR + baselines) for an asset under a given spread, without persisting."""
    qme.replay_strategy = functools.partial(ORIG_REPLAY, spread=spread)
    state, market, date_to_idx = walk.load_data(currency)
    oos_idx = walk.oos_state_indices(len(state))
    summary = {}
    for method in walk.METHOD_KEYS:
        records = walk.RUNNERS[method](state, market, date_to_idx, oos_idx)
        metrics = walk.evaluate(records, market, date_to_idx)
        metrics["method"] = method
        summary[method] = metrics
        logger.info(f"  [{currency}][spread={spread*100:.1f}%][{method}] "
                    f"n={metrics['n']} Sharpe={metrics['sharpe']:.3f} "
                    f"annual={metrics['annual_return']*100:.1f}%")
    return summary


def main():
    results = {"baseline_spread": qme.SPREAD, "stress": {}}
    for spread in STRESS_SPREADS:
        results["stress"][f"{spread:.3f}"] = {}
        for sym in SYMBOLS:
            logger.info(f"===== {sym} @ spread={spread*100:.1f}% =====")
            results["stress"][f"{spread:.3f}"][sym] = run_symbol_eval(sym, spread)

    out = ROOT / "data" / "results" / "stress_test.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {out}")

    # Print a compact comparison: CBR vs static baselines (per stress observation)
    print("\n========== Stress-test summary (Sharpe) ==========")
    for spread in STRESS_SPREADS:
        for sym in SYMBOLS:
            d = results["stress"][f"{spread:.3f}"][sym]
            cbr = d["cbr"]["sharpe"]
            gb = d["global_best"]["sharpe"]
            ew = d["equal_weight"]["sharpe"]
            bh = d["buy_hold"]["sharpe"]
            print(f"  spread={spread*100:.1f}%  {sym}: CBR={cbr:6.2f} | "
                  f"Global-Best={gb:6.2f} Equal-Weight={ew:6.2f} Buy-Hold={bh:6.2f}")


if __name__ == "__main__":
    main()