"""
Script 05: Ablation study

Verify the incremental contribution of each module by removing components of the system:
    Ablation A: remove the microstructure subspace (S_mic) — 9-dim vector
    Ablation B: remove the on-chain flow subspace (S_flow) — 11-dim vector
    Ablation C: remove the revise module (Revise) — use fixed parameters
    Ablation D: remove the retain module (Retain) — the case library is not updated
    Full Model: full 12-dim + 4R cycle

Output: data/results/ablation_{symbol}.csv
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RESULTS = Path(__file__).parent.parent / "data" / "results"

ABLATIONS = {
    "full":        {"dims": 11, "revise": True,  "retain": True},
    "no_S_mic":    {"dims": 8,  "revise": True,  "retain": True},   # remove FR, LS, ΔOI
    "no_S_vol":    {"dims": 7,  "revise": True,  "retain": True},   # remove IVP, VRP, Slope, Skew
    "no_revise":   {"dims": 11, "revise": False, "retain": True},
    "no_retain":   {"dims": 11, "revise": True,  "retain": False},
}


def run_ablation(symbol: str, ablation_name: str, config: dict) -> dict:
    """Run a single ablation configuration."""
    logger.info(f"[{symbol}] ablation: {ablation_name} (config={config})")

    # TODO: call walk_forward_backtest, passing the ablation configuration
    # return the aggregate metrics
    return {
        "ablation": ablation_name,
        "symbol": symbol,
        "sharpe": None,        # to be filled
        "annual_return": None, # to be filled
        "max_drawdown": None,  # to be filled
        "win_rate": None,      # to be filled
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    results = []
    for name, config in ABLATIONS.items():
        result = run_ablation(args.symbol, name, config)
        results.append(result)

    df = pd.DataFrame(results)
    output_path = DATA_RESULTS / f"ablation_{args.symbol.lower()}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Ablation results saved: {output_path}")


if __name__ == "__main__":
    main()
