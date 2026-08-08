"""
Script 04: Rolling-window out-of-sample backtest (Walk-Forward Backtesting)

Core experiment script: at each decision point, use CBR to retrieve historically similar states,
replay the strategies, recommend the best strategy, and record the forward 30-day realized return.

Input: data/processed/state_db_{symbol}.csv
Output: data/results/walk_forward_{symbol}.csv (daily recommendations and realized returns)
      data/results/walk_forward_summary_{symbol}.json (aggregate metrics)
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_RESULTS = Path(__file__).parent.parent / "data" / "results"

# Experiment parameters
HOLDING_PERIOD = 30       # holding period (days)
WARMUP_DAYS = 500         # initial training window (days)
STEP_DAYS = 1             # rolling step (days)
PROXIMITY_THRESHOLD = 70.0  # proximity threshold


def walk_forward_backtest(symbol: str) -> pd.DataFrame:
    """Run the rolling-window out-of-sample backtest.

    Args:
        symbol: "BTC" or "ETH"

    Returns:
        DataFrame of daily recommendation records
    """
    logger.info(f"Starting {symbol} rolling-window backtest")

    # TODO: load the state database
    state_db_path = DATA_PROCESSED / f"state_db_{symbol.lower()}.csv"
    if not state_db_path.exists():
        raise FileNotFoundError(f"State database not found: {state_db_path}. Please run 01_build_state_db.py first")

    df = pd.read_csv(state_db_path, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} daily records: {df.date.min()} ~ {df.date.max()}")

    records = []

    # TODO: rolling-window loop
    # for t in range(WARMUP_DAYS, len(df) - HOLDING_PERIOD, STEP_DAYS):
    #     1. current state vector V_now = df.iloc[t]
    #     2. historical window = df.iloc[:t]  (strictly no look-ahead)
    #     3. retrieve: Mahalanobis distance + time decay -> H_match
    #     4. replay: replay 30-day returns for each strategy over H_match
    #     5. rank: sort by Sharpe, select the best strategy θ*
    #     6. adapt: adapt parameters to the current market
    #     7. record: (date, θ*, params, forward_30d_return)
    #     8. retain: feed the realized return back into the case library after 30 days

    logger.warning("TODO: to be implemented once the algorithm module and state database are ready")

    result_df = pd.DataFrame(records)
    return result_df


def compute_summary(result_df: pd.DataFrame) -> dict:
    """Compute aggregate performance metrics."""
    if result_df.empty:
        return {}

    # TODO: compute
    # - annualized Sharpe ratio
    # - annualized return
    # - maximum drawdown
    # - win rate
    # - average holding period
    return {}


def main():
    parser = argparse.ArgumentParser(description="Rolling-window out-of-sample backtest")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    result_df = walk_forward_backtest(args.symbol)

    # Save the details
    output_csv = DATA_RESULTS / f"walk_forward_{args.symbol.lower()}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Details saved: {output_csv}")

    # Save the aggregate metrics
    summary = compute_summary(result_df)
    output_json = DATA_RESULTS / f"walk_forward_summary_{args.symbol.lower()}.json"
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Aggregate metrics saved: {output_json}")


if __name__ == "__main__":
    main()
