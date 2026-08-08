"""
Script 06: SOTA baseline comparison

Compare our method (V3.1 CBR) against the following baselines on the same platform:
    1. Cosine Similarity (K-Line)     — traditional K-line cosine similarity
    2. DTW (Dynamic Time Warping)     — dynamic time warping
    3. MS-GARCH                        — Markov-switching GARCH
    4. Time2Vec + k-NN                — deep embedding + nearest neighbor
    5. End-to-End DL (Tan et al.)     — end-to-end deep learning
    6. Global Best Single Strategy    — global best single strategy
    7. Equal-Weight Ensemble          — equal-weighted strategy combination
    8. Buy-and-Hold                    — buy and hold

Output: data/results/sota_comparison_{symbol}.csv
      data/results/sota_comparison_table.tex  (LaTeX table ready to be \\input)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RESULTS = Path(__file__).parent.parent / "data" / "results"

BASELINES = [
    "cosine_kline",
    "dtw",
    "ms_garch",
    "time2vec_knn",
    "e2e_dl",
    "global_best",
    "equal_weight",
    "buy_hold",
    "ours_v31",  # our method
]


def run_baseline(symbol: str, method: str) -> dict:
    """Run a single baseline method."""
    logger.info(f"[{symbol}] running baseline: {method}")

    # TODO: call the corresponding baseline implementation by method name
    return {
        "method": method,
        "symbol": symbol,
        "sharpe": None,           # to be filled
        "annual_return": None,    # to be filled
        "max_drawdown": None,     # to be filled
        "win_rate": None,         # to be filled
        "latency_ms": None,       # to be filled
        "interpretability": None, # high/medium/low
    }


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table ready to be \\input."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{SOTA baseline comparison}",
        r"\label{tab:sota}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{l c c c c c}",
        r"\toprule",
        r"Method & Sharpe & Annual return & Max drawdown & Win rate & Interpretability \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['method']} & {row['sharpe']} & {row['annual_return']} "
            f"& {row['max_drawdown']} & {row['win_rate']} & {row['interpretability']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SOTA baseline comparison")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    results = []
    for method in BASELINES:
        result = run_baseline(args.symbol, method)
        results.append(result)

    df = pd.DataFrame(results)

    # Save CSV
    csv_path = DATA_RESULTS / f"sota_comparison_{args.symbol.lower()}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path}")

    # Generate the LaTeX table
    tex = generate_latex_table(df)
    tex_path = DATA_RESULTS / "sota_comparison_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    logger.info(f"LaTeX table saved: {tex_path}")


if __name__ == "__main__":
    main()
