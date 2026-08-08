"""
Script 01b: Plot the evolution of 11-dim state features over time (data-quality check)

Checks:
  1. Temporal continuity: features evolve smoothly over time, no anomalous jumps
  2. Scale consistency: all features are within the [-1, +1] normalized space
  3. Missing values: no NaN (already guaranteed by the previous step)

Output: figures/paper/feature_evolution_{symbol}.png
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Font configuration (macOS)
from matplotlib import font_manager
for _f in ["PingFang HK", "Arial Unicode MS", "Hiragino Sans GB", "STHeiti"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
FIG_DIR = Path(__file__).parent.parent / "figures" / "paper"

FEATURE_NAMES = {
    "ivp": "IVP (IV Percentile)",
    "vrp": "VRP (IV - HV)",
    "slope": "Slope (Term-Structure Proxy)",
    "skew": "Skew (Realized Skewness)",
    "r_7d": "R_7d (7-Day Return)",
    "r_30d": "R_30d (30-Day Return)",
    "rsi": "RSI (14-day)",
    "hv": "HV (20d Annualized)",
    "fr": "FR (Funding Rate)",
    "ls": "LS (Long/Short Ratio)",
    "d_oi": "ΔOI (OI Change)",
}
SUBSPACE = {
    "ivp": "S_vol", "vrp": "S_vol", "slope": "S_vol", "skew": "S_vol",
    "r_7d": "S_mkt", "r_30d": "S_mkt", "rsi": "S_mkt", "hv": "S_mkt",
    "fr": "S_mic", "ls": "S_mic", "d_oi": "S_mic",
}
COLORS = {"S_vol": "#d62728", "S_mkt": "#1f77b4", "S_mic": "#2ca02c"}


def plot_evolution(symbol: str) -> str:
    """Plot the feature-evolution figure for one asset."""
    df = pd.read_csv(DATA_PROCESSED / f"state_db_{symbol.lower()}.csv")
    df["date"] = pd.to_datetime(df["date"])
    features = list(FEATURE_NAMES.keys())

    # Data-quality check
    issues = []
    if df[features].isna().sum().sum() > 0:
        issues.append(f"NaN found: {df[features].isna().sum().sum()} values")
    for col in features:
        if df[col].min() < -1.0 or df[col].max() > 1.0:
            issues.append(f"{col} out of [-1,1] range")
    logger.info(f"[{symbol}] data check: {'PASS' if not issues else '!!! ' + '; '.join(issues)}")

    # Dynamic subplot layout: 11 features -> 3 rows x 4 cols
    ncols = 4
    nrows = -(-len(features) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.2))
    fig.suptitle(f"{symbol} Evolution of {len(features)} Market-State Features (2023-12 ~ 2026-06)", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        ax.plot(df["date"], df[col], color=COLORS[SUBSPACE[col]], linewidth=0.8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(f"{FEATURE_NAMES[col]}\n[{SUBSPACE[col]}]", fontsize=10)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide unused empty subplots
    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = FIG_DIR / f"feature_evolution_{symbol.lower()}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"[{symbol}] figure saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    for sym in ["BTC", "ETH"]:
        plot_evolution(sym)
