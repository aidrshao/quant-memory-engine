"""
脚本 06: SOTA 基线对比

将本方法 (V3.1 CBR) 与以下基线进行同台对比:
    1. Cosine Similarity (K-Line)     — 传统 K 线余弦相似度
    2. DTW (Dynamic Time Warping)     — 动态时间规整
    3. MS-GARCH                        — 马尔可夫切换 GARCH
    4. Time2Vec + k-NN                — 深度嵌入 + 最近邻
    5. End-to-End DL (Tan et al.)     — 端到端深度学习
    6. Global Best Single Strategy    — 全局最优单策略
    7. Equal-Weight Ensemble          — 等权策略组合
    8. Buy-and-Hold                    — 买入持有

输出: data/results/sota_comparison_{symbol}.csv
      data/results/sota_comparison_table.tex  (可直接 \\input 的 LaTeX 表格)
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
    "ours_v31",  # 本方法
]


def run_baseline(symbol: str, method: str) -> dict:
    """运行单个基线方法。"""
    logger.info(f"[{symbol}] 运行基线: {method}")

    # TODO: 根据方法名调用对应的基线实现
    return {
        "method": method,
        "symbol": symbol,
        "sharpe": None,           # 待填
        "annual_return": None,    # 待填
        "max_drawdown": None,     # 待填
        "win_rate": None,         # 待填
        "latency_ms": None,       # 待填
        "interpretability": None, # 高/中/低
    }


def generate_latex_table(df: pd.DataFrame) -> str:
    """生成可直接 \\input 的 LaTeX 表格。"""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{SOTA 基线对比}",
        r"\label{tab:sota}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{l c c c c c}",
        r"\toprule",
        r"方法 & 夏普比率 & 年化收益 & 最大回撤 & 胜率 & 可解释性 \\",
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
    parser = argparse.ArgumentParser(description="SOTA 基线对比")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    results = []
    for method in BASELINES:
        result = run_baseline(args.symbol, method)
        results.append(result)

    df = pd.DataFrame(results)

    # 保存 CSV
    csv_path = DATA_RESULTS / f"sota_comparison_{args.symbol.lower()}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV 已保存: {csv_path}")

    # 生成 LaTeX 表格
    tex = generate_latex_table(df)
    tex_path = DATA_RESULTS / "sota_comparison_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    logger.info(f"LaTeX 表格已保存: {tex_path}")


if __name__ == "__main__":
    main()
