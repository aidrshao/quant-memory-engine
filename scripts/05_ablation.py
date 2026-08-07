"""
脚本 05: 消融实验

通过移除系统的各个组件，验证每个模块的增量贡献:
    Ablation A: 移除微观结构子空间 (S_mic) — 9维向量
    Ablation B: 移除链上资金流子空间 (S_flow) — 11维向量
    Ablation C: 移除修正模块 (Revise) — 使用固定参数
    Ablation D: 移除留存模块 (Retain) — 案例库不更新
    Full Model: 完整12维 + 4R循环

输出: data/results/ablation_{symbol}.csv
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
    "no_S_mic":    {"dims": 8,  "revise": True,  "retain": True},   # 移除 FR, LS, ΔOI
    "no_S_vol":    {"dims": 7,  "revise": True,  "retain": True},   # 移除 IVP, VRP, Slope, Skew
    "no_revise":   {"dims": 11, "revise": False, "retain": True},
    "no_retain":   {"dims": 11, "revise": True,  "retain": False},
}


def run_ablation(symbol: str, ablation_name: str, config: dict) -> dict:
    """运行单个消融配置。"""
    logger.info(f"[{symbol}] 消融实验: {ablation_name} (config={config})")

    # TODO: 调用 walk_forward_backtest，传入消融配置
    # 返回聚合指标
    return {
        "ablation": ablation_name,
        "symbol": symbol,
        "sharpe": None,        # 待填
        "annual_return": None, # 待填
        "max_drawdown": None,  # 待填
        "win_rate": None,      # 待填
    }


def main():
    parser = argparse.ArgumentParser(description="消融实验")
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
    logger.info(f"消融实验结果已保存: {output_path}")


if __name__ == "__main__":
    main()
