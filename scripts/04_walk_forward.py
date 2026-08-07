"""
脚本 04: 滚动窗口样本外回测（Walk-Forward Backtesting）

核心实验脚本: 在每个决策点用 CBR 检索历史相似状态，
回放策略，推荐最优策略，记录前向30日实现收益。

输入: data/processed/state_db_{symbol}.csv
输出: data/results/walk_forward_{symbol}.csv (每日推荐与实现收益)
      data/results/walk_forward_summary_{symbol}.json (聚合指标)
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

# 实验参数
HOLDING_PERIOD = 30       # 持有期（天）
WARMUP_DAYS = 500         # 初始训练窗口（天）
STEP_DAYS = 1             # 滚动步长（天）
PROXIMITY_THRESHOLD = 70.0  # 接近度阈值


def walk_forward_backtest(symbol: str) -> pd.DataFrame:
    """执行滚动窗口样本外回测。

    Args:
        symbol: "BTC" 或 "ETH"

    Returns:
        每日推荐记录 DataFrame
    """
    logger.info(f"开始 {symbol} 滚动窗口回测")

    # TODO: 加载状态数据库
    state_db_path = DATA_PROCESSED / f"state_db_{symbol.lower()}.csv"
    if not state_db_path.exists():
        raise FileNotFoundError(f"状态数据库不存在: {state_db_path}，请先运行 01_build_state_db.py")

    df = pd.read_csv(state_db_path, parse_dates=["date"])
    logger.info(f"已加载 {len(df)} 条日频记录: {df.date.min()} ~ {df.date.max()}")

    records = []

    # TODO: 滚动窗口循环
    # for t in range(WARMUP_DAYS, len(df) - HOLDING_PERIOD, STEP_DAYS):
    #     1. 当前状态向量 V_now = df.iloc[t]
    #     2. 历史窗口 = df.iloc[:t]  （严禁未来函数）
    #     3. 检索: Mahalanobis 距离 + 时间衰减 → H_match
    #     4. 回放: 对每个策略在 H_match 上回放30日收益
    #     5. 排行: 按 Sharpe 排序，选最优策略 θ*
    #     6. 修正: 参数适配当前市场
    #     7. 记录: (date, θ*, params, forward_30d_return)
    #     8. 留存: 30日后将实际收益反馈至案例库

    logger.warning("⚠️ 待实现: 等待算法模块和状态数据库就绪后填充")

    result_df = pd.DataFrame(records)
    return result_df


def compute_summary(result_df: pd.DataFrame) -> dict:
    """计算聚合绩效指标。"""
    if result_df.empty:
        return {}

    # TODO: 计算
    # - 年化夏普比率
    # - 年化收益率
    # - 最大回撤
    # - 胜率
    # - 平均持仓天数
    return {}


def main():
    parser = argparse.ArgumentParser(description="滚动窗口样本外回测")
    parser.add_argument("--symbol", choices=["BTC", "ETH"], required=True)
    args = parser.parse_args()

    result_df = walk_forward_backtest(args.symbol)

    # 保存明细
    output_csv = DATA_RESULTS / f"walk_forward_{args.symbol.lower()}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    logger.info(f"明细已保存: {output_csv}")

    # 保存聚合指标
    summary = compute_summary(result_df)
    output_json = DATA_RESULTS / f"walk_forward_summary_{args.symbol.lower()}.json"
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"聚合指标已保存: {output_json}")


if __name__ == "__main__":
    main()
