"""
脚本 08: 摩擦成本压力测试（Stress Test）

目的：
  回应审稿人对"BS 合成期权定价 + 单一买卖价差假设"的质疑。
  在更高买卖价差 δ_spread = 0.5% / 1.0% 下，重跑完整 CBR 检索-排行-实现
  与全部基线，验证：即使在极端流动性干涸下，CBR 对静态基线
  （Global-Best / Equal-Weight / Buy-Hold）的 Alpha 依然稳健成立。

实现：
  - 复用 02 回放引擎与 03 滚动窗口基线 runner；
  - 通过 functools.partial 将 qme.replay_strategy 的默认 spread 动态替换
    （02 内部 replay_and_rank / 03 的 realize_return 均按调用时全局解析，
    因此替换后所有策略选择与实现收益均按新价差重算）；
  - 只写独立的 stress_test.json，绝不覆盖 data/results/sota_*.json。

用法:
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

# ---- 加载 02 引擎与 03 实验（数字开头文件名，需 importlib）----
def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).parent / file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

walk = _load("walk", "03_run_walk_forward.py")
# 03 内部以 importlib 自建了独立的 qme 模块（walk.qme），
# 必须补丁 walk.qme 而非另外加载的实例，否则不生效。
qme = walk.qme

ORIG_REPLAY = qme.replay_strategy

# 压力测试价差水平：0.5% 与 1.0%（基准为 0.2%）
STRESS_SPREADS = [0.005, 0.010]
SYMBOLS = ["BTC", "ETH"]


def run_symbol_eval(currency: str, spread: float) -> dict:
    """在给定价差下重算某标的全部方法（CBR + 基线）的聚合指标，不落盘。"""
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
                    f"年化={metrics['annual_return']*100:.1f}%")
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
    logger.info(f"已保存: {out}")

    # 打印精简对比：CBR vs 静态基线（每次做空/买方压力观察）
    print("\n========== 压力测试摘要（Sharpe） ==========")
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