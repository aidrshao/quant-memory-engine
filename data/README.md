# 数据目录说明

## 目录结构

```
data/
├── raw/          # 原始数据（从 API 下载，不入版本控制）
│   ├── deribit/  # Deribit DVOL 历史
│   ├── binance/  # Binance K线 / 资金费率 / metrics(OI+多空比)
│   └── coinglass/# (预留) Coinglass 交易所净流入
├── processed/    # 11维日频状态快照 (state_db_{btc,eth}.csv)
└── results/      # 实验输出（回测结果、图表数据）
```

## 数据覆盖要求

- 标的: BTC, ETH
- 粒度: 日频
- 历史窗口: 2023-10-05 ~ 2026-06-30（约 1000 天，由 Deribit DVOL 公开 API 的 1000 条上限决定）
- 处理后状态库: 2023-12-03 ~ 2026-06-30（941 行，丢弃 60 天滚动预热期）
- 必须包含的危机/事件期间: 2024-01 现货 ETF 获批、2024-08 日元套息平仓闪崩

## 12 维特征数据源映射

| 特征 | 数据源 | API/文件 |
|---|---|---|
| IVP | Deribit DVOL | `GET /public/get_historical_volatility` |
| VRP | Deribit IV - Binance HV | 计算: IV - HV(20d) |
| Slope | Deribit 多到期日 IV | `GET /public/get_order_book` (多 expiry) |
| Skew | Deribit put/call IV | `GET /public/get_order_book` (25-delta) |
| R_7d / R_30d | Binance 日线 | `GET /api/v3/klines` (1d) |
| RSI | Binance 日线 | 计算: 14周期 RSI |
| σ_rv | Binance 日线 | 计算: 20日年化波动率 |
| FR | Binance 永续 | `GET /fapi/v1/fundingRate` |
| ΔOI | Deribit/Binance | `GET /public/get_open_interest` / `GET /fapi/v1/openInterestHist` |
| L/S | Binance/Coinglass | `GET /futures/data/topLongShortPositionRatio` |
| NetFlow | Coinglass/Glassnode | Coinglass API / Glassnode API |

## 数据验证清单

- [ ] 无缺失日期（交易日历对齐）
- [ ] 无价格跳变（>50% 单日波动已过滤）
- [ ] 时间戳 UTC 对齐
- [ ] 12 维特征均有值（缺失率 < 5%）
