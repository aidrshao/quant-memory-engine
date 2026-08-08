# Data Directory

## Structure

```
data/
├── raw/          # Raw data downloaded from public APIs (not version-controlled)
│   ├── deribit/  # Deribit DVOL history
│   └── binance/  # Binance K-lines / funding rates / metrics (OI, long-short ratio)
├── processed/    # 11-dim daily state snapshots (state_db_{btc,eth}.csv)
└── results/      # Experiment outputs (backtest results, chart data)
```

## Coverage

- Instruments: BTC, ETH
- Granularity: daily
- History window: 2023-10-05 ~ 2026-06-30 (~1000 days, limited by the Deribit DVOL public API cap)
- Processed state database: 2023-12-03 ~ 2026-06-30 (941 rows, dropping the 60-day rolling warm-up period)
- Required crisis/event periods: 2024-01 spot-ETF approval, 2024-08 yen-carry-trade unwind flash crash

## Feature → Data Source Mapping

| Feature | Source | API / File |
|---|---|---|
| IVP | Deribit DVOL | `GET /public/get_historical_volatility` |
| VRP | Deribit IV - Binance HV | computed: IV - HV(20d) |
| Slope | Deribit multi-expiry IV | `GET /public/get_order_book` (multi expiry) |
| Skew | Deribit put/call IV | `GET /public/get_order_book` (25-delta) |
| R_7d / R_30d | Binance daily | `GET /api/v3/klines` (1d) |
| RSI | Binance daily | computed: 14-period RSI |
| σ_rv | Binance daily | computed: 20-day annualized volatility |
| FR | Binance perpetual | `GET /fapi/v1/fundingRate` |
| ΔOI | Deribit/Binance | `GET /public/get_open_interest` / `GET /fapi/v1/openInterestHist` |
| L/S | Binance | `GET /futures/data/topLongShortPositionRatio` |

## Validation Checklist

- No missing dates (trading-calendar alignment)
- No price jumps (>50% single-day moves filtered)
- Timestamps aligned to UTC
- All features populated (missing rate < 5%)
