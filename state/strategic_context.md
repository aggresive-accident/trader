## Strategic Context

Generated: 2026-01-30 19:11:57

This file describes the full state of the trader system for cross-session strategic context.

---

## Strategy Zoo

### Active Strategies

| Strategy | Allocation | Stop ATR× | Trailing | MA Exit | Giveback |
|----------|-----------|-----------|----------|---------|----------|
| momentum | 50% | 0 | no | off | 0 |
| bollinger | 50% | 0 | no | off | 0 |

**Max positions:** 4
**Risk per trade:** 0.03
**Universe:** META, NVDA, AMD, GOOGL, AAPL, MSFT, AMZN, XOM, XLE, TSLA

---

## Current Positions

| Symbol | Qty | Entry | Current | P/L | P/L% | Strategy |
|--------|-----|-------|---------|-----|------|----------|
| AMAT | 9.0 | $334.93 | $328.28 | $-59.85 | -1.99% | unknown |
| AMD | 12.0 | $241.06 | $239.70 | $-16.35 | -0.57% | unknown |
| DOW | 110.0 | $27.58 | $27.68 | $+10.45 | +0.34% | unknown |
| FCX | 50.0 | $61.70 | $60.51 | $-59.69 | -1.93% | unknown |
| GME | 420.0 | $23.80 | $23.95 | $+63.00 | +0.63% | unknown |
| KLAC | 2.0 | $1575.71 | $1451.64 | $-248.13 | -7.87% | unknown |
| LHX | 8.0 | $353.33 | $343.90 | $-75.44 | -2.67% | unknown |
| LRCX | 13.0 | $248.88 | $238.42 | $-135.97 | -4.20% | unknown |
| META | 73.0 | $671.29 | $717.35 | $+3,362.38 | +6.86% | momentum |
| MU | 8.0 | $449.44 | $429.22 | $-161.76 | -4.50% | unknown |
| NVDA | 2.0 | $189.61 | $191.63 | $+4.03 | +1.06% | momentum |
| SLB | 62.0 | $48.15 | $48.07 | $-4.70 | -0.16% | unknown |
| TER | 13.0 | $252.50 | $243.69 | $-114.53 | -3.49% | unknown |
| XLE | 1.0 | $50.79 | $50.54 | $-0.25 | -0.49% | momentum |
| XOM | 1.0 | $139.75 | $140.13 | $+0.38 | +0.27% | momentum |

**Total market value:** $93,235.96
**Unrealized P/L:** $+2,563.57

---

## Account State

| Metric | Value |
|--------|-------|
| Equity | $102,719.23 |
| Cash | $9,497.29 |
| Buying Power | $283,820.02 |
| Day P/L | $-1,933.39 |
| Market | OPEN |

---

## Performance

### Per-Strategy P/L (from ledger)

| Strategy | Trades | Closed | Realized P/L | Win Rate |
|----------|--------|--------|-------------|----------|
| momentum | 24 | 10 | $-722.12 | 60% |
| unknown | 10 | 0 | $+0.00 | 0% |

### Autopilot Trade Log Summary

- Total entries: 11 buys, 9 sells
- Strategies used: momentum

---

## Backtests

### Backtest Registry

#### backtest_2022

- Return: -54.5%
- Sharpe: -1.62
- Max DD: 58.6%
- Trades: 467, Win: 49%, PF: 0.50
- Period: 2022-01-03 to 2022-12-30

#### backtest_2023

- Return: -41.8%
- Sharpe: -1.26
- Max DD: 46.1%
- Trades: 550, Win: 46%, PF: 0.54
- Period: 2023-01-03 to 2023-12-29

#### backtest_2024

- Return: -56.1%
- Sharpe: -1.12
- Max DD: 61.7%
- Trades: 522, Win: 55%, PF: 0.54
- Period: 2024-01-02 to 2024-12-30

#### backtest_2025

- Return: -10.5%
- Sharpe: -0.13
- Max DD: 29.2%
- Trades: 598, Win: 54%, PF: 0.91
- Period: 2025-01-02 to 2026-01-23

#### backtest_all_strategies

| Strategy | Return | Sharpe | MaxDD | Trades | Win% | PF |
|----------|--------|--------|-------|--------|------|----|
| momentum | -92.1% | -1.24 | 92.8% | 2534 | 53% | 0.49 |
| bollinger | +27.8% | 0.35 | 46.8% | 52 | 0% | 0.00 |
| adaptive | -80.2% | -1.05 | 81.1% | 1620 | 52% | 0.56 |

#### backtest_blend

- **Blend return:** +31.6%
- **Blend Sharpe:** 0.40
- **Blend max DD:** 29.7%
- **Correlation:** +0.354
- **SPY B&H:** +44.3%

#### backtest_signal_exit

| Strategy | Return | Sharpe | MaxDD | Trades | Win% | PF |
|----------|--------|--------|-------|--------|------|----|
| mom_signal | +36.5% | 0.40 | 44.6% | 434 | 44% | 1.09 |
| mom_stops | -92.1% | -1.24 | 92.8% | 2534 | 53% | 0.49 |
| boll_signal | +22.1% | 0.32 | 29.7% | 216 | 58% | 1.15 |
| boll_stops | +27.8% | 0.35 | 46.8% | 52 | 0% | 0.00 |

### Verdict Summary

- **ATR stop-based exits** destroy all edge across every regime (PF 0.49-0.56)
- **Signal-based exits + 20d max hold** produce slight positive expectancy (PF 1.09-1.15)
- **No strategy beats SPY buy-and-hold** over 2022-2025 (+44.3%)
- **50/50 momentum+bollinger blend** returns +31.6% with 29.7% max DD (vs 44.6% momentum alone)
- **Correlation between strategies:** +0.354 (partial diversification)

---

## Thesis Trades

_Discretionary trades managed separately from autopilot/strategy zoo._

### GME [OPEN]

- **Entry:** 2026-01-28 | 420 shares @ $23.8 | $9,996.0
- **Thesis:** Insider buying + settlement pressure + technical breakout
- **Invalidation:** close below $19.0 OR no $25 by this date (by 2026-03-14)
- **Targets:** T1: $30 (trim 25%) | T2: $50 (trim 25%) | T3: hold (trim 50%)


---

## Recent Activity

### Recent Trades

| Time | Action | Symbol | Qty | Price | Strategy | Reason |
|------|--------|--------|-----|-------|----------|--------|
| 2026-01-27T18:50:28 | BUY | MSFT | 40 | $481.77 | momentum | Router entry: momentum str=+0.60, up 6.0 |
| 2026-01-27T18:50:32 | BUY | AMZN | 1 | $243.47 | momentum | Router entry: momentum str=+0.53, up 5.3 |
| 2026-01-27T18:55:27 | SELL | AMZN | 1.0 | $243.42 | momentum | stop hit at $243.41 (ATR×2.0) |
| 2026-01-27T18:55:28 | SELL | MSFT | 40.0 | $482.26 | momentum | stop hit at $482.06 (ATR×2.0) |
| 2026-01-27T18:55:42 | BUY | AAPL | 152 | $259.62 | momentum | Router entry: momentum str=+0.53, up 5.3 |
| 2026-01-27T18:55:46 | BUY | XLE | 3 | $49.45 | momentum | Router entry: momentum str=+0.39, up 3.9 |
| 2026-01-27T19:00:24 | SELL | AAPL | 152.0 | $259.87 | momentum | stop hit at $259.75 (ATR×2.0) |
| 2026-01-27T19:00:25 | SELL | XLE | 3.0 | $49.46 | momentum | stop hit at $49.45 (ATR×2.0) |
| 2026-01-27T19:00:26 | BUY | GOOGL | 117 | $337.00 | momentum | Router entry: momentum str=+0.42, up 4.2 |
| 2026-01-27T19:10:24 | SELL | GOOGL | 117.0 | $335.55 | momentum | stop hit at $335.48 (ATR×2.0) |
| 2026-01-27T19:15:25 | BUY | TSLA | 90 | $439.00 | momentum | Router entry: momentum str=+0.30, up 3.0 |
| 2026-01-27T19:20:24 | SELL | TSLA | 90.0 | $431.46 | momentum | closed below 10 MA ($439.57) |
| 2026-01-28T17:55:40 | BUY | META | 73 | $676.50 | momentum | Router entry: momentum str=+0.95, up 9.5 |
| 2026-01-29T16:11:37 | BUY | XOM | 1 | $139.74 | momentum | Router entry: momentum str=+0.44, up 4.4 |
| 2026-01-29T16:11:38 | BUY | XLE | 1 | $50.79 | momentum | Router entry: momentum str=+0.36, up 3.6 |

### Recent Autopilot Log

```
2026-01-30 19:10:07,622 INFO   GME [?]: 420.0 @ $23.93 (+0.6%)
2026-01-30 19:10:07,622 INFO   KLAC [unknown]: 2.0 @ $1453.45 (-7.8%)
2026-01-30 19:10:07,622 INFO   LHX [unknown]: 8.0 @ $344.06 (-2.6%)
2026-01-30 19:10:07,622 INFO   LRCX [unknown]: 13.0 @ $238.54 (-4.2%)
2026-01-30 19:10:07,622 INFO   META [momentum]: 73.0 @ $717.93 (+6.9%)
2026-01-30 19:10:07,622 INFO   MU [unknown]: 8.0 @ $430.43 (-4.2%)
2026-01-30 19:10:07,622 INFO   NVDA [momentum]: 2.0 @ $191.81 (+1.2%)
2026-01-30 19:10:07,622 INFO   SLB [unknown]: 62.0 @ $48.04 (-0.2%)
2026-01-30 19:10:07,622 INFO   TER [unknown]: 13.0 @ $244.14 (-3.3%)
2026-01-30 19:10:07,622 INFO   XLE [momentum]: 1.0 @ $50.48 (-0.6%)
2026-01-30 19:10:07,622 INFO   XOM [momentum]: 1.0 @ $139.92 (+0.1%)
2026-01-30 19:10:07,622 INFO Trades today: 0
2026-01-30 19:10:07,622 INFO   Strategy momentum: 10 closed, $-722.12 realized, 60% win
2026-01-30 19:10:07,622 INFO   Strategy unknown: 0 closed, $+0.00 realized, 0% win
2026-01-30 19:10:07,623 INFO Autopilot run complete.
```

---

## System Status

- **Last autopilot run:** 2026-01-30T19:10:07.622979
- **Trades today:** 0
- **Errors:** none in recent log

### High Water Marks

- NVDA: $192.96
- META: $740.26
- XLE: $51.05
- XOM: $141.03
- AMAT: $340.15
- AMD: $244.04
- DOW: $27.91
- FCX: $61.53
- KLAC: $1556.78
- LHX: $352.56
- LRCX: $251.52
- MU: $452.32
- SLB: $48.13
- TER: $254.29
