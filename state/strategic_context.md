## Strategic Context

Generated: 2026-02-20 18:42:16

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
| AMAT | 9.0 | $334.93 | $374.34 | $+354.69 | +11.77% | xs |
| DE | 11.0 | $606.00 | $658.35 | $+575.85 | +8.64% | xs |
| DOW | 110.0 | $27.58 | $30.39 | $+309.65 | +10.21% | xs |
| FCX | 50.0 | $61.70 | $63.30 | $+80.01 | +2.59% | xs |
| FDX | 18.0 | $374.97 | $387.69 | $+228.96 | +3.39% | xs |
| GILD | 45.0 | $153.74 | $152.01 | $-77.85 | -1.12% | xs |
| GME | 420.0 | $23.80 | $23.50 | $-126.00 | -1.26% | thesis |
| INTC | 136.0 | $46.36 | $44.27 | $-284.47 | -4.51% | xs |
| LRCX | 13.0 | $248.88 | $243.47 | $-70.32 | -2.17% | xs |
| NVDA | 260.0 | $191.26 | $189.37 | $-491.40 | -0.99% | momentum |
| SLB | 62.0 | $48.15 | $50.88 | $+169.52 | +5.68% | xs |
| TER | 13.0 | $252.50 | $321.89 | $+902.07 | +27.48% | xs |

**Total market value:** $106,569.40
**Unrealized P/L:** $+1,570.71

---

## Account State

| Metric | Value |
|--------|-------|
| Equity | $100,449.29 |
| Cash | $-6,127.62 |
| Buying Power | $244,847.72 |
| Day P/L | $+339.55 |
| Market | OPEN |

---

## Performance

### Per-Strategy P/L (from ledger)

| Strategy | Trades | Closed | Realized P/L | Win Rate |
|----------|--------|--------|-------------|----------|
| momentum | 29 | 14 | $-1,283.29 | 57% |

### Autopilot Trade Log Summary

- Total entries: 12 buys, 13 sells
- Strategies used: momentum

### Equity Curve

| Date | Equity | Return | Positions |
|------|--------|--------|-----------|
| 2026-01-24 | $100,000.00 | +0.00% | 0 |
| 2026-02-18 | $99,290.77 | +0.00% | 12 |

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
| 2026-02-04T18:30:21 | SELL | NVDA | 2.0 | $173.73 | momentum | signal exit: strength=-0.43, down -4.3% |
| 2026-02-06T14:30:52 | SELL | META | 73.0 | $669.09 | momentum | signal exit: strength=-0.92, down -9.2% |
| 2026-02-09T19:25:41 | BUY | NVDA | 260 | $191.21 | momentum | Router entry: momentum str=+0.30, up 3.0 |
| 2026-02-18T14:30:47 | SELL | XOM | 1.0 | $147.90 | momentum | signal exit: strength=-0.33, down -3.3% |
| 2026-02-18T16:15:46 | SELL | XLE | 1.0 | $54.49 | momentum | max hold 20d reached (held 20d) |

### Recent Autopilot Log

```
2026-02-20 18:41:44,671 INFO   AMAT [?]: 9.0 @ $373.82 (+11.6%)
2026-02-20 18:41:44,671 INFO   DE [?]: 11.0 @ $658.35 (+8.6%)
2026-02-20 18:41:44,671 INFO   DOW [?]: 110.0 @ $30.38 (+10.1%)
2026-02-20 18:41:44,671 INFO   FCX [?]: 50.0 @ $63.27 (+2.5%)
2026-02-20 18:41:44,671 INFO   FDX [?]: 18.0 @ $387.69 (+3.4%)
2026-02-20 18:41:44,671 INFO   GILD [?]: 45.0 @ $151.96 (-1.2%)
2026-02-20 18:41:44,671 INFO   GME [?]: 420.0 @ $23.49 (-1.3%)
2026-02-20 18:41:44,671 INFO   INTC [?]: 136.0 @ $44.23 (-4.6%)
2026-02-20 18:41:44,671 INFO   LRCX [?]: 13.0 @ $243.19 (-2.3%)
2026-02-20 18:41:44,671 INFO   NVDA [momentum]: 260.0 @ $189.26 (-1.0%)
2026-02-20 18:41:44,671 INFO   SLB [?]: 62.0 @ $50.85 (+5.6%)
2026-02-20 18:41:44,671 INFO   TER [?]: 13.0 @ $321.27 (+27.2%)
2026-02-20 18:41:44,671 INFO Trades today: 0
2026-02-20 18:41:44,672 INFO   Strategy momentum: 14 closed, $-1283.29 realized, 57% win
2026-02-20 18:41:44,672 INFO Autopilot run complete.
```

---

## System Status

- **Last autopilot run:** 2026-02-20T18:41:44.672030
- **Trades today:** 0
- **Errors:** none in recent log

### High Water Marks

- AMAT: $340.15
- AMD: $249.83
- DOW: $32.83
- FCX: $64.90
- KLAC: $1556.78
- LHX: $352.56
- LRCX: $251.52
- MU: $452.32
- SLB: $51.41
- TER: $311.66
- NVDA: $193.21
