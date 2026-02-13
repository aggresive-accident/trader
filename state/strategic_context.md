## Strategic Context

Generated: 2026-02-13 21:43:25

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
| AMAT | 9.0 | $334.93 | $354.49 | $+176.04 | +5.84% | xs |
| AMD | 12.0 | $241.06 | $207.50 | $-402.75 | -13.92% | xs |
| DOW | 110.0 | $27.58 | $32.74 | $+567.60 | +18.71% | xs |
| FCX | 50.0 | $61.70 | $62.99 | $+64.51 | +2.09% | xs |
| GME | 420.0 | $23.80 | $23.60 | $-84.00 | -0.84% | thesis |
| KLAC | 2.0 | $1575.71 | $1464.13 | $-223.16 | -7.08% | xs |
| LHX | 8.0 | $353.33 | $345.50 | $-62.64 | -2.22% | xs |
| LRCX | 13.0 | $248.88 | $235.53 | $-173.54 | -5.36% | xs |
| MU | 8.0 | $449.44 | $411.76 | $-301.44 | -8.38% | xs |
| NVDA | 260.0 | $191.26 | $183.00 | $-2,147.60 | -4.32% | momentum |
| SLB | 62.0 | $48.15 | $50.29 | $+132.93 | +4.45% | xs |
| TER | 13.0 | $252.50 | $314.01 | $+799.63 | +24.36% | xs |
| XLE | 1.0 | $50.79 | $54.43 | $+3.64 | +7.18% | momentum |
| XOM | 1.0 | $139.75 | $148.76 | $+9.01 | +6.45% | momentum |

**Total market value:** $89,374.84
**Unrealized P/L:** $-1,641.76

---

## Account State

| Metric | Value |
|--------|-------|
| Equity | $98,235.72 |
| Cash | $8,860.88 |
| Buying Power | $259,586.60 |
| Day P/L | $-554.66 |
| Market | CLOSED |

---

## Performance

### Per-Strategy P/L (from ledger)

| Strategy | Trades | Closed | Realized P/L | Win Rate |
|----------|--------|--------|-------------|----------|
| momentum | 27 | 12 | $-1,295.15 | 50% |

### Autopilot Trade Log Summary

- Total entries: 12 buys, 11 sells
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
| 2026-02-04T18:30:21 | SELL | NVDA | 2.0 | $173.73 | momentum | signal exit: strength=-0.43, down -4.3% |
| 2026-02-06T14:30:52 | SELL | META | 73.0 | $669.09 | momentum | signal exit: strength=-0.92, down -9.2% |
| 2026-02-09T19:25:41 | BUY | NVDA | 260 | $191.21 | momentum | Router entry: momentum str=+0.30, up 3.0 |

### Recent Autopilot Log

```
2026-02-13 20:55:40,048 INFO   DOW [?]: 110.0 @ $32.48 (+17.8%)
2026-02-13 20:55:40,048 INFO   FCX [?]: 50.0 @ $62.72 (+1.7%)
2026-02-13 20:55:40,048 INFO   GME [?]: 420.0 @ $23.50 (-1.3%)
2026-02-13 20:55:40,048 INFO   KLAC [?]: 2.0 @ $1459.68 (-7.4%)
2026-02-13 20:55:40,048 INFO   LHX [?]: 8.0 @ $344.24 (-2.6%)
2026-02-13 20:55:40,048 INFO   LRCX [?]: 13.0 @ $234.96 (-5.6%)
2026-02-13 20:55:40,048 INFO   MU [?]: 8.0 @ $410.51 (-8.7%)
2026-02-13 20:55:40,048 INFO   NVDA [momentum]: 260.0 @ $182.52 (-4.6%)
2026-02-13 20:55:40,049 INFO   SLB [?]: 62.0 @ $50.41 (+4.7%)
2026-02-13 20:55:40,049 INFO   TER [?]: 13.0 @ $314.36 (+24.5%)
2026-02-13 20:55:40,049 INFO   XLE [momentum]: 1.0 @ $54.30 (+6.9%)
2026-02-13 20:55:40,049 INFO   XOM [momentum]: 1.0 @ $148.24 (+6.1%)
2026-02-13 20:55:40,049 INFO Trades today: 0
2026-02-13 20:55:40,049 INFO   Strategy momentum: 12 closed, $-1295.15 realized, 50% win
2026-02-13 20:55:40,049 INFO Autopilot run complete.
```

---

## System Status

- **Last autopilot run:** 2026-02-13T20:55:40.049122
- **Trades today:** 0
- **Errors:** none in recent log

### High Water Marks

- XLE: $55.22
- XOM: $156.86
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
