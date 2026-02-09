## Strategic Context

Generated: 2026-02-09 20:35:06

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
| AMAT | 9.0 | $334.93 | $330.76 | $-37.53 | -1.24% | xs |
| AMD | 12.0 | $241.06 | $215.91 | $-301.77 | -10.43% | xs |
| DOW | 110.0 | $27.58 | $32.28 | $+517.55 | +17.06% | xs |
| FCX | 50.0 | $61.70 | $63.49 | $+89.51 | +2.90% | xs |
| GME | 420.0 | $23.80 | $24.79 | $+417.65 | +4.18% | thesis |
| KLAC | 2.0 | $1575.71 | $1436.99 | $-277.44 | -8.80% | xs |
| LHX | 8.0 | $353.33 | $350.01 | $-26.56 | -0.94% | xs |
| LRCX | 13.0 | $248.88 | $229.47 | $-252.32 | -7.80% | xs |
| MU | 8.0 | $449.44 | $385.89 | $-508.40 | -14.14% | xs |
| NVDA | 260.0 | $191.26 | $189.68 | $-410.80 | -0.83% | momentum |
| SLB | 62.0 | $48.15 | $50.48 | $+144.41 | +4.84% | xs |
| TER | 13.0 | $252.50 | $309.43 | $+740.09 | +22.55% | xs |
| XLE | 1.0 | $50.79 | $53.59 | $+2.80 | +5.52% | momentum |
| XOM | 1.0 | $139.75 | $150.82 | $+11.07 | +7.92% | momentum |

**Total market value:** $91,124.86
**Unrealized P/L:** $+108.26

---

## Account State

| Metric | Value |
|--------|-------|
| Equity | $99,980.01 |
| Cash | $8,860.89 |
| Buying Power | $271,493.92 |
| Day P/L | $-124.21 |
| Market | OPEN |

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
2026-02-09 20:31:07,267 INFO   DOW [?]: 110.0 @ $32.34 (+17.3%)
2026-02-09 20:31:07,267 INFO   FCX [?]: 50.0 @ $63.63 (+3.1%)
2026-02-09 20:31:07,267 INFO   GME [?]: 420.0 @ $24.80 (+4.2%)
2026-02-09 20:31:07,267 INFO   KLAC [?]: 2.0 @ $1438.02 (-8.7%)
2026-02-09 20:31:07,267 INFO   LHX [?]: 8.0 @ $350.31 (-0.9%)
2026-02-09 20:31:07,267 INFO   LRCX [?]: 13.0 @ $229.45 (-7.8%)
2026-02-09 20:31:07,267 INFO   MU [?]: 8.0 @ $385.25 (-14.3%)
2026-02-09 20:31:07,267 INFO   NVDA [momentum]: 260.0 @ $189.96 (-0.7%)
2026-02-09 20:31:07,267 INFO   SLB [?]: 62.0 @ $50.45 (+4.8%)
2026-02-09 20:31:07,267 INFO   TER [?]: 13.0 @ $309.10 (+22.4%)
2026-02-09 20:31:07,267 INFO   XLE [momentum]: 1.0 @ $53.63 (+5.6%)
2026-02-09 20:31:07,267 INFO   XOM [momentum]: 1.0 @ $150.85 (+7.9%)
2026-02-09 20:31:07,267 INFO Trades today: 1
2026-02-09 20:31:07,267 INFO   Strategy momentum: 12 closed, $-1295.15 realized, 50% win
2026-02-09 20:31:07,267 INFO Autopilot run complete.
```

---

## System Status

- **Last autopilot run:** 2026-02-09T20:31:07.267407
- **Trades today:** 1
- **Errors:** none in recent log

### High Water Marks

- XLE: $53.70
- XOM: $150.97
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
- NVDA: $191.31
