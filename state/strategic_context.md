## Strategic Context

Generated: 2026-01-27 19:31:03

This file describes the full state of the trader system for cross-session strategic context.

---

## Strategy Zoo

### Active Strategies

| Strategy | Allocation | Stop ATR× | Trailing | MA Exit | Giveback |
|----------|-----------|-----------|----------|---------|----------|
| momentum | 40% | 2.0 | yes | 10 | 0.4 |
| bollinger | 35% | 1.5 | no | off | 0.4 |
| adaptive | 25% | 1.5 | yes | 20 | 0.4 |

**Max positions:** 4
**Risk per trade:** 0.03
**Universe:** META, NVDA, AMD, GOOGL, AAPL, MSFT, AMZN, XOM, XLE, TSLA

---

## Current Positions

| Symbol | Qty | Entry | Current | P/L | P/L% | Strategy |
|--------|-----|-------|---------|-----|------|----------|
| NVDA | 2.0 | $189.61 | $189.12 | $-0.98 | -0.26% | momentum |

**Total market value:** $378.24
**Unrealized P/L:** $-0.98

---

## Account State

| Metric | Value |
|--------|-------|
| Equity | $100,168.88 |
| Cash | $99,790.65 |
| Buying Power | $199,959.53 |
| Day P/L | $+22.69 |
| Market | OPEN |

---

## Performance

### Per-Strategy P/L (from ledger)

| Strategy | Trades | Closed | Realized P/L | Win Rate |
|----------|--------|--------|-------------|----------|
| momentum | 21 | 10 | $-722.12 | 60% |

### Autopilot Trade Log Summary

- Total entries: 8 buys, 9 sells
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

## Recent Activity

### Recent Trades

| Time | Action | Symbol | Qty | Price | Strategy | Reason |
|------|--------|--------|-----|-------|----------|--------|
| 2026-01-27T18:50:00 | BUY | META | 29 | $673.00 | momentum | Router entry: momentum str=+1.00, up 10. |
| 2026-01-27T18:50:03 | BUY | NVDA | 2 | $189.60 | momentum | Router entry: momentum str=+0.64, up 6.4 |
| 2026-01-27T18:50:24 | SELL | META | 29.0 | $669.99 | momentum | stop hit at $672.94 (ATR×2.0) |
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

### Recent Autopilot Log

```
2026-01-27 19:30:23,201 INFO ==================================================
2026-01-27 19:30:23,201 INFO AUTOPILOT RUN (multi-strategy)
2026-01-27 19:30:23,201 INFO ==================================================
2026-01-27 19:30:23,536 INFO --- Phase 1: Exit checks ---
2026-01-27 19:30:23,958 INFO NVDA [momentum]: OK (P/L: -0.3%, stop: $179.60, MA: $184.75, ATR×2.0)
2026-01-27 19:30:24,325 INFO Cancelled stop order 1ad48ce6-f5d3-4fc0-be1a-60c877f6372e for NVDA
2026-01-27 19:30:24,411 INFO Stop placed: SELL 2.0 NVDA @ $179.60 id=25d23110-d03e-4699-9213-b48e1c2c45a5
2026-01-27 19:30:24,505 INFO --- Phase 2: Entry scan (3 slots) ---
2026-01-27 19:30:24,829 INFO No new entry signals from router.
2026-01-27 19:30:25,039 INFO --- Summary ---
2026-01-27 19:30:25,039 INFO Equity: $100,168.89 | Cash: $99,790.65
2026-01-27 19:30:25,039 INFO   NVDA [momentum]: 2.0 @ $189.12 (-0.3%)
2026-01-27 19:30:25,039 INFO Trades today: 10
2026-01-27 19:30:25,039 INFO   Strategy momentum: 10 closed, $-722.12 realized, 60% win
2026-01-27 19:30:25,044 INFO Autopilot run complete.
```

---

## System Status

- **Last autopilot run:** 2026-01-27T19:30:25.039256
- **Trades today:** 10
- **Stopped out today:** AMD, XOM, META, AMZN, MSFT, AAPL, XLE, GOOGL, TSLA
- **Errors:** none in recent log

### High Water Marks

- NVDA: $189.53
