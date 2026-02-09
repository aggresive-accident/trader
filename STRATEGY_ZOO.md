# Strategy Zoo

Rules-based strategies that run on autopilot. Each strategy implements `signal(bars, idx) -> Signal(strength, reason, confidence)` and is managed by the StrategyRouter.

## Current Configuration

**Source:** `router_config.json`

| Strategy | Allocation | Exit Mode | Status |
|----------|-----------|-----------|--------|
| momentum (SimpleMomentum) | 50% | signal | wind-down (exits only, R039) |
| bollinger (BollingerReversion) | 50% | signal | active (no current positions) |
| adaptive (AdaptiveStrategy) | -- | -- | disabled |

**Universe:** META, NVDA, AMD, GOOGL, AAPL, MSFT, AMZN, XOM, XLE, TSLA
**Max positions:** 4
**Risk per trade:** 3%
**Capital:** Live equity (was hardcoded $100k, now uses broker account value)

**Note:** Zoo shares the account with Cross-Sectional XS autopilot (70% allocation). Zoo's effective capital is ~30%. New momentum entries are disabled per R039 (mega-cap momentum PF 0.84, no edge). Existing positions exit on normal signals.

## How router_config.json Works

```json
{
  "active_strategies": ["momentum", "bollinger"],
  "allocation": {"momentum": 0.50, "bollinger": 0.50},
  "symbols": ["META", "NVDA", ...],
  "max_positions": 4,
  "risk_per_trade": 0.03,
  "total_capital": 100000,  // vestigial — router.py uses live equity
  "exit_defaults": {
    "exit_mode": "signal",
    "max_hold_days": 20,
    "stop_atr_multiplier": 0,
    "trailing_stop_enabled": false,
    "profit_giveback_pct": 0,
    "ma_exit_period": null
  },
  "exit_overrides": {},
  "exclusions": ["GME"]
}
```

**Resolution order for exit params:** `exit_overrides[strategy]` > `exit_defaults` > hardcoded fallback in `autopilot.py`.

To give a specific strategy different exit rules, add to `exit_overrides`:
```json
"exit_overrides": {
  "bollinger": {
    "exit_mode": "stops",
    "stop_atr_multiplier": 2.0,
    "trailing_stop_enabled": true
  }
}
```

## Exit Modes

### Signal Mode (current, all strategies)

```
exit_mode: "signal"
```

- `check_exit()` calls `strategy_obj.signal(bars, -1)`
- Exit if `signal.strength < 0`
- Hard exit at `max_hold_days` (20 days) regardless of signal
- **No broker stop orders placed**
- `stop` field in diagnostics is always `None`

### Stop Mode (disabled, legacy)

```
exit_mode: "stops"
```

- ATR-based initial stop: `entry_price - (ATR * stop_atr_multiplier)`
- Trailing stop: `high_water - (gain * profit_giveback_pct)`
- MA cross exit: close below `ma_exit_period` SMA
- Places GTC stop orders at Alpaca broker
- Refreshes stops every autopilot run

### Why Stops Are Disabled

4-year backtests (2022-2025) across all strategies proved ATR stops destroy edge:

| Exit Mode | PF | Return | Trades |
|-----------|----|--------|--------|
| ATR stops (1.0x) | 0.49 | -92% | 2534 |
| Signal + 20d hold | 1.09 | +37% | 434 |

1x ATR stop is too tight for daily bars. Triggers on normal noise, causes death by a thousand small losses. Signal exits let the strategy decide when the thesis is invalidated.

## Autopilot Execution Flow

Every 5 minutes (systemd timer `trader-monitor.timer` → `cron_monitor.sh`):

```
1. Load config, init router, init ledger
2. Check exclusions list + dynamically exclude XS holdings (load_xs_symbols())
3. Check market clock (skip if closed)
4. Reconcile ledger vs Alpaca (skip excluded + XS symbols)
5. PHASE 1 - EXIT CHECKS:
   For each Alpaca position (skip excluded):
     - Get owning strategy from ledger
     - Resolve exit params (override > default > hardcoded)
     - Get strategy object from router
     - Get entry date from ledger
     - Fetch bars, call check_exit()
     - If exit: market sell, record in ledger
     - If signal mode: no broker stop refresh
     - If stop mode: cancel old stop, place new one
6. PHASE 2 - ENTRY SCAN (currently disabled):
   - Momentum wind-down: no new entries (R039)
   - When enabled: count managed positions, scan for signals,
     size via calculate_position_size() (per-position cap + cash guard),
     pre-trade guards block oversized orders (structural_health.py)
7. Log summary
```

## Adding a Strategy

### Step 1: Implement in `strategies/`

```python
# strategies/my_strategy.py
from strategies.base import Strategy, Signal

class MyStrategy(Strategy):
    def name(self) -> str:
        return "my_strategy"

    def warmup_period(self) -> int:
        return 30  # bars needed before first signal

    def signal(self, bars, idx) -> Signal:
        # bars is a list of bar objects, idx is current position
        # Return Signal(strength=[-1,+1], reason="why", confidence=[0,1])
        ...
```

### Step 2: Register in router.py

Add to `get_strategy_class()`:
```python
from strategies import MyStrategy
name_map = {
    ...
    "my_strategy": MyStrategy,
}
```

### Step 3: Activate in router_config.json

```json
{
  "active_strategies": ["momentum", "bollinger", "my_strategy"],
  "allocation": {"momentum": 0.34, "bollinger": 0.33, "my_strategy": 0.33}
}
```

### Step 4: Backtest

```bash
python3 backtest_strategies.py  # per-strategy 4-year backtest
```

## Backtest Results Summary

### Per-Strategy (ATR stops, 2022-2025)

| Strategy | Return | PF | Verdict |
|----------|--------|----|---------|
| momentum | -92% | 0.49 | stops destroy edge |
| bollinger | +28% | -- | few trades, unclear |
| adaptive | -80% | 0.56 | stops destroy edge |

### Per-Strategy (Signal exits + 20d hold, 2022-2025)

| Strategy | Return | PF | Verdict |
|----------|--------|----|---------|
| momentum | +37% | 1.09 | slight positive edge |
| bollinger | +22% | 1.15 | slight positive edge |

### 50/50 Blend (Signal exits, 2022-2025)

| Metric | Value |
|--------|-------|
| Return | +31.6% |
| Sharpe | 0.40 |
| Max DD | 29.7% |
| Correlation | +0.354 |
| SPY B&H | +44.3% |

### Lessons

1. ATR stops at 1x are suicide on daily bars. Too tight for normal volatility.
2. Signal-based exits outperform mechanical stops across all strategies and regimes.
3. No zoo strategy beats SPY buy-and-hold over 4 years.
4. The zoo's value is risk management (capped drawdown, defined exits), not alpha.
5. Cross-sectional mean reversion shows more promise -- see [CROSS_SECTIONAL.md](CROSS_SECTIONAL.md).
