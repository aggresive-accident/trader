# Cross-Sectional Backtesting

Factor-based strategy research across 200 S&P 500 names. Ranks the universe by a factor score, goes long the top N, rebalances periodically. All offline -- not connected to autopilot.

## What Cross-Sectional Means

Instead of asking "should I buy AAPL?" (time-series), cross-sectional asks "which stocks rank highest on this factor right now?" The portfolio is always long top-N, equal-weighted, with periodic rebalancing.

This is how most institutional equity strategies work. It tests whether a factor (momentum, mean reversion, low volatility) has persistent cross-sectional predictive power.

## Bar Cache

All data is served from a local parquet cache. No API calls during backtests.

```bash
python3 bar_cache.py status          # cache stats
python3 bar_cache.py seed            # seed 200 symbols (2022-present)
python3 bar_cache.py update          # append new bars since last cache
python3 bar_cache.py load AAPL       # inspect cached data
```

**Location:** `data/bars/{SYMBOL}.parquet`
**Schema:** date, open, high, low, close, volume
**Size:** ~8 MB for 201 symbols x 4 years
**Load time:** <250ms for all 200 symbols

### Python API

```python
from bar_cache import cache_symbols, load_bars, update_cache, SP500_TOP200

# Seed (one-time)
cache_symbols(SP500_TOP200, "2022-01-01", "2026-01-27")

# Load (never hits Alpaca)
df = load_bars("AAPL", "2023-01-01", "2024-01-01")
# Returns DataFrame: date, open, high, low, close, volume

# Daily update
update_cache()
```

### Memory Constraint

This machine has 2 GB RAM. Alpaca SDK loads all bars into memory before returning.
- 200 symbols in one pull: safe (~450 MB peak)
- 500 symbols: OOM kill. Must batch in chunks of 100.

`cache_symbols()` automatically batches and frees memory between batches.

## Running Factor Tests

### Single Factor

```python
from cross_backtest import run_backtest, load_universe, factor_momentum

universe = load_universe(SP500_TOP200, "2022-01-01", "2025-12-31")
result = run_backtest(
    universe,
    factor_fn=factor_momentum,
    start="2022-01-03",
    end="2025-12-31",
    top_n=10,              # hold top 10 stocks
    sell_threshold=20,     # persistence band: sell only if drops below rank 20
    rebalance_days=5,      # weekly rebalance
)
```

### Grid Search

```bash
python3 run_grid_test.py
```

Tests 5 factors x 4 turnover configs = 20 combinations. Reports in-sample (2022-23), out-of-sample (2024-25), and full period metrics. Results saved to `grid_results.json`.

### Walk-Forward Validation

```bash
python3 run_walkforward.py
```

Two tests:
1. **Walk-forward mean reversion:** 6-month rolling train window, 1-month trade window, stepped through 2022-2025. Tests whether edge persists out-of-sample.
2. **Regime switch:** SPY 20-day return < 0 uses mean reversion, > 0 uses momentum. Weekly rebalance with persistence band.

## Available Factors

Defined in `cross_backtest.py`:

| Factor | Function | Score Logic |
|--------|----------|-------------|
| `momentum` | `factor_momentum` | 20-day return (higher = stronger) |
| `mean_rev` | `factor_mean_reversion` | Inverse 20-day return (buy losers) |
| `low_vol` | `factor_low_volatility` | Inverse 20-day return std dev (lower vol = higher) |
| `vol_mom` | `factor_volume_momentum` | 20-day volume change vs prior 20 days |
| `composite` | Rank-blended | Avg percentile rank of momentum + low vol |

### Adding a Factor

```python
# In cross_backtest.py
def factor_my_factor(df: pd.DataFrame, period: int = 20) -> float | None:
    """Higher score = more desirable to hold."""
    if len(df) < period + 1:
        return None
    # compute score from df (has columns: date, open, high, low, close, volume)
    return score

FACTORS["my_factor"] = factor_my_factor
```

Then add to `run_grid_test.py` FACTOR_FNS dict.

## Turnover Configs

| Label | Rebalance | Persistence Band | Effect |
|-------|-----------|-------------------|--------|
| W/no-band | Weekly (5 days) | None (sell if not top 10) | High turnover, responsive |
| W/band | Weekly | Top 10 buy / top 20 sell | ~25% less turnover |
| M/no-band | Monthly (21 days) | None | Low turnover, slow |
| M/band | Monthly | Top 10 buy / top 20 sell | Lowest turnover |

Persistence bands reduce whipsaw: a stock stays in the portfolio as long as it's in the top 20, but new stocks must rank in the top 10 to enter.

## Results Summary

### Grid Search (ranked by OOS Sharpe)

Top 5 of 20 combinations:

| # | Factor | Config | OOS Sharpe | OOS Return | Full Alpha vs SPY |
|---|--------|--------|-----------|------------|-------------------|
| 1 | momentum | W/band | 1.081 | +47.2% | -33.1% |
| 2 | low_vol | M/band | 1.069 | +28.1% | -43.1% |
| 3 | low_vol | M/no-band | 0.983 | +25.7% | -37.0% |
| 4 | momentum | W/no-band | 0.868 | +39.4% | -43.2% |
| 5 | mean_rev | W/no-band | 0.693 | +33.1% | +15.7% |

**Mean reversion is the only factor with positive full-period alpha** (+15.7% to +120% depending on config). All others trail SPY B&H.

### Walk-Forward Mean Reversion

| Metric | Value |
|--------|-------|
| Return | +58.0% |
| Sharpe | 0.672 |
| Max DD | 26.1% |
| PF | 1.33 |
| Alpha vs SPY | +14.2% |
| Positive months | 24/42 |
| Edge first half | +1.79%/mo |
| Edge second half | +1.61%/mo |

Edge persists across the full walk-forward period. Minimal decay between first and second halves.

### Regime Switch (momentum when SPY up, mean reversion when SPY down)

| Metric | Value |
|--------|-------|
| Return | +69.3% |
| Sharpe | 0.622 |
| Max DD | 24.6% |
| Alpha vs SPY | +25.5% |

Counterintuitively, the momentum leg generates most P/L (+$98k), while mean reversion limits damage (-$7k) during drawdowns. The regime signal acts as a momentum confirmation filter.

### Transaction Costs (0.1% slippage per trade)

| Metric | No Costs | With 0.1% Slippage |
|--------|---------|-------------------|
| Return | +58.0% | +32.1% |
| Sharpe | 0.672 | 0.457 |
| PF | 1.327 | 1.270 |
| Edge retained | 100% | 55.3% |

The edge is real but thin. 0.1% slippage per trade eats 45% of gross returns. After costs, mean reversion no longer beats SPY B&H. Reducing turnover (monthly rebalance, persistence bands) would improve net performance.

## Live Trading (Implemented 2026-01)

The cross-sectional autopilot is live via `autopilot_xs.py`:

- **70% capital allocation** (expanded from 30% on 2026-02-09 per R039)
- **Top 10 holdings**, equal weight, 200-symbol S&P universe
- **Monday 9:35 ET rebalance** via `cron_monitor.sh` (systemd timer, every 5 min)
- **Persistence bands**: buy if rank ≤10, sell if rank >15 (reduces turnover)
- **Separate ledger**: `ledger_xs.json` (not in zoo's `trades_ledger.json`)
- **Decision journal integration**: rebalance decisions logged to `decisions/`

```bash
python3 autopilot_xs.py status      # portfolio state + P&L
python3 autopilot_xs.py rankings    # current factor rankings
python3 autopilot_xs.py preview     # preview rebalance trades
python3 autopilot_xs.py run --force # force rebalance any day
```

See `README.md` for full architecture and `CODEBASE.md` for API reference.
