# Trader

Automated multi-strategy paper trading system on Alpaca. $100k paper account.

Three parallel tracks:
- **Strategy Zoo**: rules-based strategies (momentum, bollinger) running on autopilot via systemd timer
- **Cross-Sectional Autopilot**: top-10 momentum factor portfolio, 30% allocation, Monday rebalance
- **Thesis Trades**: discretionary positions managed manually via `thesis_execute.py`

Plus offline research:
- **Research Queue**: systematic backtesting and parameter sensitivity analysis (see `~/workspace/research/`)

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Alpaca Paper API           │
                    └──────┬──────────────┬───────────────┘
                           │              │
              ┌────────────┴──┐    ┌────────────────┐    ┌──────────────────┐
              │  autopilot.py │    │autopilot_xs.py │    │ thesis_execute.py │
              │  (systemd 5m) │    │ (Monday 9:35)  │    │  (manual/claude)  │
              └──┬────┬───┬──┘    └───────┬────────┘    └──────────────────┘
                 │    │   │               │
              ┌──┴────┴───┴───────────────┘
                 │    │   │
      ┌──────────┘    │   └──────────┐
      │               │              │
 ┌────┴────┐   ┌──────┴──────┐  ┌───┴───────┐
 │router.py│   │  ledger.py  │  │ config.py │
 │ scans   │   │  ownership  │  │ API keys  │
 └────┬────┘   └─────────────┘  └───────────┘
      │
 ┌────┴──────────────┐
 │    strategies/     │
 │ momentum.py        │
 │ mean_reversion.py  │
 │ adaptive.py  ...   │
 └────────────────────┘

Offline research:
 ┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐
 │ bar_cache.py │──>│ cross_backtest.py │──>│ run_grid_test.py │
 │ 201 parquets │   │ factor framework  │   │ run_walkforward  │
 └──────────────┘   └───────────────────┘   └──────────────────┘
```

## Quick Start

```bash
# Check current state
python3 trader.py status              # account + positions
python3 autopilot.py status           # last run, trades today
python3 thesis_execute.py status      # thesis positions

# View logs
tail -50 autopilot.log                # recent autopilot output
tail -20 autopilot_trades.jsonl       # recent trades

# Run autopilot manually (normally runs via timer)
python3 autopilot.py run              # live execution
python3 autopilot.py run --dry-run    # log only, no orders

# Systemd timer control
systemctl --user status trader-monitor.timer   # is it running?
systemctl --user stop trader-monitor.timer     # pause
systemctl --user start trader-monitor.timer    # resume

# Strategic context (full state dump)
python3 state_export.py --stdout      # print to terminal
python3 state_export.py               # write to state/strategic_context.md
```

## File Structure

### Core Trading

| File | Purpose |
|------|---------|
| `autopilot.py` | Main trading loop. Runs every 5 min via systemd. Phase 1: exit checks. Phase 2: entry scans. |
| `router.py` | Strategy scanner. Loads active strategies, fetches bars, returns attributed signals. |
| `router_config.json` | Zoo configuration: active strategies, allocation, symbols, exit params, exclusions. |
| `ledger.py` | Position ownership. Tracks which strategy owns which position. Persists to `trades_ledger.json`. |
| `trader.py` | Low-level Alpaca API wrapper. Positions, account, quotes, orders. |
| `config.py` | API key loader. Reads `~/.alpaca-keys`. |
| `monitor.py` | High water mark tracking for trailing stops. |
| `thesis_execute.py` | Manual thesis trade execution. Buy/sell/status with `--confirm` flag. |
| `thesis_trades.json` | Thesis trade state: entries, targets, invalidation, outcomes. |
| `autopilot_xs.py` | Cross-sectional momentum autopilot. Top-10 factor portfolio, 30% allocation. |
| `ledger_xs.py` | XS position ledger. Tracks holdings, rebalances, P&L separately from zoo. |
| `autopilot_xs_state.json` | XS state: holdings, last rebalance, rankings history. |
| `ledger_xs.json` | XS trade history and position tracking. |

### Strategies

| File | Strategy | Description |
|------|----------|-------------|
| `strategies/momentum.py` | SimpleMomentum | 20-day return breakout |
| `strategies/mean_reversion.py` | BollingerReversion | Bollinger band mean reversion |
| `strategies/adaptive.py` | AdaptiveStrategy | Regime-aware (disabled in zoo) |
| `strategies/base.py` | Strategy base class | `signal(bars, idx) -> Signal(strength, reason, confidence)` |

### Research / Backtesting

| File | Purpose |
|------|---------|
| `bar_cache.py` | Parquet cache for historical bars. 201 symbols, 2022-present. |
| `cross_backtest.py` | Cross-sectional factor backtester with persistence bands. |
| `run_grid_test.py` | 5 factors x 4 turnover configs grid search. |
| `run_walkforward.py` | Walk-forward validation + regime-switch test. |
| `backtest_strategies.py` | Per-strategy backtest with configurable exits. |
| `state_export.py` | Generates `state/strategic_context.md` with full system state. |

### State & Logs

| File | Format | Written By |
|------|--------|-----------|
| `autopilot.log` | Text log | `autopilot.py` (every run) |
| `autopilot_state.json` | JSON | `autopilot.py` (trades today, stopped out list) |
| `autopilot_trades.jsonl` | JSONL | `autopilot.py` (every trade) |
| `trades_ledger.json` | JSON | `ledger.py` (position ownership + trade history) |
| `high_water_marks.json` | JSON | `monitor.py` (peak prices for trailing stops) |
| `thesis_trades.json` | JSON | `thesis_execute.py` (discretionary trades) |
| `thesis_trades.log` | Text log | `thesis_execute.py` (execution log) |
| `data/bars/*.parquet` | Parquet | `bar_cache.py` (historical OHLCV) |

### Results

| File | Contents |
|------|----------|
| `grid_results.json` | Factor grid search: 20 combos ranked by OOS Sharpe |
| `walkforward_results.json` | Walk-forward + regime switch metrics |
| `cross_backtest_momentum.json` | Cross-sectional momentum IS/OOS/full results |
| `backtest_signal_exit.json` | Signal vs stop exit comparison |
| `backtest_blend.json` | 50/50 momentum+bollinger blend |

## Key Concepts

### Zoo vs XS vs Thesis Trades

**Zoo** (`autopilot.py` + `router.py`): Rules-based per-symbol strategies (momentum, bollinger). Strategies generate signals, autopilot executes. Every position is attributed to a strategy via the ledger. Runs every 5 minutes via systemd timer.

**Cross-Sectional** (`autopilot_xs.py` + `ledger_xs.py`): Factor-based ranking portfolio. Ranks all 200 S&P symbols by 25-day momentum, buys top 10, equal weight. 30% of capital. Rebalances Monday 9:35 ET with persistence bands (buy ≤10, sell >15) to reduce turnover. Separate ledger from zoo.

**Thesis** (`thesis_execute.py` + `thesis_trades.json`): Discretionary, manual. Human or Claude decides entry/exit based on a stated thesis with explicit invalidation criteria. Completely separate from both autopilots. Protected by the `exclusions` list in `router_config.json`.

### Signal-Exit Mode

ATR stop-based exits were proven to destroy all edge (PF 0.49 over 4 years). The zoo now uses signal-based exits:
- Exit when `strategy.signal()` returns negative strength
- Hard exit at `max_hold_days` (20 days)
- No broker stop orders placed
- Configured via `exit_defaults.exit_mode: "signal"` in `router_config.json`

### Ledger Ownership

Every zoo position is owned by exactly one strategy. `ledger.py` enforces single-strategy ownership per symbol. On every autopilot run, `reconcile_ledger()` syncs the ledger against Alpaca's actual positions. Orphan positions (manual buys, crashes) get imported as strategy "unknown" -- unless excluded.

### Exclusions

`router_config.json.exclusions` is a list of symbols invisible to autopilot. Excluded symbols are skipped in:
1. Reconciliation (won't import into ledger)
2. Phase 1 exit checks (won't evaluate or trade)
3. Phase 2 entry scans (won't buy)
4. Position counting (don't consume zoo slots)

## Secrets

API keys live in `~/.alpaca-keys`:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

Paper trading only. No real money.

## Quick Commands - Cross-Sectional

```bash
python3 autopilot_xs.py status      # XS portfolio state
python3 autopilot_xs.py rankings    # current factor rankings (top 20)
python3 autopilot_xs.py preview     # preview rebalance trades
python3 autopilot_xs.py run         # execute rebalance (Monday only by default)
python3 autopilot_xs.py run --force # force rebalance any day
```

## Other Docs

- [THESIS_TRADES.md](THESIS_TRADES.md) - Discretionary trade management
- [STRATEGY_ZOO.md](STRATEGY_ZOO.md) - Strategy configuration and backtests
- [CROSS_SECTIONAL.md](CROSS_SECTIONAL.md) - Factor research framework
- [../research/README.md](../research/README.md) - Research queue system
