# Agent Instructions - Trader

## Before Writing Code

1. **Read `CODEBASE.md`** for correct interfaces before importing local modules
2. **Do not assume** class/function names - verify against documentation first
3. **Check `PRINCIPLES.md`** for design decisions that may constrain implementation

## Key Documentation

| File | Purpose |
|------|---------|
| `CODEBASE.md` | Module interfaces, function signatures, examples |
| `~/workspace/specs/PRINCIPLES.md` | 13 architectural principles |
| `~/workspace/specs/WAYS_OF_WORKING.md` | Ceremonies, workflows |
| `README.md` | System architecture overview |

## Quick Commands

```bash
# Status checks
python3 trader.py status          # account + positions
python3 morning.py --quiet        # pre-market health
python3 evening.py --quiet        # EOD summary
python3 thesis_execute.py status  # thesis trades

# Operations
python3 autopilot.py run          # execute trading loop
python3 autopilot.py run --dry-run
python3 state_export.py           # generate context

# Debugging
python3 ledger.py status          # position ownership
systemctl --user status trader-monitor.timer
```

## Common Patterns

### Checking positions
```python
from trader import Trader
from ledger import Ledger

t = Trader()
l = Ledger()

for p in t.get_positions():
    strategy = l.get_position_strategy(p["symbol"]) or "unknown"
    print(f"{p['symbol']} [{strategy}]: {p['unrealized_pl_pct']:+.1f}%")
```

### Getting entry signals
```python
from router import StrategyRouter
r = StrategyRouter()
signals = r.get_entry_signals()
for s in signals:
    print(f"{s.symbol} [{s.strategy}]: strength={s.strength:.2f}")
```

### Loading cached bars
```python
import bar_cache
import pandas as pd

df = bar_cache.load_bars('META')
df['date'] = pd.to_datetime(df['date'])
```

## Exclusions

- **GME** is a thesis trade, excluded from autopilot via `router_config.json`
- Thesis trades are NOT in the ledger - check `thesis_trades.json` separately

## State Files

| File | Purpose | Managed By |
|------|---------|------------|
| `autopilot_state.json` | Daily state (trades_today, stopped_out) | autopilot.py |
| `trades_ledger.json` | Position ownership, trade history | ledger.py |
| `thesis_trades.json` | Discretionary trades | thesis_execute.py |
| `router_config.json` | Strategy allocation, symbols, exclusions | Manual |
| `state/strategic_context.md` | Cross-session context | state_export.py |

## Principles (Summary)

1. No AI where determinism suffices
2. Prove edge before deploying capital
3. Signal-based exits > rule-based stops
4. Thesis trades separate from systematic
5. Self-healing state
6. Fail loud, not silent
7. Decisions should be traceable

See `~/workspace/specs/PRINCIPLES.md` for full list with rationales.
