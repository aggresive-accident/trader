# Thesis Trades

Discretionary trades with explicit theses, invalidation criteria, and targets. Managed separately from the strategy zoo. The autopilot never touches thesis positions.

## What This Is

A thesis trade is a position entered because of a specific belief about a stock's near-term trajectory. Unlike zoo trades (rules-based, automated), thesis trades require:

- A **stated thesis** (why this stock, why now)
- **Invalidation criteria** (price or time conditions that prove the thesis wrong)
- **Targets** with trim percentages (staged exits)
- **Manual execution** via `thesis_execute.py`

## Execution

### Preview (dry run, no `--confirm`)

```bash
python3 thesis_execute.py buy GME 10000
```

Output:
```
  THESIS BUY: GME
  Shares:    437
  Price:     ~$22.86
  Notional:  ~$9,989.82
  Thesis:    Insider buying + settlement pressure + technical breakout

  [Preview only - pass --confirm to execute]
```

### Execute (with `--confirm`)

```bash
python3 thesis_execute.py buy GME 10000 --confirm
```

This:
1. Validates symbol is in `router_config.json` exclusions list
2. Refuses if symbol is managed by autopilot
3. Calculates shares from dollar amount at current price
4. Places market order via Alpaca
5. Updates `thesis_trades.json` with fill price, shares, timestamp, order ID

### Trim / Exit

```bash
# Trim 25% at T1
python3 thesis_execute.py sell GME 25% --confirm

# Sell specific share count
python3 thesis_execute.py sell GME 100 --confirm

# Full exit
python3 thesis_execute.py sell GME 100% --confirm
```

Sell commands:
- Calculate realized P/L on the trimmed portion
- Update `thesis_trades.json` status (`open` -> `partial` -> `closed`)
- Mark targets as hit if current price >= target price
- Append to `executions` array in the trade record
- Log to `thesis_trades.log`

### Status

```bash
python3 thesis_execute.py status     # all thesis trades
python3 thesis_execute.py check GME  # single symbol detail
```

Shows: current price, unrealized P/L, distance to stop, distance to each target, days until deadline, execution history.

## Isolation From Autopilot

Thesis symbols must be in `router_config.json` `exclusions` list. This blocks autopilot at four points:

| Autopilot Phase | Behavior for Excluded Symbols |
|-----------------|-------------------------------|
| `reconcile_ledger()` | Skips -- won't import into ledger as "unknown" |
| Phase 1 exit checks | Skips -- won't evaluate signals or sell |
| Phase 2 entry scans | Blocked -- won't appear in entry candidates |
| Position counting | Excluded -- doesn't consume `max_positions` slots |

**If a symbol is NOT in exclusions, autopilot will claim it on the next run.** Always add to exclusions before buying.

## Adding a New Thesis Trade

### Step 1: Add to exclusions

Edit `router_config.json`:
```json
{
  "exclusions": ["GME", "NEW_SYMBOL"]
}
```

### Step 2: Add trade entry to thesis_trades.json

```json
{
  "id": "SYMBOL-YYYY-MM",
  "symbol": "SYMBOL",
  "status": "pending",
  "entry": {
    "date": "YYYY-MM-DD",
    "price": null,
    "shares": null,
    "notional": 10000,
    "method": "market open"
  },
  "thesis": {
    "summary": "Why this trade exists",
    "details": null
  },
  "invalidation": {
    "price_below": 0.0,
    "price_below_trigger": "close",
    "time_deadline": "YYYY-MM-DD",
    "time_condition": "description of time-based invalidation"
  },
  "targets": [
    {"label": "T1", "price": 0.0, "trim_pct": 25, "hit": false, "hit_date": null},
    {"label": "T2", "price": 0.0, "trim_pct": 25, "hit": false, "hit_date": null},
    {"label": "T3", "price": null, "trim_pct": 50, "hit": false, "hit_date": null}
  ],
  "outcome": {
    "realized_pnl": 0,
    "unrealized_pnl": 0,
    "thesis_correct": null,
    "notes": null,
    "closed_date": null
  }
}
```

### Step 3: Cache historical data

```bash
python3 bar_cache.py seed --symbols SYMBOL
```

### Step 4: Execute

```bash
python3 thesis_execute.py buy SYMBOL 10000 --confirm
```

## State Files

| File | Purpose |
|------|---------|
| `thesis_trades.json` | Trade definitions, status, targets, outcomes |
| `thesis_trades.log` | Execution log (appended on every buy/sell) |
| `router_config.json` | Exclusions list (must contain thesis symbols) |

## Current Active Trades

See `thesis_trades.json` for current state. Also visible in:
```bash
python3 thesis_execute.py status
python3 state_export.py --stdout   # includes Thesis Trades section
```

## Principles

- **State the thesis before entry.** If you can't articulate why, don't trade.
- **Define invalidation before entry.** Price stop + time stop. No moving goalposts.
- **Trim into strength.** Staged exits at predefined targets.
- **No capitulation.** If the thesis is intact and invalidation hasn't triggered, hold.
- **Track outcomes.** Every closed trade gets `thesis_correct: true/false` and notes.
