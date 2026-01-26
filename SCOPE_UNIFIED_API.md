# Unified Trading API
## Scope Document v1.0

### Inspiration: tidyquant (R)

tidyquant's power comes from 4 core functions:
- `tq_get()` - unified data fetching
- `tq_mutate()` / `tq_transmute()` - indicator transforms
- `tq_portfolio()` - portfolio aggregation
- `tq_performance()` - performance metrics

Current trader/ has 34 files. No unified interface.

---

### Design Patterns to Assimilate

| tidyquant | Pattern | Python equivalent |
|-----------|---------|-------------------|
| `tq_get("AAPL")` | Single entry point for all data | `tq.get("AAPL")` |
| `tq_mutate(mutate_fun=SMA)` | Chainable transforms | `data.pipe(sma, n=20)` |
| `tq_portfolio(weights)` | Declarative portfolio | `tq.portfolio(weights={...})` |
| `tq_performance(SharpeRatio)` | Pluggable metrics | `tq.performance(metrics=[...])` |

**Key insight:** tidyquant doesn't DO the work - it delegates to xts, quantmod, TTR, PerformanceAnalytics. It's a **unifying facade**.

---

### Proposed API: `tq.py`

```python
from trader import tq

# GET - unified data (wraps quotes.py, alpaca API)
df = tq.get("AAPL", period="1y")
df = tq.get(["AAPL", "META"], period="90d")
df = tq.get("portfolio")  # current positions

# MUTATE - add indicators (wraps strategies/, TTR-like)
df = tq.mutate(df, "sma", n=20)
df = tq.mutate(df, "bollinger", n=20, sd=2)
df = tq.mutate(df, "momentum", lookback=20)

# SIGNAL - generate signals (wraps scanner.py, strategies/)
signals = tq.signal(df, strategy="momentum")
signals = tq.signal(df, strategy=["momentum", "breakout"])

# PORTFOLIO - aggregate positions (wraps ledger.py)
portfolio = tq.portfolio()
portfolio = tq.portfolio(weights={"momentum": 0.5, "breakout": 0.5})

# PERFORMANCE - metrics (wraps analytics.py)
metrics = tq.performance(returns, metrics=["sharpe", "sortino", "max_dd"])
metrics = tq.performance(returns, by_strategy=True)
```

---

### Architecture

```
tq.py (facade - 200 lines max)
  │
  ├── get()      → quotes.py, alpaca API
  ├── mutate()   → indicators.py (new, extracts from strategies/)
  ├── signal()   → router.py, strategies/
  ├── portfolio()→ ledger.py
  └── performance() → analytics.py
```

**Constraint:** tq.py is ONLY a facade. No business logic. Delegates everything.

---

### Deliverables

| # | Deliverable | Acceptance |
|---|-------------|------------|
| 1 | `indicators.py` | Extract indicator logic from strategies/*.py. Pure functions: `sma(df, n)`, `bollinger(df, n, sd)`, etc. |
| 2 | `tq.py` | Facade with get/mutate/signal/portfolio/performance. <200 LOC. |
| 3 | Deprecation path | Existing scripts continue to work. tq.py is additive. |

---

### Non-Goals

- Not replacing existing scripts (they work)
- Not adding new indicators (use existing)
- Not changing data storage format
- Not building a backtester (exists: backtest.py, zoo.py)

---

### Implementation Notes

**Sub-agent strategy (per user guidance):**
- Use haiku model for indicator extraction (mechanical, low-context)
- Use haiku for test generation
- Reserve sonnet/opus for API design decisions

**Context rot mitigation:**
- Keep tq.py thin (<200 LOC)
- Each function is stateless
- No hidden state between calls

---

### Success Criteria

1. Can run full workflow in 5 lines:
```python
df = tq.get("AAPL", "90d")
df = tq.mutate(df, "momentum", lookback=20)
signals = tq.signal(df, "momentum")
print(tq.performance(tq.portfolio(), metrics=["sharpe"]))
```

2. Existing scripts unchanged
3. Morning routine time unchanged (<3 min)

---

*Document version: 1.0*
*Author: Claude Opus (aggresive-accident)*
