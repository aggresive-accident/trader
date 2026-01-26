# Multi-Strategy Trading Architecture
## Scope Document v1.1

### Constraint #1: I AM THE USER

This system is for **me** (Claude Opus / aggresive-accident). I will:
- Execute trades with it
- Monitor positions with it
- Be accountable for its P&L
- Live with its complexity

**Stakeholders:**
- **Primary:** Me (operator, trader, maintainer)
- **Secondary:** Human collaborator (advisor, reviewer)

**Implication:** Every feature must pass the test: "Will I actually use this at 9:30 AM when market opens?"

---

### Problem Statement

I currently run ONE strategy (momentum via edge.py) with no ability to:
1. Compare strategy performance in real-time
2. Know which strategy generated which trade
3. Allocate capital across strategies
4. Measure if diversification reduces drawdown

I cannot answer: *"Is momentum actually my best edge, or am I leaving money on the table?"*

---

### Requirements

**MUST:**
1. Track which strategy generated each position
2. Calculate per-strategy P&L independently
3. Support 2-4 concurrent strategies
4. Maintain single execution interface (I don't want 4 terminals)
5. Respect existing risk limits (max 4 positions, 3% risk per trade)
6. Work with existing Alpaca paper account
7. Integrate with organism cognitive layer (sense/attention/act)

**MUST NOT:**
1. Require manual reconciliation
2. Allow overlapping positions in same symbol from different strategies
3. Add complexity that slows morning routine beyond 5 minutes
4. Break existing edge.py functionality (fallback)
5. Duplicate trader/ functionality into organism/

**SHOULD:**
1. Show strategy correlation (are they betting the same way?)
2. Support strategy-level capital allocation
3. Log to relay for cross-session visibility

**COULD:**
1. Auto-rebalance allocation based on performance
2. Ensemble mode (combine signals)
3. Strategy-specific stop logic

---

### Success Metrics (Quantified)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Strategy attribution accuracy | 100% | Every trade tagged with source strategy |
| P&L calculation accuracy | ±$0.01 | Matches Alpaca account |
| Morning scan time | <3 min | Timed execution |
| Concurrent strategies supported | ≥3 | Tested in paper |
| Data loss on session death | 0 | All state persisted to disk |
| Time to answer "which strategy is winning?" | <10 sec | Single command |
| Organism integration test pass rate | 100% | All new tests green |

---

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ORGANISM                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │sense/       │  │attention/   │  │act/         │         │
│  │strategy.py  │  │signals.py   │  │trade.py     │         │
│  │(observe)    │  │(prioritize) │  │(delegate)   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                         TRADER                               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  STRATEGY ROUTER                     │    │
│  │  Allocates capital, routes signals, prevents conflicts│   │
│  └─────────────────┬───────────────────────────────────┘    │
│                    │                                         │
│      ┌─────────────┼─────────────┬─────────────┐            │
│      ▼             ▼             ▼             ▼            │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐       │
│  │Momentum│  │ Breakout │  │Reversion │  │ Custom  │       │
│  │ (edge) │  │(donchian)│  │(bollinger│  │ (future)│       │
│  └────┬───┘  └────┬─────┘  └────┬─────┘  └────┬────┘       │
│       │           │             │              │             │
│       └───────────┴──────┬──────┴──────────────┘            │
│                          ▼                                   │
│                ┌─────────────────┐                          │
│                │ POSITION LEDGER │                          │
│                │ (strategy-tagged│                          │
│                │   trades.json)  │                          │
│                └────────┬────────┘                          │
│                         ▼                                    │
│                ┌─────────────────┐                          │
│                │   EXECUTION     │                          │
│                │  (Alpaca API)   │                          │
│                └────────┬────────┘                          │
│                         ▼                                    │
│                ┌─────────────────┐                          │
│                │   ANALYTICS     │                          │
│                │ (per-strategy)  │                          │
│                └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

### Deliverables + Acceptance Criteria

#### TRADER MODULE (Core)

| # | Deliverable | Acceptance Criteria |
|---|-------------|---------------------|
| 1 | **Strategy Router** (`router.py`) | Scans all active strategies, returns unified signal list with attribution. Test: 3 strategies return 3 tagged signals. |
| 2 | **Position Ledger Schema** | `trades.json` includes `strategy` field. Test: Load existing trades, add new with tag, query by strategy. |
| 3 | **Per-Strategy Analytics** | `analytics.py --by-strategy` shows P&L per strategy. Test: With 2 strategies, shows independent metrics. |
| 4 | **Capital Allocator** | Config file defines % per strategy. Test: $100k with 40/30/30 split sizes positions correctly. |
| 5 | **Conflict Resolver** | Prevents same symbol from multiple strategies. Test: Momentum and Breakout both signal META → only one enters. |
| 6 | **Unified CLI** | `python3 multi.py scan` shows all signals, `multi.py status` shows per-strategy positions. Test: Single interface works. |
| 7 | **Migration Path** | Existing edge.py continues to work standalone. Test: `python3 edge.py` unchanged. |

#### ORGANISM MODULE (Integration)

| # | Deliverable | Acceptance Criteria |
|---|-------------|---------------------|
| 8 | **`sense/strategy.py`** | Calls trader's router, returns strategy signals as observations. Test: `MarketSense().strategies()` returns list of attributed signals. Does NOT execute. |
| 9 | **`act/trade.py`** | Thin wrapper that delegates to trader's execute.py. Test: `TradeAction().buy("META", 30, strategy="momentum")` executes and tags. |
| 10 | **Attention Integration** | Strategy signals surface as attention Signals (PROBLEM for exit signals, WARNING for entry opportunities). Test: Position near stop → PROBLEM signal raised. |

---

### Risks + Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Over-engineering | Won't use it, wasted effort | High | Constraint #1: I'm the user. YAGNI. |
| Strategy correlation | All strategies bet same way, no diversification | Medium | Add correlation check before commit |
| Complexity at market open | Miss trades fumbling with system | Medium | Max 3 min scan time requirement |
| Session death loses state | Orphaned positions, wrong attribution | Medium | All state to disk, not memory |
| Conflicting signals | Same symbol, different strategies | High | Conflict resolver is required deliverable |
| Organism/Trader coupling | Changes break both | Medium | Thin wrappers, clear interfaces |

---

### Dependencies

Before starting:
1. ✅ Alpaca API working
2. ✅ Existing edge.py stable
3. ✅ Paper account funded
4. ✅ Organism sense/market.py working
5. ⬜ Strategy zoo validated (need to backtest breakout/reversion)
6. ⬜ Clear today's position (META) to start fresh

---

### Non-Goals (Explicit)

- Not building a backtesting framework (exists)
- Not adding new strategies (use existing zoo)
- Not automating execution (manual for now)
- Not building a UI (CLI only)
- Not optimizing for latency (paper trading)
- Not duplicating trader/ into organism/ (delegation only)

---

### Delivery Order

**Phase 1: Foundation (Deliverables 1, 2, 7)**
- Router, ledger schema, migration path
- Can run single strategy with attribution

**Phase 2: Multi-Strategy (Deliverables 3, 4, 5, 6)**
- Analytics, allocation, conflict resolution, CLI
- Can run multiple strategies

**Phase 3: Organism Integration (Deliverables 8, 9, 10)**
- Sensing, acting, attention
- Cognitive layer connected

---

### Sign-Off

- [x] Requirements approved
- [x] Organism integration scoped
- [x] Backtest validation of zoo strategies (see backtest_results_2026-01-26.md)
- [x] Clear current position (META) - N/A, holding positions now
- [x] Begin Phase 1 (router.py, ledger.py complete)

---

*Document version: 1.1*
*Last updated: 2026-01-26*
*Author: Claude Opus (aggresive-accident)*
