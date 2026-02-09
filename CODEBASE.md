# Codebase Reference

Module interfaces for the trader system. Use this to reduce trial-and-error tool calls.

---

## bar_cache.py

**Purpose:** Local parquet cache for historical daily bars. Avoids hitting Alpaca API for backtests.

**Public functions:**
- `load_bars(symbol: str) -> pd.DataFrame` - Load cached bars for symbol
- `cache_symbols(symbols: list, start: str, end: str)` - Fetch and cache from Alpaca
- `update_cache()` - Append new bars since last cache date

**DataFrame schema:** `date`, `open`, `high`, `low`, `close`, `volume`

**Example:**
```python
import bar_cache

df = bar_cache.load_bars('META')
df['date'] = pd.to_datetime(df['date'])
df['ma20'] = df['close'].rolling(20).mean()
```

**CLI:**
```bash
python3 bar_cache.py seed          # seed top 200 S&P500
python3 bar_cache.py update        # append new bars
python3 bar_cache.py status        # cache stats
python3 bar_cache.py load AAPL     # print bars
```

**Dependencies:** pandas, pyarrow, alpaca-py

---

## trader.py

**Purpose:** Alpaca API wrapper. All broker communication goes through this.

**Class: `Trader`**

```python
class Trader:
    def get_account(self) -> dict
    def get_positions(self) -> list[dict]
    def get_quote(self, symbol: str) -> dict
    def get_clock(self) -> dict
    def buy(self, symbol: str, qty: float, limit_price: float = None) -> dict
    def sell(self, symbol: str, qty: float, limit_price: float = None) -> dict
    def get_orders(self, status: str = "open") -> list[dict]
    def cancel_order(self, order_id: str) -> bool
    def cancel_all_orders(self) -> int
```

**Return schemas:**

`get_account()`:
```python
{"cash", "buying_power", "portfolio_value", "equity", "last_equity", "pl_today", "pl_today_pct"}
```

`get_positions()`:
```python
[{"symbol", "qty", "market_value", "cost_basis", "unrealized_pl", "unrealized_pl_pct", "current_price", "avg_entry"}]
```

`get_clock()`:
```python
{"is_open", "phase", "timestamp", "next_open", "next_close"}
# phase: "regular", "pre-market", "after-hours", "closed"
```

**Example:**
```python
from trader import Trader
t = Trader()
print(t.get_account()["portfolio_value"])
for p in t.get_positions():
    print(f"{p['symbol']}: {p['unrealized_pl_pct']:+.1f}%")
```

**CLI:**
```bash
python3 trader.py status    # account + positions
python3 trader.py quote NVDA
python3 trader.py buy AAPL 10
python3 trader.py sell AAPL 10
```

**Dependencies:** alpaca-py

---

## ledger.py

**Purpose:** Position ledger with strategy attribution. Tracks which strategy owns each position.

**Classes:**

```python
@dataclass
class Trade:
    id: str
    timestamp: str
    symbol: str
    action: str  # "BUY" or "SELL"
    qty: float
    price: float
    strategy: str
    reason: str = ""
    order_id: str = ""
    closed_at: str = None
    close_price: float = None
    pnl: float = None
    pnl_pct: float = None

@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    strategy: str
    opened_at: str
    trade_ids: list

class Ledger:
    def has_position(self, symbol: str) -> bool
    def get_position(self, symbol: str) -> Optional[Position]
    def get_position_strategy(self, symbol: str) -> Optional[str]
    def record_buy(self, symbol: str, qty: float, price: float, strategy: str, reason: str = "") -> Trade
    def record_sell(self, symbol: str, qty: float, price: float, reason: str = "") -> Trade
    def get_trades_by_strategy(self, strategy: str) -> list[Trade]
    def get_positions_by_strategy(self, strategy: str) -> list[Position]
    def calculate_strategy_pnl(self, strategy: str) -> dict
    def summary(self) -> dict
```

**Example:**
```python
from ledger import Ledger
l = Ledger()

# Check ownership
strategy = l.get_position_strategy('NVDA')  # Returns "momentum" or None

# Get positions by strategy
momentum_positions = l.get_positions_by_strategy('momentum')

# P&L by strategy
pnl = l.calculate_strategy_pnl('momentum')
print(f"Realized P&L: ${pnl['realized_pnl']:,.2f}")
```

**CLI:**
```bash
python3 ledger.py status      # summary
python3 ledger.py positions   # open positions
python3 ledger.py trades      # all trades
python3 ledger.py pnl         # per-strategy P&L
```

**Dependencies:** None (stdlib only)

---

## router.py

**Purpose:** Strategy router. Scans universe, generates signals, resolves conflicts, sizes positions.

**Classes:**

```python
@dataclass
class StrategySignal:
    strategy: str
    symbol: str
    strength: float      # 0 to 1
    confidence: float
    reason: str
    timestamp: str
    allocation_pct: float

class StrategyRouter:
    def __init__(self, config: dict = None)
    def scan(self, symbols: list = None) -> list[StrategySignal]
    def resolve_conflicts(self, signals: list) -> list[StrategySignal]
    def get_entry_signals(self) -> list[StrategySignal]
    def get_exit_signals(self) -> list[StrategySignal]
    def calculate_position_size(self, signal: StrategySignal, price: float) -> dict
    def get_sized_entries(self) -> list[dict]
```

**Config file:** `router_config.json`
```json
{
  "active_strategies": ["momentum", "bollinger"],
  "allocation": {"momentum": 0.50, "bollinger": 0.50},
  "symbols": ["META", "NVDA", ...],
  "max_positions": 4,
  "risk_per_trade": 0.03,
  "exclusions": ["GME"]
}
```

**Example:**
```python
from router import StrategyRouter
r = StrategyRouter()

# Get entry signals
signals = r.get_entry_signals()
for s in signals:
    print(f"{s.symbol} [{s.strategy}]: strength={s.strength:.2f}")

# Size a position
sizing = r.calculate_position_size(signals[0], price=150.0)
print(f"Buy {sizing['shares']} shares (max {sizing['max_per_position']:,.0f} per position)")

# Get sized entries (signals + sizing combined)
entries = r.get_sized_entries()
for e in entries:
    print(f"{e['signal']['symbol']}: {e['sizing']['shares']} shares @ ${e['price']:.2f}")
```

**`calculate_position_size()` return schema:**
```python
{
    "shares": int,
    "notional": float,
    "strategy_allocation": float,
    "strategy_capital": float,
    "available_capital": float,
    "used_capital": float,
    "max_per_position": float,  # strategy_capital / max_positions
}
```

**Sizing logic:**
1. Strategy gets a % of live equity (from allocation config)
2. Per-position cap = strategy_capital / max_positions
3. Capped by available cash (buying power) from broker
4. risk_per_trade scales down from the per-position cap

**Dependencies:** strategies module

---

## autopilot.py

**Purpose:** Main trading loop. Checks exits, enters new positions, reconciles ledger.

**Key functions:**
```python
def run()           # Main loop - one pass (Phase 2 entries currently disabled)
def reset_daily()   # Reset daily state (trades_today, stopped_out)
def load_state() -> dict
def save_state(state: dict)
def load_xs_symbols() -> set  # Load XS holdings from ledger_xs.json (prevents split-brain)
def reconcile_ledger(trader: Trader, ledger: Ledger, exclusions: set)  # Skips XS symbols
def buy_position(client, ledger, symbol, qty, price, stop_price, reason, strategy, exit_mode="stops") -> bool
def check_exit(bars, entry_price, high_water, exit_params, ...) -> dict
def fetch_bars(data_client, symbol: str, days: int = 60) -> list
```

**State file:** `autopilot_state.json`
```json
{"trades_today": 0, "last_run": "2026-01-29T...", "stopped_out": []}
```

**CLI:**
```bash
python3 autopilot.py run       # execute one pass
python3 autopilot.py reset     # reset daily state
python3 autopilot.py status    # show state
python3 autopilot.py run --dry-run  # log only, no trades
```

**Key behavior:**
- Auto-resets daily state if `last_run` was a different day
- Excludes symbols in `config.exclusions` (thesis trades) + dynamic XS exclusion via `load_xs_symbols()`
- Signal-based exits: no broker stops, exits when signal reverses
- Phase 2 (new entries) currently disabled (momentum wind-down, R039)
- Pre-trade guards in `buy_position()`: checks concentration limit (25%) and cash via `structural_health`

**Dependencies:** trader, ledger, router, strategies

---

## thesis_execute.py

**Purpose:** Discretionary thesis trade execution. Separate from autopilot/zoo.

**Key functions:**
```python
def cmd_buy(symbol: str, dollar_amount: float, auto_confirm: bool = False)
def cmd_sell(symbol: str, amount_str: str, auto_confirm: bool = False)
def cmd_status()
def cmd_check(symbol: str)
```

**Data file:** `thesis_trades.json`
```json
{
  "trades": [{
    "symbol": "GME",
    "status": "open",  // pending, open, closed
    "thesis": {"summary": "...", "catalysts": [...]},
    "entry": {"date": "...", "shares": 420, "price": 23.80, "notional": 9996},
    "invalidation": {"price_below": 19.0, "time_deadline": "..."},
    "targets": [{"label": "T1", "price": 30, "trim_pct": 25, "hit": false}]
  }]
}
```

**CLI:**
```bash
python3 thesis_execute.py status
python3 thesis_execute.py buy GME 10000 --confirm
python3 thesis_execute.py sell GME 25%
python3 thesis_execute.py sell GME all
python3 thesis_execute.py check GME
```

**Dependencies:** alpaca-py (direct, not via trader.py)

---

## morning.py

**Purpose:** Pre-market wrapper. System health check, reconciliation, anomaly detection.

**Key functions:**
```python
def check_memory() -> dict
def check_alpaca() -> dict
def reconcile_positions() -> dict
def get_overnight_changes() -> dict
def get_pending_thesis() -> dict
def check_autopilot() -> dict
def check_strategy_health() -> dict   # Strategy performance vs expectations
def check_decision_journal() -> dict  # Decision journal status
def detect_anomalies() -> dict
def build_report(data: dict) -> str   # Includes structural health section
```

**Output files:**
- `state/morning_report.md`
- `state/morning_report.json`

**CLI:**
```bash
python3 morning.py              # full report
python3 morning.py --quiet      # one-line summary
python3 morning.py --json       # structured JSON
```

**Exit codes:** 0=healthy, 1=warnings, 2=errors (includes structural health: PROBLEM=2, WARNING=1)

**Dependencies:** trader, ledger, structural_health

---

## evening.py

**Purpose:** End-of-day wrapper. Activity summary, equity delta, state archiving.

**Key functions:**
```python
def get_days_activity() -> dict
def compare_to_previous_eod() -> dict
def check_thesis_targets() -> dict
def check_autopilot_xs() -> dict
def check_decision_journal() -> dict  # Today's decisions summary
def check_log_rotation() -> dict
def archive_snapshot() -> dict
def build_report(data: dict) -> str
```

**Output files:**
- `state/evening_report.md`
- `state/evening_report.json`
- `state/archive/YYYY-MM-DD/` (archived state files)

**CLI:**
```bash
python3 evening.py              # full report
python3 evening.py --quiet      # one-line summary
python3 evening.py --json       # structured JSON
```

**Exit codes:** 0=healthy, 1=warnings (stops hit, log rotation needed), 2=errors

**Dependencies:** trader, ledger

---

## state_export.py

**Purpose:** Generate strategic context markdown for cross-session continuity.

**Key functions:**
```python
def generate_context() -> str   # Returns full markdown
@section                        # Decorator to register section generators
```

**Sections generated:**
- Strategy Zoo configuration
- Current positions with P&L (reads all three ledgers: trades_ledger.json, ledger_xs.json, thesis_trades.json)
- Account state
- Performance by strategy
- Backtest results
- Thesis trades
- Recent activity
- System status

**Output:** `state/strategic_context.md`

**CLI:**
```bash
python3 state_export.py              # write to file
python3 state_export.py --stdout     # print to stdout
```

**Dependencies:** trader, ledger

---

## structural_health.py

**Purpose:** Structural health assertions. Checks for concentration risk, orphan positions, stale rebalances, and sizing violations. Used by morning.py (reporting) and autopilot.py (pre-trade guards).

**Key functions:**
```python
def check_concentration(positions: list, equity: float, limit: float = 0.25) -> dict
    """No single position > limit% of account equity."""
    # Returns {"status": "PASS"|"PROBLEM", "violations": [...], "max_pct": float}

def check_attribution(positions: list) -> dict
    """Every Alpaca position must have a strategy owner in one of the ledgers."""
    # Returns {"status": "PASS"|"WARNING", "unattributed": [...]}

def check_xs_freshness(max_days: int = 8) -> dict
    """XS must have rebalanced within max_days."""
    # Returns {"status": "PASS"|"WARNING", "days_since": float|None}

def check_pretrade_concentration(symbol: str, order_shares: int,
                                 order_price: float, positions: list,
                                 equity: float, limit: float = 0.25) -> dict
    """Pre-trade guard: would this order push a position past the concentration limit?"""
    # Returns {"allowed": bool, "reason": str, "projected_pct": float}

def check_pretrade_cash(order_shares: int, order_price: float, cash: float) -> dict
    """Pre-trade guard: does the account have enough cash for this order?"""
    # Returns {"allowed": bool, "reason": str, "order_cost": float, "cash": float}

def run_all_checks() -> list
    """Run all structural health checks. Returns list of check results."""
    # Each: {"name": str, "status": "PASS"|"WARNING"|"PROBLEM", "detail": str}
```

**Constants:**
```python
CONCENTRATION_LIMIT = 0.25  # No single position > 25% of account
```

**Dependencies:** trader (for run_all_checks only)

---

## strategies/

**Purpose:** Strategy zoo with 20+ strategies across multiple categories.

**Base class:** `strategies/base.py`
```python
@dataclass
class Signal:
    strength: float  # -1 (strong sell) to +1 (strong buy)
    reason: str
    confidence: float = 0.5

class Strategy(ABC):
    name: str = "base"
    params: dict = {}

    @abstractmethod
    def signal(self, bars: list, idx: int) -> Signal:
        """Generate signal for symbol at bar index"""
        pass

    def warmup_period(self) -> int:
        """Minimum bars needed"""
        return 10
```

**Available strategies:**
```python
from strategies import (
    # Momentum
    SimpleMomentum, AcceleratingMomentum,
    # Mean Reversion
    BollingerReversion, RSIReversion,
    # Trend
    MovingAverageCross, TrendStrength,
    # Volume
    VolumeBreakout, OnBalanceVolume, VWAP,
    # Volatility
    ATRBreakout, VolatilityContraction, KeltnerChannel,
    # Breakout
    DonchianBreakout, RangeBreakout, GapStrategy, HighLowBreakout,
    # Patterns
    CandlestickPatterns, ThreeBarPatterns, SwingHighLow,
    # Ensemble
    VotingEnsemble, ConfirmationEnsemble, BestOfEnsemble,
    # Adaptive
    AdaptiveStrategy, RegimeStrategy,
)

from strategies import ALL_STRATEGIES, CATEGORIES
```

**Registering in router:**

Strategies are registered by name in `router_config.json`:
```json
{"active_strategies": ["momentum", "bollinger"]}
```

The router maps names to classes via `get_strategy_class()`:
- `"momentum"` → `SimpleMomentum`
- `"bollinger"` → `BollingerReversion`

**Example:**
```python
from strategies import SimpleMomentum

m = SimpleMomentum()
bars = [...]  # List of bar objects
signal = m.signal(bars, idx=-1)
print(f"Strength: {signal.strength}, Reason: {signal.reason}")
```

**Dependencies:** None (stdlib only)

---

## autopilot_xs.py

**Purpose:** Cross-sectional momentum autopilot. Ranks universe by 25-day momentum factor, holds top 10 with persistence bands.

**Key functions:**
```python
def compute_rankings() -> list[dict]
    """Rank all symbols by momentum factor. Returns [{symbol, momentum, rank}, ...]"""

def get_current_prices(symbols: list[str]) -> dict[str, float]
    """Get current prices, falls back to bar_cache if market closed"""

def calculate_rebalance_trades(rankings: list, current_holdings: dict, target_capital: float) -> list[dict]
    """Calculate buy/sell orders to rebalance. Uses persistence bands."""

def execute_rebalance(trades: list[dict], dry_run: bool = False) -> list[dict]
    """Execute rebalance trades via Alpaca"""

def cmd_status() / cmd_rankings() / cmd_preview() / cmd_run()
    """CLI commands"""
```

**Configuration (constants in file):**
```python
XS_ALLOCATION_PCT = 0.70      # 70% of portfolio (expanded from 30%, R039)
XS_TOP_N = 10                 # Hold top 10
XS_LOOKBACK = 25              # 25-day momentum
XS_BUY_THRESHOLD = 10         # Buy if rank ≤ 10
XS_SELL_THRESHOLD = 15        # Sell if rank > 15
REBALANCE_DAY = 0             # Monday
REBALANCE_HOUR = 9            # 9:35 ET
REBALANCE_MINUTE = 35
```

**State file:** `autopilot_xs_state.json`
```json
{
  "holdings": {"MU": {"shares": 50, "avg_price": 98.50}, ...},
  "last_rebalance": "2026-01-27T09:35:00",
  "rankings_history": [...]
}
```

**CLI:**
```bash
python3 autopilot_xs.py status      # portfolio state + P&L
python3 autopilot_xs.py rankings    # current factor rankings
python3 autopilot_xs.py preview     # preview rebalance trades
python3 autopilot_xs.py run         # execute (Monday only)
python3 autopilot_xs.py run --force # force any day
python3 autopilot_xs.py history     # past rebalances
```

**Dependencies:** trader, bar_cache

**Ledger:** XS positions tracked in `ledger_xs.json` (inline, no separate module)

---

## decision_journal.py

**Purpose:** Decision logging system for autopilot traceability. Records trading decisions with context, reasoning, and outcomes.

**Classes:**
```python
@dataclass
class Decision:
    strategy: str           # xs | zoo | thesis
    action: str             # rebalance | entry | exit | hold
    symbol: Optional[str]
    context: dict           # Rankings, positions, allocation info
    outcome: dict           # Buys, sells, holds, orders proposed
    reasoning: str          # Human-readable explanation
    executed: bool = True
    execution_note: str = ""
    timestamp: str          # ISO format

class DecisionJournal:
    def __init__(self, base_dir: Path = None)
    def log(self, decision: Decision) -> None
    def get_decisions(self, date: str = None) -> list[Decision]
    def search(self, symbol: str) -> list[Decision]
    def get_last_by_strategy(self, strategy: str) -> Optional[Decision]
```

**Helper functions:**
```python
def build_xs_rebalance_decision(...) -> Decision
def get_journal_summary_for_morning() -> dict
def get_journal_summary_for_evening() -> dict
```

**Data files:** `decisions/YYYY-MM-DD.jsonl` (one file per day, append-only)

**CLI:**
```bash
python3 decision_journal.py list             # today's decisions
python3 decision_journal.py list 2026-02-03  # specific date
python3 decision_journal.py show 2026-02-03 0  # show decision at index
python3 decision_journal.py search MU        # find decisions for symbol
python3 decision_journal.py summary          # recent summary
python3 decision_journal.py health           # check for issues
```

**Dependencies:** None (stdlib only)

---

## health.py

**Purpose:** Strategy health monitoring. Compares live performance to backtest expectations, emits signals to organism.

**Key functions:**
```python
def get_strategy_stats() -> dict
    """Get per-strategy realized P&L and trade counts from ledger"""

def get_overall_stats() -> dict
    """Aggregate stats across all strategies"""

def check_deviation(actual: float, expected: float, metric_name: str) -> dict | None
    """Check if metric deviates from expected value"""

def run_health_check() -> dict
    """Run full health check, return findings"""

def emit_signals(health_data: dict) -> Path
    """Write health signals to ~/.organism/signals/trader_health.json"""
```

**Configuration (constants):**
```python
EXPECTED = {
    "overall": {"win_rate": 0.47, "profit_factor": 2.92}
}
WARNING_THRESHOLD = 0.15  # 15% deviation
PROBLEM_THRESHOLD = 0.25  # 25% deviation
MIN_TRADES = 20           # Minimum trades before checking
```

**Signal file:** `~/.organism/signals/trader_health.json`

**CLI:**
```bash
python3 health.py          # run health check + emit signals
python3 health.py --json   # JSON output
```

**Dependencies:** ledger

---

## yf_cache.py

**Purpose:** yfinance data layer with parquet caching. Provides fundamental data (PE, PB, dividend yield, etc.) and alternative bar source.

**Public functions:**
- `fetch_bars(symbol: str, period: str = '5y', force_refresh: bool = False) -> pd.DataFrame` - Fetch OHLCV bars
- `fetch_info(symbol: str, force_refresh: bool = False) -> dict` - Fetch fundamental info
- `get_sp500_symbols() -> list[str]` - Get S&P 500 constituents from Wikipedia
- `get_cached_symbols() -> list[str]` - List symbols with cached data
- `clear_cache(symbol: str = None)` - Clear cache (all or specific symbol)
- `seed_cache(symbols: list = None, period: str = '5y')` - Pre-fetch multiple symbols
- `cache_status() -> dict` - Get cache statistics

**DataFrame schema (fetch_bars):** `date`, `open`, `high`, `low`, `close`, `volume`

**Info dict keys (fetch_info):**
```python
['marketCap', 'trailingPE', 'forwardPE', 'priceToBook', 'dividendYield',
 'returnOnEquity', 'returnOnAssets', 'debtToEquity', 'sector', 'industry', ...]
```

**Example:**
```python
from yf_cache import fetch_bars, fetch_info

# Get bars
df = fetch_bars('AAPL')

# Get fundamentals
info = fetch_info('AAPL')
pe = info.get('trailingPE')
div_yield = info.get('dividendYield')
```

**CLI:**
```bash
python3 yf_cache.py status            # cache stats
python3 yf_cache.py seed              # seed S&P 500
python3 yf_cache.py fetch AAPL        # fetch and display bars
python3 yf_cache.py info AAPL         # fetch and display info
python3 yf_cache.py sp500             # list S&P 500 symbols
python3 yf_cache.py clear [symbol]    # clear cache
```

**Cache location:** `data/yf_cache/` (bars in parquet, info in JSON)

**Cache freshness:** Bars refresh after 1 day, info after 7 days

**Dependencies:** yfinance, pandas, pyarrow

---

## Quick Reference

| Task | Command |
|------|---------|
| Account status | `python3 trader.py status` |
| Morning check | `python3 morning.py --quiet` |
| Evening wrap | `python3 evening.py --quiet` |
| Run autopilot | `python3 autopilot.py run` |
| Thesis status | `python3 thesis_execute.py status` |
| Generate context | `python3 state_export.py` |
| Update bar cache | `python3 bar_cache.py update` |
| Check ledger | `python3 ledger.py status` |
| XS status | `python3 autopilot_xs.py status` |
| XS rankings | `python3 autopilot_xs.py rankings` |
| XS rebalance | `python3 autopilot_xs.py run` |
| YF cache status | `python3 yf_cache.py status` |
| YF seed S&P 500 | `python3 yf_cache.py seed` |
