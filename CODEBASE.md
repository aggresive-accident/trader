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
print(f"Buy {sizing['shares']} shares")
```

**Dependencies:** strategies module

---

## autopilot.py

**Purpose:** Main trading loop. Checks exits, enters new positions, reconciles ledger.

**Key functions:**
```python
def run()           # Main loop - one pass
def reset_daily()   # Reset daily state (trades_today, stopped_out)
def load_state() -> dict
def save_state(state: dict)
def reconcile_ledger(trader: Trader, ledger: Ledger, exclusions: set)
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
- Excludes symbols in `config.exclusions` (thesis trades)
- Signal-based exits: no broker stops, exits when signal reverses

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
def detect_anomalies() -> dict
def build_report(data: dict) -> str
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

**Exit codes:** 0=healthy, 1=warnings, 2=errors

**Dependencies:** trader, ledger

---

## evening.py

**Purpose:** End-of-day wrapper. Activity summary, equity delta, state archiving.

**Key functions:**
```python
def get_days_activity() -> dict
def compare_to_previous_eod() -> dict
def check_thesis_targets() -> dict
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
- Current positions with P&L
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
XS_ALLOCATION_PCT = 0.30      # 30% of portfolio
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

**Dependencies:** trader, bar_cache, ledger_xs

---

## ledger_xs.py

**Purpose:** Cross-sectional position ledger. Separate from zoo ledger.

**Class: `LedgerXS`**
```python
class LedgerXS:
    def __init__(self, path: str = "ledger_xs.json")

    def record_buy(self, symbol: str, shares: float, price: float, reason: str = "") -> dict
    def record_sell(self, symbol: str, shares: float, price: float, reason: str = "") -> dict

    def get_position(self, symbol: str) -> Optional[dict]
    def get_all_positions() -> dict[str, dict]

    def get_trades() -> list[dict]
    def calculate_pnl() -> dict  # realized + unrealized

    def summary() -> dict
```

**Data file:** `ledger_xs.json`
```json
{
  "positions": {
    "MU": {"shares": 50, "avg_price": 98.50, "opened_at": "..."},
    ...
  },
  "trades": [
    {"timestamp": "...", "symbol": "MU", "action": "BUY", "shares": 50, "price": 98.50, "reason": "rebalance"}
  ]
}
```

**CLI:**
```bash
python3 ledger_xs.py status     # summary
python3 ledger_xs.py positions  # open positions
python3 ledger_xs.py trades     # trade history
python3 ledger_xs.py pnl        # P&L calculation
```

**Dependencies:** None (stdlib only)

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
