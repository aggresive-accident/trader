#!/usr/bin/env python3
"""
autopilot_xs.py - Cross-sectional momentum autopilot.

Weekly rebalancing system that ranks all 200 symbols by 25-day momentum,
holds top 10 positions equal-weighted.

Separate from per-symbol zoo autopilot - parallel system, no interference.

Configuration:
  - 70% capital allocation (percentage of total equity, expanded from 30% per R039)
  - Top 10 holdings, equal weight (7% each)
  - Weekly rebalance: Monday 9:35 ET
  - Persistence band: buy if rank ≤10, sell if rank >15
  - Excludes thesis trades

Usage:
  python3 autopilot_xs.py status      # Current state
  python3 autopilot_xs.py rankings    # Top 20 by momentum
  python3 autopilot_xs.py preview     # Show trades without executing
  python3 autopilot_xs.py run         # Execute rebalance (if due)
  python3 autopilot_xs.py run --force # Force rebalance now
  python3 autopilot_xs.py run --dry-run
  python3 autopilot_xs.py history     # Past rebalances
"""

import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader
from bar_cache import load_bars, SP500_TOP200, cache_status
from decision_journal import DecisionJournal, build_xs_rebalance_decision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("autopilot_xs")

# === Configuration ===

XS_ALLOCATION_PCT = 0.70      # 70% of portfolio for XS (expanded from 30%, R039)
XS_TOP_N = 10                 # Hold top 10 positions
XS_BUY_THRESHOLD = 10         # Buy if rank <= 10
XS_SELL_THRESHOLD = 15        # Sell if rank > 15 (persistence band)
LOOKBACK_DAYS = 25            # Momentum lookback
REBALANCE_DAY = 0             # Monday (0=Mon, 6=Sun)
REBALANCE_HOUR = 9
REBALANCE_MINUTE = 35
MIN_VOLUME = 100_000          # Minimum avg daily volume

ET = ZoneInfo("America/New_York")

# === File Paths ===

BASE_DIR = Path(__file__).parent
STATE_FILE = BASE_DIR / "autopilot_xs_state.json"
LEDGER_FILE = BASE_DIR / "ledger_xs.json"
THESIS_FILE = BASE_DIR / "thesis_trades.json"


# === Data Structures ===

@dataclass
class XSPosition:
    symbol: str
    shares: float
    entry_price: float
    entry_date: str
    entry_rank: int


@dataclass
class XSTrade:
    id: str
    timestamp: str
    symbol: str
    action: str  # BUY or SELL
    shares: float
    price: float
    reason: str
    pnl: float = 0.0


# === State Management ===

def load_state() -> dict:
    """Load XS autopilot state."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_rebalance": None,
        "next_rebalance": None,
        "rebalance_count": 0,
    }


def save_state(state: dict):
    """Save XS autopilot state."""
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def load_ledger() -> dict:
    """Load XS ledger."""
    if LEDGER_FILE.exists():
        return json.loads(LEDGER_FILE.read_text())
    return {
        "positions": [],
        "trades": [],
        "closed_trades": [],
        "stats": {
            "total_rebalances": 0,
            "total_trades": 0,
            "realized_pnl": 0.0,
        }
    }


def save_ledger(ledger: dict):
    """Save XS ledger."""
    LEDGER_FILE.write_text(json.dumps(ledger, indent=2, default=str))


def load_thesis_exclusions() -> set[str]:
    """Load symbols excluded due to thesis trades."""
    if not THESIS_FILE.exists():
        return set()
    try:
        data = json.loads(THESIS_FILE.read_text())
        return {t["symbol"] for t in data.get("trades", []) if t.get("status") == "open"}
    except Exception:
        return set()


# === Core Functions ===

def compute_momentum(symbol: str) -> float | None:
    """Compute 25-day momentum for a symbol."""
    df = load_bars(symbol)
    if df.empty or len(df) < LOOKBACK_DAYS + 1:
        return None

    # 25-day return
    current = df["close"].iloc[-1]
    past = df["close"].iloc[-LOOKBACK_DAYS - 1]

    if past <= 0:
        return None

    return (current / past) - 1.0


def compute_avg_volume(symbol: str, days: int = 20) -> float:
    """Compute average daily volume."""
    df = load_bars(symbol)
    if df.empty or len(df) < days:
        return 0
    return df["volume"].iloc[-days:].mean()


def rank_universe(exclusions: set[str] = None) -> list[dict]:
    """
    Rank all symbols by 25-day momentum.

    Returns sorted list: [{"symbol": "NVDA", "score": 0.0823, "rank": 1, "volume": 1234567}, ...]
    """
    exclusions = exclusions or set()
    rankings = []

    for symbol in SP500_TOP200:
        if symbol in exclusions:
            continue

        score = compute_momentum(symbol)
        if score is None:
            continue

        volume = compute_avg_volume(symbol)
        if volume < MIN_VOLUME:
            continue

        rankings.append({
            "symbol": symbol,
            "score": score,
            "volume": volume,
        })

    # Sort by score descending
    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Add ranks
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    return rankings


def get_current_prices(symbols: list[str]) -> dict[str, float]:
    """Get current prices for symbols. Falls back to bar_cache if market closed."""
    trader = Trader()
    prices = {}

    for symbol in symbols:
        price = 0
        try:
            quote = trader.get_quote(symbol)
            price = quote.get("price") or quote.get("last") or 0
        except Exception:
            pass

        # Fallback to last close from cache if no live price
        if price <= 0:
            df = load_bars(symbol)
            if not df.empty:
                price = float(df["close"].iloc[-1])

        if price > 0:
            prices[symbol] = price

    return prices


def calculate_xs_allocation(trader: Trader, ledger: dict) -> dict:
    """
    Calculate XS buying power and target position size.

    Returns: {
        "total_equity": float,
        "xs_allocation": float,
        "xs_current_value": float,
        "xs_available": float,
        "target_per_position": float,
    }
    """
    account = trader.get_account()
    total_equity = float(account["equity"])

    # Calculate current XS position value
    xs_positions = ledger.get("positions", [])
    xs_symbols = [p["symbol"] for p in xs_positions]

    if xs_symbols:
        prices = get_current_prices(xs_symbols)
        xs_current_value = sum(
            p["shares"] * prices.get(p["symbol"], p["entry_price"])
            for p in xs_positions
        )
    else:
        xs_current_value = 0

    xs_allocation = total_equity * XS_ALLOCATION_PCT
    xs_available = xs_allocation - xs_current_value
    target_per_position = xs_allocation / XS_TOP_N

    return {
        "total_equity": total_equity,
        "xs_allocation": xs_allocation,
        "xs_current_value": xs_current_value,
        "xs_available": xs_available,
        "target_per_position": target_per_position,
        "cash": float(account["cash"]),
    }


def calculate_rebalance_trades(
    rankings: list[dict],
    ledger: dict,
    allocation: dict,
    prices: dict,
) -> list[dict]:
    """
    Calculate trades needed for rebalance.

    Logic:
    - SELL: positions with rank > 15 (persistence band)
    - BUY: symbols with rank <= 10 not currently held
    - Size to equal weight (target_per_position)
    """
    trades = []
    current_positions = {p["symbol"]: p for p in ledger.get("positions", [])}

    # Build rank lookup
    rank_lookup = {r["symbol"]: r["rank"] for r in rankings}

    # Determine sells (rank > 15)
    for symbol, pos in current_positions.items():
        rank = rank_lookup.get(symbol, 999)
        if rank > XS_SELL_THRESHOLD:
            price = prices.get(symbol, pos["entry_price"])
            pnl = (price - pos["entry_price"]) * pos["shares"]
            trades.append({
                "action": "SELL",
                "symbol": symbol,
                "shares": pos["shares"],
                "price": price,
                "reason": f"dropped to rank {rank} (>{XS_SELL_THRESHOLD})",
                "pnl": pnl,
            })

    # Determine buys (rank <= 10, not held)
    target_value = allocation["target_per_position"]
    held_symbols = set(current_positions.keys())
    sells_symbols = {t["symbol"] for t in trades if t["action"] == "SELL"}

    # After sells, these will be open slots
    remaining_held = held_symbols - sells_symbols
    open_slots = XS_TOP_N - len(remaining_held)

    # Get top candidates not already held (after sells)
    candidates = [
        r for r in rankings
        if r["rank"] <= XS_BUY_THRESHOLD and r["symbol"] not in remaining_held
    ]

    # Buy top candidates to fill slots
    for r in candidates[:open_slots]:
        symbol = r["symbol"]
        price = prices.get(symbol)
        if not price or price <= 0:
            continue

        shares = int(target_value / price)
        if shares < 1:
            continue

        trades.append({
            "action": "BUY",
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "reason": f"entered top {XS_BUY_THRESHOLD} (rank {r['rank']}, score {r['score']:+.2%})",
            "rank": r["rank"],
            "score": r["score"],
        })

    return trades


def execute_trades(trades: list[dict], dry_run: bool = False) -> list[dict]:
    """Execute trades and return results."""
    if not trades:
        return []

    trader = Trader()
    results = []

    for trade in trades:
        if dry_run:
            results.append({**trade, "status": "dry_run"})
            continue

        try:
            if trade["action"] == "SELL":
                result = trader.sell(trade["symbol"], trade["shares"])
            else:
                result = trader.buy(trade["symbol"], trade["shares"])

            results.append({
                **trade,
                "status": "executed",
                "order_id": result.get("id"),
            })
            log.info(f"{trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")

        except Exception as e:
            results.append({
                **trade,
                "status": "failed",
                "error": str(e),
            })
            log.error(f"Failed to {trade['action']} {trade['symbol']}: {e}")

    return results


def update_ledger_from_trades(ledger: dict, trades: list[dict], timestamp: str):
    """Update ledger with executed trades."""
    positions = {p["symbol"]: p for p in ledger.get("positions", [])}

    for trade in trades:
        if trade.get("status") != "executed":
            continue

        trade_record = {
            "id": f"xs_{ledger['stats']['total_trades'] + 1:04d}",
            "timestamp": timestamp,
            "symbol": trade["symbol"],
            "action": trade["action"],
            "shares": trade["shares"],
            "price": trade["price"],
            "reason": trade["reason"],
            "pnl": trade.get("pnl", 0),
        }
        ledger["trades"].append(trade_record)
        ledger["stats"]["total_trades"] += 1

        if trade["action"] == "SELL":
            # Remove position, record P&L
            if trade["symbol"] in positions:
                pos = positions.pop(trade["symbol"])
                ledger["closed_trades"].append({
                    **pos,
                    "exit_date": timestamp,
                    "exit_price": trade["price"],
                    "pnl": trade.get("pnl", 0),
                })
                ledger["stats"]["realized_pnl"] += trade.get("pnl", 0)

        elif trade["action"] == "BUY":
            # Add position
            positions[trade["symbol"]] = {
                "symbol": trade["symbol"],
                "shares": trade["shares"],
                "entry_price": trade["price"],
                "entry_date": timestamp,
                "entry_rank": trade.get("rank", 0),
            }

    ledger["positions"] = list(positions.values())


def should_rebalance(state: dict, force: bool = False) -> tuple[bool, str]:
    """
    Check if rebalance is due.

    Returns: (should_rebalance, reason)
    """
    if force:
        return True, "forced"

    now = datetime.now(ET)

    # Check day of week
    if now.weekday() != REBALANCE_DAY:
        return False, f"not rebalance day (today={now.strftime('%A')}, rebalance=Monday)"

    # Check time
    if now.hour < REBALANCE_HOUR or (now.hour == REBALANCE_HOUR and now.minute < REBALANCE_MINUTE):
        return False, f"before rebalance time ({REBALANCE_HOUR}:{REBALANCE_MINUTE:02d} ET)"

    # Check if already rebalanced today
    last = state.get("last_rebalance")
    if last:
        last_dt = datetime.fromisoformat(last)
        if last_dt.date() == now.date():
            return False, "already rebalanced today"

    # Check market hours
    trader = Trader()
    clock = trader.get_clock()
    if not clock.get("is_open"):
        return False, f"market closed ({clock.get('phase', 'unknown')})"

    return True, "rebalance due"


# === CLI Commands ===

def cmd_status():
    """Show current XS autopilot status."""
    state = load_state()
    ledger = load_ledger()
    trader = Trader()

    print("=" * 60)
    print("CROSS-SECTIONAL AUTOPILOT STATUS")
    print("=" * 60)

    # Allocation
    allocation = calculate_xs_allocation(trader, ledger)
    print(f"\nAllocation ({XS_ALLOCATION_PCT:.0%} of portfolio):")
    print(f"  Total equity:     ${allocation['total_equity']:,.2f}")
    print(f"  XS allocation:    ${allocation['xs_allocation']:,.2f}")
    print(f"  XS current value: ${allocation['xs_current_value']:,.2f}")
    print(f"  Target/position:  ${allocation['target_per_position']:,.2f}")

    # Positions
    positions = ledger.get("positions", [])
    print(f"\nPositions ({len(positions)}/{XS_TOP_N}):")

    if positions:
        symbols = [p["symbol"] for p in positions]
        prices = get_current_prices(symbols)

        total_value = 0
        total_pnl = 0

        for p in positions:
            price = prices.get(p["symbol"], p["entry_price"])
            value = p["shares"] * price
            pnl = (price - p["entry_price"]) * p["shares"]
            pnl_pct = (price / p["entry_price"] - 1) * 100
            total_value += value
            total_pnl += pnl

            print(f"  {p['symbol']:6} {p['shares']:4} shares @ ${p['entry_price']:.2f} "
                  f"-> ${price:.2f} ({pnl_pct:+.1f}%) P&L: ${pnl:+,.2f}")

        print(f"\n  Total value: ${total_value:,.2f}")
        print(f"  Unrealized P&L: ${total_pnl:+,.2f}")
    else:
        print("  (no positions)")

    # Rebalance info
    print(f"\nRebalance:")
    print(f"  Last: {state.get('last_rebalance', 'never')}")
    print(f"  Total rebalances: {ledger['stats']['total_rebalances']}")
    print(f"  Total trades: {ledger['stats']['total_trades']}")
    print(f"  Realized P&L: ${ledger['stats']['realized_pnl']:+,.2f}")

    # Next rebalance
    should, reason = should_rebalance(state)
    print(f"\n  Status: {reason}")


def cmd_rankings():
    """Show current top 20 by momentum."""
    print("Loading universe and computing rankings...")
    exclusions = load_thesis_exclusions()
    rankings = rank_universe(exclusions)

    print(f"\nTop 20 by {LOOKBACK_DAYS}-day momentum:")
    print(f"{'Rank':<6} {'Symbol':<8} {'Momentum':<12} {'Avg Volume':<15}")
    print("-" * 45)

    for r in rankings[:20]:
        print(f"{r['rank']:<6} {r['symbol']:<8} {r['score']:+.2%}       {r['volume']:>12,.0f}")

    print(f"\nUniverse: {len(rankings)} symbols (after exclusions)")

    # Show current holdings ranks
    ledger = load_ledger()
    positions = ledger.get("positions", [])
    if positions:
        print(f"\nCurrent holdings ranks:")
        rank_lookup = {r["symbol"]: r for r in rankings}
        for p in positions:
            r = rank_lookup.get(p["symbol"])
            if r:
                status = "HOLD" if r["rank"] <= XS_SELL_THRESHOLD else "SELL"
                print(f"  {p['symbol']:6} rank {r['rank']:3} ({r['score']:+.2%}) -> {status}")
            else:
                print(f"  {p['symbol']:6} not ranked (would SELL)")


def cmd_preview():
    """Preview rebalance trades without executing."""
    print("Computing rebalance preview...")

    trader = Trader()
    ledger = load_ledger()
    exclusions = load_thesis_exclusions()

    # Get rankings
    rankings = rank_universe(exclusions)

    # Get allocation
    allocation = calculate_xs_allocation(trader, ledger)

    # Get prices
    all_symbols = set(p["symbol"] for p in ledger.get("positions", []))
    all_symbols.update(r["symbol"] for r in rankings[:XS_TOP_N])
    prices = get_current_prices(list(all_symbols))

    # Calculate trades
    trades = calculate_rebalance_trades(rankings, ledger, allocation, prices)

    print(f"\n{'='*60}")
    print("REBALANCE PREVIEW")
    print(f"{'='*60}")

    print(f"\nAllocation:")
    print(f"  XS allocation: ${allocation['xs_allocation']:,.2f}")
    print(f"  Current value: ${allocation['xs_current_value']:,.2f}")
    print(f"  Available cash: ${allocation['cash']:,.2f}")
    print(f"  Target/position: ${allocation['target_per_position']:,.2f}")

    if not trades:
        print("\nNo trades needed - portfolio is balanced.")
        return

    sells = [t for t in trades if t["action"] == "SELL"]
    buys = [t for t in trades if t["action"] == "BUY"]

    if sells:
        print(f"\nSELLS ({len(sells)}):")
        total_sell = 0
        total_pnl = 0
        for t in sells:
            value = t["shares"] * t["price"]
            total_sell += value
            total_pnl += t.get("pnl", 0)
            pnl_str = f"P&L: ${t.get('pnl', 0):+,.2f}" if t.get("pnl") else ""
            print(f"  SELL {t['shares']:4} {t['symbol']:6} @ ${t['price']:>8.2f} = ${value:>10,.2f}  {t['reason']} {pnl_str}")
        print(f"  Total sell value: ${total_sell:,.2f}, P&L: ${total_pnl:+,.2f}")

    if buys:
        print(f"\nBUYS ({len(buys)}):")
        total_buy = 0
        for t in buys:
            value = t["shares"] * t["price"]
            total_buy += value
            print(f"  BUY  {t['shares']:4} {t['symbol']:6} @ ${t['price']:>8.2f} = ${value:>10,.2f}  {t['reason']}")
        print(f"  Total buy value: ${total_buy:,.2f}")

    print(f"\nNet cash flow: ${sum(t['shares']*t['price'] for t in sells) - sum(t['shares']*t['price'] for t in buys):+,.2f}")

    # Show resulting portfolio
    current_holdings = {p["symbol"] for p in ledger.get("positions", [])}
    sell_symbols = {t["symbol"] for t in sells}
    buy_symbols = {t["symbol"] for t in buys}

    final_holdings = (current_holdings - sell_symbols) | buy_symbols
    print(f"\nResulting portfolio ({len(final_holdings)} positions):")

    rank_lookup = {r["symbol"]: r["rank"] for r in rankings}
    for symbol in sorted(final_holdings, key=lambda s: rank_lookup.get(s, 999)):
        rank = rank_lookup.get(symbol, "?")
        status = "NEW" if symbol in buy_symbols else "HOLD"
        print(f"  {symbol:6} rank {rank:3} [{status}]")


def cmd_run(force: bool = False, dry_run: bool = False):
    """Execute rebalance."""
    state = load_state()

    # Check if rebalance is due
    should, reason = should_rebalance(state, force)
    if not should:
        print(f"Rebalance not due: {reason}")
        if not force:
            print("Use --force to override")
            return

    print(f"Starting rebalance ({reason})...")

    trader = Trader()
    ledger = load_ledger()
    exclusions = load_thesis_exclusions()
    timestamp = datetime.now(ET).isoformat()

    # Get rankings
    log.info("Computing rankings...")
    rankings = rank_universe(exclusions)

    # Get allocation
    allocation = calculate_xs_allocation(trader, ledger)
    log.info(f"XS allocation: ${allocation['xs_allocation']:,.2f}")

    # Get prices
    all_symbols = set(p["symbol"] for p in ledger.get("positions", []))
    all_symbols.update(r["symbol"] for r in rankings[:XS_TOP_N])
    prices = get_current_prices(list(all_symbols))

    # Calculate trades
    trades = calculate_rebalance_trades(rankings, ledger, allocation, prices)

    if not trades:
        print("No trades needed.")

        # Log decision even when no trades
        journal = DecisionJournal()
        current_positions = ledger.get("positions", [])
        decision = build_xs_rebalance_decision(
            rankings=rankings,
            current_positions=current_positions,
            trades=[],
            results=[],
            allocation=allocation,
            dry_run=False,
        )
        journal.log(decision)
        log.info("Decision logged: rebalance (no trades)")

        state["last_rebalance"] = timestamp
        save_state(state)
        return

    print(f"\nExecuting {len(trades)} trades...")

    # Execute
    results = execute_trades(trades, dry_run)

    # Update ledger
    if not dry_run:
        update_ledger_from_trades(ledger, results, timestamp)
        ledger["stats"]["total_rebalances"] += 1
        save_ledger(ledger)

        state["last_rebalance"] = timestamp
        state["rebalance_count"] = state.get("rebalance_count", 0) + 1
        save_state(state)

    # Log decision
    journal = DecisionJournal()
    current_positions = ledger.get("positions", [])
    decision = build_xs_rebalance_decision(
        rankings=rankings,
        current_positions=current_positions,
        trades=trades,
        results=results,
        allocation=allocation,
        dry_run=dry_run,
    )
    journal.log(decision)
    log.info(f"Decision logged: {decision.action}")

    # Summary
    executed = [r for r in results if r.get("status") == "executed"]
    failed = [r for r in results if r.get("status") == "failed"]

    print(f"\nRebalance complete:")
    print(f"  Executed: {len(executed)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed trades:")
        for t in failed:
            print(f"  {t['action']} {t['symbol']}: {t.get('error', 'unknown')}")


def cmd_history():
    """Show rebalance history."""
    ledger = load_ledger()

    print("=" * 60)
    print("REBALANCE HISTORY")
    print("=" * 60)

    print(f"\nStats:")
    print(f"  Total rebalances: {ledger['stats']['total_rebalances']}")
    print(f"  Total trades: {ledger['stats']['total_trades']}")
    print(f"  Realized P&L: ${ledger['stats']['realized_pnl']:+,.2f}")

    closed = ledger.get("closed_trades", [])
    if closed:
        print(f"\nClosed positions ({len(closed)}):")
        for c in closed[-10:]:  # Last 10
            pnl_pct = (c["exit_price"] / c["entry_price"] - 1) * 100
            print(f"  {c['symbol']:6} {c['entry_date'][:10]} -> {c['exit_date'][:10]} "
                  f"P&L: ${c.get('pnl', 0):+,.2f} ({pnl_pct:+.1f}%)")

    trades = ledger.get("trades", [])
    if trades:
        print(f"\nRecent trades ({len(trades)} total):")
        for t in trades[-10:]:
            print(f"  {t['timestamp'][:16]} {t['action']:4} {t['symbol']:6} "
                  f"x{t['shares']} @ ${t['price']:.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-sectional momentum autopilot")
    parser.add_argument("command", nargs="?", default="status",
                        choices=["status", "rankings", "preview", "run", "history"])
    parser.add_argument("--force", action="store_true", help="Force rebalance")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute trades")
    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "rankings":
        cmd_rankings()
    elif args.command == "preview":
        cmd_preview()
    elif args.command == "run":
        cmd_run(force=args.force, dry_run=args.dry_run)
    elif args.command == "history":
        cmd_history()


if __name__ == "__main__":
    main()
