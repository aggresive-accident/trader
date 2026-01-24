#!/usr/bin/env python3
"""
execute.py - trade execution

Actually places trades based on edge.py signals.
Includes entry improvement logic - don't chase extended stocks.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader
from edge import EdgeTrader

# Journal file
JOURNAL_FILE = Path(__file__).parent / "trades.json"


def load_journal() -> list:
    """Load trade journal"""
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE) as f:
            return json.load(f)
    return []


def save_journal(trades: list):
    """Save trade journal"""
    with open(JOURNAL_FILE, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def log_trade(trade: dict):
    """Log a trade to journal"""
    trades = load_journal()
    trade["timestamp"] = datetime.now().isoformat()
    trades.append(trade)
    save_journal(trades)


def execute_buy(symbol: str, shares: int, stop_price: float, reason: str, dry_run: bool = True) -> dict:
    """Execute a buy order"""
    trader = Trader()

    if dry_run:
        print(f"[DRY RUN] Would buy {shares} {symbol}")
        return {"dry_run": True, "symbol": symbol, "shares": shares}

    try:
        order = trader.buy(symbol, shares)

        trade = {
            "action": "BUY",
            "symbol": symbol,
            "shares": shares,
            "order_id": order["id"],
            "status": order["status"],
            "stop_price": stop_price,
            "reason": reason,
        }
        log_trade(trade)

        return order
    except Exception as e:
        return {"error": str(e)}


def execute_sell(symbol: str, shares: int, reason: str, dry_run: bool = True) -> dict:
    """Execute a sell order"""
    trader = Trader()

    if dry_run:
        print(f"[DRY RUN] Would sell {shares} {symbol}")
        return {"dry_run": True, "symbol": symbol, "shares": shares}

    try:
        order = trader.sell(symbol, shares)

        trade = {
            "action": "SELL",
            "symbol": symbol,
            "shares": shares,
            "order_id": order["id"],
            "status": order["status"],
            "reason": reason,
        }
        log_trade(trade)

        return order
    except Exception as e:
        return {"error": str(e)}


def check_entry_quality(setup: dict) -> dict:
    """Check if entry is good or if we should wait"""
    # If stock is up >10% in a week, it's extended - wait for pullback
    if setup["week_return"] > 10:
        pullback_target = setup["price"] * 0.95  # Wait for 5% pullback
        return {
            "quality": "WAIT",
            "reason": f"Extended (+{setup['week_return']:.1f}% week). Wait for pullback to ~${pullback_target:.0f}",
            "target_entry": pullback_target,
        }

    # If stock is up 5-10% week with volume, good entry
    if setup["week_return"] > 5 and setup.get("vol_ratio", 1) > 1.3:
        return {
            "quality": "GOOD",
            "reason": "Strong momentum with volume confirmation",
        }

    # If stock just broke out (up <5% week but above MAs), great entry
    if setup["week_return"] < 5 and setup["week_return"] > 0:
        return {
            "quality": "GREAT",
            "reason": "Early breakout, not extended",
        }

    return {
        "quality": "OK",
        "reason": "Acceptable entry",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Execute trades")
    parser.add_argument("--execute", action="store_true", help="Actually execute (default: dry run)")
    parser.add_argument("--symbol", help="Specific symbol to trade")
    parser.add_argument("--action", choices=["buy", "sell"], help="Action to take")
    parser.add_argument("--shares", type=int, help="Number of shares")
    args = parser.parse_args()

    edge = EdgeTrader()
    trader = Trader()

    print("=" * 60)
    print("TRADE EXECUTION")
    print(f"Mode: {'LIVE' if args.execute else 'DRY RUN'}")
    print("=" * 60)

    # Check market
    try:
        clock = trader.get_clock()
        print(f"\nMarket: {'OPEN' if clock['is_open'] else 'CLOSED'}")
        if not clock["is_open"]:
            print(f"Next open: {clock['next_open']}")
            if args.execute:
                print("\n[WARNING] Market is closed. Orders will queue.")
    except:
        print("\nCould not check market status")

    # Current positions
    positions = trader.get_positions()
    print(f"\nPositions: {len(positions)}")
    for p in positions:
        print(f"  {p['symbol']}: {p['qty']} @ ${p['current_price']:.2f} ({p['unrealized_pl_pct']:+.1f}%)")

    # If manual trade specified
    if args.symbol and args.action and args.shares:
        if args.action == "buy":
            result = execute_buy(args.symbol, args.shares, 0, "manual", dry_run=not args.execute)
        else:
            result = execute_sell(args.symbol, args.shares, "manual", dry_run=not args.execute)
        print(f"\nResult: {result}")
        return

    # Auto trading logic
    print("\n" + "-" * 60)
    print("CHECKING SIGNALS...")
    print("-" * 60)

    # Check for exit signals on current positions
    for p in positions:
        sym = p["symbol"]
        entry = p["avg_entry"]
        current = p["current_price"]
        high_water = max(entry, current)  # Simplified - would need to track actual high

        exit_check = edge.check_exit(sym, entry, high_water)
        if exit_check.get("exit"):
            print(f"\n[EXIT SIGNAL] {sym}: {exit_check['reason']}")
            result = execute_sell(sym, int(p["qty"]), exit_check["reason"], dry_run=not args.execute)
            print(f"Result: {result}")

    # Check for entry signals if no position
    if not positions:
        watchlist = ["AMD", "NVDA", "META", "GOOGL", "AAPL", "MSFT", "INTC", "AVGO"]
        setups = edge.scan_for_setups(watchlist)

        if setups:
            best = setups[0]
            entry_quality = check_entry_quality(best)

            print(f"\nBest setup: {best['symbol']} (score: {best['score']})")
            print(f"Entry quality: {entry_quality['quality']}")
            print(f"Reason: {entry_quality['reason']}")

            if entry_quality["quality"] in ["GOOD", "GREAT"]:
                pos = edge.calculate_position(best)
                print(f"\n[ENTRY SIGNAL] Buy {pos['shares']} {best['symbol']} @ ${best['price']:.2f}")
                print(f"Stop: ${pos['stop_price']:.2f}, Risk: ${pos['risk_dollars']:.0f}")

                if args.execute:
                    result = execute_buy(
                        best["symbol"],
                        pos["shares"],
                        pos["stop_price"],
                        ", ".join(best["reasons"]),
                        dry_run=False
                    )
                    print(f"Result: {result}")
                else:
                    print("[DRY RUN] Would execute above trade")

            elif entry_quality["quality"] == "WAIT":
                print(f"\n[WAIT] {best['symbol']} is extended")
                print(f"Set alert for pullback to ${entry_quality.get('target_entry', 0):.0f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
