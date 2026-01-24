#!/usr/bin/env python3
"""
monitor.py - position monitoring daemon

Checks positions and alerts when:
- Stop price is hit
- Momentum dying (close < 20 MA)
- Trailing stop triggered (gave back 50% of gains)

Run periodically during market hours.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader
from edge import EdgeTrader, ATR_STOP_MULT, TRAIL_GIVEBACK

# Track high water marks
HWMARKS_FILE = Path(__file__).parent / "high_water_marks.json"


def load_high_water_marks() -> dict:
    """Load high water marks for positions"""
    if HWMARKS_FILE.exists():
        with open(HWMARKS_FILE) as f:
            return json.load(f)
    return {}


def save_high_water_marks(marks: dict):
    """Save high water marks"""
    with open(HWMARKS_FILE, "w") as f:
        json.dump(marks, f, indent=2)


def update_high_water(symbol: str, current_price: float) -> float:
    """Update high water mark, return the mark"""
    marks = load_high_water_marks()

    if symbol not in marks:
        marks[symbol] = current_price
    elif current_price > marks[symbol]:
        marks[symbol] = current_price

    save_high_water_marks(marks)
    return marks[symbol]


def clear_high_water(symbol: str):
    """Clear high water mark when position closed"""
    marks = load_high_water_marks()
    if symbol in marks:
        del marks[symbol]
        save_high_water_marks(marks)


def check_position(edge: EdgeTrader, position: dict) -> dict:
    """Check a single position for exit signals"""
    symbol = position["symbol"]
    entry_price = position["avg_entry"]
    current_price = position["current_price"]
    qty = position["qty"]

    # Update high water mark
    high_water = update_high_water(symbol, current_price)

    # Get exit check from edge
    exit_check = edge.check_exit(symbol, entry_price, high_water)

    # Calculate P&L
    pnl = (current_price - entry_price) * float(qty)
    pnl_pct = (current_price - entry_price) / entry_price * 100

    # Build alert status
    alerts = []

    if exit_check.get("exit"):
        alerts.append(f"EXIT SIGNAL: {exit_check['reason']}")
    else:
        # Near-miss warnings
        stop_price = exit_check.get("stop", 0)
        ma20 = exit_check.get("ma20", 0)

        # Within 1% of stop
        if stop_price and current_price < stop_price * 1.01:
            alerts.append(f"WARNING: Near stop (${stop_price:.2f})")

        # Within 1% of MA20
        if ma20 and current_price < ma20 * 1.01:
            alerts.append(f"WARNING: Near 20 MA (${ma20:.2f})")

    return {
        "symbol": symbol,
        "qty": qty,
        "entry": entry_price,
        "current": current_price,
        "high_water": high_water,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "stop": exit_check.get("stop"),
        "ma20": exit_check.get("ma20"),
        "exit_signal": exit_check.get("exit", False),
        "exit_reason": exit_check.get("reason"),
        "alerts": alerts,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor positions")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show alerts")
    parser.add_argument("--clear", help="Clear high water mark for symbol")
    args = parser.parse_args()

    if args.clear:
        clear_high_water(args.clear)
        print(f"Cleared high water mark for {args.clear}")
        return

    edge = EdgeTrader()
    trader = Trader()

    # Check market
    clock = trader.get_clock()

    print("=" * 60)
    print("POSITION MONITOR")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market: {'OPEN' if clock['is_open'] else 'CLOSED'}")
    print("=" * 60)

    # Get positions
    positions = trader.get_positions()

    if not positions:
        if not args.quiet:
            print("\nNo positions to monitor.")
        return

    # Check each position
    has_alerts = False

    for p in positions:
        status = check_position(edge, p)

        if args.quiet and not status["alerts"]:
            continue

        print(f"\n{status['symbol']}")
        print(f"  Qty: {status['qty']}")
        print(f"  Entry: ${status['entry']:.2f}")
        print(f"  Current: ${status['current']:.2f}")
        print(f"  High Water: ${status['high_water']:.2f}")
        print(f"  P&L: ${status['pnl']:+,.2f} ({status['pnl_pct']:+.1f}%)")

        if status["stop"]:
            print(f"  Stop: ${status['stop']:.2f}")
        if status["ma20"]:
            print(f"  20 MA: ${status['ma20']:.2f}")

        if status["alerts"]:
            has_alerts = True
            print()
            for alert in status["alerts"]:
                print(f"  *** {alert} ***")

    # Summary
    print("\n" + "=" * 60)

    if has_alerts:
        print("ACTION REQUIRED - Check alerts above")
    else:
        print("All positions healthy")

    print("=" * 60)


if __name__ == "__main__":
    main()
