#!/usr/bin/env python3
"""
equity.py - track daily portfolio equity curve

Records daily portfolio value for performance tracking.
Run once per day to build equity history.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader

EQUITY_FILE = Path(__file__).parent / "equity_curve.json"
STARTING_CAPITAL = 100000.0


def load_equity() -> list:
    """Load equity history"""
    if EQUITY_FILE.exists():
        return json.loads(EQUITY_FILE.read_text())
    return []


def save_equity(data: list):
    """Save equity history"""
    EQUITY_FILE.write_text(json.dumps(data, indent=2))


def record_equity():
    """Record today's equity"""
    trader = Trader()
    account = trader.get_account()

    equity = float(account["portfolio_value"])
    cash = float(account["cash"])

    positions = trader.get_positions()
    pos_value = sum(float(p["market_value"]) for p in positions)

    today = datetime.now().strftime("%Y-%m-%d")

    history = load_equity()

    # Check if already recorded today
    if history and history[-1].get("date") == today:
        # Update existing record
        history[-1] = {
            "date": today,
            "equity": equity,
            "cash": cash,
            "positions_value": pos_value,
            "positions_count": len(positions),
            "return_pct": (equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100,
        }
    else:
        # Add new record
        history.append({
            "date": today,
            "equity": equity,
            "cash": cash,
            "positions_value": pos_value,
            "positions_count": len(positions),
            "return_pct": (equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100,
        })

    save_equity(history)
    return history[-1]


def calculate_stats(history: list) -> dict:
    """Calculate performance statistics"""
    if not history:
        return {}

    equities = [h["equity"] for h in history]

    # Returns
    total_return = (equities[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    # Daily returns
    daily_returns = []
    for i in range(1, len(equities)):
        ret = (equities[i] - equities[i-1]) / equities[i-1]
        daily_returns.append(ret)

    # Volatility
    if daily_returns:
        import math
        avg_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_ret)**2 for r in daily_returns) / len(daily_returns)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(252) * 100
    else:
        annual_vol = 0

    # Max drawdown
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe (assuming 5% risk-free rate)
    if daily_returns and annual_vol > 0:
        avg_annual_ret = (sum(daily_returns) / len(daily_returns)) * 252 * 100
        sharpe = (avg_annual_ret - 5) / annual_vol
    else:
        sharpe = 0

    return {
        "days": len(history),
        "total_return_pct": total_return,
        "annual_volatility_pct": annual_vol,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "current_equity": equities[-1],
        "peak_equity": max(equities),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Equity curve tracking")
    parser.add_argument("command", nargs="?", default="record",
                       choices=["record", "show", "stats", "export"])
    args = parser.parse_args()

    if args.command == "record":
        record = record_equity()
        print(f"Recorded: ${record['equity']:,.2f} ({record['return_pct']:+.2f}%)")
        return

    if args.command == "show":
        history = load_equity()
        if not history:
            print("No equity history")
            return

        print("=" * 60)
        print("EQUITY CURVE")
        print("=" * 60)
        print(f"{'Date':12} {'Equity':>12} {'Return':>8} {'Positions':>10}")
        print("-" * 60)

        for h in history[-30:]:  # Last 30 days
            print(f"{h['date']:12} ${h['equity']:>10,.0f} {h['return_pct']:>+7.2f}% {h['positions_count']:>10}")

        return

    if args.command == "stats":
        history = load_equity()
        if not history:
            print("No equity history")
            return

        stats = calculate_stats(history)
        print("=" * 60)
        print("PERFORMANCE STATISTICS")
        print("=" * 60)
        print(f"Days tracked:     {stats['days']}")
        print(f"Total return:     {stats['total_return_pct']:+.2f}%")
        print(f"Annual volatility: {stats['annual_volatility_pct']:.1f}%")
        print(f"Max drawdown:     {stats['max_drawdown_pct']:.1f}%")
        print(f"Sharpe ratio:     {stats['sharpe_ratio']:.2f}")
        print(f"Current equity:   ${stats['current_equity']:,.2f}")
        print(f"Peak equity:      ${stats['peak_equity']:,.2f}")
        return

    if args.command == "export":
        history = load_equity()
        if not history:
            print("No equity history")
            return

        # Export as CSV
        print("date,equity,cash,positions_value,return_pct")
        for h in history:
            print(f"{h['date']},{h['equity']:.2f},{h['cash']:.2f},{h['positions_value']:.2f},{h['return_pct']:.2f}")


if __name__ == "__main__":
    main()
