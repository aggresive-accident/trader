#!/usr/bin/env python3
"""
performance.py - portfolio performance tracking

Tracks portfolio value over time.
Calculates returns and metrics.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader

PERF_FILE = Path(__file__).parent / "performance.jsonl"


def record_snapshot() -> dict:
    """Record current portfolio snapshot"""
    try:
        trader = Trader()
        account = trader.get_account()
        positions = trader.get_positions()
        clock = trader.get_clock()

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": account["portfolio_value"],
            "cash": account["cash"],
            "equity": account["equity"],
            "pl_today": account["pl_today"],
            "pl_today_pct": account["pl_today_pct"],
            "positions": len(positions),
            "market_open": clock["is_open"],
        }

        # Append to file
        with open(PERF_FILE, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

        return snapshot
    except Exception as e:
        return {"error": str(e)}


def load_history() -> list[dict]:
    """Load performance history"""
    if not PERF_FILE.exists():
        return []

    entries = []
    for line in PERF_FILE.read_text().strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except:
                pass
    return entries


def calculate_metrics(history: list[dict]) -> dict:
    """Calculate performance metrics"""
    if len(history) < 2:
        return {"error": "not enough data"}

    values = [h["portfolio_value"] for h in history]

    # Basic metrics
    initial = values[0]
    current = values[-1]
    total_return = ((current - initial) / initial) * 100

    # Daily returns
    daily_returns = []
    for i in range(1, len(values)):
        ret = (values[i] - values[i-1]) / values[i-1]
        daily_returns.append(ret)

    # Volatility (std of daily returns)
    if daily_returns:
        avg_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
        volatility = variance ** 0.5
    else:
        volatility = 0

    # Max drawdown
    peak = values[0]
    max_drawdown = 0
    for v in values:
        if v > peak:
            peak = v
        drawdown = (peak - v) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Sharpe ratio (simplified, assumes 0 risk-free rate)
    if volatility > 0:
        sharpe = avg_return / volatility if daily_returns else 0
    else:
        sharpe = 0

    return {
        "initial_value": initial,
        "current_value": current,
        "total_return_pct": total_return,
        "data_points": len(history),
        "volatility": volatility * 100,  # as percentage
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio performance tracking")
    parser.add_argument("command", nargs="?", default="status",
                       choices=["status", "record", "history", "metrics", "clear"])
    parser.add_argument("-n", type=int, default=10, help="Number of entries")
    args = parser.parse_args()

    if args.command == "record":
        snapshot = record_snapshot()
        if "error" in snapshot:
            print(f"Error: {snapshot['error']}")
        else:
            print(f"Recorded: ${snapshot['portfolio_value']:,.2f}")
        return

    if args.command == "clear":
        if PERF_FILE.exists():
            PERF_FILE.unlink()
        print("Performance history cleared")
        return

    if args.command == "history":
        history = load_history()
        if not history:
            print("No performance history")
            return

        print(f"Performance History (last {min(args.n, len(history))})")
        print("-" * 60)
        for h in history[-args.n:]:
            ts = h["timestamp"][:19]
            value = h["portfolio_value"]
            pl = h.get("pl_today", 0)
            pos = h.get("positions", 0)
            print(f"{ts}  ${value:>12,.2f}  {pl:>+8.2f}  {pos} pos")
        return

    if args.command == "metrics":
        history = load_history()
        if not history:
            print("No performance history")
            return

        metrics = calculate_metrics(history)
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        print("Performance Metrics")
        print("-" * 40)
        print(f"Initial Value:    ${metrics['initial_value']:,.2f}")
        print(f"Current Value:    ${metrics['current_value']:,.2f}")
        print(f"Total Return:     {metrics['total_return_pct']:+.2f}%")
        print(f"Data Points:      {metrics['data_points']}")
        print(f"Volatility:       {metrics['volatility']:.2f}%")
        print(f"Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        return

    # Default: status
    history = load_history()
    if not history:
        # Record first snapshot
        print("Recording first snapshot...")
        snapshot = record_snapshot()
        if "error" in snapshot:
            print(f"Error: {snapshot['error']}")
        else:
            print(f"Portfolio: ${snapshot['portfolio_value']:,.2f}")
            print(f"Cash: ${snapshot['cash']:,.2f}")
            print(f"Positions: {snapshot['positions']}")
        return

    # Show latest and comparison
    latest = history[-1]
    first = history[0]

    total_return = ((latest["portfolio_value"] - first["portfolio_value"]) / first["portfolio_value"]) * 100

    print("Portfolio Performance")
    print("-" * 40)
    print(f"Current Value:  ${latest['portfolio_value']:,.2f}")
    print(f"Today's P&L:    ${latest.get('pl_today', 0):+,.2f} ({latest.get('pl_today_pct', 0):+.2f}%)")
    print(f"Total Return:   {total_return:+.2f}%")
    print(f"Data Points:    {len(history)}")

    # Quick metrics if enough data
    if len(history) >= 5:
        metrics = calculate_metrics(history)
        if "error" not in metrics:
            print(f"Max Drawdown:   {metrics['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    main()
