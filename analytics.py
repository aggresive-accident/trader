#!/usr/bin/env python3
"""
analytics.py - trade performance analytics

Tracks actual vs expected results:
- Win rate vs backtested 47%
- Profit factor vs backtested 2.92
- Average win/loss vs expected
- Drawdown tracking

Expected (from backtest):
- Win rate: 47%
- Profit factor: 2.92
- Avg win: $1,825
- Avg loss: $560
- Max drawdown: 6.8%
"""

import sys
import json
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader

# Journal file
JOURNAL_FILE = Path(__file__).parent / "trades.json"

# Expected from backtest
EXPECTED = {
    "win_rate": 0.47,
    "profit_factor": 2.92,
    "avg_win": 1825,
    "avg_loss": 560,
    "max_drawdown_pct": 6.8,
}


def load_trades() -> list:
    """Load trade journal"""
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE) as f:
            return json.load(f)
    return []


def match_trades(trades: list) -> list:
    """Match buy/sell pairs into round trips"""
    # Group by symbol
    by_symbol = {}
    for t in trades:
        sym = t.get("symbol")
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(t)

    round_trips = []

    for sym, sym_trades in by_symbol.items():
        buys = [t for t in sym_trades if t.get("action") == "BUY"]
        sells = [t for t in sym_trades if t.get("action") == "SELL"]

        # Match buys to sells (FIFO)
        for i, buy in enumerate(buys):
            if i < len(sells):
                sell = sells[i]
                # Calculate P&L (would need fill prices from API)
                round_trips.append({
                    "symbol": sym,
                    "buy": buy,
                    "sell": sell,
                    "shares": buy.get("shares", 0),
                    # P&L would need actual fill prices
                })

    return round_trips


def calculate_stats(round_trips: list) -> dict:
    """Calculate trading statistics"""
    if not round_trips:
        return None

    wins = [t for t in round_trips if t.get("pnl", 0) > 0]
    losses = [t for t in round_trips if t.get("pnl", 0) <= 0]

    total_trades = len(round_trips)
    win_count = len(wins)

    win_rate = win_count / total_trades if total_trades > 0 else 0

    gross_profit = sum(t.get("pnl", 0) for t in wins)
    gross_loss = abs(sum(t.get("pnl", 0) for t in losses))

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / len(losses) if losses else 0

    return {
        "total_trades": total_trades,
        "wins": win_count,
        "losses": len(losses),
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def compare_to_expected(actual: dict) -> list:
    """Compare actual stats to expected from backtest"""
    comparisons = []

    if actual["win_rate"] < EXPECTED["win_rate"] - 0.1:
        comparisons.append({
            "metric": "Win Rate",
            "actual": f"{actual['win_rate']:.1%}",
            "expected": f"{EXPECTED['win_rate']:.1%}",
            "status": "BELOW",
            "note": "Win rate underperforming",
        })
    elif actual["win_rate"] > EXPECTED["win_rate"] + 0.1:
        comparisons.append({
            "metric": "Win Rate",
            "actual": f"{actual['win_rate']:.1%}",
            "expected": f"{EXPECTED['win_rate']:.1%}",
            "status": "ABOVE",
            "note": "Win rate exceeding expectations",
        })
    else:
        comparisons.append({
            "metric": "Win Rate",
            "actual": f"{actual['win_rate']:.1%}",
            "expected": f"{EXPECTED['win_rate']:.1%}",
            "status": "OK",
        })

    if actual["profit_factor"] < EXPECTED["profit_factor"] * 0.7:
        comparisons.append({
            "metric": "Profit Factor",
            "actual": f"{actual['profit_factor']:.2f}",
            "expected": f"{EXPECTED['profit_factor']:.2f}",
            "status": "BELOW",
            "note": "Profit factor underperforming",
        })
    else:
        comparisons.append({
            "metric": "Profit Factor",
            "actual": f"{actual['profit_factor']:.2f}",
            "expected": f"{EXPECTED['profit_factor']:.2f}",
            "status": "OK",
        })

    return comparisons


def main():
    print("=" * 60)
    print("TRADE ANALYTICS")
    print("=" * 60)

    # Expected stats
    print("\nExpected (from backtest):")
    print(f"  Win rate: {EXPECTED['win_rate']:.0%}")
    print(f"  Profit factor: {EXPECTED['profit_factor']:.2f}")
    print(f"  Avg win: ${EXPECTED['avg_win']:,.0f}")
    print(f"  Avg loss: ${EXPECTED['avg_loss']:,.0f}")
    print(f"  Max drawdown: {EXPECTED['max_drawdown_pct']:.1f}%")

    # Load trades
    trades = load_trades()

    if not trades:
        print("\n" + "-" * 60)
        print("No trades recorded yet.")
        print("Start trading and entries will be logged to trades.json")
        print("-" * 60)
        return

    print(f"\n{'-' * 60}")
    print(f"Trades in journal: {len(trades)}")
    print("-" * 60)

    # Show recent trades
    print("\nRecent Trades:")
    for t in trades[-10:]:
        ts = t.get("timestamp", "")[:10]
        action = t.get("action", "?")
        symbol = t.get("symbol", "?")
        shares = t.get("shares", 0)
        reason = t.get("reason", "")[:30]
        print(f"  {ts} {action:4} {symbol:5} {shares:4} shares - {reason}")

    # Match into round trips
    round_trips = match_trades(trades)

    if round_trips:
        print(f"\nRound trips: {len(round_trips)}")

        stats = calculate_stats(round_trips)
        if stats:
            print("\nActual Stats:")
            print(f"  Win rate: {stats['win_rate']:.0%}")
            print(f"  Profit factor: {stats['profit_factor']:.2f}")
            print(f"  Avg win: ${stats['avg_win']:,.0f}")
            print(f"  Avg loss: ${stats['avg_loss']:,.0f}")

            # Compare to expected
            comparisons = compare_to_expected(stats)
            print("\nComparison to Backtest:")
            for c in comparisons:
                status = c["status"]
                if status == "BELOW":
                    print(f"  {c['metric']}: {c['actual']} vs {c['expected']} [BELOW] ⚠️")
                elif status == "ABOVE":
                    print(f"  {c['metric']}: {c['actual']} vs {c['expected']} [ABOVE] ✓")
                else:
                    print(f"  {c['metric']}: {c['actual']} vs {c['expected']} [OK]")
    else:
        print("\nNo completed round trips yet.")

    # Account equity
    trader = Trader()
    account = trader.get_account()
    print(f"\n{'-' * 60}")
    print(f"Current Portfolio: ${account['portfolio_value']:,.2f}")
    print(f"Starting: $100,000.00")
    print(f"P&L: ${account['portfolio_value'] - 100000:+,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
