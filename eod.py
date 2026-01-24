#!/usr/bin/env python3
"""
eod.py - end of day trading summary

Run after market close to review:
- Day's trades
- Position changes
- P&L for the day
- Signals that were missed
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader
from edge import EdgeTrader

JOURNAL_FILE = Path(__file__).parent / "trades.json"
EQUITY_FILE = Path(__file__).parent / "equity_curve.json"


def load_trades() -> list:
    """Load trade journal"""
    if JOURNAL_FILE.exists():
        return json.loads(JOURNAL_FILE.read_text())
    return []


def load_equity() -> list:
    """Load equity history"""
    if EQUITY_FILE.exists():
        return json.loads(EQUITY_FILE.read_text())
    return []


def get_todays_trades(trades: list) -> list:
    """Filter trades from today"""
    today = datetime.now().strftime("%Y-%m-%d")
    return [t for t in trades if t.get("timestamp", "").startswith(today)]


def calculate_daily_pnl(trader: Trader) -> dict:
    """Calculate today's P&L"""
    account = trader.get_account()
    equity = float(account["portfolio_value"])

    # Get yesterday's equity from history
    history = load_equity()
    if len(history) >= 2:
        yesterday_equity = history[-2]["equity"]
    else:
        yesterday_equity = 100000.0  # Starting capital

    daily_pnl = equity - yesterday_equity
    daily_pnl_pct = (daily_pnl / yesterday_equity) * 100

    return {
        "equity": equity,
        "yesterday": yesterday_equity,
        "pnl": daily_pnl,
        "pnl_pct": daily_pnl_pct,
    }


def get_position_summary(trader: Trader) -> list:
    """Get current position summary"""
    positions = trader.get_positions()
    summary = []

    for p in positions:
        entry = float(p["avg_entry"])
        current = float(p["current_price"])
        qty = float(p["qty"])
        pnl = (current - entry) * qty
        pnl_pct = (current - entry) / entry * 100

        summary.append({
            "symbol": p["symbol"],
            "qty": qty,
            "entry": entry,
            "current": current,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "market_value": float(p["market_value"]),
        })

    return summary


def get_missed_signals(edge: EdgeTrader) -> list:
    """Get signals that appeared today but weren't traded"""
    watchlist = [
        "AMD", "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "AVGO", "CRM", "ORCL", "NFLX", "ADBE", "INTC", "QCOM",
        "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
    ]

    setups = edge.scan_for_setups(watchlist)

    # Load today's trades
    trades = get_todays_trades(load_trades())
    traded_symbols = {t.get("symbol") for t in trades}

    # Filter to signals not traded
    missed = [s for s in setups if s["symbol"] not in traded_symbols]

    return missed


def main():
    trader = Trader()
    edge = EdgeTrader()

    print("=" * 60)
    print("END OF DAY SUMMARY")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)

    # Market status
    clock = trader.get_clock()
    print(f"\nMarket: {'OPEN' if clock['is_open'] else 'CLOSED'}")

    # Daily P&L
    pnl = calculate_daily_pnl(trader)
    print(f"\n{'='*60}")
    print("DAILY P&L")
    print("-" * 60)
    print(f"  Starting equity: ${pnl['yesterday']:,.2f}")
    print(f"  Ending equity:   ${pnl['equity']:,.2f}")
    print(f"  Daily P&L:       ${pnl['pnl']:+,.2f} ({pnl['pnl_pct']:+.2f}%)")

    # Today's trades
    trades = get_todays_trades(load_trades())
    print(f"\n{'='*60}")
    print(f"TODAY'S TRADES ({len(trades)})")
    print("-" * 60)

    if trades:
        for t in trades:
            action = t.get("action", "?")
            symbol = t.get("symbol", "?")
            shares = t.get("shares", 0)
            reason = t.get("reason", "")[:40]
            ts = t.get("timestamp", "")[-8:-3]  # HH:MM
            print(f"  {ts} {action:4} {symbol:5} {shares:4} shares - {reason}")
    else:
        print("  No trades today")

    # Current positions
    positions = get_position_summary(trader)
    print(f"\n{'='*60}")
    print(f"POSITIONS ({len(positions)})")
    print("-" * 60)

    if positions:
        for p in positions:
            print(f"  {p['symbol']:5} {p['qty']:4.0f} @ ${p['entry']:.2f}")
            print(f"        Now ${p['current']:.2f} | P&L: ${p['pnl']:+,.0f} ({p['pnl_pct']:+.1f}%)")
    else:
        print("  No positions")

    # Missed signals
    missed = get_missed_signals(edge)
    print(f"\n{'='*60}")
    print(f"SIGNALS NOT TRADED ({len(missed)})")
    print("-" * 60)

    if missed:
        for m in missed[:5]:  # Top 5
            print(f"  {m['symbol']:5} (score {m['score']}) @ ${m['price']:.2f}")
            print(f"        {', '.join(m['reasons'][:3])}")
    else:
        print("  No signals missed")

    # Record equity
    from equity import record_equity
    record = record_equity()
    print(f"\n{'='*60}")
    print("EQUITY RECORDED")
    print("-" * 60)
    print(f"  ${record['equity']:,.2f} ({record['return_pct']:+.2f}% total)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
