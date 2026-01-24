#!/usr/bin/env python3
"""
visualize.py - ASCII visualization for trading data

Creates text-based charts for:
- Equity curve
- Trade distribution
- Monthly returns
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

EQUITY_FILE = Path(__file__).parent / "equity_curve.json"
JOURNAL_FILE = Path(__file__).parent / "trades.json"


def load_equity() -> list:
    """Load equity history"""
    if EQUITY_FILE.exists():
        return json.loads(EQUITY_FILE.read_text())
    return []


def load_trades() -> list:
    """Load trade journal"""
    if JOURNAL_FILE.exists():
        return json.loads(JOURNAL_FILE.read_text())
    return []


def draw_sparkline(values: list, width: int = 50) -> str:
    """Draw a simple ASCII sparkline"""
    if not values:
        return ""

    # Sparkline characters (8 levels)
    chars = " ▁▂▃▄▅▆▇█"

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    # Sample to width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    # Map to characters
    result = ""
    for v in sampled:
        level = int((v - min_val) / range_val * 7)
        result += chars[level + 1]

    return result


def draw_bar_chart(data: dict, width: int = 40, title: str = "") -> str:
    """Draw horizontal bar chart"""
    if not data:
        return "No data"

    lines = []
    if title:
        lines.append(title)
        lines.append("-" * (width + 15))

    max_val = max(abs(v) for v in data.values()) if data.values() else 1
    max_label = max(len(str(k)) for k in data.keys()) if data else 5

    for label, value in data.items():
        bar_width = int(abs(value) / max_val * width) if max_val > 0 else 0
        bar = "█" * bar_width

        if value >= 0:
            lines.append(f"{str(label):>{max_label}} │{bar} {value:+.1f}")
        else:
            lines.append(f"{str(label):>{max_label}} │{bar} {value:.1f}")

    return "\n".join(lines)


def equity_curve_chart(history: list) -> str:
    """Draw equity curve"""
    if not history:
        return "No equity data"

    lines = []
    lines.append("EQUITY CURVE")
    lines.append("=" * 60)

    # Extract values
    dates = [h["date"] for h in history]
    values = [h["equity"] for h in history]

    # Stats
    start = values[0]
    end = values[-1]
    peak = max(values)
    low = min(values)
    total_return = (end - start) / start * 100

    # Sparkline
    spark = draw_sparkline(values, 50)
    lines.append(f"[{spark}]")
    lines.append(f" {dates[0][:7]}{'':>36}{dates[-1][:7]}")
    lines.append("")

    # Stats
    lines.append(f"Start:  ${start:>10,.0f}  |  End:   ${end:>10,.0f}")
    lines.append(f"Peak:   ${peak:>10,.0f}  |  Low:   ${low:>10,.0f}")
    lines.append(f"Return: {total_return:>+10.2f}%")

    return "\n".join(lines)


def monthly_returns_chart(history: list) -> str:
    """Draw monthly returns"""
    if len(history) < 2:
        return "Need more data for monthly returns"

    # Group by month
    monthly = defaultdict(list)
    prev_equity = history[0]["equity"]

    for h in history[1:]:
        month = h["date"][:7]  # YYYY-MM
        equity = h["equity"]
        ret = (equity - prev_equity) / prev_equity * 100
        monthly[month].append(ret)
        prev_equity = equity

    # Sum returns per month
    month_totals = {m: sum(rets) for m, rets in monthly.items()}

    return draw_bar_chart(month_totals, 30, "MONTHLY RETURNS (%)")


def trade_distribution_chart(trades: list) -> str:
    """Draw trade distribution by symbol"""
    if not trades:
        return "No trades"

    # Count by symbol
    by_symbol = defaultdict(int)
    for t in trades:
        sym = t.get("symbol", "?")
        by_symbol[sym] += 1

    # Sort by count
    sorted_symbols = dict(sorted(by_symbol.items(), key=lambda x: -x[1]))

    return draw_bar_chart(sorted_symbols, 30, "TRADES BY SYMBOL")


def win_loss_chart(trades: list) -> str:
    """Simple win/loss visualization"""
    if not trades:
        return "No trades"

    lines = []
    lines.append("TRADE RESULTS")
    lines.append("-" * 40)

    # Match buys and sells (simplified)
    buys = [t for t in trades if t.get("action") == "BUY"]
    sells = [t for t in trades if t.get("action") == "SELL"]

    lines.append(f"Buys:  {len(buys)}")
    lines.append(f"Sells: {len(sells)}")

    # Show recent trades
    if trades:
        lines.append("")
        lines.append("Recent (last 10):")
        for t in trades[-10:]:
            action = t.get("action", "?")
            symbol = t.get("symbol", "?")
            ts = t.get("timestamp", "")[:10]
            marker = "→" if action == "BUY" else "←"
            lines.append(f"  {ts} {marker} {action:4} {symbol}")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trading visualizations")
    parser.add_argument("chart", nargs="?", default="all",
                       choices=["all", "equity", "monthly", "trades"])
    args = parser.parse_args()

    history = load_equity()
    trades = load_trades()

    charts = []

    if args.chart in ["all", "equity"]:
        charts.append(equity_curve_chart(history))

    if args.chart in ["all", "monthly"]:
        charts.append(monthly_returns_chart(history))

    if args.chart in ["all", "trades"]:
        charts.append(trade_distribution_chart(trades))
        charts.append(win_loss_chart(trades))

    print("\n\n".join(charts))


if __name__ == "__main__":
    main()
