#!/usr/bin/env python3
"""
daily.py - daily trading summary

Run once per day to get a summary of trading activity.
Saves to daily_log.md for history.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader
from scanner import Scanner
from performance import record_snapshot, load_history, calculate_metrics

DAILY_LOG = Path(__file__).parent / "daily_log.md"
LAST_RUN_FILE = Path(__file__).parent / ".daily_last_run"


def already_run_today() -> bool:
    """Check if daily summary already ran today"""
    if not LAST_RUN_FILE.exists():
        return False
    last_run = LAST_RUN_FILE.read_text().strip()
    today = datetime.now().strftime("%Y-%m-%d")
    return last_run == today


def mark_run():
    """Mark that daily ran today"""
    LAST_RUN_FILE.write_text(datetime.now().strftime("%Y-%m-%d"))


def generate_summary() -> str:
    """Generate daily summary"""
    lines = []
    now = datetime.now()

    lines.append(f"# Daily Summary - {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Portfolio status
    try:
        trader = Trader()
        account = trader.get_account()
        positions = trader.get_positions()
        clock = trader.get_clock()

        lines.append("## Portfolio")
        lines.append(f"- Value: ${account['portfolio_value']:,.2f}")
        lines.append(f"- Cash: ${account['cash']:,.2f}")
        lines.append(f"- Today's P&L: ${account['pl_today']:+,.2f} ({account['pl_today_pct']:+.2f}%)")
        lines.append(f"- Market: {'OPEN' if clock['is_open'] else 'CLOSED'}")
        lines.append("")

        # Positions
        if positions:
            lines.append("## Positions")
            for p in positions:
                lines.append(f"- {p['symbol']}: {p['qty']:.0f} shares @ ${p['current_price']:.2f} ({p['unrealized_pl_pct']:+.2f}%)")
            lines.append("")
        else:
            lines.append("## Positions")
            lines.append("- None (all cash)")
            lines.append("")

    except Exception as e:
        lines.append(f"## Portfolio")
        lines.append(f"- Error: {e}")
        lines.append("")

    # Signals
    try:
        scanner = Scanner()
        buy_candidates = scanner.buy_candidates()
        if buy_candidates:
            lines.append("## Buy Signals")
            for s in buy_candidates[:5]:
                lines.append(f"- {s['symbol']}: ${s['price']:.2f} ({s['momentum']:+.2f}%) - {s['signal']}")
            lines.append("")
    except Exception as e:
        pass

    # Performance metrics (if enough data)
    try:
        # Record today's snapshot
        record_snapshot()

        history = load_history()
        if len(history) >= 5:
            metrics = calculate_metrics(history)
            if "error" not in metrics:
                lines.append("## Performance")
                lines.append(f"- Total Return: {metrics['total_return_pct']:+.2f}%")
                lines.append(f"- Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
                lines.append(f"- Data Points: {metrics['data_points']}")
                lines.append("")
    except Exception:
        pass

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def save_summary(summary: str):
    """Append summary to daily log"""
    with open(DAILY_LOG, "a") as f:
        f.write(summary)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily trading summary")
    parser.add_argument("command", nargs="?", default="run",
                       choices=["run", "force", "view", "check"])
    args = parser.parse_args()

    if args.command == "check":
        if already_run_today():
            print("Daily summary already ran today")
        else:
            print("Daily summary has not run today")
        return

    if args.command == "view":
        if DAILY_LOG.exists():
            # Show last entry
            content = DAILY_LOG.read_text()
            entries = content.split("# Daily Summary")
            if len(entries) > 1:
                last_entry = "# Daily Summary" + entries[-1]
                print(last_entry)
            else:
                print("No entries yet")
        else:
            print("No daily log yet")
        return

    if args.command == "run":
        if already_run_today():
            print("Already ran today. Use 'force' to run again.")
            return

    # Generate and save
    print("Generating daily summary...")
    summary = generate_summary()
    save_summary(summary)
    mark_run()

    # Print summary
    print(summary)


if __name__ == "__main__":
    main()
