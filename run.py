#!/usr/bin/env python3
"""
run.py - trading runner

Run the strategy and log results.
Can be called periodically or manually.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from strategy import Strategy

LOG_FILE = Path(__file__).parent / "trade_log.jsonl"


def log_run(result: dict) -> None:
    """Append run result to log"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "status": result["status"],
        "actions": result["actions"],
        "portfolio_value": result["state"]["portfolio_value"],
        "cash": result["state"]["cash"],
        "position_count": result["state"]["position_count"],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def show_log(n: int = 10) -> None:
    """Show recent log entries"""
    if not LOG_FILE.exists():
        print("No log file yet")
        return

    lines = LOG_FILE.read_text().strip().split("\n")
    entries = [json.loads(line) for line in lines[-n:]]

    print(f"Recent Runs (last {len(entries)})")
    print("-" * 60)
    for entry in entries:
        ts = entry["timestamp"][:19]
        status = entry["status"]
        actions = len(entry["actions"])
        value = entry["portfolio_value"]
        print(f"{ts}  {status:12}  {actions} actions  ${value:,.2f}")
        for action in entry["actions"]:
            print(f"    {action['type']:4} {action['symbol']:6} x{action['qty']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trading runner")
    parser.add_argument("command", nargs="?", default="check",
                       choices=["check", "run", "log"])
    parser.add_argument("-n", type=int, default=10, help="Log entries to show")
    args = parser.parse_args()

    if args.command == "log":
        show_log(args.n)
        return

    # Run strategy
    strategy = Strategy(dry_run=(args.command != "run"))
    result = strategy.run()

    # Log result
    log_run(result)

    # Output
    print(f"[{datetime.now().isoformat()[:19]}] {result['status']}")
    if result["actions"]:
        for action in result["actions"]:
            cost = action["qty"] * action["price"]
            print(f"  {action['type']:4} {action['symbol']:6} x{action['qty']:4}  ${cost:,.2f}")
    else:
        market_status = "open" if result["state"]["market_open"] else "closed"
        print(f"  no actions (market {market_status})")


if __name__ == "__main__":
    main()
