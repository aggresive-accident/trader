#!/usr/bin/env python3
"""
status.py - one-line trading status

Quick market/portfolio status for glance checks.
"""

import sys
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader


def time_until(target: datetime, now: datetime) -> str:
    """Human-readable time until target"""
    delta = target - now
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        return "now"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def format_market_phase(clock: dict) -> str:
    """Format market phase with countdown"""
    phase = clock.get("phase", "unknown").upper()

    # Parse timestamps
    now = datetime.fromisoformat(clock["timestamp"].replace("Z", "+00:00"))
    next_open = datetime.fromisoformat(clock["next_open"].replace("Z", "+00:00"))
    next_close = datetime.fromisoformat(clock["next_close"].replace("Z", "+00:00"))

    if clock["is_open"]:
        countdown = time_until(next_close, now)
        return f"OPEN (closes in {countdown})"
    else:
        day_name = next_open.strftime("%a")
        time_str = next_open.strftime("%-I:%M%p").lower()
        countdown = time_until(next_open, now)
        return f"{phase} (opens {day_name} {time_str}, {countdown})"


def format_positions(positions: list) -> str:
    """Format positions as compact string"""
    if not positions:
        return "no positions"

    parts = []
    for p in positions:
        pct = p["unrealized_pl_pct"]
        sign = "+" if pct >= 0 else ""
        parts.append(f"{p['symbol']} {sign}{pct:.1f}%")

    return f"{len(positions)} pos: " + ", ".join(parts)


def status():
    """Print one-line trading status"""
    try:
        t = Trader()
        account = t.get_account()
        clock = t.get_clock()
        positions = t.get_positions()

        market = format_market_phase(clock)
        pf_value = f"${account['portfolio_value']:,.0f}"
        pf_pct = f"{account['pl_today_pct']:+.2f}%"
        pos_str = format_positions(positions)

        print(f"{market} | {pf_value} ({pf_pct}) | {pos_str}")

    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    status()
