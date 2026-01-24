#!/usr/bin/env python3
"""
alerts.py - trade signal alerts

Compares current signals to previous signals.
Notifies when something changes.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from scanner import Scanner

STATE_FILE = Path(__file__).parent / ".signal_state.json"


def load_state() -> dict:
    """Load previous signal state"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"signals": {}, "timestamp": None}


def save_state(signals: dict) -> None:
    """Save current signal state"""
    state = {
        "signals": signals,
        "timestamp": datetime.now().isoformat(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_alerts() -> list[dict]:
    """Get alerts for signal changes"""
    scanner = Scanner()
    current = scanner.scan()

    previous = load_state()
    prev_signals = previous.get("signals", {})

    # Convert current to dict
    curr_signals = {s["symbol"]: s for s in current}

    alerts = []

    # Check for signal changes
    for symbol, curr in curr_signals.items():
        prev = prev_signals.get(symbol)

        if prev is None:
            # New symbol
            if curr["signal"] in ("BUY", "STRONG BUY"):
                alerts.append({
                    "type": "NEW_BUY",
                    "symbol": symbol,
                    "signal": curr["signal"],
                    "momentum": curr["momentum"],
                    "price": curr["price"],
                })
        else:
            # Signal changed?
            if prev["signal"] != curr["signal"]:
                # Upgrade to buy
                if curr["signal"] in ("BUY", "STRONG BUY") and prev["signal"] not in ("BUY", "STRONG BUY"):
                    alerts.append({
                        "type": "UPGRADE",
                        "symbol": symbol,
                        "from": prev["signal"],
                        "to": curr["signal"],
                        "momentum": curr["momentum"],
                        "price": curr["price"],
                    })
                # Downgrade from buy
                elif prev["signal"] in ("BUY", "STRONG BUY") and curr["signal"] not in ("BUY", "STRONG BUY"):
                    alerts.append({
                        "type": "DOWNGRADE",
                        "symbol": symbol,
                        "from": prev["signal"],
                        "to": curr["signal"],
                        "momentum": curr["momentum"],
                        "price": curr["price"],
                    })
                # Strong buy upgrade
                elif curr["signal"] == "STRONG BUY" and prev["signal"] == "BUY":
                    alerts.append({
                        "type": "STRONG_BUY",
                        "symbol": symbol,
                        "momentum": curr["momentum"],
                        "price": curr["price"],
                    })
                # Turned to avoid
                elif curr["signal"] == "AVOID" and prev["signal"] != "AVOID":
                    alerts.append({
                        "type": "AVOID",
                        "symbol": symbol,
                        "from": prev["signal"],
                        "momentum": curr["momentum"],
                        "price": curr["price"],
                    })

    # Save current state for next comparison
    save_state(curr_signals)

    return alerts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trade alerts")
    parser.add_argument("command", nargs="?", default="check",
                       choices=["check", "reset", "status"])
    args = parser.parse_args()

    if args.command == "reset":
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("Signal state reset")
        return

    if args.command == "status":
        state = load_state()
        if state["timestamp"]:
            print(f"Last check: {state['timestamp']}")
            print(f"Symbols tracked: {len(state['signals'])}")
            buys = [s for s, d in state["signals"].items() if d["signal"] in ("BUY", "STRONG BUY")]
            if buys:
                print(f"Current buys: {', '.join(buys)}")
        else:
            print("No previous state")
        return

    # Check for alerts
    alerts = get_alerts()

    if not alerts:
        print("No alerts")
        return

    print(f"ALERTS ({len(alerts)})")
    print("-" * 50)

    for alert in alerts:
        atype = alert["type"]
        symbol = alert["symbol"]
        price = alert["price"]
        mom = alert["momentum"]

        if atype == "NEW_BUY":
            print(f"🟢 NEW   {symbol:6} ${price:8.2f}  {mom:+6.2f}%  {alert['signal']}")
        elif atype == "UPGRADE":
            print(f"🟡 UP    {symbol:6} ${price:8.2f}  {mom:+6.2f}%  {alert['from']} → {alert['to']}")
        elif atype == "STRONG_BUY":
            print(f"🟢 STRONG {symbol:6} ${price:8.2f}  {mom:+6.2f}%")
        elif atype == "DOWNGRADE":
            print(f"🟠 DOWN  {symbol:6} ${price:8.2f}  {mom:+6.2f}%  {alert['from']} → {alert['to']}")
        elif atype == "AVOID":
            print(f"🔴 AVOID {symbol:6} ${price:8.2f}  {mom:+6.2f}%  was {alert['from']}")


if __name__ == "__main__":
    main()
