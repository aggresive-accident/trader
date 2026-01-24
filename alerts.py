#!/usr/bin/env python3
"""
alerts.py - trade signal alerts

Compares current edge signals to previous.
Notifies when entry/exit signals appear.
Supports optional webhook (ntfy.sh, Discord).
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from edge import EdgeTrader
from trader import Trader

STATE_FILE = Path(__file__).parent / ".alert_state.json"
ALERT_LOG = Path(__file__).parent / "alerts_log.json"

# Optional webhook URL (e.g., "https://ntfy.sh/your-topic")
WEBHOOK_URL = None


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


def log_alert(alert: dict):
    """Log alert to history file"""
    alerts = []
    if ALERT_LOG.exists():
        alerts = json.loads(ALERT_LOG.read_text())

    alert["timestamp"] = datetime.now().isoformat()
    alerts.append(alert)

    # Keep last 200 alerts
    alerts = alerts[-200:]
    ALERT_LOG.write_text(json.dumps(alerts, indent=2, default=str))


def send_webhook(message: str, title: str = "Trader Alert"):
    """Send alert to webhook if configured"""
    if not WEBHOOK_URL:
        return

    try:
        import urllib.request
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=message.encode(),
            headers={"Title": title},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")


def get_alerts() -> list[dict]:
    """Get alerts for signal changes"""
    edge = EdgeTrader()
    trader = Trader()

    # Watchlist (same as edge.py)
    watchlist = [
        "AMD", "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "AVGO", "CRM", "ORCL", "NFLX", "ADBE", "INTC", "QCOM",
        "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
    ]

    # Get current setups
    setups = edge.scan_for_setups(watchlist)
    curr_signals = {s["symbol"]: s for s in setups}

    # Load previous state
    previous = load_state()
    prev_signals = previous.get("signals", {})

    alerts = []

    # Check for new entry signals
    for symbol, setup in curr_signals.items():
        if symbol not in prev_signals:
            # New entry signal
            pos = edge.calculate_position(setup)
            alerts.append({
                "type": "NEW_ENTRY",
                "symbol": symbol,
                "score": setup["score"],
                "price": setup["price"],
                "stop": pos["stop_price"],
                "risk_pct": pos["risk_pct"],
                "reasons": setup["reasons"],
            })

    # Check for signals that disappeared
    for symbol in prev_signals:
        if symbol not in curr_signals:
            alerts.append({
                "type": "SIGNAL_LOST",
                "symbol": symbol,
                "was_score": prev_signals[symbol].get("score", 0),
            })

    # Check positions for exit signals
    positions = trader.get_positions()
    for p in positions:
        sym = p["symbol"]
        entry = p["avg_entry"]
        current = p["current_price"]
        high_water = max(entry, current)

        exit_check = edge.check_exit(sym, entry, high_water)

        if exit_check.get("exit"):
            alerts.append({
                "type": "EXIT_SIGNAL",
                "symbol": sym,
                "reason": exit_check["reason"],
                "price": current,
                "pnl_pct": (current - entry) / entry * 100,
            })

    # Save current state
    save_state(curr_signals)

    return alerts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trade alerts")
    parser.add_argument("command", nargs="?", default="check",
                       choices=["check", "reset", "status", "history"])
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
            print(f"Entry signals: {len(state['signals'])}")
            if state['signals']:
                for sym, data in list(state['signals'].items())[:5]:
                    print(f"  {sym}: score {data.get('score', '?')}")
        else:
            print("No previous state")
        return

    if args.command == "history":
        if not ALERT_LOG.exists():
            print("No alert history")
            return

        alerts = json.loads(ALERT_LOG.read_text())
        print(f"ALERT HISTORY (last 20 of {len(alerts)})")
        print("-" * 60)

        for a in alerts[-20:]:
            ts = a.get("timestamp", "")[:16]
            atype = a.get("type", "?")
            sym = a.get("symbol", "?")

            if atype == "NEW_ENTRY":
                print(f"{ts}  ENTRY   {sym:5}  score {a.get('score', '?')}")
            elif atype == "EXIT_SIGNAL":
                print(f"{ts}  EXIT    {sym:5}  {a.get('reason', '')}")
            elif atype == "SIGNAL_LOST":
                print(f"{ts}  LOST    {sym:5}  was score {a.get('was_score', '?')}")

        return

    # Check for alerts
    print("=" * 60)
    print("CHECKING ALERTS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    alerts = get_alerts()

    if not alerts:
        print("\nNo new alerts")
        return

    print(f"\nALERTS ({len(alerts)})")
    print("-" * 60)

    for alert in alerts:
        atype = alert["type"]
        symbol = alert.get("symbol", "?")

        if atype == "NEW_ENTRY":
            print(f"  ENTRY  {symbol:5} (score {alert['score']}) @ ${alert['price']:.2f}")
            print(f"         Stop: ${alert['stop']:.2f}, Risk: {alert['risk_pct']:.1f}%")
            print(f"         {', '.join(alert['reasons'])}")

            # Send webhook
            msg = f"NEW ENTRY: {symbol}\n"
            msg += f"Score: {alert['score']}, Price: ${alert['price']:.2f}\n"
            msg += f"Stop: ${alert['stop']:.2f}\n"
            msg += f"{', '.join(alert['reasons'])}"
            send_webhook(msg, f"BUY {symbol}")

        elif atype == "EXIT_SIGNAL":
            print(f"  EXIT   {symbol:5} - {alert['reason']}")
            print(f"         P&L: {alert['pnl_pct']:+.1f}%")

            msg = f"EXIT SIGNAL: {symbol}\n"
            msg += f"Reason: {alert['reason']}\n"
            msg += f"P&L: {alert['pnl_pct']:+.1f}%"
            send_webhook(msg, f"SELL {symbol}")

        elif atype == "SIGNAL_LOST":
            print(f"  LOST   {symbol:5} - no longer meets criteria")

        log_alert(alert)

    print("=" * 60)


if __name__ == "__main__":
    main()
