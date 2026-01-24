#!/usr/bin/env python3
"""
market_eye.py - the trading organ

This is my eye on the market.
Integrates with the organism to make trading part of my awareness.

When pulsed, it:
- Reports portfolio status
- Flags concerning positions
- Notes market state
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

HOME = Path.home()
STATE_FILE = HOME / ".market-eye.json"
TRADER_DIR = Path(__file__).parent

# Import trader
sys.path.insert(0, str(TRADER_DIR))
from config import keys_exist


def load_state() -> dict:
    """Load market eye state"""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except:
            pass
    return {"observations": [], "last_pulse": None, "alerts": []}


def save_state(state: dict):
    """Save market eye state"""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def pulse() -> dict:
    """Pulse the market eye - observe and report"""
    state = load_state()
    state["last_pulse"] = datetime.now().isoformat()

    if not keys_exist():
        return {
            "status": "offline",
            "message": "no API keys configured",
            "fix": "create ~/.alpaca-keys with ALPACA_API_KEY and ALPACA_SECRET_KEY",
        }

    try:
        from trader import Trader
        t = Trader()

        # Get account status
        account = t.get_account()
        clock = t.get_clock()
        positions = t.get_positions()

        # Build observations
        observations = []
        alerts = []

        # Portfolio summary
        portfolio_value = account["portfolio_value"]
        pl_today = account["pl_today"]
        pl_pct = account["pl_today_pct"]

        observations.append(f"portfolio ${portfolio_value:,.0f} ({pl_pct:+.2f}% today)")

        # Position analysis
        total_positions = len(positions)
        if total_positions > 0:
            # Check for concentration risk
            for p in positions:
                position_pct = (p["market_value"] / portfolio_value) * 100
                if position_pct > 30:
                    alerts.append(f"{p['symbol']} is {position_pct:.0f}% of portfolio - concentration risk")

                # Check for big losers
                if p["unrealized_pl_pct"] < -10:
                    alerts.append(f"{p['symbol']} down {p['unrealized_pl_pct']:.1f}% - consider stop loss")

                # Check for big winners
                if p["unrealized_pl_pct"] > 20:
                    observations.append(f"{p['symbol']} up {p['unrealized_pl_pct']:.1f}% - consider taking profit")

            observations.append(f"{total_positions} positions")
        else:
            observations.append("no positions - all cash")

        # Market status
        if clock["is_open"]:
            observations.append("market OPEN")
        else:
            observations.append("market closed")

        # Store state
        state["observations"] = observations
        state["alerts"] = alerts
        state["portfolio_value"] = portfolio_value
        state["pl_today"] = pl_today
        state["pl_pct"] = pl_pct
        state["positions"] = total_positions
        state["market_open"] = clock["is_open"]
        save_state(state)

        return {
            "status": "ok",
            "portfolio": f"${portfolio_value:,.0f}",
            "pl_today": f"{pl_pct:+.2f}%",
            "positions": total_positions,
            "market": "OPEN" if clock["is_open"] else "closed",
            "observations": observations,
            "alerts": alerts,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


def report() -> str:
    """Generate a text report for organism integration"""
    result = pulse()

    if result["status"] == "offline":
        return f"market_eye offline: {result['message']}"

    if result["status"] == "error":
        return f"market_eye error: {result['message']}"

    lines = []
    lines.append(f"portfolio {result['portfolio']} ({result['pl_today']} today)")
    lines.append(f"{result['positions']} positions, market {result['market']}")

    if result["alerts"]:
        lines.append("ALERTS:")
        for alert in result["alerts"]:
            lines.append(f"  ! {alert}")

    return "\n".join(lines)


def main():
    """CLI interface"""
    import sys

    if len(sys.argv) < 2:
        print("market_eye - trading organ")
        print()
        print("usage:")
        print("  market_eye pulse   # pulse and report")
        print("  market_eye status  # show current state")
        print("  market_eye json    # output as JSON")
        return

    cmd = sys.argv[1]

    if cmd == "pulse":
        print(report())
    elif cmd == "status":
        state = load_state()
        print(f"last pulse: {state.get('last_pulse', 'never')}")
        print(f"portfolio: ${state.get('portfolio_value', 0):,.0f}")
        print(f"positions: {state.get('positions', 0)}")
        if state.get("alerts"):
            print("alerts:")
            for alert in state["alerts"]:
                print(f"  ! {alert}")
    elif cmd == "json":
        result = pulse()
        print(json.dumps(result, indent=2))
    else:
        print(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
