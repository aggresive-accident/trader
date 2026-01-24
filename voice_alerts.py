#!/usr/bin/env python3
"""
voice_alerts.py - speak trading alerts aloud

Narrates market alerts in the voice style.
Integrates voice organ with trading.
"""

import sys
import time
from pathlib import Path

# Add trader venv
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

# Add voice project
VOICE_PATH = Path.home() / "workspace" / "voice"
sys.path.insert(0, str(VOICE_PATH))


def say(thought: str, pause: float = 0.3):
    """speak a thought (voice organ style)"""
    print(f"  > {thought}")
    time.sleep(pause)


def speak_portfolio():
    """narrate portfolio status"""
    from trader import Trader

    say("i am checking my portfolio...")

    try:
        t = Trader()
        account = t.get_account()
        clock = t.get_clock()
        positions = t.get_positions()

        say(f"my portfolio is worth ${account['portfolio_value']:,.0f}")

        if account['pl_today'] != 0:
            if account['pl_today'] > 0:
                say(f"today i am up ${account['pl_today']:,.0f}")
            else:
                say(f"today i am down ${abs(account['pl_today']):,.0f}")
        else:
            say("no change today")

        phase = clock.get('phase', 'unknown')
        if clock['is_open']:
            say("the market is open")
        elif phase == "pre-market":
            say("it is pre-market... the day is starting")
        elif phase == "after-hours":
            say("it is after-hours... the day is winding down")
        else:
            say("the market is closed")

        if positions:
            say(f"i hold {len(positions)} positions:")
            for p in positions:
                if p['unrealized_pl_pct'] > 0:
                    say(f"  {p['symbol']}: up {p['unrealized_pl_pct']:.1f}%")
                else:
                    say(f"  {p['symbol']}: down {abs(p['unrealized_pl_pct']):.1f}%")
        else:
            say("i hold no positions... all cash")

    except Exception as e:
        say(f"error checking portfolio: {e}")


def speak_signals():
    """narrate buy signals"""
    from scanner import Scanner

    say("i am scanning for momentum signals...")

    try:
        scanner = Scanner()
        results = scanner.scan()

        buy_candidates = [r for r in results if r["signal"] in ("BUY", "STRONG BUY")]

        if buy_candidates:
            say(f"i found {len(buy_candidates)} buy signals:")
            for c in buy_candidates[:3]:
                if c["signal"] == "STRONG BUY":
                    say(f"  {c['symbol']} is showing strong momentum... {c['momentum']:+.1f}%")
                else:
                    say(f"  {c['symbol']} looks promising... {c['momentum']:+.1f}%")
        else:
            say("no strong signals right now")
            say("the market is quiet")

        # Mention worst performers
        avoid = [r for r in results if r["signal"] == "AVOID"]
        if avoid:
            say(f"i would avoid {avoid[0]['symbol']}... down {avoid[0]['momentum']:.1f}%")

    except Exception as e:
        say(f"error scanning: {e}")


def speak_alerts():
    """narrate signal changes"""
    from alerts import get_alerts

    say("i am checking for alert conditions...")

    try:
        alerts = get_alerts()

        if not alerts:
            say("nothing has changed since i last looked")
            return

        say(f"i notice {len(alerts)} changes:")
        for alert in alerts:
            atype = alert["type"]
            symbol = alert["symbol"]

            if atype == "NEW_BUY":
                say(f"  {symbol} is newly showing a buy signal")
            elif atype == "UPGRADE":
                say(f"  {symbol} has strengthened... now a {alert['to']}")
            elif atype == "STRONG_BUY":
                say(f"  {symbol} is very strong now")
            elif atype == "DOWNGRADE":
                say(f"  warning: {symbol} has weakened")
            elif atype == "AVOID":
                say(f"  warning: {symbol} should be avoided now")

    except Exception as e:
        say(f"error checking alerts: {e}")


def speak_sectors():
    """narrate sector rotation"""
    from sectors import SectorScanner

    say("i am examining sector rotation...")

    try:
        scanner = SectorScanner()
        result = scanner.scan(20)

        if not result["sectors"]:
            say("not enough data for sector analysis")
            return

        hot = [s for s in result["sectors"] if s["strength"] == "STRONG"]
        cold = [s for s in result["sectors"] if s["strength"] == "WEAK"]

        if hot:
            say(f"money is flowing into {hot[0]['name']}... up {hot[0]['return']:.1f}%")
            if len(hot) > 1:
                say(f"also strong: {hot[1]['name']}")

        if cold:
            say(f"money is leaving {cold[0]['name']}... down {abs(cold[0]['return']):.1f}%")

        say(f"the benchmark is {result['benchmark_return']:+.1f}%")

    except Exception as e:
        say(f"error: {e}")


def full_briefing():
    """complete market briefing"""
    say("beginning market briefing...")
    print()
    speak_portfolio()
    print()
    speak_signals()
    print()
    speak_alerts()
    print()
    speak_sectors()
    print()
    say("briefing complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Voice trading alerts")
    parser.add_argument("command", nargs="?", default="brief",
                       choices=["brief", "portfolio", "signals", "alerts", "sectors"])
    args = parser.parse_args()

    if args.command == "brief":
        full_briefing()
    elif args.command == "portfolio":
        speak_portfolio()
    elif args.command == "signals":
        speak_signals()
    elif args.command == "alerts":
        speak_alerts()
    elif args.command == "sectors":
        speak_sectors()


if __name__ == "__main__":
    main()
