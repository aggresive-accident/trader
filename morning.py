#!/usr/bin/env python3
"""
morning.py - pre-market morning routine

Runs all checks before market opens:
1. Gap scanner - overnight moves
2. Market regime - trend/volatility
3. Edge signals - entry opportunities
4. Position status - exit signals

Run before 9:30 AM ET.
"""

import sys
import subprocess
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader


def run_script(name: str, script: str):
    """Run a script and print its output"""
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            ["python3", script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent,
        )
        print(result.stdout)
        if result.stderr:
            print(f"[STDERR] {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {script} took too long")
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    print("=" * 60)
    print(" MORNING ROUTINE - PRE-MARKET CHECKLIST")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check market status
    try:
        trader = Trader()
        clock = trader.get_clock()
        print(f"\nMarket: {'OPEN' if clock['is_open'] else 'CLOSED'}")
        if not clock['is_open']:
            print(f"Next open: {clock['next_open']}")
    except Exception as e:
        print(f"Could not get market status: {e}")

    # Account status
    try:
        account = trader.get_account()
        print(f"Portfolio: ${account['portfolio_value']:,.2f}")
        print(f"Cash: ${account['cash']:,.2f}")
        print(f"Buying power: ${account['buying_power']:,.2f}")
    except Exception as e:
        print(f"Could not get account: {e}")

    # 1. Gap scanner
    run_script("1. GAP SCANNER", "premarket.py")

    # 2. Market regime
    run_script("2. MARKET REGIME", "market_report.py")

    # 3. Edge signals
    run_script("3. EDGE SIGNALS", "edge.py")

    # 4. Position status
    run_script("4. POSITION MONITOR", "monitor.py")

    # Summary
    print("\n" + "=" * 60)
    print(" MORNING CHECKLIST COMPLETE")
    print("=" * 60)
    print("""
Next steps:
- Review signals above
- If entry signal: execute.py --execute
- If exit signal: sell position
- Set alerts for price levels
- Re-run at market open for final check
""")


if __name__ == "__main__":
    main()
