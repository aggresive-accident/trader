#!/usr/bin/env python3
"""
dashboard.py - one-page trading overview

Everything important at a glance.
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
from sectors import SectorScanner


def dashboard():
    """Generate dashboard view"""
    now = datetime.now()
    lines = []

    # Header
    lines.append("╔" + "═" * 58 + "╗")
    lines.append("║" + f" TRADING DASHBOARD - {now.strftime('%Y-%m-%d %H:%M')}".ljust(58) + "║")
    lines.append("╠" + "═" * 58 + "╣")

    # Portfolio
    try:
        t = Trader()
        account = t.get_account()
        clock = t.get_clock()
        positions = t.get_positions()

        phase = clock.get("phase", "unknown")
        lines.append("║" + f" PORTFOLIO: ${account['portfolio_value']:>10,.0f}  Today: {account['pl_today_pct']:+.2f}%".ljust(58) + "║")
        lines.append("║" + f" Phase: {phase.upper():12}  Positions: {len(positions)}".ljust(58) + "║")

        if positions:
            lines.append("╠" + "─" * 58 + "╣")
            for p in positions[:3]:
                pct = p['unrealized_pl_pct']
                indicator = "▲" if pct > 0 else "▼" if pct < 0 else "─"
                lines.append("║" + f" {indicator} {p['symbol']:6} {p['qty']:5.0f} @ ${p['current_price']:8.2f}  {pct:+6.2f}%".ljust(58) + "║")
    except Exception as e:
        lines.append("║" + f" Portfolio error: {str(e)[:45]}".ljust(58) + "║")

    # Buy signals
    lines.append("╠" + "═" * 58 + "╣")
    lines.append("║" + " BUY SIGNALS".ljust(58) + "║")
    lines.append("╠" + "─" * 58 + "╣")

    try:
        scanner = Scanner()
        buy = scanner.buy_candidates()
        if buy:
            for b in buy[:3]:
                signal = "★" if b["signal"] == "STRONG BUY" else "●"
                lines.append("║" + f" {signal} {b['symbol']:6} ${b['price']:8.2f}  {b['momentum']:+6.2f}%  {b['signal']}".ljust(58) + "║")
        else:
            lines.append("║" + "   No buy signals".ljust(58) + "║")
    except Exception as e:
        lines.append("║" + f"   Error: {str(e)[:45]}".ljust(58) + "║")

    # Sectors
    lines.append("╠" + "═" * 58 + "╣")
    lines.append("║" + " SECTOR LEADERS".ljust(58) + "║")
    lines.append("╠" + "─" * 58 + "╣")

    try:
        sector_scanner = SectorScanner()
        result = sector_scanner.scan(20)
        hot = [s for s in result["sectors"][:3] if s["relative"] > 0]
        if hot:
            for h in hot:
                lines.append("║" + f"   {h['symbol']:5} {h['name'][:14]:14} {h['relative']:+5.2f}% vs SPY".ljust(58) + "║")
        else:
            lines.append("║" + "   No strong sectors".ljust(58) + "║")
    except Exception as e:
        lines.append("║" + f"   Error: {str(e)[:45]}".ljust(58) + "║")

    # Avoid
    lines.append("╠" + "═" * 58 + "╣")
    lines.append("║" + " AVOID".ljust(58) + "║")
    lines.append("╠" + "─" * 58 + "╣")

    try:
        avoid = [r for r in scanner.scan() if r["signal"] == "AVOID"]
        if avoid:
            for a in avoid[:2]:
                lines.append("║" + f"   ✗ {a['symbol']:6} {a['momentum']:+6.2f}%".ljust(58) + "║")
        else:
            lines.append("║" + "   None".ljust(58) + "║")
    except:
        pass

    # Footer
    lines.append("╚" + "═" * 58 + "╝")

    return "\n".join(lines)


def main():
    print(dashboard())


if __name__ == "__main__":
    main()
