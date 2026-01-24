#!/usr/bin/env python3
"""
risk.py - risk management

Rules:
1. Max 5% portfolio per position
2. Max 20% total exposure
3. Stop loss at 3% per position
4. Take profit at 10% per position
"""

import sys
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader


class RiskManager:
    """Enforce risk rules on the portfolio"""

    def __init__(
        self,
        max_position_pct: float = 0.05,   # 5% max per position
        max_exposure_pct: float = 0.20,   # 20% max total exposure
        stop_loss_pct: float = 0.03,      # 3% stop loss
        take_profit_pct: float = 0.10,    # 10% take profit
    ):
        self.trader = Trader()
        self.max_position_pct = max_position_pct
        self.max_exposure_pct = max_exposure_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def get_status(self) -> dict:
        """Get current risk status"""
        account = self.trader.get_account()
        positions = self.trader.get_positions()

        portfolio_value = account["portfolio_value"]
        total_exposure = sum(abs(p["market_value"]) for p in positions)
        exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Check each position
        position_risks = []
        for p in positions:
            position_pct = abs(p["market_value"]) / portfolio_value
            pl_pct = p["unrealized_pl_pct"] / 100

            risk = {
                "symbol": p["symbol"],
                "qty": p["qty"],
                "value": p["market_value"],
                "position_pct": position_pct * 100,
                "pl_pct": p["unrealized_pl_pct"],
                "oversize": position_pct > self.max_position_pct,
                "stop_loss_hit": pl_pct < -self.stop_loss_pct,
                "take_profit_hit": pl_pct > self.take_profit_pct,
            }
            position_risks.append(risk)

        return {
            "portfolio_value": portfolio_value,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct * 100,
            "over_exposed": exposure_pct > self.max_exposure_pct,
            "cash": account["cash"],
            "available_for_new": max(0, portfolio_value * self.max_exposure_pct - total_exposure),
            "positions": position_risks,
        }

    def check_position_size(self, symbol: str, value: float) -> dict:
        """Check if a proposed position size is acceptable"""
        account = self.trader.get_account()
        portfolio_value = account["portfolio_value"]
        positions = self.trader.get_positions()

        current_exposure = sum(abs(p["market_value"]) for p in positions)
        new_total_exposure = current_exposure + value

        position_pct = value / portfolio_value
        new_exposure_pct = new_total_exposure / portfolio_value

        acceptable = True
        reasons = []

        if position_pct > self.max_position_pct:
            acceptable = False
            reasons.append(f"Position too large ({position_pct:.1%} > {self.max_position_pct:.1%})")

        if new_exposure_pct > self.max_exposure_pct:
            acceptable = False
            reasons.append(f"Would exceed max exposure ({new_exposure_pct:.1%} > {self.max_exposure_pct:.1%})")

        max_allowed = min(
            portfolio_value * self.max_position_pct,
            portfolio_value * self.max_exposure_pct - current_exposure
        )

        return {
            "symbol": symbol,
            "proposed_value": value,
            "acceptable": acceptable,
            "reasons": reasons,
            "max_allowed_value": max(0, max_allowed),
            "position_pct": position_pct * 100,
            "new_exposure_pct": new_exposure_pct * 100,
        }

    def get_actions(self) -> list:
        """Get recommended actions based on risk rules"""
        status = self.get_status()
        actions = []

        for p in status["positions"]:
            if p["stop_loss_hit"]:
                actions.append({
                    "action": "SELL",
                    "symbol": p["symbol"],
                    "qty": abs(p["qty"]),
                    "reason": f"Stop loss triggered ({p['pl_pct']:+.1f}%)",
                    "priority": "HIGH",
                })
            elif p["take_profit_hit"]:
                actions.append({
                    "action": "SELL",
                    "symbol": p["symbol"],
                    "qty": abs(p["qty"]),
                    "reason": f"Take profit triggered ({p['pl_pct']:+.1f}%)",
                    "priority": "MEDIUM",
                })
            elif p["oversize"]:
                # Calculate how much to trim
                target_value = status["portfolio_value"] * self.max_position_pct
                excess_value = p["value"] - target_value
                price = p["value"] / p["qty"]
                trim_qty = int(excess_value / price)

                if trim_qty > 0:
                    actions.append({
                        "action": "TRIM",
                        "symbol": p["symbol"],
                        "qty": trim_qty,
                        "reason": f"Position oversize ({p['position_pct']:.1f}% > {self.max_position_pct*100:.0f}%)",
                        "priority": "LOW",
                    })

        return actions

    def execute_actions(self, dry_run: bool = True) -> list:
        """Execute recommended risk actions"""
        actions = self.get_actions()
        results = []

        for action in actions:
            if dry_run:
                results.append({
                    "dry_run": True,
                    "would_execute": action,
                })
            else:
                try:
                    if action["action"] in ["SELL", "TRIM"]:
                        order = self.trader.sell(action["symbol"], action["qty"])
                        results.append({
                            "executed": True,
                            "action": action,
                            "order": order,
                        })
                except Exception as e:
                    results.append({
                        "executed": False,
                        "action": action,
                        "error": str(e),
                    })

        return results


def main():
    """Show risk status"""
    import argparse

    parser = argparse.ArgumentParser(description="Risk management")
    parser.add_argument("--execute", action="store_true", help="Execute risk actions")
    args = parser.parse_args()

    rm = RiskManager()

    # Show status
    status = rm.get_status()

    print("=" * 50)
    print("RISK STATUS")
    print("=" * 50)
    print(f"Portfolio: ${status['portfolio_value']:,.2f}")
    print(f"Cash: ${status['cash']:,.2f}")
    print(f"Exposure: ${status['total_exposure']:,.2f} ({status['exposure_pct']:.1f}%)")
    print(f"Available for new: ${status['available_for_new']:,.2f}")

    if status["over_exposed"]:
        print("\n[WARNING] OVER-EXPOSED")

    # Show positions
    if status["positions"]:
        print("\nPositions:")
        print("-" * 50)
        for p in status["positions"]:
            flags = []
            if p["stop_loss_hit"]:
                flags.append("STOP")
            if p["take_profit_hit"]:
                flags.append("TP")
            if p["oversize"]:
                flags.append("OVERSIZE")

            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {p['symbol']}: ${p['value']:,.2f} ({p['position_pct']:.1f}%) P&L: {p['pl_pct']:+.1f}%{flag_str}")
    else:
        print("\nNo positions")

    # Show recommended actions
    actions = rm.get_actions()
    if actions:
        print("\nRecommended Actions:")
        print("-" * 50)
        for a in actions:
            print(f"  [{a['priority']}] {a['action']} {a['qty']} {a['symbol']}: {a['reason']}")

        if args.execute:
            print("\nExecuting actions...")
            results = rm.execute_actions(dry_run=False)
            for r in results:
                if r.get("executed"):
                    o = r["order"]
                    print(f"  Executed: {o['side']} {o['qty']} {o['symbol']}")
                else:
                    print(f"  Failed: {r.get('error', 'unknown')}")
    else:
        print("\nNo risk actions needed")


if __name__ == "__main__":
    main()
