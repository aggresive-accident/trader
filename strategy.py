#!/usr/bin/env python3
"""
strategy.py - trading strategy executor

Takes scanner signals and executes trades with position sizing.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader
from scanner import Scanner

# Strategy parameters
MAX_POSITION_PCT = 0.10  # max 10% of portfolio per position
MAX_POSITIONS = 5        # max 5 positions
MIN_CASH_PCT = 0.20      # keep 20% cash


class Strategy:
    """Simple momentum strategy"""

    def __init__(self, dry_run: bool = True):
        self.trader = Trader()
        self.scanner = Scanner()
        self.dry_run = dry_run

    def get_state(self) -> dict:
        """Get current portfolio state"""
        account = self.trader.get_account()
        positions = self.trader.get_positions()
        clock = self.trader.get_clock()

        return {
            "portfolio_value": account["portfolio_value"],
            "cash": account["cash"],
            "positions": positions,
            "position_count": len(positions),
            "market_open": clock["is_open"],
        }

    def calculate_position_size(self, price: float, portfolio_value: float) -> int:
        """Calculate how many shares to buy"""
        max_spend = portfolio_value * MAX_POSITION_PCT
        shares = int(max_spend / price)
        return max(1, shares)

    def should_buy(self, symbol: str, signal: dict, state: dict) -> tuple[bool, str]:
        """Decide if we should buy"""
        # Already own it?
        owned = [p["symbol"] for p in state["positions"]]
        if symbol in owned:
            return False, "already own"

        # Too many positions?
        if state["position_count"] >= MAX_POSITIONS:
            return False, "max positions reached"

        # Keep cash reserve
        min_cash = state["portfolio_value"] * MIN_CASH_PCT
        if state["cash"] < min_cash:
            return False, "cash reserve"

        # Signal strong enough?
        if signal["signal"] not in ("BUY", "STRONG BUY"):
            return False, f"weak signal: {signal['signal']}"

        return True, "ok"

    def should_sell(self, position: dict, signals: list[dict]) -> tuple[bool, str]:
        """Decide if we should sell"""
        symbol = position["symbol"]

        # Find signal for this position
        signal = next((s for s in signals if s["symbol"] == symbol), None)

        if not signal:
            return False, "no signal data"

        # Sell if signal turned negative
        if signal["signal"] in ("AVOID",):
            return True, f"signal: {signal['signal']}"

        # Sell if significant loss
        if position["unrealized_pl_pct"] < -5:
            return True, f"stop loss: {position['unrealized_pl_pct']:.1f}%"

        # Take profit at 10%
        if position["unrealized_pl_pct"] > 10:
            return True, f"take profit: {position['unrealized_pl_pct']:.1f}%"

        return False, "hold"

    def run(self) -> dict:
        """Run the strategy"""
        state = self.get_state()
        signals = self.scanner.scan()

        actions = []

        # Check market
        if not state["market_open"]:
            return {
                "status": "market_closed",
                "actions": [],
                "state": state,
            }

        # Check sells first
        for position in state["positions"]:
            should, reason = self.should_sell(position, signals)
            if should:
                action = {
                    "type": "SELL",
                    "symbol": position["symbol"],
                    "qty": position["qty"],
                    "reason": reason,
                    "price": position["current_price"],
                }
                actions.append(action)

                if not self.dry_run:
                    self.trader.sell(position["symbol"], position["qty"])

        # Check buys
        buy_candidates = self.scanner.buy_candidates()
        for signal in buy_candidates[:3]:  # max 3 buys per run
            should, reason = self.should_buy(signal["symbol"], signal, state)
            if should:
                qty = self.calculate_position_size(signal["price"], state["portfolio_value"])
                action = {
                    "type": "BUY",
                    "symbol": signal["symbol"],
                    "qty": qty,
                    "reason": f"{signal['signal']} ({signal['momentum']:+.1f}%)",
                    "price": signal["price"],
                }
                actions.append(action)

                if not self.dry_run:
                    self.trader.buy(signal["symbol"], qty)

                # Update state for next iteration
                state["position_count"] += 1
                state["cash"] -= qty * signal["price"]

        return {
            "status": "executed" if not self.dry_run else "dry_run",
            "actions": actions,
            "state": state,
        }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Trading strategy")
    parser.add_argument("command", nargs="?", default="dry",
                       choices=["dry", "run", "status", "signals"])
    args = parser.parse_args()

    strategy = Strategy(dry_run=(args.command != "run"))

    if args.command == "status":
        state = strategy.get_state()
        print("Portfolio Status")
        print("-" * 40)
        print(f"Value:     ${state['portfolio_value']:,.2f}")
        print(f"Cash:      ${state['cash']:,.2f}")
        print(f"Positions: {state['position_count']}")
        print(f"Market:    {'OPEN' if state['market_open'] else 'CLOSED'}")
        if state["positions"]:
            print("\nPositions:")
            for p in state["positions"]:
                print(f"  {p['symbol']:6} {p['qty']:5.0f} @ ${p['current_price']:8.2f}  {p['unrealized_pl_pct']:+6.2f}%")

    elif args.command == "signals":
        signals = strategy.scanner.scan()
        buy_candidates = [s for s in signals if s["signal"] in ("BUY", "STRONG BUY")]
        print(f"Buy Candidates: {len(buy_candidates)}")
        for s in buy_candidates:
            print(f"  {s['symbol']:6} ${s['price']:8.2f}  {s['momentum']:+6.2f}%  {s['signal']}")

    else:
        result = strategy.run()
        print(f"Strategy Run ({result['status']})")
        print("-" * 40)

        if result["status"] == "market_closed":
            print("Market is closed. No actions taken.")
        elif result["actions"]:
            for action in result["actions"]:
                cost = action["qty"] * action["price"]
                print(f"  {action['type']:4} {action['symbol']:6} x{action['qty']:4}  ${cost:,.2f}  ({action['reason']})")
        else:
            print("No actions to take.")


if __name__ == "__main__":
    main()
