#!/usr/bin/env python3
"""
live.py - live trading harness

Connects strategy signals to real trades.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trader import Trader
from config import load_keys
from strategies import BestOfEnsemble, DonchianBreakout, ALL_STRATEGIES
from strategies.base import Signal


class LiveTrader:
    """Live trading harness with strategy-based decisions"""

    def __init__(self, strategy=None, max_position_pct: float = 0.10):
        self.trader = Trader()
        self.strategy = strategy or BestOfEnsemble()
        self.max_position_pct = max_position_pct  # Max 10% of portfolio per position

        api_key, secret_key = load_keys()
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbol: str, days: int = 90) -> list:
        """Fetch historical bars for analysis"""
        end = datetime.now() - timedelta(days=1)  # Avoid subscription issues
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self.data.get_stock_bars(request)
        if hasattr(bars, 'data') and symbol in bars.data:
            return list(bars.data[symbol])
        elif symbol in bars:
            return list(bars[symbol])
        return []

    def analyze(self, symbol: str) -> dict:
        """Analyze a symbol and get trading signal"""
        bars = self.get_bars(symbol)

        if len(bars) < self.strategy.warmup_period():
            return {
                "symbol": symbol,
                "signal": None,
                "error": f"Insufficient data ({len(bars)} bars, need {self.strategy.warmup_period()})"
            }

        signal = self.strategy.signal(bars, len(bars) - 1)

        # Get current price
        try:
            quote = self.trader.get_quote(symbol)
            current_price = quote["mid"]
        except:
            current_price = float(bars[-1].close)

        return {
            "symbol": symbol,
            "signal": {
                "strength": signal.strength,
                "reason": signal.reason,
                "confidence": signal.confidence,
            },
            "price": current_price,
            "recommendation": self._get_recommendation(signal),
        }

    def _get_recommendation(self, signal: Signal) -> str:
        """Convert signal to actionable recommendation"""
        if signal.strength > 0.6 and signal.confidence > 0.6:
            return "STRONG BUY"
        elif signal.strength > 0.3 and signal.confidence > 0.5:
            return "BUY"
        elif signal.strength < -0.6 and signal.confidence > 0.6:
            return "STRONG SELL"
        elif signal.strength < -0.3 and signal.confidence > 0.5:
            return "SELL"
        else:
            return "HOLD"

    def calculate_position_size(self, symbol: str, signal: Signal) -> int:
        """Calculate position size based on signal strength and risk limits"""
        account = self.trader.get_account()
        portfolio_value = account["portfolio_value"]

        # Max position value based on signal strength
        strength = abs(signal.strength)
        confidence = signal.confidence

        # Base: 10% of portfolio, scaled by signal strength and confidence
        max_value = portfolio_value * self.max_position_pct * strength * confidence

        # Get price
        try:
            quote = self.trader.get_quote(symbol)
            price = quote["ask"] if signal.strength > 0 else quote["bid"]
        except:
            return 0

        # Calculate shares (round down for safety)
        shares = int(max_value / price)
        return max(shares, 0)

    def execute_signal(self, symbol: str, dry_run: bool = True) -> dict:
        """Execute a trade based on strategy signal"""
        analysis = self.analyze(symbol)

        if analysis.get("error"):
            return {"executed": False, "reason": analysis["error"]}

        signal = Signal(
            strength=analysis["signal"]["strength"],
            reason=analysis["signal"]["reason"],
            confidence=analysis["signal"]["confidence"]
        )

        recommendation = analysis["recommendation"]

        if recommendation == "HOLD":
            return {
                "executed": False,
                "reason": f"Hold signal: {signal.reason}",
                "signal": analysis["signal"],
            }

        # Check existing position
        positions = {p["symbol"]: p for p in self.trader.get_positions()}
        has_position = symbol in positions

        # Determine action
        if recommendation in ["BUY", "STRONG BUY"]:
            if has_position:
                return {
                    "executed": False,
                    "reason": f"Already have position in {symbol}",
                    "signal": analysis["signal"],
                }

            qty = self.calculate_position_size(symbol, signal)
            if qty < 1:
                return {
                    "executed": False,
                    "reason": "Position size too small",
                    "signal": analysis["signal"],
                }

            action = "BUY"

        elif recommendation in ["SELL", "STRONG SELL"]:
            if not has_position:
                return {
                    "executed": False,
                    "reason": f"No position to sell in {symbol}",
                    "signal": analysis["signal"],
                }

            qty = int(positions[symbol]["qty"])
            action = "SELL"
        else:
            return {"executed": False, "reason": "No clear action"}

        # Execute or simulate
        if dry_run:
            return {
                "executed": False,
                "dry_run": True,
                "would_execute": {
                    "action": action,
                    "symbol": symbol,
                    "qty": qty,
                    "price": analysis["price"],
                    "value": qty * analysis["price"],
                },
                "signal": analysis["signal"],
            }
        else:
            if action == "BUY":
                result = self.trader.buy(symbol, qty)
            else:
                result = self.trader.sell(symbol, qty)

            return {
                "executed": True,
                "order": result,
                "signal": analysis["signal"],
            }

    def scan_and_trade(self, symbols: list, dry_run: bool = True) -> list:
        """Scan multiple symbols and execute trades"""
        results = []

        for symbol in symbols:
            print(f"Analyzing {symbol}...", end=" ", flush=True)
            try:
                result = self.execute_signal(symbol, dry_run=dry_run)
                result["symbol"] = symbol
                results.append(result)

                rec = result.get("would_execute", {}).get("action", "HOLD")
                if result.get("executed"):
                    rec = result["order"]["side"].upper()

                print(f"{rec}")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"symbol": symbol, "error": str(e)})

        return results


def main():
    """Run live trading analysis"""
    import argparse

    parser = argparse.ArgumentParser(description="Live trading harness")
    parser.add_argument("symbols", nargs="*", default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--execute", action="store_true", help="Execute real trades (default: dry run)")
    parser.add_argument("--strategy", default="best_of", help="Strategy to use")
    args = parser.parse_args()

    # Select strategy
    strategy_map = {
        "best_of": BestOfEnsemble(),
        "donchian": DonchianBreakout(),
    }
    # Add all strategies dynamically
    for s in ALL_STRATEGIES:
        if hasattr(s, "name"):
            strategy_map[s().name] = s()

    strategy = strategy_map.get(args.strategy, BestOfEnsemble())

    print(f"Live Trading Harness")
    print(f"Strategy: {strategy.name}")
    print(f"Mode: {'LIVE' if args.execute else 'DRY RUN'}")
    print("=" * 50)

    live = LiveTrader(strategy=strategy)

    # Show account
    account = live.trader.get_account()
    print(f"\nAccount: ${account['portfolio_value']:,.2f}")
    print(f"Cash: ${account['cash']:,.2f}")

    # Check market
    clock = live.trader.get_clock()
    print(f"Market: {clock['phase'].upper()}")

    if not clock["is_open"] and args.execute:
        print("\nWARNING: Market is closed. Orders may not fill.")

    # Scan symbols
    print(f"\nScanning {len(args.symbols)} symbols...")
    print("-" * 50)

    results = live.scan_and_trade(args.symbols, dry_run=not args.execute)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("-" * 50)

    buys = [r for r in results if r.get("would_execute", {}).get("action") == "BUY"]
    sells = [r for r in results if r.get("would_execute", {}).get("action") == "SELL"]
    executed = [r for r in results if r.get("executed")]

    if args.execute and executed:
        print(f"Executed: {len(executed)} trades")
        for r in executed:
            o = r["order"]
            print(f"  {o['side'].upper()} {o['qty']} {o['symbol']}")
    else:
        if buys:
            print(f"Would BUY ({len(buys)}):")
            for r in buys:
                w = r["would_execute"]
                print(f"  {w['qty']} {w['symbol']} @ ${w['price']:.2f} = ${w['value']:.2f}")

        if sells:
            print(f"Would SELL ({len(sells)}):")
            for r in sells:
                w = r["would_execute"]
                print(f"  {w['qty']} {w['symbol']} @ ${w['price']:.2f}")

        if not buys and not sells:
            print("No trades recommended.")


if __name__ == "__main__":
    main()
