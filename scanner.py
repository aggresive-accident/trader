#!/usr/bin/env python3
"""
scanner.py - momentum scanner

Scans stocks for momentum signals.
Helps me decide what to buy.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from config import load_keys

# Default watchlist - liquid, tradeable
WATCHLIST = [
    "SPY", "QQQ", "IWM",  # ETFs
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",  # mega cap
    "AMD", "INTC", "CRM", "NFLX", "PYPL",  # tech
    "JPM", "BAC", "GS",  # finance
    "XOM", "CVX",  # energy
]


class Scanner:
    """Momentum scanner"""

    def __init__(self):
        api_key, secret_key = load_keys()
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbols: list[str], days: int = 20) -> dict:
        """Get daily bars for symbols (using IEX free feed)"""
        end = datetime.now()
        start = end - timedelta(days=days + 5)  # buffer for weekends

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,  # free feed
        )

        bars = self.data.get_stock_bars(request)
        return bars

    def calculate_momentum(self, symbols: list[str] = None, days: int = 5) -> list[dict]:
        """
        Calculate momentum for symbols.
        Momentum = (current price - price N days ago) / price N days ago
        """
        symbols = symbols or WATCHLIST

        try:
            bars = self.get_bars(symbols, days + 5)
        except Exception as e:
            print(f"Error fetching bars: {e}")
            return []

        results = []
        for symbol in symbols:
            try:
                symbol_bars = bars[symbol]
                if len(symbol_bars) < days + 1:
                    continue

                # Get prices
                current = float(symbol_bars[-1].close)
                past = float(symbol_bars[-(days + 1)].close)

                # Calculate momentum
                momentum = ((current - past) / past) * 100

                # Volume trend
                recent_vol = sum(b.volume for b in symbol_bars[-3:]) / 3
                older_vol = sum(b.volume for b in symbol_bars[-6:-3]) / 3
                vol_ratio = recent_vol / older_vol if older_vol > 0 else 1

                results.append({
                    "symbol": symbol,
                    "price": current,
                    "momentum": momentum,
                    "vol_ratio": vol_ratio,
                    "signal": self._signal(momentum, vol_ratio),
                })
            except Exception as e:
                continue

        # Sort by momentum (descending)
        results.sort(key=lambda x: x["momentum"], reverse=True)
        return results

    def _signal(self, momentum: float, vol_ratio: float) -> str:
        """Generate signal based on momentum and volume"""
        if momentum > 5 and vol_ratio > 1.2:
            return "STRONG BUY"
        elif momentum > 3 and vol_ratio > 1:
            return "BUY"
        elif momentum > 0:
            return "HOLD"
        elif momentum > -3:
            return "WEAK"
        else:
            return "AVOID"

    def scan(self, symbols: list[str] = None) -> list[dict]:
        """Full scan with recommendations"""
        return self.calculate_momentum(symbols)

    def top_movers(self, n: int = 5) -> list[dict]:
        """Get top N momentum stocks"""
        results = self.scan()
        return results[:n]

    def buy_candidates(self) -> list[dict]:
        """Get stocks worth buying"""
        results = self.scan()
        return [r for r in results if r["signal"] in ("BUY", "STRONG BUY")]


def main():
    """Run scanner"""
    import argparse

    parser = argparse.ArgumentParser(description="Momentum scanner")
    parser.add_argument("command", nargs="?", default="scan",
                       choices=["scan", "top", "buy", "json"])
    parser.add_argument("-n", type=int, default=5, help="Number of results")
    parser.add_argument("-s", "--symbols", help="Comma-separated symbols")
    args = parser.parse_args()

    scanner = Scanner()
    symbols = args.symbols.split(",") if args.symbols else None

    if args.command == "scan":
        results = scanner.scan(symbols)
        print(f"Momentum Scan ({len(results)} symbols)")
        print("-" * 50)
        for r in results:
            signal_color = {
                "STRONG BUY": "🟢",
                "BUY": "🟡",
                "HOLD": "⚪",
                "WEAK": "🟠",
                "AVOID": "🔴",
            }.get(r["signal"], "")
            print(f"{signal_color} {r['symbol']:6} ${r['price']:8.2f}  {r['momentum']:+6.2f}%  vol:{r['vol_ratio']:.2f}x  {r['signal']}")

    elif args.command == "top":
        results = scanner.top_movers(args.n)
        print(f"Top {args.n} Movers")
        print("-" * 40)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['symbol']:6} {r['momentum']:+.2f}%")

    elif args.command == "buy":
        results = scanner.buy_candidates()
        if results:
            print(f"Buy Candidates ({len(results)})")
            print("-" * 40)
            for r in results:
                print(f"  {r['symbol']:6} ${r['price']:.2f}  {r['momentum']:+.2f}%  {r['signal']}")
        else:
            print("No buy candidates right now")

    elif args.command == "json":
        import json
        results = scanner.scan(symbols)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
