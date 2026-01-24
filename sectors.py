#!/usr/bin/env python3
"""
sectors.py - sector rotation scanner

Track which sectors are hot using sector ETFs.
Helps identify where money is flowing.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from config import load_keys

# Sector ETFs (SPDR Select Sector ETFs)
SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication",
}

# Benchmark
BENCHMARK = "SPY"


class SectorScanner:
    """Sector rotation scanner"""

    def __init__(self):
        api_key, secret_key = load_keys()
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbols: list[str], days: int = 30) -> dict:
        """Get daily bars for symbols"""
        end = datetime.now()
        start = end - timedelta(days=days + 5)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )

        return self.data.get_stock_bars(request)

    def calculate_returns(self, days: int = 20) -> list[dict]:
        """Calculate sector returns vs benchmark"""
        symbols = list(SECTORS.keys()) + [BENCHMARK]

        try:
            bars = self.get_bars(symbols, days + 5)
        except Exception as e:
            print(f"Error: {e}")
            return [], 0

        # Get benchmark return
        try:
            benchmark_bars = bars[BENCHMARK]
        except KeyError:
            return [], 0

        # Use available bars, minimum 5 days
        available_days = min(days, len(benchmark_bars) - 1)
        if available_days < 5:
            return [], 0

        benchmark_return = ((float(benchmark_bars[-1].close) - float(benchmark_bars[-(available_days+1)].close))
                           / float(benchmark_bars[-(available_days+1)].close)) * 100

        results = []
        for symbol, name in SECTORS.items():
            try:
                sector_bars = bars[symbol]
                sector_days = min(available_days, len(sector_bars) - 1)
                if sector_days < 5:
                    continue

                current = float(sector_bars[-1].close)
                past = float(sector_bars[-(sector_days+1)].close)
                sector_return = ((current - past) / past) * 100

                # Relative strength vs benchmark
                relative_strength = sector_return - benchmark_return

                results.append({
                    "symbol": symbol,
                    "name": name,
                    "return": sector_return,
                    "relative": relative_strength,
                    "strength": self._strength_signal(relative_strength),
                })
            except Exception:
                continue

        # Sort by relative strength
        results.sort(key=lambda x: x["relative"], reverse=True)
        return results, benchmark_return

    def _strength_signal(self, relative: float) -> str:
        """Convert relative strength to signal"""
        if relative > 3:
            return "STRONG"
        elif relative > 1:
            return "OUTPERFORM"
        elif relative > -1:
            return "NEUTRAL"
        elif relative > -3:
            return "UNDERPERFORM"
        else:
            return "WEAK"

    def scan(self, days: int = 20) -> dict:
        """Full sector scan"""
        results, benchmark = self.calculate_returns(days)
        return {
            "sectors": results,
            "benchmark_return": benchmark,
            "days": days,
            "timestamp": datetime.now().isoformat(),
        }

    def hot_sectors(self, days: int = 20) -> list[dict]:
        """Get sectors outperforming benchmark"""
        results, _ = self.calculate_returns(days)
        return [r for r in results if r["strength"] in ("STRONG", "OUTPERFORM")]

    def cold_sectors(self, days: int = 20) -> list[dict]:
        """Get sectors underperforming benchmark"""
        results, _ = self.calculate_returns(days)
        return [r for r in results if r["strength"] in ("WEAK", "UNDERPERFORM")]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sector rotation scanner")
    parser.add_argument("command", nargs="?", default="scan",
                       choices=["scan", "hot", "cold", "json"])
    parser.add_argument("-d", "--days", type=int, default=20, help="Lookback period")
    args = parser.parse_args()

    scanner = SectorScanner()

    if args.command == "scan":
        result = scanner.scan(args.days)
        print(f"Sector Rotation ({args.days} days)")
        print(f"Benchmark (SPY): {result['benchmark_return']:+.2f}%")
        print("-" * 55)

        for s in result["sectors"]:
            indicator = {
                "STRONG": "🟢",
                "OUTPERFORM": "🟡",
                "NEUTRAL": "⚪",
                "UNDERPERFORM": "🟠",
                "WEAK": "🔴",
            }.get(s["strength"], "")

            print(f"{indicator} {s['symbol']:5} {s['name']:16} {s['return']:+6.2f}%  ({s['relative']:+5.2f}% vs SPY)  {s['strength']}")

    elif args.command == "hot":
        sectors = scanner.hot_sectors(args.days)
        if sectors:
            print(f"Hot Sectors ({args.days} days)")
            print("-" * 40)
            for s in sectors:
                print(f"  {s['symbol']:5} {s['name']:16} {s['relative']:+.2f}%")
        else:
            print("No sectors outperforming")

    elif args.command == "cold":
        sectors = scanner.cold_sectors(args.days)
        if sectors:
            print(f"Cold Sectors ({args.days} days)")
            print("-" * 40)
            for s in sectors:
                print(f"  {s['symbol']:5} {s['name']:16} {s['relative']:+.2f}%")
        else:
            print("No sectors underperforming")

    elif args.command == "json":
        import json
        result = scanner.scan(args.days)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
