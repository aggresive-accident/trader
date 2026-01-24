#!/usr/bin/env python3
"""
signal_grid.py - show all strategy signals in a grid

Quick visual comparison of what each strategy thinks.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import load_keys
from strategies import ALL_STRATEGIES, CATEGORIES


def get_bars(client, symbol: str, days: int = 90) -> list:
    """Fetch historical bars"""
    # Use slightly older end date to avoid subscription issues
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    if hasattr(bars, 'data') and symbol in bars.data:
        return list(bars.data[symbol])
    elif symbol in bars:
        return list(bars[symbol])
    return []


def signal_char(strength: float) -> str:
    """Convert signal strength to visual character"""
    if strength > 0.6:
        return "+"  # Strong buy
    elif strength > 0.3:
        return "+"  # Buy
    elif strength < -0.6:
        return "-"  # Strong sell
    elif strength < -0.3:
        return "-"  # Sell
    else:
        return "."  # Neutral


def signal_color(strength: float) -> str:
    """ANSI color for signal"""
    if strength > 0.6:
        return "\033[92m"  # Bright green
    elif strength > 0.3:
        return "\033[32m"  # Green
    elif strength < -0.6:
        return "\033[91m"  # Bright red
    elif strength < -0.3:
        return "\033[31m"  # Red
    else:
        return "\033[90m"  # Gray


def reset_color() -> str:
    return "\033[0m"


def generate_grid(symbols: list, categories: list = None) -> None:
    """Generate signal grid for symbols"""
    api_key, secret_key = load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    # Select strategies
    if categories:
        strategies = []
        for cat in categories:
            if cat in CATEGORIES:
                strategies.extend([s() for s in CATEGORIES[cat]])
    else:
        # Use key strategies from each category
        strategies = [
            s() for s in [
                ALL_STRATEGIES[0],   # SimpleMomentum
                ALL_STRATEGIES[2],   # BollingerReversion
                ALL_STRATEGIES[4],   # MovingAverageCross
                ALL_STRATEGIES[6],   # VolumeBreakout
                ALL_STRATEGIES[9],   # ATRBreakout
                ALL_STRATEGIES[12],  # DonchianBreakout
                ALL_STRATEGIES[16],  # CandlestickPatterns
                ALL_STRATEGIES[19],  # VotingEnsemble
            ]
        ]

    # Header
    strat_names = [s.name[:8] for s in strategies]
    header = f"{'Symbol':<8} " + " ".join(f"{n:>8}" for n in strat_names) + "  Score"
    print("=" * len(header))
    print("SIGNAL GRID")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Process each symbol
    results = []
    for symbol in symbols:
        try:
            bars = get_bars(client, symbol)
            if len(bars) < 30:
                print(f"{symbol:<8} [insufficient data]")
                continue

            signals = []
            row = f"{symbol:<8} "

            for strat in strategies:
                if len(bars) >= strat.warmup_period():
                    try:
                        sig = strat.signal(bars, len(bars) - 1)
                        signals.append(sig.strength)
                        char = signal_char(sig.strength)
                        color = signal_color(sig.strength)
                        row += f"{color}{char:>8}{reset_color()} "
                    except:
                        signals.append(0)
                        row += f"{'?':>8} "
                else:
                    signals.append(0)
                    row += f"{'':>8} "

            # Composite score
            if signals:
                score = sum(signals) / len(signals)
                score_color = signal_color(score)
                row += f" {score_color}{score:+.2f}{reset_color()}"
                results.append((symbol, score, signals))

            print(row)

        except Exception as e:
            print(f"{symbol:<8} [error: {str(e)[:40]}]")

    # Summary
    print("-" * len(header))
    if results:
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        bullish = [r for r in results if r[1] > 0.2]
        bearish = [r for r in results if r[1] < -0.2]

        if bullish:
            print(f"\nMost bullish: {', '.join(r[0] for r in bullish[:3])}")
        if bearish:
            print(f"Most bearish: {', '.join(r[0] for r in bearish[:3])}")

        # Legend
        print(f"\nLegend: {signal_color(0.7)}+ buy{reset_color()}  {signal_color(-0.7)}- sell{reset_color()}  {signal_color(0)}. neutral{reset_color()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strategy signal grid")
    parser.add_argument("symbols", nargs="*", default=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"])
    parser.add_argument("--categories", "-c", nargs="*", help="Strategy categories to show")
    args = parser.parse_args()

    generate_grid(args.symbols, args.categories)


if __name__ == "__main__":
    main()
