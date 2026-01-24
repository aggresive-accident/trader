#!/usr/bin/env python3
"""
zoo.py - strategy zoo comparison

Compare multiple strategies via backtesting.
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
from strategies import ALL_STRATEGIES, Strategy, Signal

# Backtest parameters
INITIAL_CASH = 100000
MAX_POSITION_PCT = 0.10
MAX_POSITIONS = 5


class StrategyBacktest:
    """Backtest a single strategy"""

    def __init__(self, strategy: Strategy, bars: dict, symbols: list):
        self.strategy = strategy
        self.bars = bars
        self.symbols = symbols

    def run(self) -> dict:
        """Run backtest and return results"""
        # Get date range from first symbol
        first = self.symbols[0]
        num_bars = len(self.bars[first])
        warmup = self.strategy.warmup_period()

        # State
        cash = INITIAL_CASH
        positions = {}  # symbol -> {qty, entry_price}
        trades = []
        daily_values = []

        for idx in range(warmup, num_bars):
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, pos in positions.items():
                try:
                    price = float(self.bars[symbol][idx].close)
                    portfolio_value += pos["qty"] * price
                except:
                    pass

            daily_values.append(portfolio_value)

            # Generate signals for all symbols
            signals = {}
            for symbol in self.symbols:
                try:
                    sig = self.strategy.signal(self.bars[symbol], idx)
                    signals[symbol] = sig
                except Exception as e:
                    signals[symbol] = Signal(0, str(e), 0)

            # Sell positions with negative signals
            for symbol in list(positions.keys()):
                sig = signals.get(symbol)
                if sig and sig.strength < -0.3:
                    try:
                        price = float(self.bars[symbol][idx].close)
                        qty = positions[symbol]["qty"]
                        cash += qty * price
                        trades.append(("SELL", symbol, qty, price, sig.reason))
                        del positions[symbol]
                    except:
                        pass

            # Buy based on positive signals
            ranked = sorted(
                [(s, sig) for s, sig in signals.items() if sig.strength > 0.3],
                key=lambda x: x[1].strength,
                reverse=True
            )

            for symbol, sig in ranked[:3]:
                if symbol in positions:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break

                try:
                    price = float(self.bars[symbol][idx].close)
                    max_spend = portfolio_value * MAX_POSITION_PCT
                    if cash < max_spend * 0.5:
                        continue

                    qty = int(min(max_spend, cash * 0.8) / price)
                    if qty < 1:
                        continue

                    cost = qty * price
                    cash -= cost
                    positions[symbol] = {"qty": qty, "entry_price": price}
                    trades.append(("BUY", symbol, qty, price, sig.reason))
                except:
                    pass

        # Final value
        final_value = cash
        for symbol, pos in positions.items():
            try:
                price = float(self.bars[symbol][-1].close)
                final_value += pos["qty"] * price
            except:
                pass

        # Calculate metrics
        returns = [(daily_values[i] - daily_values[i-1]) / daily_values[i-1]
                   for i in range(1, len(daily_values))]

        avg_return = sum(returns) / len(returns) if returns else 0
        std_return = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5 if returns else 1

        sharpe = (avg_return / std_return * (252**0.5)) if std_return > 0 else 0

        # Max drawdown
        peak = INITIAL_CASH
        max_dd = 0
        for val in daily_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

        return {
            "strategy": str(self.strategy),
            "initial": INITIAL_CASH,
            "final": final_value,
            "return_pct": ((final_value - INITIAL_CASH) / INITIAL_CASH) * 100,
            "trades": len(trades),
            "sharpe": sharpe,
            "max_drawdown": max_dd * 100,
            "win_rate": self._calculate_win_rate(trades),
        }

    def _calculate_win_rate(self, trades: list) -> float:
        """Calculate win rate from trades"""
        # Track P/L per symbol
        entries = {}
        wins = 0
        total = 0

        for action, symbol, qty, price, reason in trades:
            if action == "BUY":
                entries[symbol] = price
            elif action == "SELL" and symbol in entries:
                if price > entries[symbol]:
                    wins += 1
                total += 1
                del entries[symbol]

        return (wins / total * 100) if total > 0 else 0


def fetch_data(symbols: list, days: int) -> dict:
    """Fetch historical data for all symbols"""
    api_key, secret_key = load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now()
    start = end - timedelta(days=days + 30)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    return client.get_stock_bars(request)


def run_zoo(symbols: list, days: int = 60) -> list:
    """Run all strategies and compare"""
    print(f"Fetching {days} days of data for {len(symbols)} symbols...")
    bars = fetch_data(symbols, days)

    results = []
    for strategy_class in ALL_STRATEGIES:
        strategy = strategy_class()
        print(f"Testing {strategy.name}...", end=" ", flush=True)

        bt = StrategyBacktest(strategy, bars, symbols)
        result = bt.run()
        results.append(result)
        print(f"{result['return_pct']:+.2f}%")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strategy zoo comparison")
    parser.add_argument("-d", "--days", type=int, default=60, help="Days to backtest")
    parser.add_argument("-s", "--symbols", default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,NVDA,META,AMD,TSLA",
                       help="Comma-separated symbols")
    args = parser.parse_args()

    symbols = args.symbols.split(",")

    results = run_zoo(symbols, args.days)

    # Sort by return
    results.sort(key=lambda x: x["return_pct"], reverse=True)

    print()
    print("=" * 70)
    print("STRATEGY ZOO RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Win%':>8}")
    print("-" * 70)

    for r in results:
        name = r["strategy"][:24]
        print(f"{name:<25} {r['return_pct']:>+9.2f}% {r['sharpe']:>8.2f} {r['max_drawdown']:>7.1f}% {r['trades']:>8} {r['win_rate']:>7.1f}%")

    print()
    print("Best strategy:", results[0]["strategy"])


if __name__ == "__main__":
    main()
