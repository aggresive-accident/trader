#!/usr/bin/env python3
"""
backtest.py - strategy backtesting

Test momentum strategy on historical data.
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

# Strategy parameters (same as strategy.py)
MAX_POSITION_PCT = 0.10
MAX_POSITIONS = 5
INITIAL_CASH = 100000


class Backtest:
    """Simple backtester for momentum strategy"""

    def __init__(self, symbols: list[str], days: int = 60):
        api_key, secret_key = load_keys()
        self.data = StockHistoricalDataClient(api_key, secret_key)
        self.symbols = symbols
        self.days = days
        self.bars = None

    def fetch_data(self) -> None:
        """Fetch historical data"""
        end = datetime.now()
        start = end - timedelta(days=self.days + 10)

        request = StockBarsRequest(
            symbol_or_symbols=self.symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )

        self.bars = self.data.get_stock_bars(request)

    def calculate_momentum(self, symbol: str, day_idx: int, lookback: int = 5) -> float:
        """Calculate momentum at a specific day"""
        symbol_bars = self.bars[symbol]
        if day_idx < lookback:
            return 0

        current = float(symbol_bars[day_idx].close)
        past = float(symbol_bars[day_idx - lookback].close)
        return ((current - past) / past) * 100

    def run(self) -> dict:
        """Run backtest"""
        if self.bars is None:
            self.fetch_data()

        # Get the date range from first symbol
        first_symbol = self.symbols[0]
        all_dates = [bar.timestamp.date() for bar in self.bars[first_symbol]]

        # State
        cash = INITIAL_CASH
        positions = {}  # symbol -> {qty, entry_price}
        trades = []
        daily_values = []

        # Skip first 10 days for momentum calculation
        for day_idx in range(10, len(all_dates)):
            date = all_dates[day_idx]

            # Calculate portfolio value
            portfolio_value = cash
            for symbol, pos in positions.items():
                try:
                    current_price = float(self.bars[symbol][day_idx].close)
                    portfolio_value += pos["qty"] * current_price
                except:
                    pass

            daily_values.append({"date": str(date), "value": portfolio_value})

            # Calculate momentum for all symbols
            momentum = {}
            for symbol in self.symbols:
                try:
                    momentum[symbol] = self.calculate_momentum(symbol, day_idx)
                except:
                    momentum[symbol] = 0

            # Sort by momentum
            ranked = sorted(momentum.items(), key=lambda x: x[1], reverse=True)

            # Sell positions with negative momentum
            for symbol in list(positions.keys()):
                if momentum.get(symbol, 0) < -3:
                    try:
                        price = float(self.bars[symbol][day_idx].close)
                        qty = positions[symbol]["qty"]
                        cash += qty * price
                        trades.append({
                            "date": str(date),
                            "action": "SELL",
                            "symbol": symbol,
                            "qty": qty,
                            "price": price,
                            "reason": f"momentum {momentum[symbol]:.1f}%",
                        })
                        del positions[symbol]
                    except:
                        pass

            # Buy top momentum stocks
            for symbol, mom in ranked[:3]:
                if symbol in positions:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    break
                if mom < 3:  # need positive momentum
                    continue

                try:
                    price = float(self.bars[symbol][day_idx].close)
                    max_spend = portfolio_value * MAX_POSITION_PCT
                    if cash < max_spend * 0.5:
                        continue

                    qty = int(min(max_spend, cash * 0.8) / price)
                    if qty < 1:
                        continue

                    cost = qty * price
                    cash -= cost
                    positions[symbol] = {"qty": qty, "entry_price": price}
                    trades.append({
                        "date": str(date),
                        "action": "BUY",
                        "symbol": symbol,
                        "qty": qty,
                        "price": price,
                        "reason": f"momentum {mom:.1f}%",
                    })
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

        # Calculate returns
        total_return = ((final_value - INITIAL_CASH) / INITIAL_CASH) * 100

        return {
            "initial": INITIAL_CASH,
            "final": final_value,
            "return_pct": total_return,
            "trades": len(trades),
            "trade_log": trades,
            "daily_values": daily_values,
            "final_positions": positions,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest momentum strategy")
    parser.add_argument("-d", "--days", type=int, default=60, help="Days to backtest")
    parser.add_argument("-s", "--symbols", default="SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,NVDA,META,AMD,TSLA",
                       help="Comma-separated symbols")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show trades")
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    print(f"Backtesting {len(symbols)} symbols over {args.days} days...")

    bt = Backtest(symbols, args.days)
    result = bt.run()

    print()
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Period:      {args.days} days")
    print(f"Symbols:     {len(symbols)}")
    print(f"Initial:     ${result['initial']:,.2f}")
    print(f"Final:       ${result['final']:,.2f}")
    print(f"Return:      {result['return_pct']:+.2f}%")
    print(f"Trades:      {result['trades']}")

    if result["final_positions"]:
        print(f"\nOpen Positions:")
        for symbol, pos in result["final_positions"].items():
            print(f"  {symbol}: {pos['qty']} @ ${pos['entry_price']:.2f}")

    if args.verbose and result["trade_log"]:
        print(f"\nTrade Log:")
        for trade in result["trade_log"]:
            print(f"  {trade['date']} {trade['action']:4} {trade['symbol']:6} x{trade['qty']:4} @ ${trade['price']:8.2f}  ({trade['reason']})")


if __name__ == "__main__":
    main()
