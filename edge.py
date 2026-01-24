#!/usr/bin/env python3
"""
edge.py - the actual trading system

ONE EDGE: Momentum concentration with discipline.

Rules:
1. One position at a time, max 20% of portfolio
2. Only buy top momentum stocks breaking out on volume
3. Stop: 1.5x ATR below entry, max 3% portfolio risk
4. Trail stop, never give back >50% of gains
5. Exit when momentum dies (close < 20 MA)
"""

import sys
import math
from datetime import datetime, timedelta
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trader import Trader
from config import load_keys

# Risk parameters
MAX_POSITION_PCT = 0.20  # 20% max position
MAX_RISK_PCT = 0.03      # 3% max risk per trade
ATR_STOP_MULT = 1.5      # Stop at 1.5x ATR
TRAIL_GIVEBACK = 0.50    # Don't give back more than 50% of gains


class EdgeTrader:
    """The real trading system"""

    def __init__(self):
        self.trader = Trader()
        api_key, secret_key = load_keys()
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbol: str, days: int = 60) -> list:
        """Fetch bars"""
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        result = self.data.get_stock_bars(request)
        if hasattr(result, 'data') and symbol in result.data:
            return list(result.data[symbol])
        return []

    def calculate_atr(self, bars: list, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(bars) < period + 1:
            return 0

        trs = []
        for i in range(-period, 0):
            high = float(bars[i].high)
            low = float(bars[i].low)
            prev_close = float(bars[i-1].close)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        return sum(trs) / len(trs)

    def calculate_momentum(self, bars: list) -> dict:
        """Calculate momentum metrics"""
        if len(bars) < 21:
            return None

        current = float(bars[-1].close)
        week_ago = float(bars[-6].close) if len(bars) > 5 else current
        month_ago = float(bars[-22].close) if len(bars) > 21 else current

        # Moving averages
        ma20 = sum(float(bars[i].close) for i in range(-20, 0)) / 20
        ma50 = sum(float(bars[i].close) for i in range(-50, 0)) / 50 if len(bars) >= 50 else ma20

        # Volume
        avg_vol = sum(float(bars[i].volume) for i in range(-20, 0)) / 20
        current_vol = float(bars[-1].volume)
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

        # ATR
        atr = self.calculate_atr(bars)
        atr_pct = atr / current * 100

        return {
            "price": current,
            "week_return": (current - week_ago) / week_ago * 100,
            "month_return": (current - month_ago) / month_ago * 100,
            "above_ma20": current > ma20,
            "above_ma50": current > ma50,
            "ma20": ma20,
            "ma50": ma50,
            "vol_ratio": vol_ratio,
            "atr": atr,
            "atr_pct": atr_pct,
        }

    def scan_for_setups(self, symbols: list) -> list:
        """Find momentum breakout setups"""
        setups = []

        for sym in symbols:
            bars = self.get_bars(sym, days=60)
            if len(bars) < 30:
                continue

            mom = self.calculate_momentum(bars)
            if not mom:
                continue

            # Momentum criteria
            score = 0
            reasons = []

            # Strong 1-week return
            if mom["week_return"] > 5:
                score += 2
                reasons.append(f"+{mom['week_return']:.1f}% week")
            elif mom["week_return"] > 2:
                score += 1

            # Strong 1-month return
            if mom["month_return"] > 15:
                score += 2
                reasons.append(f"+{mom['month_return']:.1f}% month")
            elif mom["month_return"] > 8:
                score += 1

            # Above moving averages
            if mom["above_ma20"] and mom["above_ma50"]:
                score += 1
                reasons.append("above MAs")

            # Volume confirmation
            if mom["vol_ratio"] > 1.5:
                score += 1
                reasons.append(f"{mom['vol_ratio']:.1f}x volume")

            # Only consider high scores
            if score >= 3:
                setups.append({
                    "symbol": sym,
                    "score": score,
                    "price": mom["price"],
                    "atr": mom["atr"],
                    "atr_pct": mom["atr_pct"],
                    "reasons": reasons,
                    "week_return": mom["week_return"],
                    "month_return": mom["month_return"],
                    "ma20": mom["ma20"],
                })

        # Sort by score
        setups.sort(key=lambda x: x["score"], reverse=True)
        return setups

    def calculate_position(self, setup: dict) -> dict:
        """Calculate position size and stops"""
        account = self.trader.get_account()
        portfolio = account["portfolio_value"]

        price = setup["price"]
        atr = setup["atr"]

        # Stop distance: 1.5x ATR
        stop_distance = atr * ATR_STOP_MULT
        stop_price = price - stop_distance
        stop_pct = stop_distance / price * 100

        # Risk-based position sizing
        # Risk = position_size * stop_pct
        # Max risk = portfolio * MAX_RISK_PCT
        # position_size = (portfolio * MAX_RISK_PCT) / stop_pct
        max_risk_dollars = portfolio * MAX_RISK_PCT
        position_value = max_risk_dollars / (stop_pct / 100)

        # Cap at max position size
        max_position = portfolio * MAX_POSITION_PCT
        position_value = min(position_value, max_position)

        shares = int(position_value / price)

        return {
            "symbol": setup["symbol"],
            "price": price,
            "shares": shares,
            "position_value": shares * price,
            "position_pct": (shares * price) / portfolio * 100,
            "stop_price": stop_price,
            "stop_pct": stop_pct,
            "risk_dollars": shares * stop_distance,
            "risk_pct": (shares * stop_distance) / portfolio * 100,
        }

    def check_exit(self, symbol: str, entry_price: float, high_water: float) -> dict:
        """Check if we should exit a position"""
        bars = self.get_bars(symbol, days=30)
        if not bars:
            return {"exit": False, "reason": "no data"}

        current = float(bars[-1].close)
        atr = self.calculate_atr(bars)
        ma20 = sum(float(bars[i].close) for i in range(-20, 0)) / 20

        # Initial stop: 1.5x ATR below entry
        initial_stop = entry_price - (atr * ATR_STOP_MULT)

        # Trailing stop: don't give back more than 50% of gains
        if high_water > entry_price:
            gain = high_water - entry_price
            trail_stop = high_water - (gain * TRAIL_GIVEBACK)
            stop_price = max(initial_stop, trail_stop)
        else:
            stop_price = initial_stop

        # Exit conditions
        if current < stop_price:
            return {
                "exit": True,
                "reason": f"stop hit at ${stop_price:.2f}",
                "price": current,
            }

        if current < ma20:
            return {
                "exit": True,
                "reason": f"closed below 20 MA (${ma20:.2f})",
                "price": current,
            }

        return {
            "exit": False,
            "current": current,
            "stop": stop_price,
            "ma20": ma20,
            "gain_pct": (current - entry_price) / entry_price * 100,
        }


def main():
    print("=" * 60)
    print("EDGE TRADING SYSTEM")
    print("=" * 60)

    edge = EdgeTrader()

    # Account status
    account = edge.trader.get_account()
    print(f"\nPortfolio: ${account['portfolio_value']:,.2f}")
    print(f"Cash: ${account['cash']:,.2f}")

    # Current positions
    positions = edge.trader.get_positions()
    if positions:
        print(f"\nCurrent Positions:")
        for p in positions:
            print(f"  {p['symbol']}: {p['qty']} shares @ ${p['current_price']:.2f} ({p['unrealized_pl_pct']:+.1f}%)")
    else:
        print("\nNo current positions")

    # Scan for setups
    print("\n" + "-" * 60)
    print("SCANNING FOR SETUPS...")
    print("-" * 60)

    watchlist = [
        "AMD", "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "AVGO", "CRM", "ORCL", "NFLX", "ADBE", "INTC", "QCOM",
        "SPY", "QQQ", "IWM",
    ]

    setups = edge.scan_for_setups(watchlist)

    if setups:
        print(f"\nTop Setups (score >= 3):")
        for s in setups[:5]:
            print(f"\n  {s['symbol']} (score: {s['score']})")
            print(f"    Price: ${s['price']:.2f}")
            print(f"    Reasons: {', '.join(s['reasons'])}")
            print(f"    ATR: ${s['atr']:.2f} ({s['atr_pct']:.1f}%)")

            pos = edge.calculate_position(s)
            print(f"    Position: {pos['shares']} shares (${pos['position_value']:,.0f}, {pos['position_pct']:.1f}%)")
            print(f"    Stop: ${pos['stop_price']:.2f} ({pos['stop_pct']:.1f}% risk)")
            print(f"    Risk: ${pos['risk_dollars']:.0f} ({pos['risk_pct']:.1f}% of portfolio)")
    else:
        print("\nNo setups found. Wait for better conditions.")

    # Trading recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if not positions and setups:
        best = setups[0]
        pos = edge.calculate_position(best)
        print(f"\nBUY: {best['symbol']}")
        print(f"  Entry: ${best['price']:.2f}")
        print(f"  Shares: {pos['shares']}")
        print(f"  Stop: ${pos['stop_price']:.2f}")
        print(f"  Risk: ${pos['risk_dollars']:.0f} ({pos['risk_pct']:.1f}%)")
    elif positions:
        print("\nAlready in position. Monitor for exit signals.")
    else:
        print("\nNo action. Wait for setup.")


if __name__ == "__main__":
    main()
