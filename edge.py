#!/usr/bin/env python3
"""
edge.py - the actual trading system

EDGE: Concentrated momentum, full margin, no fear.

Rules:
1. Up to 3 positions, use full buying power
2. Only the strongest momentum names - high beta, high volume
3. Stop: 1.0x ATR below entry, 5% portfolio risk per trade
4. Trail tight - don't give back >40% of gains
5. Exit when momentum dies (close < 10 MA for speed)
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

# Risk parameters - AGGRESSIVE
# Target: maximum daily return using full margin
MAX_POSITIONS = 3        # Concentrated - fewer, bigger
MAX_POSITION_PCT = 0.50  # 50% of buying power per position
MAX_RISK_PCT = 0.05      # 5% portfolio risk per trade
ATR_STOP_MULT = 1.0      # Tight stop at 1.0x ATR
TRAIL_GIVEBACK = 0.40    # Don't give back more than 40% of gains
WEEK_MIN = 3             # Lower entry bar - catch more momentum
WEEK_MAX = 20            # Don't cap upside - ride the runners
USE_MARGIN = True        # Use full buying power


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

            # Entry criteria: WEEK_MIN < week_return < WEEK_MAX
            week_ret = mom["week_return"]

            if week_ret < WEEK_MIN or week_ret > WEEK_MAX:
                continue  # Outside entry zone

            if not mom["above_ma20"]:
                continue  # Must be above 20 MA

            # Score the setup
            score = 0
            reasons = []

            # Week return in sweet spot
            score += 2
            reasons.append(f"+{week_ret:.1f}% week")

            # Strong 1-month return
            if mom["month_return"] > 15:
                score += 2
                reasons.append(f"+{mom['month_return']:.1f}% month")
            elif mom["month_return"] > 8:
                score += 1

            # Above both MAs
            if mom["above_ma50"]:
                score += 1
                reasons.append("above MAs")

            # Volume confirmation
            if mom["vol_ratio"] > 1.5:
                score += 1
                reasons.append(f"{mom['vol_ratio']:.1f}x volume")

            # All setups meeting criteria are valid
            if score >= 2:
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
        """Calculate position size and stops - uses full buying power"""
        account = self.trader.get_account()
        portfolio = account["portfolio_value"]
        buying_power = account["buying_power"] if USE_MARGIN else account["cash"]

        price = setup["price"]
        atr = setup["atr"]

        # Stop distance: tighter ATR stop
        stop_distance = atr * ATR_STOP_MULT
        stop_price = price - stop_distance
        stop_pct = stop_distance / price * 100

        # Risk-based position sizing against EQUITY (not buying power)
        max_risk_dollars = portfolio * MAX_RISK_PCT
        position_value = max_risk_dollars / (stop_pct / 100)

        # Cap at max position pct of BUYING POWER (not equity)
        max_position = buying_power * MAX_POSITION_PCT
        position_value = min(position_value, max_position)

        # Also cap at available buying power
        position_value = min(position_value, buying_power)

        shares = int(position_value / price)

        return {
            "symbol": setup["symbol"],
            "price": price,
            "shares": shares,
            "position_value": shares * price,
            "position_pct": (shares * price) / portfolio * 100,
            "buying_power_pct": (shares * price) / buying_power * 100 if buying_power > 0 else 0,
            "stop_price": stop_price,
            "stop_pct": stop_pct,
            "risk_dollars": shares * stop_distance,
            "risk_pct": (shares * stop_distance) / portfolio * 100,
            "margin_used": shares * price > account["cash"],
        }

    def check_exit(self, symbol: str, entry_price: float, high_water: float) -> dict:
        """Check if we should exit a position"""
        bars = self.get_bars(symbol, days=30)
        if not bars or len(bars) < 20:
            return {"exit": False, "reason": "insufficient data", "stop": None, "ma20": None}

        current = float(bars[-1].close)
        atr = self.calculate_atr(bars)
        ma10 = sum(float(bars[i].close) for i in range(-10, 0)) / 10
        ma20 = sum(float(bars[i].close) for i in range(-20, 0)) / 20

        # Initial stop: 1.0x ATR below entry (tight)
        initial_stop = entry_price - (atr * ATR_STOP_MULT)

        # Trailing stop: don't give back more than 40% of gains
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

        # Use 10 MA for faster exit on momentum death
        if current < ma10:
            return {
                "exit": True,
                "reason": f"closed below 10 MA (${ma10:.2f})",
                "price": current,
            }

        return {
            "exit": False,
            "current": current,
            "stop": stop_price,
            "ma10": ma10,
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

    # Aggressive universe - high beta momentum names
    watchlist = [
        # Mega-cap momentum
        "NVDA", "META", "TSLA", "AMD", "AVGO", "NFLX", "AMZN", "GOOGL", "AAPL", "MSFT",
        # Semis (high beta)
        "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "QCOM", "INTC", "ARM", "SMCI",
        # High beta tech
        "CRM", "ORCL", "ADBE", "PLTR", "COIN", "MSTR", "SNOW", "CRWD", "NET",
        # Energy momentum
        "XOM", "CVX", "OXY", "SLB", "HAL",
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
    print(f"\nBuying power: ${account['buying_power']:,.0f} (margin: {account['buying_power']/account['portfolio_value']:.1f}x)")

    current_symbols = [p['symbol'] for p in positions]
    available_slots = MAX_POSITIONS - len(positions)

    if available_slots > 0 and setups:
        new_setups = [s for s in setups if s['symbol'] not in current_symbols]

        if new_setups:
            print(f"Slots available: {available_slots}/{MAX_POSITIONS}")
            total_deploy = 0
            for setup in new_setups[:available_slots]:
                pos = edge.calculate_position(setup)
                total_deploy += pos['position_value']
                margin_tag = " [MARGIN]" if pos.get('margin_used') else ""
                print(f"\n  BUY: {setup['symbol']} (score: {setup['score']})")
                print(f"    Entry: ${setup['price']:.2f}")
                print(f"    Shares: {pos['shares']} (${pos['position_value']:,.0f} = {pos['position_pct']:.0f}% equity){margin_tag}")
                print(f"    Stop: ${pos['stop_price']:.2f} ({pos['stop_pct']:.1f}%)")
                print(f"    Risk: ${pos['risk_dollars']:.0f} ({pos['risk_pct']:.1f}% of equity)")
            print(f"\n  Total new deployment: ${total_deploy:,.0f}")
        else:
            print("\nNo new setups. Hold current positions.")
    elif len(positions) >= MAX_POSITIONS:
        print(f"\nFully loaded ({MAX_POSITIONS} positions). Monitor for exits.")
    else:
        print("\nNo setups meeting criteria. Scanning again at next interval.")


if __name__ == "__main__":
    main()
