#!/usr/bin/env python3
"""
market_report.py - comprehensive market analysis

Generates a full market report with:
- Current regime
- Sector performance
- Strategy recommendations
- Timing factors (day of week, month)
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

from config import load_keys
from strategies import RegimeStrategy, AdaptiveStrategy, BestOfEnsemble


def get_bars(client, symbols, days=90):
    """Fetch bars for multiple symbols"""
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    result = client.get_stock_bars(request)

    bars = {}
    for sym in symbols:
        if sym in result.data:
            bars[sym] = list(result.data[sym])
    return bars


def analyze_regime(bars):
    """Detect market regime"""
    if len(bars) < 20:
        return "UNKNOWN", "UNKNOWN", 0, 0

    # 20-day metrics
    current = float(bars[-1].close)
    past = float(bars[-20].close)
    price_change = (current - past) / past * 100

    # Volatility
    returns = []
    for i in range(-19, 0):
        ret = (float(bars[i].close) - float(bars[i-1].close)) / float(bars[i-1].close)
        returns.append(ret)
    avg_ret = sum(returns) / len(returns)
    variance = sum((r - avg_ret)**2 for r in returns) / len(returns)
    vol = math.sqrt(variance) * math.sqrt(252) * 100

    trend = "BULL" if price_change > 3 else "BEAR" if price_change < -3 else "RANGE"
    vol_regime = "HIGH_VOL" if vol > 25 else "LOW_VOL" if vol < 15 else "NORMAL"

    return trend, vol_regime, price_change, vol


def calculate_returns(bars, period):
    """Calculate returns over period"""
    if len(bars) < period:
        return None
    return (float(bars[-1].close) - float(bars[-period].close)) / float(bars[-period].close) * 100


def main():
    api_key, secret_key = load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    # Symbols to analyze
    core = ['SPY', 'QQQ']
    stocks = ['AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AMZN', 'AMD', 'TSLA']
    all_symbols = core + stocks

    print("=" * 70)
    print("MARKET REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Fetch data
    print("\nFetching market data...")
    bars = get_bars(client, all_symbols, days=180)

    # Market Regime
    print("\n" + "=" * 70)
    print("MARKET REGIME")
    print("-" * 70)

    spy_bars = bars.get('SPY', [])
    if spy_bars:
        trend, vol_regime, change, vol = analyze_regime(spy_bars)
        print(f"SPY Regime: {trend} / {vol_regime}")
        print(f"  20-day change: {change:+.1f}%")
        print(f"  20-day volatility: {vol:.1f}% (annualized)")

        # Strategy recommendation based on regime
        print(f"\n  Recommended approach:")
        if trend == "BULL":
            print("    -> Momentum/trend-following strategies")
            print("    -> Favor strong stocks (AMD, NVDA)")
        elif trend == "BEAR":
            print("    -> Mean reversion strategies")
            print("    -> Reduce position sizes, favor defensive")
        else:
            print("    -> Mean reversion strategies")
            print("    -> Range-bound trading, smaller positions")

    # Performance Table
    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("-" * 70)
    print(f"{'Symbol':8} {'1-Week':>10} {'1-Month':>10} {'3-Month':>10} {'Signal':>10}")
    print("-" * 70)

    regime = RegimeStrategy()
    adaptive = AdaptiveStrategy()

    for sym in all_symbols:
        if sym not in bars or len(bars[sym]) < 60:
            continue

        sym_bars = bars[sym]
        w1 = calculate_returns(sym_bars, 5)
        m1 = calculate_returns(sym_bars, 21)
        m3 = calculate_returns(sym_bars, 63)

        # Get signal
        regime.set_symbol(sym) if hasattr(regime, 'set_symbol') else None
        adaptive.set_symbol(sym)
        sig = adaptive.signal(sym_bars, len(sym_bars) - 1)

        signal_str = "BUY" if sig.strength > 0.3 else "SELL" if sig.strength < -0.3 else "HOLD"

        print(f"{sym:8} {w1:>+9.1f}% {m1:>+9.1f}% {m3:>+9.1f}% {signal_str:>10}")

    # Timing Factors
    print("\n" + "=" * 70)
    print("TIMING FACTORS")
    print("-" * 70)

    dow = datetime.now().weekday()
    month = datetime.now().month
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"Today: {days[dow]}")

    # Day of week edge (based on historical analysis)
    day_edges = {
        0: ("Positive", "+0.21% avg"),
        1: ("Neutral", "-0.01% avg"),
        2: ("Best day", "+0.30% avg"),
        3: ("Negative", "-0.18% avg"),
        4: ("Neutral", "-0.04% avg"),
    }
    if dow < 5:
        edge, stat = day_edges[dow]
        print(f"  Historical edge: {edge} ({stat})")

    print(f"\nCurrent month: {months[month-1]}")
    month_edges = {
        1: ("Neutral", "+0.1%"),
        2: ("Bearish", "-1.2%"),
        3: ("Very bearish", "-5.9%"),
        4: ("Neutral", "+0.2%"),
        5: ("Best month", "+6.2%"),
        6: ("Strong", "+4.8%"),
        7: ("Positive", "+2.3%"),
        8: ("Positive", "+2.1%"),
        9: ("Strong", "+3.2%"),
        10: ("Positive", "+2.4%"),
        11: ("Neutral", "+0.3%"),
        12: ("Neutral", "-0.2%"),
    }
    edge, stat = month_edges[month]
    print(f"  Historical edge: {edge} ({stat})")

    # Action Items
    print("\n" + "=" * 70)
    print("ACTION ITEMS")
    print("-" * 70)

    buys = []
    sells = []

    for sym in stocks:
        if sym not in bars:
            continue
        sym_bars = bars[sym]
        adaptive.set_symbol(sym)
        sig = adaptive.signal(sym_bars, len(sym_bars) - 1)

        if sig.strength > 0.3:
            buys.append((sym, sig.strength, sig.reason))
        elif sig.strength < -0.3:
            sells.append((sym, sig.strength, sig.reason))

    if buys:
        print("BUY candidates:")
        for sym, strength, reason in sorted(buys, key=lambda x: -x[1]):
            print(f"  {sym}: {reason}")

    if sells:
        print("SELL candidates:")
        for sym, strength, reason in sorted(sells, key=lambda x: x[1]):
            print(f"  {sym}: {reason}")

    if not buys and not sells:
        print("No strong signals. Consider waiting or using mean reversion on pullbacks.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
