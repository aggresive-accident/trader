#!/usr/bin/env python3
"""
premarket.py - pre-market scanner

Checks overnight action:
- Gap up/down from previous close
- Pre-market volume (if available)
- News sentiment (placeholder for future)

Run before market opens to adjust trading plan.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

from config import load_keys
from edge import WEEK_MIN, WEEK_MAX

# Gap thresholds
GAP_THRESHOLD = 0.02  # 2% gap is significant


def get_previous_close(client, symbols):
    """Get yesterday's close for each symbol"""
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=5)  # Get a few days to ensure we have data

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )

    result = client.get_stock_bars(request)
    closes = {}

    for sym in symbols:
        if hasattr(result, "data") and sym in result.data:
            bars = list(result.data[sym])
            if bars:
                closes[sym] = float(bars[-1].close)

    return closes


def get_latest_prices(client, symbols):
    """Get latest prices (pre-market if available)"""
    try:
        request = StockLatestBarRequest(symbol_or_symbols=symbols)
        result = client.get_stock_latest_bar(request)

        prices = {}
        for sym in symbols:
            if sym in result:
                prices[sym] = float(result[sym].close)

        return prices
    except Exception as e:
        print(f"Could not get latest prices: {e}")
        return {}


def analyze_gap(prev_close: float, current: float) -> dict:
    """Analyze the gap"""
    gap_pct = (current - prev_close) / prev_close * 100
    gap_type = "UP" if gap_pct > 0 else "DOWN" if gap_pct < 0 else "FLAT"

    # Impact on entry decision
    if gap_type == "UP" and gap_pct > 3:
        impact = "DO NOT CHASE - wait for pullback"
    elif gap_type == "DOWN" and gap_pct < -3:
        impact = "POTENTIAL ENTRY - check momentum intact"
    elif gap_type == "UP" and gap_pct > 1:
        impact = "Watch first 30 min - may extend or fade"
    else:
        impact = "Normal open - follow standard plan"

    return {
        "prev_close": prev_close,
        "current": current,
        "gap_pct": gap_pct,
        "gap_type": gap_type,
        "impact": impact,
    }


def main():
    api_key, secret_key = load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    # Watchlist (same as edge.py)
    watchlist = [
        "AMD", "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA",
        "AVGO", "CRM", "ORCL", "NFLX", "ADBE", "INTC", "QCOM",
        "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
    ]

    print("=" * 60)
    print("PRE-MARKET SCANNER")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Get previous closes
    print("\nFetching previous close prices...")
    prev_closes = get_previous_close(client, watchlist)

    # Get current/pre-market prices
    print("Checking latest prices...")
    latest = get_latest_prices(client, watchlist)

    # Analyze gaps
    gaps = []
    for sym in watchlist:
        if sym in prev_closes and sym in latest:
            gap = analyze_gap(prev_closes[sym], latest[sym])
            gap["symbol"] = sym
            gaps.append(gap)

    # Sort by gap magnitude
    gaps.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)

    # Display results
    print(f"\n{'Symbol':6} {'Prev Close':>12} {'Current':>12} {'Gap':>8} {'Impact'}")
    print("-" * 70)

    significant_gaps = []
    for g in gaps:
        gap_str = f"{g['gap_pct']:+.1f}%"
        if abs(g['gap_pct']) >= GAP_THRESHOLD * 100:
            significant_gaps.append(g)
            flag = " ***" if g['gap_type'] == 'UP' and g['gap_pct'] > 3 else ""
            flag = " !!!" if g['gap_type'] == 'DOWN' and g['gap_pct'] < -3 else flag
        else:
            flag = ""

        print(f"{g['symbol']:6} ${g['prev_close']:>11.2f} ${g['current']:>11.2f} {gap_str:>8}{flag}")

    # Action items
    print("\n" + "=" * 60)
    print("ACTION ITEMS")
    print("-" * 60)

    if significant_gaps:
        for g in significant_gaps:
            if g['gap_type'] == 'UP' and g['gap_pct'] > 3:
                print(f"\n{g['symbol']}: Gapped UP {g['gap_pct']:+.1f}%")
                print(f"  -> {g['impact']}")
            elif g['gap_type'] == 'DOWN' and g['gap_pct'] < -3:
                print(f"\n{g['symbol']}: Gapped DOWN {g['gap_pct']:+.1f}%")
                print(f"  -> {g['impact']}")
            elif abs(g['gap_pct']) > 2:
                print(f"\n{g['symbol']}: Gap {g['gap_pct']:+.1f}%")
                print(f"  -> {g['impact']}")
    else:
        print("\nNo significant gaps. Normal open expected.")
        print("Follow standard entry criteria from edge.py")

    print("\n" + "=" * 60)
    print("NOTE: Pre-market data may have limited availability.")
    print("Re-run closer to open for more accurate prices.")
    print("=" * 60)


if __name__ == "__main__":
    main()
