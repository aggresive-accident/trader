#!/usr/bin/env python3
"""
research.py - stock research module

Deep analysis before committing capital. Answers:
- Is the momentum real or a dead cat bounce?
- How volatile is this name? What's the daily range?
- Where are support/resistance levels?
- What's the relative strength vs SPY?
- Is volume confirming the move?
- What's the risk/reward at current price?

Usage:
  python3 research.py AMD
  python3 research.py AMD NVDA META
  python3 research.py --full AMD
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


class Researcher:
    """Deep stock research before committing capital."""

    def __init__(self):
        api_key, secret_key = load_keys()
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbol: str, days: int = 120) -> list:
        """Fetch daily bars."""
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

    def analyze(self, symbol: str) -> dict:
        """Full research report on a symbol."""
        bars = self.get_bars(symbol, days=120)
        if len(bars) < 30:
            return {"symbol": symbol, "error": "insufficient data", "bars": len(bars)}

        closes = [float(b.close) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        volumes = [float(b.volume) for b in bars]
        current = closes[-1]

        report = {"symbol": symbol, "price": current, "bars": len(bars)}

        # --- MOMENTUM ---
        report["momentum"] = self._momentum(closes)

        # --- VOLATILITY ---
        report["volatility"] = self._volatility(closes, highs, lows, bars)

        # --- VOLUME ---
        report["volume"] = self._volume_analysis(volumes)

        # --- SUPPORT / RESISTANCE ---
        report["levels"] = self._support_resistance(closes, highs, lows)

        # --- MOVING AVERAGES ---
        report["ma"] = self._moving_averages(closes)

        # --- RELATIVE STRENGTH vs SPY ---
        report["relative_strength"] = self._relative_strength(symbol, closes)

        # --- RISK/REWARD ---
        report["risk_reward"] = self._risk_reward(closes, highs, lows, bars)

        # --- VERDICT ---
        report["verdict"] = self._verdict(report)

        return report

    def _momentum(self, closes: list) -> dict:
        """Momentum metrics across timeframes."""
        current = closes[-1]

        def pct_change(n):
            if len(closes) > n:
                return (current - closes[-n-1]) / closes[-n-1] * 100
            return 0

        d1 = pct_change(1)
        d5 = pct_change(5)
        d10 = pct_change(10)
        d20 = pct_change(20)
        d60 = pct_change(60) if len(closes) > 60 else 0

        # Momentum acceleration: is recent momentum > older momentum?
        recent = d5
        older = d20 - d5 if d20 != 0 else 0
        accelerating = recent > older / 3 if older != 0 else recent > 0

        return {
            "1d": round(d1, 2),
            "5d": round(d5, 2),
            "10d": round(d10, 2),
            "20d": round(d20, 2),
            "60d": round(d60, 2),
            "accelerating": accelerating,
        }

    def _volatility(self, closes: list, highs: list, lows: list, bars: list) -> dict:
        """Volatility analysis."""
        # Daily returns
        returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]

        # ATR (14-period)
        atr_period = min(14, len(bars) - 1)
        trs = []
        for i in range(-atr_period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        atr = sum(trs) / len(trs) if trs else 0
        atr_pct = atr / closes[-1] * 100

        # Daily range
        daily_ranges = [(highs[i] - lows[i]) / closes[i] * 100 for i in range(-20, 0) if abs(i) <= len(closes)]
        avg_range = sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0

        # Std dev of returns
        if len(returns) > 1:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0

        # Max daily moves
        max_up = max(returns) if returns else 0
        max_down = min(returns) if returns else 0

        return {
            "atr": round(atr, 2),
            "atr_pct": round(atr_pct, 2),
            "avg_daily_range_pct": round(avg_range, 2),
            "daily_std_dev": round(std_dev, 2),
            "max_up_day": round(max_up, 2),
            "max_down_day": round(max_down, 2),
            "annualized_vol": round(std_dev * math.sqrt(252), 1),
        }

    def _volume_analysis(self, volumes: list) -> dict:
        """Volume trends and confirmation."""
        avg_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        avg_50 = sum(volumes[-50:]) / 50 if len(volumes) >= 50 else avg_20
        latest = volumes[-1]

        # Volume trend: is recent volume expanding?
        recent_avg = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else latest
        volume_expanding = recent_avg > avg_20 * 1.1

        # Up-volume vs down-volume (last 20 days)
        up_vol = 0
        down_vol = 0
        closes = None  # we don't have closes here, use volume direction proxy
        for i in range(-min(20, len(volumes)), 0):
            if volumes[i] > avg_20:
                up_vol += 1
            else:
                down_vol += 1

        return {
            "latest": int(latest),
            "avg_20d": int(avg_20),
            "avg_50d": int(avg_50),
            "ratio_vs_avg": round(latest / avg_20, 2) if avg_20 > 0 else 0,
            "expanding": volume_expanding,
            "above_avg_days": up_vol,
        }

    def _support_resistance(self, closes: list, highs: list, lows: list) -> dict:
        """Key support and resistance levels."""
        current = closes[-1]

        # Recent high/low
        high_20 = max(highs[-20:])
        low_20 = min(lows[-20:])
        high_60 = max(highs[-60:]) if len(highs) >= 60 else high_20
        low_60 = min(lows[-60:]) if len(lows) >= 60 else low_20

        # Where are we in the range?
        range_60 = high_60 - low_60
        position_in_range = (current - low_60) / range_60 * 100 if range_60 > 0 else 50

        # Key MAs as support
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20

        # Distance to key levels
        dist_to_high = (high_60 - current) / current * 100
        dist_to_low = (current - low_60) / current * 100
        dist_to_ma20 = (current - ma20) / current * 100
        dist_to_ma50 = (current - ma50) / current * 100

        return {
            "high_20d": round(high_20, 2),
            "low_20d": round(low_20, 2),
            "high_60d": round(high_60, 2),
            "low_60d": round(low_60, 2),
            "range_position_pct": round(position_in_range, 1),
            "dist_to_60d_high_pct": round(dist_to_high, 2),
            "dist_to_60d_low_pct": round(dist_to_low, 2),
            "dist_to_ma20_pct": round(dist_to_ma20, 2),
            "dist_to_ma50_pct": round(dist_to_ma50, 2),
        }

    def _moving_averages(self, closes: list) -> dict:
        """Moving average analysis."""
        current = closes[-1]
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes[-10:]) / 10
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20

        # MA slope (5-day change in MA20)
        if len(closes) >= 25:
            ma20_5ago = sum(closes[-25:-5]) / 20
            ma20_slope = (ma20 - ma20_5ago) / ma20_5ago * 100
        else:
            ma20_slope = 0

        # Stacked MAs (bullish: price > 5 > 10 > 20 > 50)
        stacked_bull = current > ma5 > ma10 > ma20 > ma50
        stacked_bear = current < ma5 < ma10 < ma20 < ma50

        return {
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "above_ma10": current > ma10,
            "above_ma20": current > ma20,
            "above_ma50": current > ma50,
            "ma20_slope_pct": round(ma20_slope, 2),
            "stacked_bullish": stacked_bull,
            "stacked_bearish": stacked_bear,
        }

    def _relative_strength(self, symbol: str, closes: list) -> dict:
        """Relative strength vs SPY."""
        if symbol == "SPY":
            return {"vs_spy": "N/A (is SPY)"}

        try:
            spy_bars = self.get_bars("SPY", days=120)
            if len(spy_bars) < 20:
                return {"vs_spy": "no data"}

            spy_closes = [float(b.close) for b in spy_bars]

            # Match lengths
            n = min(len(closes), len(spy_closes))
            closes = closes[-n:]
            spy_closes = spy_closes[-n:]

            # Relative performance over different periods
            def rel_perf(days):
                if n < days + 1:
                    return 0
                stock_ret = (closes[-1] - closes[-days-1]) / closes[-days-1] * 100
                spy_ret = (spy_closes[-1] - spy_closes[-days-1]) / spy_closes[-days-1] * 100
                return stock_ret - spy_ret

            rs_5 = rel_perf(5)
            rs_20 = rel_perf(20)
            rs_60 = rel_perf(60) if n > 60 else 0

            # RS rating (0-100 percentile - simplified)
            # Just use 60d relative as proxy
            rs_rating = min(99, max(1, 50 + rs_60))

            return {
                "vs_spy_5d": round(rs_5, 2),
                "vs_spy_20d": round(rs_20, 2),
                "vs_spy_60d": round(rs_60, 2),
                "rs_rating": round(rs_rating),
                "outperforming": rs_20 > 0,
            }
        except Exception:
            return {"vs_spy": "error"}

    def _risk_reward(self, closes: list, highs: list, lows: list, bars: list) -> dict:
        """Risk/reward analysis at current price."""
        current = closes[-1]

        # ATR for stop calculation
        atr_period = min(14, len(bars) - 1)
        trs = []
        for i in range(-atr_period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        atr = sum(trs) / len(trs) if trs else 0

        # Risk: 1 ATR stop
        stop = current - atr
        risk_pct = atr / current * 100

        # Reward: distance to 60d high or 2x ATR target
        high_60 = max(highs[-60:]) if len(highs) >= 60 else max(highs[-20:])
        target_high = high_60
        target_2atr = current + (atr * 2)
        target = max(target_high, target_2atr)

        reward_pct = (target - current) / current * 100
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0

        # Position size for 5% portfolio risk on $100k
        portfolio = 100000
        risk_per_share = atr
        shares_for_5pct = int((portfolio * 0.05) / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares_for_5pct * current

        return {
            "stop_1atr": round(stop, 2),
            "risk_pct": round(risk_pct, 2),
            "target": round(target, 2),
            "reward_pct": round(reward_pct, 2),
            "rr_ratio": round(rr_ratio, 2),
            "shares_5pct_risk": shares_for_5pct,
            "position_value": round(position_value),
        }

    def _verdict(self, report: dict) -> dict:
        """Overall buy/hold/avoid verdict."""
        score = 0
        reasons = []

        # Momentum
        mom = report["momentum"]
        if mom["5d"] > 3:
            score += 2
            reasons.append(f"+{mom['5d']}% week")
        elif mom["5d"] > 0:
            score += 1
        if mom["accelerating"]:
            score += 1
            reasons.append("accelerating")

        # Trend
        ma = report["ma"]
        if ma["stacked_bullish"]:
            score += 2
            reasons.append("MAs stacked bullish")
        elif ma["above_ma20"]:
            score += 1
            reasons.append("above MA20")
        if ma["ma20_slope_pct"] > 0.5:
            score += 1
            reasons.append("MA20 rising")

        # Volume
        vol = report["volume"]
        if vol["expanding"]:
            score += 1
            reasons.append("volume expanding")

        # Relative strength
        rs = report["relative_strength"]
        if rs.get("outperforming"):
            score += 1
            reasons.append("beating SPY")

        # Risk/reward
        rr = report["risk_reward"]
        if rr["rr_ratio"] > 2:
            score += 1
            reasons.append(f"R:R {rr['rr_ratio']}:1")

        # Range position (not too extended)
        levels = report["levels"]
        if levels["range_position_pct"] > 90:
            score -= 1
            reasons.append("extended (>90% of range)")

        # Verdict
        if score >= 6:
            action = "STRONG BUY"
        elif score >= 4:
            action = "BUY"
        elif score >= 2:
            action = "WATCH"
        else:
            action = "AVOID"

        return {
            "score": score,
            "max_score": 9,
            "action": action,
            "reasons": reasons,
        }


def print_report(report: dict, full: bool = False):
    """Print a research report."""
    if "error" in report:
        print(f"{report['symbol']}: {report['error']}")
        return

    sym = report["symbol"]
    price = report["price"]
    v = report["verdict"]

    print(f"\n{'=' * 60}")
    print(f"  {sym}  ${price:.2f}  |  {v['action']} ({v['score']}/{v['max_score']})")
    print(f"  {', '.join(v['reasons'])}")
    print(f"{'=' * 60}")

    # Momentum
    m = report["momentum"]
    print(f"\n  MOMENTUM")
    print(f"    1d: {m['1d']:+.1f}%  5d: {m['5d']:+.1f}%  10d: {m['10d']:+.1f}%  20d: {m['20d']:+.1f}%  60d: {m['60d']:+.1f}%")
    print(f"    {'Accelerating' if m['accelerating'] else 'Decelerating'}")

    # Volatility
    vol = report["volatility"]
    print(f"\n  VOLATILITY")
    print(f"    ATR: ${vol['atr']} ({vol['atr_pct']:.1f}%)  Daily range: {vol['avg_daily_range_pct']:.1f}%")
    print(f"    Max up: +{vol['max_up_day']:.1f}%  Max down: {vol['max_down_day']:.1f}%")
    print(f"    Annualized vol: {vol['annualized_vol']}%")

    # Moving averages
    ma = report["ma"]
    print(f"\n  MOVING AVERAGES")
    flags = []
    if ma["above_ma10"]:
        flags.append(">MA10")
    if ma["above_ma20"]:
        flags.append(">MA20")
    if ma["above_ma50"]:
        flags.append(">MA50")
    if ma["stacked_bullish"]:
        flags.append("STACKED BULL")
    print(f"    MA10: ${ma['ma10']:.2f}  MA20: ${ma['ma20']:.2f}  MA50: ${ma['ma50']:.2f}")
    print(f"    {' | '.join(flags) if flags else 'below key MAs'}")
    print(f"    MA20 slope: {ma['ma20_slope_pct']:+.2f}%")

    # Volume
    v = report["volume"]
    print(f"\n  VOLUME")
    print(f"    Latest: {v['latest']:,}  Avg 20d: {v['avg_20d']:,}  Ratio: {v['ratio_vs_avg']:.1f}x")
    print(f"    {'Expanding' if v['expanding'] else 'Contracting'}  |  {v['above_avg_days']}/20 days above avg")

    # Levels
    lv = report["levels"]
    print(f"\n  LEVELS")
    print(f"    20d range: ${lv['low_20d']:.2f} - ${lv['high_20d']:.2f}")
    print(f"    60d range: ${lv['low_60d']:.2f} - ${lv['high_60d']:.2f}")
    print(f"    Position in range: {lv['range_position_pct']:.0f}%")
    print(f"    Dist to 60d high: +{lv['dist_to_60d_high_pct']:.1f}%  Dist to 60d low: -{lv['dist_to_60d_low_pct']:.1f}%")

    # Relative strength
    rs = report["relative_strength"]
    if "vs_spy_20d" in rs:
        print(f"\n  VS SPY")
        print(f"    5d: {rs['vs_spy_5d']:+.1f}%  20d: {rs['vs_spy_20d']:+.1f}%  60d: {rs['vs_spy_60d']:+.1f}%")
        print(f"    RS Rating: {rs['rs_rating']}  {'OUTPERFORMING' if rs.get('outperforming') else 'underperforming'}")

    # Risk/reward
    rr = report["risk_reward"]
    print(f"\n  RISK/REWARD")
    print(f"    Stop (1 ATR): ${rr['stop_1atr']:.2f} ({rr['risk_pct']:.1f}% risk)")
    print(f"    Target: ${rr['target']:.2f} ({rr['reward_pct']:.1f}% reward)")
    print(f"    R:R ratio: {rr['rr_ratio']:.1f}:1")
    print(f"    Position size (5% risk): {rr['shares_5pct_risk']} shares (${rr['position_value']:,})")

    print()


def main():
    symbols = [s.upper() for s in sys.argv[1:] if not s.startswith("-")]
    full = "--full" in sys.argv

    if not symbols:
        print("Usage: python3 research.py SYMBOL [SYMBOL2 ...]")
        print("       python3 research.py --full AMD")
        return 1

    r = Researcher()

    for sym in symbols:
        report = r.analyze(sym)
        print_report(report, full=full)

    # If multiple symbols, show comparison
    if len(symbols) > 1:
        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        print(f"  {'Symbol':<8} {'Price':>8} {'5d':>7} {'20d':>7} {'ATR%':>6} {'R:R':>5} {'Score':>6} {'Action':<12}")
        print(f"  {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*12}")

        reports = []
        for sym in symbols:
            rpt = r.analyze(sym) if sym != symbols[0] else report  # reuse last if single
            # Actually re-analyze all for comparison
            rpt = r.analyze(sym)
            reports.append(rpt)

        # Sort by verdict score
        reports.sort(key=lambda x: x.get("verdict", {}).get("score", 0), reverse=True)

        for rpt in reports:
            if "error" in rpt:
                print(f"  {rpt['symbol']:<8} {'ERROR':>8}")
                continue
            s = rpt["symbol"]
            p = rpt["price"]
            m5 = rpt["momentum"]["5d"]
            m20 = rpt["momentum"]["20d"]
            atr = rpt["volatility"]["atr_pct"]
            rr = rpt["risk_reward"]["rr_ratio"]
            sc = rpt["verdict"]["score"]
            act = rpt["verdict"]["action"]
            print(f"  {s:<8} ${p:>7.2f} {m5:>+6.1f}% {m20:>+6.1f}% {atr:>5.1f}% {rr:>4.1f}x {sc:>3}/{9} {act}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
