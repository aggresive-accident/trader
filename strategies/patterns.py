#!/usr/bin/env python3
"""
patterns.py - candlestick and price pattern strategies

Recognize common patterns and trade them.
"""

from .base import Strategy, Signal


class CandlestickPatterns(Strategy):
    """Recognize basic candlestick patterns"""

    name = "candlestick"

    def __init__(self):
        self.params = {}

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < 3:
            return Signal(0, "insufficient data", 0)

        # Current candle
        o = float(bars[idx].open)
        h = float(bars[idx].high)
        l = float(bars[idx].low)
        c = float(bars[idx].close)

        # Previous candle
        po = float(bars[idx - 1].open)
        ph = float(bars[idx - 1].high)
        pl = float(bars[idx - 1].low)
        pc = float(bars[idx - 1].close)

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l if h > l else 0.01

        prev_body = abs(pc - po)

        # Hammer (bullish reversal)
        if lower_wick > 2 * body and upper_wick < body * 0.3 and c > o:
            if pc < po:  # After downtrend
                return Signal(0.7, "hammer (bullish)", 0.7)

        # Shooting star (bearish reversal)
        if upper_wick > 2 * body and lower_wick < body * 0.3 and c < o:
            if pc > po:  # After uptrend
                return Signal(-0.7, "shooting star (bearish)", 0.7)

        # Engulfing patterns
        if c > o and pc < po:  # Bullish candle after bearish
            if o < pc and c > po and body > prev_body:
                return Signal(0.8, "bullish engulfing", 0.75)

        if c < o and pc > po:  # Bearish candle after bullish
            if o > pc and c < po and body > prev_body:
                return Signal(-0.8, "bearish engulfing", 0.75)

        # Doji (indecision)
        if body < total_range * 0.1:
            return Signal(0, "doji (indecision)", 0.4)

        # Marubozu (strong conviction)
        if body > total_range * 0.9:
            if c > o:
                return Signal(0.5, "bullish marubozu", 0.6)
            else:
                return Signal(-0.5, "bearish marubozu", 0.6)

        return Signal(0, "no pattern", 0.3)

    def warmup_period(self) -> int:
        return 4


class ThreeBarPatterns(Strategy):
    """Recognize 3-bar patterns"""

    name = "three_bar"

    def __init__(self):
        self.params = {}

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < 4:
            return Signal(0, "insufficient data", 0)

        # Get last 3 candles
        candles = []
        for i in range(idx - 2, idx + 1):
            candles.append({
                "o": float(bars[i].open),
                "h": float(bars[i].high),
                "l": float(bars[i].low),
                "c": float(bars[i].close),
                "bullish": float(bars[i].close) > float(bars[i].open),
            })

        c1, c2, c3 = candles

        # Morning star (bullish reversal)
        if not c1["bullish"] and c3["bullish"]:
            c1_body = abs(c1["c"] - c1["o"])
            c2_body = abs(c2["c"] - c2["o"])
            c3_body = abs(c3["c"] - c3["o"])
            if c2_body < c1_body * 0.3 and c3_body > c1_body * 0.5:
                if c3["c"] > (c1["o"] + c1["c"]) / 2:
                    return Signal(0.8, "morning star", 0.75)

        # Evening star (bearish reversal)
        if c1["bullish"] and not c3["bullish"]:
            c1_body = abs(c1["c"] - c1["o"])
            c2_body = abs(c2["c"] - c2["o"])
            c3_body = abs(c3["c"] - c3["o"])
            if c2_body < c1_body * 0.3 and c3_body > c1_body * 0.5:
                if c3["c"] < (c1["o"] + c1["c"]) / 2:
                    return Signal(-0.8, "evening star", 0.75)

        # Three white soldiers
        if c1["bullish"] and c2["bullish"] and c3["bullish"]:
            if c2["c"] > c1["c"] and c3["c"] > c2["c"]:
                if c2["o"] > c1["o"] and c3["o"] > c2["o"]:
                    return Signal(0.7, "three white soldiers", 0.7)

        # Three black crows
        if not c1["bullish"] and not c2["bullish"] and not c3["bullish"]:
            if c2["c"] < c1["c"] and c3["c"] < c2["c"]:
                if c2["o"] < c1["o"] and c3["o"] < c2["o"]:
                    return Signal(-0.7, "three black crows", 0.7)

        return Signal(0, "no pattern", 0.3)

    def warmup_period(self) -> int:
        return 5


class SwingHighLow(Strategy):
    """Trade swing high/low reversals"""

    name = "swing"

    def __init__(self, lookback: int = 5):
        self.params = {"lookback": lookback}
        self.lookback = lookback

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.lookback * 2:
            return Signal(0, "insufficient data", 0)

        current_high = float(bars[idx].high)
        current_low = float(bars[idx].low)
        current_close = float(bars[idx].close)

        # Check for swing high (local maximum)
        is_swing_high = True
        is_swing_low = True

        for i in range(idx - self.lookback, idx):
            if float(bars[i].high) >= current_high:
                is_swing_high = False
            if float(bars[i].low) <= current_low:
                is_swing_low = False

        # Get prior swing for context
        prior_highs = [float(bars[i].high) for i in range(idx - self.lookback * 2, idx - self.lookback)]
        prior_lows = [float(bars[i].low) for i in range(idx - self.lookback * 2, idx - self.lookback)]

        if is_swing_high:
            prior_high = max(prior_highs)
            if current_high > prior_high:
                return Signal(0.5, "higher high (uptrend)", 0.6)
            else:
                return Signal(-0.6, "lower high (reversal)", 0.65)

        if is_swing_low:
            prior_low = min(prior_lows)
            if current_low < prior_low:
                return Signal(-0.5, "lower low (downtrend)", 0.6)
            else:
                return Signal(0.6, "higher low (reversal)", 0.65)

        return Signal(0, "no swing", 0.3)

    def warmup_period(self) -> int:
        return self.lookback * 2 + 1
