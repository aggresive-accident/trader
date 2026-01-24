#!/usr/bin/env python3
"""
breakout.py - breakout and range strategies

Trade when price breaks out of ranges or patterns.
"""

from .base import Strategy, Signal


class DonchianBreakout(Strategy):
    """Trade Donchian Channel breakouts (turtle trading)"""

    name = "donchian"

    def __init__(self, period: int = 20):
        self.params = {"period": period}
        self.period = period

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period:
            return Signal(0, "insufficient data", 0)

        # Donchian channel
        highs = [float(bars[i].high) for i in range(idx - self.period, idx)]
        lows = [float(bars[i].low) for i in range(idx - self.period, idx)]

        upper = max(highs)
        lower = min(lows)
        current = float(bars[idx].close)
        current_high = float(bars[idx].high)
        current_low = float(bars[idx].low)

        # Breakout detection
        if current_high > upper:
            channel_width = upper - lower
            excess = (current - upper) / channel_width if channel_width > 0 else 0
            strength = min(0.6 + excess, 1.0)
            return Signal(strength, f"Donchian breakout", 0.75)
        elif current_low < lower:
            channel_width = upper - lower
            excess = (lower - current) / channel_width if channel_width > 0 else 0
            strength = -min(0.6 + excess, 1.0)
            return Signal(strength, f"Donchian breakdown", 0.75)

        # Position in channel
        pos = (current - lower) / (upper - lower) if upper > lower else 0.5
        if pos > 0.8:
            return Signal(0.3, f"near resistance ({pos:.0%})", 0.4)
        elif pos < 0.2:
            return Signal(-0.3, f"near support ({pos:.0%})", 0.4)

        return Signal(0, f"mid-channel ({pos:.0%})", 0.3)

    def warmup_period(self) -> int:
        return self.period + 1


class RangeBreakout(Strategy):
    """Trade consolidation range breakouts"""

    name = "range_breakout"

    def __init__(self, lookback: int = 10, threshold: float = 0.03):
        self.params = {"lookback": lookback, "threshold": threshold}
        self.lookback = lookback
        self.threshold = threshold

    def _is_consolidating(self, bars: list, idx: int) -> tuple:
        """Check if in consolidation, return (is_consolidating, range_high, range_low)"""
        if idx < self.lookback:
            return False, 0, 0

        highs = [float(bars[i].high) for i in range(idx - self.lookback, idx)]
        lows = [float(bars[i].low) for i in range(idx - self.lookback, idx)]

        range_high = max(highs)
        range_low = min(lows)
        range_pct = (range_high - range_low) / range_low if range_low > 0 else 0

        return range_pct < self.threshold, range_high, range_low

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.lookback + 1:
            return Signal(0, "insufficient data", 0)

        # Check previous consolidation
        was_consolidating, range_high, range_low = self._is_consolidating(bars, idx - 1)

        if not was_consolidating:
            return Signal(0, "not consolidating", 0.3)

        current = float(bars[idx].close)
        current_high = float(bars[idx].high)
        current_low = float(bars[idx].low)

        if current_high > range_high:
            breakout_pct = (current - range_high) / range_high * 100
            strength = min(0.7 + breakout_pct / 2, 1.0)
            return Signal(strength, f"range breakout +{breakout_pct:.1f}%", 0.8)
        elif current_low < range_low:
            breakout_pct = (range_low - current) / range_low * 100
            strength = -min(0.7 + breakout_pct / 2, 1.0)
            return Signal(strength, f"range breakdown -{breakout_pct:.1f}%", 0.8)

        return Signal(0, "still consolidating", 0.3)

    def warmup_period(self) -> int:
        return self.lookback + 2


class GapStrategy(Strategy):
    """Trade gaps (gap and go vs gap fill)"""

    name = "gap"

    def __init__(self, min_gap_pct: float = 1.0, fill_threshold: float = 0.5):
        self.params = {"min_gap_pct": min_gap_pct, "fill_threshold": fill_threshold}
        self.min_gap_pct = min_gap_pct
        self.fill_threshold = fill_threshold

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < 2:
            return Signal(0, "insufficient data", 0)

        prev_close = float(bars[idx - 1].close)
        current_open = float(bars[idx].open)
        current = float(bars[idx].close)

        gap_pct = (current_open - prev_close) / prev_close * 100

        if abs(gap_pct) < self.min_gap_pct:
            return Signal(0, f"no gap ({gap_pct:.1f}%)", 0.3)

        # Gap fill analysis
        if gap_pct > 0:
            # Gap up
            fill_pct = (current_open - current) / (current_open - prev_close) if current_open != prev_close else 0
            if fill_pct > self.fill_threshold:
                # Gap filling - bearish
                return Signal(-0.5, f"gap up filling ({fill_pct:.0%})", 0.6)
            else:
                # Gap and go - bullish
                strength = min(gap_pct / 3, 1.0)
                return Signal(strength, f"gap up holding +{gap_pct:.1f}%", 0.65)
        else:
            # Gap down
            fill_pct = (current - current_open) / (prev_close - current_open) if prev_close != current_open else 0
            if fill_pct > self.fill_threshold:
                # Gap filling - bullish
                return Signal(0.5, f"gap down filling ({fill_pct:.0%})", 0.6)
            else:
                # Gap and go - bearish
                strength = max(gap_pct / 3, -1.0)
                return Signal(strength, f"gap down continuing {gap_pct:.1f}%", 0.65)

    def warmup_period(self) -> int:
        return 3


class HighLowBreakout(Strategy):
    """Trade 52-week or N-day high/low breakouts"""

    name = "highlow"

    def __init__(self, period: int = 52):
        self.params = {"period": period}
        self.period = period

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period:
            return Signal(0, "insufficient data", 0)

        highs = [float(bars[i].high) for i in range(idx - self.period, idx)]
        lows = [float(bars[i].low) for i in range(idx - self.period, idx)]

        period_high = max(highs)
        period_low = min(lows)
        current_high = float(bars[idx].high)
        current_low = float(bars[idx].low)
        current = float(bars[idx].close)

        if current_high > period_high:
            excess = (current - period_high) / period_high * 100
            return Signal(0.8, f"{self.period}-day high +{excess:.1f}%", 0.75)
        elif current_low < period_low:
            excess = (period_low - current) / period_low * 100
            return Signal(-0.8, f"{self.period}-day low -{excess:.1f}%", 0.75)

        # Distance from extremes
        to_high = (period_high - current) / current * 100
        to_low = (current - period_low) / current * 100

        if to_high < 2:
            return Signal(0.4, f"near {self.period}-day high", 0.5)
        elif to_low < 2:
            return Signal(-0.4, f"near {self.period}-day low", 0.5)

        return Signal(0, f"mid-range", 0.3)

    def warmup_period(self) -> int:
        return self.period + 1
