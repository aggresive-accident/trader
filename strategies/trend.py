#!/usr/bin/env python3
"""
trend.py - trend following strategies

Follow the trend. The trend is your friend.
"""

from .base import Strategy, Signal


class MovingAverageCross(Strategy):
    """Trade when short MA crosses long MA"""

    name = "ma_cross"

    def __init__(self, short: int = 10, long: int = 30):
        self.params = {"short": short, "long": long}
        self.short = short
        self.long = long

    def _sma(self, bars: list, idx: int, period: int) -> float:
        """Calculate simple moving average"""
        if idx < period - 1:
            return 0
        closes = [float(bars[i].close) for i in range(idx - period + 1, idx + 1)]
        return sum(closes) / len(closes)

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.long + 1:
            return Signal(0, "insufficient data", 0)

        short_ma = self._sma(bars, idx, self.short)
        long_ma = self._sma(bars, idx, self.long)
        prev_short = self._sma(bars, idx - 1, self.short)
        prev_long = self._sma(bars, idx - 1, self.long)

        # Check for crossover
        crossed_up = prev_short <= prev_long and short_ma > long_ma
        crossed_down = prev_short >= prev_long and short_ma < long_ma

        # Also consider distance from MA
        distance = (short_ma - long_ma) / long_ma * 100

        if crossed_up:
            return Signal(0.8, f"golden cross +{distance:.1f}%", 0.75)
        elif crossed_down:
            return Signal(-0.8, f"death cross {distance:.1f}%", 0.75)
        elif short_ma > long_ma:
            strength = min(distance / 5, 0.5)
            return Signal(strength, f"bullish +{distance:.1f}%", 0.5)
        else:
            strength = max(distance / 5, -0.5)
            return Signal(strength, f"bearish {distance:.1f}%", 0.5)

    def warmup_period(self) -> int:
        return self.long + 2


class TrendStrength(Strategy):
    """Measure trend strength using ADX-like calculation"""

    name = "trend_strength"

    def __init__(self, period: int = 14, threshold: float = 25):
        self.params = {"period": period, "threshold": threshold}
        self.period = period
        self.threshold = threshold

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period + 1:
            return Signal(0, "insufficient data", 0)

        # Calculate directional movement
        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(idx - self.period + 1, idx + 1):
            high = float(bars[i].high)
            low = float(bars[i].low)
            close = float(bars[i].close)
            prev_high = float(bars[i - 1].high)
            prev_low = float(bars[i - 1].low)
            prev_close = float(bars[i - 1].close)

            # True range
            tr.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

            # Directional movement
            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        # Average
        atr = sum(tr) / len(tr) if tr else 1
        avg_plus = sum(plus_dm) / len(plus_dm) if plus_dm else 0
        avg_minus = sum(minus_dm) / len(minus_dm) if minus_dm else 0

        # Directional indicators
        plus_di = (avg_plus / atr * 100) if atr > 0 else 0
        minus_di = (avg_minus / atr * 100) if atr > 0 else 0

        # DX and ADX approximation
        di_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum > 0 else 0

        # Signal based on trend strength and direction
        if dx > self.threshold:
            if plus_di > minus_di:
                strength = min(dx / 50, 1.0)
                return Signal(strength, f"strong uptrend ADX={dx:.0f}", 0.7)
            else:
                strength = -min(dx / 50, 1.0)
                return Signal(strength, f"strong downtrend ADX={dx:.0f}", 0.7)
        else:
            return Signal(0, f"weak trend ADX={dx:.0f}", 0.3)

    def warmup_period(self) -> int:
        return self.period + 2
