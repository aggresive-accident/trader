#!/usr/bin/env python3
"""
volatility.py - volatility-based strategies

Trade volatility expansion and contraction.
"""

from .base import Strategy, Signal


class ATRBreakout(Strategy):
    """Trade breakouts based on ATR"""

    name = "atr_breakout"

    def __init__(self, period: int = 14, mult: float = 2.0):
        self.params = {"period": period, "mult": mult}
        self.period = period
        self.mult = mult

    def _calculate_atr(self, bars: list, idx: int) -> float:
        """Calculate Average True Range"""
        trs = []
        for i in range(idx - self.period + 1, idx + 1):
            high = float(bars[i].high)
            low = float(bars[i].low)
            prev_close = float(bars[i - 1].close) if i > 0 else low
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs)

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period + 1:
            return Signal(0, "insufficient data", 0)

        atr = self._calculate_atr(bars, idx)
        current = float(bars[idx].close)
        prev_close = float(bars[idx - 1].close)

        # Check for breakout
        change = current - prev_close
        atr_mult = abs(change) / atr if atr > 0 else 0

        if atr_mult > self.mult:
            if change > 0:
                strength = min(atr_mult / 4, 1.0)
                return Signal(strength, f"ATR breakout +{atr_mult:.1f}x", 0.7)
            else:
                strength = -min(atr_mult / 4, 1.0)
                return Signal(strength, f"ATR breakdown {atr_mult:.1f}x", 0.7)

        return Signal(0, f"normal range {atr_mult:.1f}x ATR", 0.3)

    def warmup_period(self) -> int:
        return self.period + 2


class VolatilityContraction(Strategy):
    """Trade after volatility contracts (anticipating expansion)"""

    name = "vol_contraction"

    def __init__(self, short: int = 5, long: int = 20, threshold: float = 0.6):
        self.params = {"short": short, "long": long, "threshold": threshold}
        self.short = short
        self.long = long
        self.threshold = threshold

    def _volatility(self, bars: list, idx: int, period: int) -> float:
        """Calculate volatility as std of returns"""
        if idx < period:
            return 0
        returns = []
        for i in range(idx - period + 1, idx + 1):
            ret = (float(bars[i].close) - float(bars[i - 1].close)) / float(bars[i - 1].close)
            returns.append(ret)
        avg = sum(returns) / len(returns)
        variance = sum((r - avg) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.long + 1:
            return Signal(0, "insufficient data", 0)

        short_vol = self._volatility(bars, idx, self.short)
        long_vol = self._volatility(bars, idx, self.long)

        ratio = short_vol / long_vol if long_vol > 0 else 1

        # Volatility contraction - expect expansion
        if ratio < self.threshold:
            # Direction based on recent trend
            recent_change = (float(bars[idx].close) - float(bars[idx - 3].close)) / float(bars[idx - 3].close) * 100
            if recent_change > 0:
                return Signal(0.5, f"vol squeeze bullish {ratio:.2f}", 0.65)
            else:
                return Signal(-0.5, f"vol squeeze bearish {ratio:.2f}", 0.65)

        return Signal(0, f"normal vol {ratio:.2f}", 0.3)

    def warmup_period(self) -> int:
        return self.long + 2


class KeltnerChannel(Strategy):
    """Trade Keltner Channel breakouts"""

    name = "keltner"

    def __init__(self, period: int = 20, mult: float = 2.0):
        self.params = {"period": period, "mult": mult}
        self.period = period
        self.mult = mult

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period + 1:
            return Signal(0, "insufficient data", 0)

        # EMA of close
        closes = [float(bars[i].close) for i in range(idx - self.period + 1, idx + 1)]
        ema = sum(closes) / len(closes)  # simplified - use SMA

        # ATR
        trs = []
        for i in range(idx - self.period + 1, idx + 1):
            high = float(bars[i].high)
            low = float(bars[i].low)
            prev_close = float(bars[i - 1].close) if i > 0 else low
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        atr = sum(trs) / len(trs)

        upper = ema + (self.mult * atr)
        lower = ema - (self.mult * atr)
        current = float(bars[idx].close)

        if current > upper:
            # Breakout above
            excess = (current - upper) / atr if atr > 0 else 0
            strength = min(0.5 + excess / 2, 1.0)
            return Signal(strength, f"above Keltner +{excess:.1f}ATR", 0.65)
        elif current < lower:
            # Breakdown below - could be oversold
            excess = (lower - current) / atr if atr > 0 else 0
            # Mean reversion - buy oversold
            strength = min(0.5 + excess / 2, 1.0)
            return Signal(strength, f"below Keltner -{excess:.1f}ATR (oversold)", 0.6)

        return Signal(0, "inside Keltner", 0.3)

    def warmup_period(self) -> int:
        return self.period + 2
