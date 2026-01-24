#!/usr/bin/env python3
"""
mean_reversion.py - mean reversion strategies

Buy oversold, sell overbought. Assume prices revert to mean.
"""

from .base import Strategy, Signal


class BollingerReversion(Strategy):
    """Trade when price touches Bollinger Bands"""

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.params = {"period": period, "num_std": num_std}
        self.period = period
        self.num_std = num_std

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period:
            return Signal(0, "insufficient data", 0)

        # Calculate SMA and standard deviation
        closes = [float(bars[i].close) for i in range(idx - self.period + 1, idx + 1)]
        sma = sum(closes) / len(closes)
        variance = sum((c - sma) ** 2 for c in closes) / len(closes)
        std = variance ** 0.5

        current = float(bars[idx].close)
        upper = sma + (self.num_std * std)
        lower = sma - (self.num_std * std)

        # Calculate position relative to bands
        if std > 0:
            z_score = (current - sma) / std
        else:
            z_score = 0

        if current <= lower:
            # Oversold - buy signal
            strength = min(abs(z_score) / 3, 1.0)
            return Signal(strength, f"oversold z={z_score:.1f}", 0.65)
        elif current >= upper:
            # Overbought - sell signal
            strength = -min(abs(z_score) / 3, 1.0)
            return Signal(strength, f"overbought z={z_score:.1f}", 0.65)
        else:
            return Signal(0, f"neutral z={z_score:.1f}", 0.3)

    def warmup_period(self) -> int:
        return self.period + 1


class RSIReversion(Strategy):
    """Trade based on RSI overbought/oversold"""

    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.params = {"period": period, "oversold": oversold, "overbought": overbought}
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _calculate_rsi(self, bars: list, idx: int) -> float:
        """Calculate RSI"""
        gains = []
        losses = []

        for i in range(idx - self.period + 1, idx + 1):
            change = float(bars[i].close) - float(bars[i - 1].close)
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period + 1:
            return Signal(0, "insufficient data", 0)

        rsi = self._calculate_rsi(bars, idx)

        if rsi <= self.oversold:
            # Oversold - buy
            strength = (self.oversold - rsi) / self.oversold
            return Signal(min(strength, 1.0), f"RSI={rsi:.0f} oversold", 0.6)
        elif rsi >= self.overbought:
            # Overbought - sell
            strength = (rsi - self.overbought) / (100 - self.overbought)
            return Signal(-min(strength, 1.0), f"RSI={rsi:.0f} overbought", 0.6)
        else:
            return Signal(0, f"RSI={rsi:.0f} neutral", 0.3)

    def warmup_period(self) -> int:
        return self.period + 2
