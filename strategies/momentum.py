#!/usr/bin/env python3
"""
momentum.py - momentum-based strategies

Buy what's going up, sell what's going down.
"""

from .base import Strategy, Signal


class SimpleMomentum(Strategy):
    """Simple price momentum over lookback period"""

    name = "momentum"

    def __init__(self, lookback: int = 5, threshold: float = 3.0):
        self.params = {"lookback": lookback, "threshold": threshold}
        self.lookback = lookback
        self.threshold = threshold

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.lookback:
            return Signal(0, "insufficient data", 0)

        current = float(bars[idx].close)
        past = float(bars[idx - self.lookback].close)
        momentum = ((current - past) / past) * 100

        if momentum > self.threshold:
            strength = min(momentum / 10, 1.0)  # cap at +1
            return Signal(strength, f"up {momentum:.1f}%", 0.6)
        elif momentum < -self.threshold:
            strength = max(momentum / 10, -1.0)  # cap at -1
            return Signal(strength, f"down {momentum:.1f}%", 0.6)
        else:
            return Signal(0, f"flat {momentum:.1f}%", 0.3)

    def warmup_period(self) -> int:
        return self.lookback + 1


class AcceleratingMomentum(Strategy):
    """Momentum that's accelerating (second derivative)"""

    name = "accel_momentum"

    def __init__(self, short: int = 3, long: int = 10):
        self.params = {"short": short, "long": long}
        self.short = short
        self.long = long

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.long:
            return Signal(0, "insufficient data", 0)

        current = float(bars[idx].close)
        short_past = float(bars[idx - self.short].close)
        long_past = float(bars[idx - self.long].close)

        short_mom = ((current - short_past) / short_past) * 100
        long_mom = ((current - long_past) / long_past) * 100

        # Acceleration = short momentum - scaled long momentum
        long_mom_scaled = long_mom * (self.short / self.long)
        acceleration = short_mom - long_mom_scaled

        if acceleration > 1 and short_mom > 0:
            strength = min(acceleration / 5, 1.0)
            return Signal(strength, f"accelerating +{acceleration:.1f}", 0.7)
        elif acceleration < -1 and short_mom < 0:
            strength = max(acceleration / 5, -1.0)
            return Signal(strength, f"decelerating {acceleration:.1f}", 0.7)
        else:
            return Signal(0, f"steady {acceleration:.1f}", 0.3)

    def warmup_period(self) -> int:
        return self.long + 1
