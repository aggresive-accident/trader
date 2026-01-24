#!/usr/bin/env python3
"""
regime.py - regime-aware strategy

Detects market regime (bull/bear/range) and volatility (high/low)
and picks the appropriate strategy.

Based on backtesting:
- BULL: use momentum
- BEAR: use mean reversion
- RANGE: use mean reversion
- HIGH_VOL: reduce position sizes
"""

import math
from .base import Strategy, Signal
from .momentum import SimpleMomentum
from .mean_reversion import BollingerReversion


class RegimeStrategy(Strategy):
    """Switch strategies based on market regime"""

    name = "regime"

    def __init__(self, lookback: int = 20, trend_threshold: float = 3.0, vol_high: float = 25.0):
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.vol_high = vol_high

        self.momentum = SimpleMomentum()
        self.bollinger = BollingerReversion()

        self.params = {
            "lookback": lookback,
            "trend_threshold": trend_threshold,
            "vol_high": vol_high,
        }

    def _detect_regime(self, bars: list, idx: int) -> tuple:
        """Detect current market regime: (trend, volatility, trend_pct, vol_pct)"""
        if idx < self.lookback:
            return "RANGE", "NORMAL", 0, 20

        # Price change over lookback
        current = float(bars[idx].close)
        past = float(bars[idx - self.lookback].close)
        price_change = (current - past) / past * 100

        # Volatility over lookback
        returns = []
        for i in range(idx - self.lookback + 1, idx + 1):
            ret = (float(bars[i].close) - float(bars[i-1].close)) / float(bars[i-1].close)
            returns.append(ret)

        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret)**2 for r in returns) / len(returns)
        vol = math.sqrt(variance) * math.sqrt(252) * 100

        # Classify trend
        if price_change > self.trend_threshold:
            trend = "BULL"
        elif price_change < -self.trend_threshold:
            trend = "BEAR"
        else:
            trend = "RANGE"

        # Classify volatility
        if vol > self.vol_high:
            vol_regime = "HIGH_VOL"
        elif vol < 15:
            vol_regime = "LOW_VOL"
        else:
            vol_regime = "NORMAL"

        return trend, vol_regime, price_change, vol

    def signal(self, bars: list, idx: int) -> Signal:
        trend, vol_regime, trend_pct, vol_pct = self._detect_regime(bars, idx)

        # Pick strategy based on regime
        if trend == "BULL":
            sig = self.momentum.signal(bars, idx)
            strategy_used = "momentum"
        else:
            # BEAR or RANGE - use mean reversion
            sig = self.bollinger.signal(bars, idx)
            strategy_used = "bollinger"

        # Adjust confidence based on volatility
        confidence = sig.confidence
        if vol_regime == "HIGH_VOL":
            confidence *= 0.7  # Reduce confidence in high vol
        elif vol_regime == "LOW_VOL":
            confidence *= 1.1  # Boost confidence in low vol

        confidence = min(confidence, 1.0)

        return Signal(
            sig.strength,
            f"[{trend}/{vol_regime} -> {strategy_used}] {sig.reason}",
            confidence
        )

    def warmup_period(self) -> int:
        return max(
            self.lookback + 1,
            self.momentum.warmup_period(),
            self.bollinger.warmup_period()
        )
