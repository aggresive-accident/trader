#!/usr/bin/env python3
"""
volume.py - volume-based strategies

Volume confirms price moves. High volume = conviction.
"""

from .base import Strategy, Signal


class VolumeBreakout(Strategy):
    """Trade when volume spikes above average"""

    name = "volume_breakout"

    def __init__(self, period: int = 20, volume_mult: float = 2.0):
        self.params = {"period": period, "volume_mult": volume_mult}
        self.period = period
        self.volume_mult = volume_mult

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period:
            return Signal(0, "insufficient data", 0)

        # Calculate average volume
        volumes = [float(bars[i].volume) for i in range(idx - self.period, idx)]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = float(bars[idx].volume)

        # Price change
        price_change = (float(bars[idx].close) - float(bars[idx - 1].close)) / float(bars[idx - 1].close) * 100

        # Volume ratio
        vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        if vol_ratio > self.volume_mult:
            if price_change > 0.5:
                strength = min(vol_ratio / 4, 1.0)
                return Signal(strength, f"vol breakout +{price_change:.1f}% ({vol_ratio:.1f}x)", 0.7)
            elif price_change < -0.5:
                strength = -min(vol_ratio / 4, 1.0)
                return Signal(strength, f"vol breakdown {price_change:.1f}% ({vol_ratio:.1f}x)", 0.7)

        return Signal(0, f"normal vol ({vol_ratio:.1f}x)", 0.3)

    def warmup_period(self) -> int:
        return self.period + 1


class OnBalanceVolume(Strategy):
    """Trade based on OBV trend"""

    name = "obv"

    def __init__(self, period: int = 10):
        self.params = {"period": period}
        self.period = period

    def _calculate_obv(self, bars: list, end_idx: int, length: int) -> list:
        """Calculate OBV series"""
        obv = [0]
        for i in range(end_idx - length + 1, end_idx + 1):
            if i < 1:
                continue
            vol = float(bars[i].volume)
            if float(bars[i].close) > float(bars[i - 1].close):
                obv.append(obv[-1] + vol)
            elif float(bars[i].close) < float(bars[i - 1].close):
                obv.append(obv[-1] - vol)
            else:
                obv.append(obv[-1])
        return obv

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < self.period + 5:
            return Signal(0, "insufficient data", 0)

        obv = self._calculate_obv(bars, idx, self.period + 5)

        if len(obv) < 5:
            return Signal(0, "insufficient obv", 0)

        # OBV trend (recent vs older)
        recent_obv = sum(obv[-3:]) / 3
        older_obv = sum(obv[-6:-3]) / 3 if len(obv) >= 6 else recent_obv

        # Price trend
        recent_price = float(bars[idx].close)
        older_price = float(bars[idx - 3].close)

        obv_change = (recent_obv - older_obv) / abs(older_obv) * 100 if older_obv != 0 else 0
        price_change = (recent_price - older_price) / older_price * 100

        # Divergence detection
        if obv_change > 5 and price_change < -1:
            # Bullish divergence
            return Signal(0.6, f"bullish divergence OBV+{obv_change:.0f}%", 0.7)
        elif obv_change < -5 and price_change > 1:
            # Bearish divergence
            return Signal(-0.6, f"bearish divergence OBV{obv_change:.0f}%", 0.7)
        elif obv_change > 10:
            return Signal(0.4, f"OBV rising +{obv_change:.0f}%", 0.5)
        elif obv_change < -10:
            return Signal(-0.4, f"OBV falling {obv_change:.0f}%", 0.5)

        return Signal(0, f"OBV neutral {obv_change:.0f}%", 0.3)

    def warmup_period(self) -> int:
        return self.period + 6


class VWAP(Strategy):
    """Trade relative to VWAP"""

    name = "vwap"

    def __init__(self, threshold_pct: float = 1.0):
        self.params = {"threshold_pct": threshold_pct}
        self.threshold_pct = threshold_pct

    def signal(self, bars: list, idx: int) -> Signal:
        if idx < 5:
            return Signal(0, "insufficient data", 0)

        # Calculate VWAP for recent bars (simplified - full day would need intraday)
        total_pv = 0
        total_vol = 0

        for i in range(max(0, idx - 20), idx + 1):
            typical = (float(bars[i].high) + float(bars[i].low) + float(bars[i].close)) / 3
            vol = float(bars[i].volume)
            total_pv += typical * vol
            total_vol += vol

        vwap = total_pv / total_vol if total_vol > 0 else float(bars[idx].close)
        current = float(bars[idx].close)
        deviation = (current - vwap) / vwap * 100

        if deviation < -self.threshold_pct:
            # Below VWAP - potential buy
            strength = min(abs(deviation) / 3, 1.0)
            return Signal(strength, f"below VWAP {deviation:.1f}%", 0.6)
        elif deviation > self.threshold_pct:
            # Above VWAP - potential sell
            strength = -min(abs(deviation) / 3, 1.0)
            return Signal(strength, f"above VWAP +{deviation:.1f}%", 0.6)

        return Signal(0, f"at VWAP {deviation:.1f}%", 0.3)

    def warmup_period(self) -> int:
        return 21
