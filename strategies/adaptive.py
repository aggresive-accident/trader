#!/usr/bin/env python3
"""
adaptive.py - stock-adaptive strategy

Uses historical analysis to pick the right strategy per stock.
Momentum stocks get momentum strategy, mean-reversion stocks get Bollinger.
"""

from .base import Strategy, Signal
from .momentum import SimpleMomentum
from .mean_reversion import BollingerReversion
from .trend import MovingAverageCross

# Based on backtesting: which strategy wins for each stock
# when momentum and mean-reversion disagree
STOCK_PROFILES = {
    # Momentum-favoring stocks (>55% momentum wins)
    "AMD": "momentum",
    "AAPL": "momentum",
    "SPY": "momentum",

    # Mean-reversion favoring stocks (>55% mean-reversion wins)
    "MSFT": "mean_reversion",
    "META": "mean_reversion",
    "QQQ": "mean_reversion",
    "NVDA": "mean_reversion",
    "TSLA": "mean_reversion",

    # Balanced stocks - use ensemble
    "GOOGL": "balanced",
    "AMZN": "balanced",
}


class AdaptiveStrategy(Strategy):
    """Pick strategy based on stock characteristics"""

    name = "adaptive"

    def __init__(self):
        self.momentum = SimpleMomentum()
        self.bollinger = BollingerReversion()
        self.ma_cross = MovingAverageCross()
        self.params = {"profiles": len(STOCK_PROFILES)}

        # Track current symbol (set externally)
        self.current_symbol = None

    def set_symbol(self, symbol: str):
        """Set the symbol to analyze"""
        self.current_symbol = symbol.upper()

    def _get_profile(self, symbol: str) -> str:
        """Get the profile for a stock"""
        return STOCK_PROFILES.get(symbol.upper(), "balanced")

    def signal(self, bars: list, idx: int) -> Signal:
        # Get symbol from bars if available
        symbol = self.current_symbol
        if not symbol and hasattr(bars[0], 'symbol'):
            symbol = bars[0].symbol

        if not symbol:
            # Default to balanced/ensemble approach
            return self._balanced_signal(bars, idx)

        profile = self._get_profile(symbol)

        if profile == "momentum":
            sig = self.momentum.signal(bars, idx)
            return Signal(
                sig.strength,
                f"[{symbol} momentum] {sig.reason}",
                sig.confidence
            )
        elif profile == "mean_reversion":
            sig = self.bollinger.signal(bars, idx)
            return Signal(
                sig.strength,
                f"[{symbol} reversion] {sig.reason}",
                sig.confidence
            )
        else:
            return self._balanced_signal(bars, idx)

    def _balanced_signal(self, bars: list, idx: int) -> Signal:
        """Ensemble approach for balanced stocks"""
        mom_sig = self.momentum.signal(bars, idx)
        boll_sig = self.bollinger.signal(bars, idx)
        ma_sig = self.ma_cross.signal(bars, idx)

        signals = [mom_sig, boll_sig, ma_sig]

        # Average
        avg_strength = sum(s.strength for s in signals) / len(signals)
        avg_conf = sum(s.confidence for s in signals) / len(signals)

        # Count agreement
        bullish = sum(1 for s in signals if s.strength > 0.2)
        bearish = sum(1 for s in signals if s.strength < -0.2)

        if bullish >= 2:
            return Signal(avg_strength, f"[balanced] {bullish}/3 bullish", avg_conf)
        elif bearish >= 2:
            return Signal(avg_strength, f"[balanced] {bearish}/3 bearish", avg_conf)
        else:
            return Signal(0, "[balanced] no consensus", avg_conf * 0.5)

    def warmup_period(self) -> int:
        return max(
            self.momentum.warmup_period(),
            self.bollinger.warmup_period(),
            self.ma_cross.warmup_period()
        )
