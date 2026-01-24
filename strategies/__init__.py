"""
strategies - trading strategy zoo

20+ strategies across multiple categories:

Momentum:
- SimpleMomentum: buy what's going up
- AcceleratingMomentum: buy accelerating stocks

Mean Reversion:
- BollingerReversion: trade Bollinger band touches
- RSIReversion: trade RSI extremes

Trend Following:
- MovingAverageCross: trade MA crossovers
- TrendStrength: trade strong trends (ADX-like)

Volume:
- VolumeBreakout: trade volume spikes
- OnBalanceVolume: trade OBV divergences
- VWAP: trade relative to VWAP

Volatility:
- ATRBreakout: trade ATR breakouts
- VolatilityContraction: trade after squeezes
- KeltnerChannel: trade Keltner breakouts

Breakout:
- DonchianBreakout: turtle trading
- RangeBreakout: trade consolidation breakouts
- GapStrategy: trade gaps
- HighLowBreakout: trade N-day highs/lows

Patterns:
- CandlestickPatterns: hammer, engulfing, etc.
- ThreeBarPatterns: morning star, three soldiers, etc.
- SwingHighLow: trade swing reversals

Ensemble:
- VotingEnsemble: weighted voting
- ConfirmationEnsemble: require agreement
"""

from .base import Strategy, Signal
from .momentum import SimpleMomentum, AcceleratingMomentum
from .mean_reversion import BollingerReversion, RSIReversion
from .trend import MovingAverageCross, TrendStrength
from .volume import VolumeBreakout, OnBalanceVolume, VWAP
from .volatility import ATRBreakout, VolatilityContraction, KeltnerChannel
from .breakout import DonchianBreakout, RangeBreakout, GapStrategy, HighLowBreakout
from .patterns import CandlestickPatterns, ThreeBarPatterns, SwingHighLow
from .ensemble import VotingEnsemble, ConfirmationEnsemble

# All available strategies
ALL_STRATEGIES = [
    # Momentum
    SimpleMomentum,
    AcceleratingMomentum,
    # Mean Reversion
    BollingerReversion,
    RSIReversion,
    # Trend
    MovingAverageCross,
    TrendStrength,
    # Volume
    VolumeBreakout,
    OnBalanceVolume,
    VWAP,
    # Volatility
    ATRBreakout,
    VolatilityContraction,
    KeltnerChannel,
    # Breakout
    DonchianBreakout,
    RangeBreakout,
    GapStrategy,
    HighLowBreakout,
    # Patterns
    CandlestickPatterns,
    ThreeBarPatterns,
    SwingHighLow,
    # Ensemble
    VotingEnsemble,
    ConfirmationEnsemble,
]

# Strategy categories
CATEGORIES = {
    "momentum": [SimpleMomentum, AcceleratingMomentum],
    "mean_reversion": [BollingerReversion, RSIReversion],
    "trend": [MovingAverageCross, TrendStrength],
    "volume": [VolumeBreakout, OnBalanceVolume, VWAP],
    "volatility": [ATRBreakout, VolatilityContraction, KeltnerChannel],
    "breakout": [DonchianBreakout, RangeBreakout, GapStrategy, HighLowBreakout],
    "patterns": [CandlestickPatterns, ThreeBarPatterns, SwingHighLow],
    "ensemble": [VotingEnsemble, ConfirmationEnsemble],
}

__all__ = [
    "Strategy",
    "Signal",
    "ALL_STRATEGIES",
    "CATEGORIES",
    # Momentum
    "SimpleMomentum",
    "AcceleratingMomentum",
    # Mean Reversion
    "BollingerReversion",
    "RSIReversion",
    # Trend
    "MovingAverageCross",
    "TrendStrength",
    # Volume
    "VolumeBreakout",
    "OnBalanceVolume",
    "VWAP",
    # Volatility
    "ATRBreakout",
    "VolatilityContraction",
    "KeltnerChannel",
    # Breakout
    "DonchianBreakout",
    "RangeBreakout",
    "GapStrategy",
    "HighLowBreakout",
    # Patterns
    "CandlestickPatterns",
    "ThreeBarPatterns",
    "SwingHighLow",
    # Ensemble
    "VotingEnsemble",
    "ConfirmationEnsemble",
]
