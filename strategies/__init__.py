"""
strategies - trading strategy zoo

Available strategies:

Momentum:
- SimpleMomentum: buy what's going up
- AcceleratingMomentum: buy accelerating stocks

Mean Reversion:
- BollingerReversion: trade Bollinger band touches
- RSIReversion: trade RSI extremes

Trend Following:
- MovingAverageCross: trade MA crossovers
- TrendStrength: trade strong trends (ADX-like)
"""

from .base import Strategy, Signal
from .momentum import SimpleMomentum, AcceleratingMomentum
from .mean_reversion import BollingerReversion, RSIReversion
from .trend import MovingAverageCross, TrendStrength
from .ensemble import VotingEnsemble, ConfirmationEnsemble

# All available strategies
ALL_STRATEGIES = [
    SimpleMomentum,
    AcceleratingMomentum,
    BollingerReversion,
    RSIReversion,
    MovingAverageCross,
    TrendStrength,
    VotingEnsemble,
    ConfirmationEnsemble,
]

__all__ = [
    "Strategy",
    "Signal",
    "SimpleMomentum",
    "AcceleratingMomentum",
    "BollingerReversion",
    "RSIReversion",
    "MovingAverageCross",
    "TrendStrength",
    "VotingEnsemble",
    "ConfirmationEnsemble",
    "ALL_STRATEGIES",
]
