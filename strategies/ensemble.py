#!/usr/bin/env python3
"""
ensemble.py - combine multiple strategies

Aggregate signals from multiple strategies for more robust decisions.
"""

from .base import Strategy, Signal
from .momentum import SimpleMomentum, AcceleratingMomentum
from .mean_reversion import BollingerReversion, RSIReversion
from .trend import MovingAverageCross, TrendStrength


class VotingEnsemble(Strategy):
    """Combine multiple strategies by voting"""

    name = "voting_ensemble"

    def __init__(self, strategies: list = None):
        if strategies is None:
            # Default: use all strategies
            self.strategies = [
                SimpleMomentum(),
                AcceleratingMomentum(),
                BollingerReversion(),
                RSIReversion(),
                MovingAverageCross(),
                TrendStrength(),
            ]
        else:
            self.strategies = strategies

        self.params = {"n_strategies": len(self.strategies)}

    def signal(self, bars: list, idx: int) -> Signal:
        # Collect signals from all strategies
        signals = []
        for strat in self.strategies:
            if idx >= strat.warmup_period():
                try:
                    sig = strat.signal(bars, idx)
                    signals.append(sig)
                except:
                    pass

        if not signals:
            return Signal(0, "no signals", 0)

        # Weighted average of signals (weight by confidence)
        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            return Signal(0, "no confidence", 0)

        weighted_strength = sum(s.strength * s.confidence for s in signals) / total_weight
        avg_confidence = total_weight / len(signals)

        # Count votes
        bullish = sum(1 for s in signals if s.strength > 0.2)
        bearish = sum(1 for s in signals if s.strength < -0.2)

        if weighted_strength > 0.2:
            return Signal(
                weighted_strength,
                f"ensemble bullish ({bullish}/{len(signals)} vote)",
                avg_confidence
            )
        elif weighted_strength < -0.2:
            return Signal(
                weighted_strength,
                f"ensemble bearish ({bearish}/{len(signals)} vote)",
                avg_confidence
            )
        else:
            return Signal(0, f"ensemble neutral", avg_confidence * 0.5)

    def warmup_period(self) -> int:
        return max(s.warmup_period() for s in self.strategies)


class ConfirmationEnsemble(Strategy):
    """Only trade when multiple strategies agree"""

    name = "confirmation_ensemble"

    def __init__(self, min_agreement: int = 3):
        self.strategies = [
            SimpleMomentum(),
            AcceleratingMomentum(),
            BollingerReversion(),
            MovingAverageCross(),
            TrendStrength(),
        ]
        self.min_agreement = min_agreement
        self.params = {"min_agreement": min_agreement, "n_strategies": len(self.strategies)}

    def signal(self, bars: list, idx: int) -> Signal:
        signals = []
        for strat in self.strategies:
            if idx >= strat.warmup_period():
                try:
                    sig = strat.signal(bars, idx)
                    signals.append(sig)
                except:
                    pass

        if len(signals) < self.min_agreement:
            return Signal(0, "insufficient signals", 0)

        # Count strong agreement
        bullish = sum(1 for s in signals if s.strength > 0.3)
        bearish = sum(1 for s in signals if s.strength < -0.3)

        if bullish >= self.min_agreement:
            avg_strength = sum(s.strength for s in signals if s.strength > 0.3) / bullish
            return Signal(avg_strength, f"confirmed buy ({bullish} agree)", 0.8)
        elif bearish >= self.min_agreement:
            avg_strength = sum(s.strength for s in signals if s.strength < -0.3) / bearish
            return Signal(avg_strength, f"confirmed sell ({bearish} agree)", 0.8)
        else:
            return Signal(0, f"no consensus ({bullish}B/{bearish}S)", 0.3)

    def warmup_period(self) -> int:
        return max(s.warmup_period() for s in self.strategies)
