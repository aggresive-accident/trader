#!/usr/bin/env python3
"""
ensemble.py - combine multiple strategies

Aggregate signals from multiple strategies for more robust decisions.
"""

from .base import Strategy, Signal
from .momentum import SimpleMomentum, AcceleratingMomentum
from .mean_reversion import BollingerReversion, RSIReversion
from .trend import MovingAverageCross, TrendStrength
from .breakout import DonchianBreakout, RangeBreakout
from .volatility import ATRBreakout, KeltnerChannel
from .volume import VolumeBreakout


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


class BestOfEnsemble(Strategy):
    """Ensemble of top performing strategies from backtest"""

    name = "best_of_ensemble"

    def __init__(self):
        # Top performers from zoo backtest:
        # 1. DonchianBreakout (+1.60%, Sharpe 2.95)
        # 2. MovingAverageCross (+1.45%, Sharpe 8.44)
        # 3. ATRBreakout (volatility breakout)
        # 4. VolumeBreakout (volume confirmation)
        self.strategies = [
            DonchianBreakout(period=20),
            MovingAverageCross(short=10, long=30),
            ATRBreakout(period=14, mult=2.0),
            VolumeBreakout(period=20, volume_mult=2.0),
        ]
        self.params = {"n_strategies": len(self.strategies)}

    def signal(self, bars: list, idx: int) -> Signal:
        signals = []
        for strat in self.strategies:
            if idx >= strat.warmup_period():
                try:
                    sig = strat.signal(bars, idx)
                    signals.append((strat.name, sig))
                except:
                    pass

        if not signals:
            return Signal(0, "no signals", 0)

        # Weight by confidence and give bonus to agreeing signals
        bullish = [(n, s) for n, s in signals if s.strength > 0.3]
        bearish = [(n, s) for n, s in signals if s.strength < -0.3]

        # Strong consensus: 3+ agree
        if len(bullish) >= 3:
            avg = sum(s.strength for _, s in bullish) / len(bullish)
            names = ", ".join(n for n, _ in bullish[:2])
            return Signal(min(avg * 1.2, 1.0), f"strong buy ({names}...)", 0.85)

        if len(bearish) >= 3:
            avg = sum(s.strength for _, s in bearish) / len(bearish)
            names = ", ".join(n for n, _ in bearish[:2])
            return Signal(max(avg * 1.2, -1.0), f"strong sell ({names}...)", 0.85)

        # Moderate consensus: 2 agree
        if len(bullish) >= 2:
            avg = sum(s.strength for _, s in bullish) / len(bullish)
            return Signal(avg, f"buy ({len(bullish)}/4 agree)", 0.7)

        if len(bearish) >= 2:
            avg = sum(s.strength for _, s in bearish) / len(bearish)
            return Signal(avg, f"sell ({len(bearish)}/4 agree)", 0.7)

        # Single signal - weak
        if bullish:
            return Signal(bullish[0][1].strength * 0.5, f"weak buy ({bullish[0][0]})", 0.4)
        if bearish:
            return Signal(bearish[0][1].strength * 0.5, f"weak sell ({bearish[0][0]})", 0.4)

        return Signal(0, "no consensus", 0.3)

    def warmup_period(self) -> int:
        return max(s.warmup_period() for s in self.strategies)
