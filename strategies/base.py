#!/usr/bin/env python3
"""
base.py - base strategy interface

All strategies implement:
- signal(bars, idx) -> float: returns signal strength (-1 to +1)
- name: strategy name
- params: strategy parameters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Signal:
    """Trading signal"""
    strength: float  # -1 (strong sell) to +1 (strong buy)
    reason: str
    confidence: float = 0.5  # 0 to 1


class Strategy(ABC):
    """Base strategy interface"""

    name: str = "base"
    params: dict = {}

    @abstractmethod
    def signal(self, bars: list, idx: int) -> Signal:
        """
        Generate trading signal for a symbol at a specific bar index.

        Args:
            bars: List of bar objects for one symbol
            idx: Current bar index (0 = oldest)

        Returns:
            Signal with strength, reason, confidence
        """
        pass

    def warmup_period(self) -> int:
        """Minimum bars needed before generating signals"""
        return 10

    def __repr__(self):
        return f"{self.name}({self.params})"
