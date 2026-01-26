#!/usr/bin/env python3
"""
router.py - Strategy Router

Scans all active strategies and returns unified signal list with attribution.
Each signal is tagged with its source strategy for later analytics.

This is the core of multi-strategy trading.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from config import load_keys
from strategies import Strategy, Signal, CATEGORIES


# Config file for strategy allocation
CONFIG_FILE = Path(__file__).parent / "router_config.json"

# Default config
DEFAULT_CONFIG = {
    "active_strategies": ["momentum", "bollinger", "adaptive"],
    "allocation": {
        "momentum": 0.40,
        "bollinger": 0.35,
        "adaptive": 0.25,
    },
    "symbols": ["META", "NVDA", "AMD", "GOOGL", "AAPL", "MSFT", "AMZN", "XOM", "XLE", "TSLA"],
    "max_positions": 4,
    "risk_per_trade": 0.03,
}


@dataclass
class StrategySignal:
    """A signal attributed to a specific strategy"""
    strategy: str
    symbol: str
    strength: float
    confidence: float
    reason: str
    timestamp: str
    allocation_pct: float

    def to_dict(self):
        return asdict(self)


def load_config() -> dict:
    """Load router configuration"""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return DEFAULT_CONFIG


def save_config(config: dict):
    """Save router configuration"""
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_strategy_class(name: str):
    """Get strategy class by name"""
    from strategies import (
        SimpleMomentum, BollingerReversion, AdaptiveStrategy,
        DonchianBreakout, RSIReversion, RegimeStrategy
    )

    name_map = {
        "momentum": SimpleMomentum,
        "bollinger": BollingerReversion,
        "adaptive": AdaptiveStrategy,
        "donchian": DonchianBreakout,
        "rsi": RSIReversion,
        "regime": RegimeStrategy,
    }

    return name_map.get(name.lower())


class StrategyRouter:
    """
    Routes signals from multiple strategies.

    Each signal is attributed to its source strategy for:
    - Per-strategy P&L tracking
    - Conflict resolution
    - Capital allocation
    """

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.strategies = {}
        self._load_strategies()

        api_key, secret_key = load_keys()
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    def _load_strategies(self):
        """Instantiate active strategies"""
        for name in self.config["active_strategies"]:
            strategy_class = get_strategy_class(name)
            if strategy_class:
                self.strategies[name] = strategy_class()

    def fetch_bars(self, symbols: list, days: int = 30) -> dict:
        """Fetch historical bars for all symbols"""
        end = datetime.now()
        start = end - timedelta(days=days + 10)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )

        return self.data_client.get_stock_bars(request)

    def scan(self, symbols: list = None) -> list[StrategySignal]:
        """
        Scan all active strategies for all symbols.

        Returns list of StrategySignal objects sorted by strength.
        """
        symbols = symbols or self.config["symbols"]
        bars = self.fetch_bars(symbols)

        signals = []
        now = datetime.now().isoformat()
        allocation = self.config.get("allocation", {})

        for strat_name, strategy in self.strategies.items():
            alloc_pct = allocation.get(strat_name, 0.33)

            for symbol in symbols:
                try:
                    symbol_bars = bars[symbol]
                    if len(symbol_bars) < strategy.warmup_period():
                        continue

                    sig = strategy.signal(symbol_bars, len(symbol_bars) - 1)

                    # Only include signals above threshold
                    if abs(sig.strength) >= 0.3:
                        signals.append(StrategySignal(
                            strategy=strat_name,
                            symbol=symbol,
                            strength=sig.strength,
                            confidence=sig.confidence,
                            reason=sig.reason,
                            timestamp=now,
                            allocation_pct=alloc_pct,
                        ))
                except Exception as e:
                    continue

        # Sort by absolute strength descending
        signals.sort(key=lambda s: abs(s.strength), reverse=True)
        return signals

    def resolve_conflicts(self, signals: list[StrategySignal]) -> list[StrategySignal]:
        """
        Resolve conflicts when multiple strategies signal same symbol.

        Rules:
        1. Same direction: Keep strongest signal
        2. Opposite direction: No trade (conflict)
        """
        symbol_signals = {}

        for sig in signals:
            if sig.symbol not in symbol_signals:
                symbol_signals[sig.symbol] = []
            symbol_signals[sig.symbol].append(sig)

        resolved = []
        for symbol, sigs in symbol_signals.items():
            if len(sigs) == 1:
                resolved.append(sigs[0])
                continue

            # Check for direction conflict
            directions = set(1 if s.strength > 0 else -1 for s in sigs)
            if len(directions) > 1:
                # Conflict - opposite directions
                continue

            # Same direction - use strongest
            strongest = max(sigs, key=lambda s: abs(s.strength))
            resolved.append(strongest)

        return resolved

    def get_entry_signals(self, signals: list[StrategySignal] = None) -> list[StrategySignal]:
        """Get buy signals (positive strength)"""
        if signals is None:
            signals = self.scan()

        resolved = self.resolve_conflicts(signals)
        return [s for s in resolved if s.strength > 0]

    def get_exit_signals(self, signals: list[StrategySignal] = None) -> list[StrategySignal]:
        """Get sell signals (negative strength)"""
        if signals is None:
            signals = self.scan()

        resolved = self.resolve_conflicts(signals)
        return [s for s in resolved if s.strength < 0]

    def calculate_position_size(self, signal: StrategySignal, price: float) -> dict:
        """
        Calculate position size based on strategy allocation.

        Returns:
            {
                "shares": int,
                "notional": float,
                "strategy_allocation": float,
                "strategy_capital": float,
            }
        """
        from trader import Trader
        from ledger import Ledger

        trader = Trader()
        account = trader.get_account()
        ledger = Ledger()

        total_capital = self.config.get("total_capital", account["portfolio_value"])
        allocation = self.config.get("allocation", {})
        risk_per_trade = self.config.get("risk_per_trade", 0.03)

        # Strategy's allocated capital
        strat_alloc = allocation.get(signal.strategy, 0.33)
        strategy_capital = total_capital * strat_alloc

        # Capital already used by this strategy
        strategy_positions = ledger.get_positions_by_strategy(signal.strategy)
        used_capital = sum(
            p.qty * p.avg_entry for p in strategy_positions
        )

        # Available capital for this strategy
        available = max(0, strategy_capital - used_capital)

        # Position size (risk-based)
        max_position = available * risk_per_trade / risk_per_trade  # Full available
        max_position = min(max_position, available)

        shares = int(max_position / price) if price > 0 else 0

        return {
            "shares": shares,
            "notional": shares * price,
            "strategy_allocation": strat_alloc,
            "strategy_capital": strategy_capital,
            "available_capital": available,
            "used_capital": used_capital,
        }

    def get_sized_entries(self) -> list[dict]:
        """
        Get entry signals with position sizes calculated.

        Returns list of entry opportunities with sizing info.
        """
        from trader import Trader

        signals = self.get_entry_signals()
        trader = Trader()

        sized = []
        for sig in signals:
            try:
                quote = trader.get_quote(sig.symbol)
                price = quote["ask"]  # Use ask for buys

                sizing = self.calculate_position_size(sig, price)

                if sizing["shares"] > 0:
                    sized.append({
                        "signal": sig.to_dict(),
                        "price": price,
                        "sizing": sizing,
                    })
            except Exception:
                continue

        return sized


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Router")
    parser.add_argument("command", choices=["scan", "config", "entries", "exits", "sized"],
                       default="scan", nargs="?")
    parser.add_argument("-s", "--symbols", help="Comma-separated symbols")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.command == "config":
        config = load_config()
        if args.json:
            print(json.dumps(config, indent=2))
        else:
            print("Active strategies:", ", ".join(config["active_strategies"]))
            print("Allocation:")
            for name, pct in config.get("allocation", {}).items():
                print(f"  {name}: {pct*100:.0f}%")
            print("Symbols:", ", ".join(config["symbols"]))
        return

    router = StrategyRouter()
    symbols = args.symbols.split(",") if args.symbols else None

    if args.command == "sized":
        print("Getting sized entries...")
        sized = router.get_sized_entries()

        if args.json:
            print(json.dumps(sized, indent=2))
        else:
            if not sized:
                print("No entry signals or no available capital")
                return

            print()
            print(f"{'Strategy':<12} {'Symbol':<8} {'Shares':>8} {'Price':>10} {'Notional':>12} {'Avail Cap':>12}")
            print("-" * 70)
            for entry in sized:
                sig = entry["signal"]
                sizing = entry["sizing"]
                print(f"{sig['strategy']:<12} {sig['symbol']:<8} {sizing['shares']:>8} ${entry['price']:>9.2f} ${sizing['notional']:>11,.0f} ${sizing['available_capital']:>11,.0f}")
        return

    print("Scanning strategies...", end=" ", flush=True)
    signals = router.scan(symbols)
    print(f"found {len(signals)} signals")

    if args.command == "entries":
        signals = router.get_entry_signals(signals)
    elif args.command == "exits":
        signals = router.get_exit_signals(signals)
    else:
        signals = router.resolve_conflicts(signals)

    if args.json:
        print(json.dumps([s.to_dict() for s in signals], indent=2))
    else:
        if not signals:
            print("No signals")
            return

        print()
        print(f"{'Strategy':<12} {'Symbol':<8} {'Strength':>10} {'Reason':<40}")
        print("-" * 75)
        for sig in signals:
            direction = "BUY" if sig.strength > 0 else "SELL"
            print(f"{sig.strategy:<12} {sig.symbol:<8} {direction} {sig.strength:>+.2f} {sig.reason[:38]:<40}")


if __name__ == "__main__":
    main()
