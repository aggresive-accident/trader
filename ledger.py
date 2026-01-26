#!/usr/bin/env python3
"""
ledger.py - Position Ledger with Strategy Attribution

Tracks all trades with their source strategy for per-strategy P&L analytics.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

LEDGER_FILE = Path(__file__).parent / "trades_ledger.json"


@dataclass
class Trade:
    """A single trade with strategy attribution"""
    id: str
    timestamp: str
    symbol: str
    action: str  # BUY or SELL
    qty: float
    price: float
    strategy: str
    reason: str = ""
    order_id: str = ""

    # Calculated on close
    closed_at: Optional[str] = None
    close_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        return cls(**d)


@dataclass
class Position:
    """An open position"""
    symbol: str
    qty: float
    avg_entry: float
    strategy: str
    opened_at: str
    trade_ids: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


class Ledger:
    """
    Position ledger with strategy tracking.

    Features:
    - Records all trades with strategy attribution
    - Tracks open positions by strategy
    - Calculates per-strategy P&L
    - Prevents duplicate positions in same symbol
    """

    def __init__(self, path: Path = None):
        self.path = path or LEDGER_FILE
        self.trades: list[Trade] = []
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self._load()

    def _load(self):
        """Load ledger from disk"""
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.trades = [Trade.from_dict(t) for t in data.get("trades", [])]
            self.positions = {
                k: Position.from_dict(v)
                for k, v in data.get("positions", {}).items()
            }

    def _save(self):
        """Save ledger to disk"""
        data = {
            "trades": [t.to_dict() for t in self.trades],
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "updated_at": datetime.now().isoformat(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def _generate_id(self) -> str:
        """Generate unique trade ID"""
        return f"T{len(self.trades)+1:04d}"

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in symbol"""
        return symbol in self.positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol if exists"""
        return self.positions.get(symbol)

    def get_position_strategy(self, symbol: str) -> Optional[str]:
        """Get which strategy owns the position"""
        pos = self.positions.get(symbol)
        return pos.strategy if pos else None

    def record_buy(self, symbol: str, qty: float, price: float,
                   strategy: str, reason: str = "", order_id: str = "") -> Trade:
        """
        Record a buy trade.

        If position exists, adds to it (must be same strategy).
        If new position, creates it.
        """
        symbol = symbol.upper()

        # Check for conflict
        if symbol in self.positions:
            existing = self.positions[symbol]
            if existing.strategy != strategy:
                raise ValueError(
                    f"Position in {symbol} owned by {existing.strategy}, "
                    f"cannot buy from {strategy}"
                )
            # Add to existing position
            total_cost = existing.avg_entry * existing.qty + price * qty
            new_qty = existing.qty + qty
            existing.avg_entry = total_cost / new_qty
            existing.qty = new_qty
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                avg_entry=price,
                strategy=strategy,
                opened_at=datetime.now().isoformat(),
                trade_ids=[],
            )

        trade = Trade(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action="BUY",
            qty=qty,
            price=price,
            strategy=strategy,
            reason=reason,
            order_id=order_id,
        )

        self.trades.append(trade)
        self.positions[symbol].trade_ids.append(trade.id)
        self._save()
        return trade

    def record_sell(self, symbol: str, qty: float, price: float,
                    reason: str = "", order_id: str = "") -> Trade:
        """
        Record a sell trade.

        Strategy is inferred from the open position.
        """
        symbol = symbol.upper()

        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol} to sell")

        pos = self.positions[symbol]
        if qty > pos.qty:
            raise ValueError(f"Cannot sell {qty}, only have {pos.qty}")

        # Calculate P&L
        pnl = (price - pos.avg_entry) * qty
        pnl_pct = ((price / pos.avg_entry) - 1) * 100

        trade = Trade(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action="SELL",
            qty=qty,
            price=price,
            strategy=pos.strategy,
            reason=reason,
            order_id=order_id,
            closed_at=datetime.now().isoformat(),
            close_price=price,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )

        self.trades.append(trade)
        pos.trade_ids.append(trade.id)

        # Update or close position
        pos.qty -= qty
        if pos.qty <= 0:
            del self.positions[symbol]

        self._save()
        return trade

    def get_trades_by_strategy(self, strategy: str) -> list[Trade]:
        """Get all trades for a strategy"""
        return [t for t in self.trades if t.strategy == strategy]

    def get_positions_by_strategy(self, strategy: str) -> list[Position]:
        """Get open positions for a strategy"""
        return [p for p in self.positions.values() if p.strategy == strategy]

    def calculate_strategy_pnl(self, strategy: str) -> dict:
        """Calculate P&L for a strategy"""
        trades = self.get_trades_by_strategy(strategy)
        sells = [t for t in trades if t.action == "SELL" and t.pnl is not None]

        realized_pnl = sum(t.pnl for t in sells)
        win_trades = [t for t in sells if t.pnl > 0]

        return {
            "strategy": strategy,
            "total_trades": len(trades),
            "closed_trades": len(sells),
            "realized_pnl": realized_pnl,
            "win_rate": len(win_trades) / len(sells) * 100 if sells else 0,
            "avg_pnl": realized_pnl / len(sells) if sells else 0,
        }

    def summary(self) -> dict:
        """Get ledger summary"""
        strategies = set(t.strategy for t in self.trades)
        return {
            "total_trades": len(self.trades),
            "open_positions": len(self.positions),
            "strategies": list(strategies),
            "positions": [p.to_dict() for p in self.positions.values()],
            "by_strategy": {s: self.calculate_strategy_pnl(s) for s in strategies},
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Position Ledger")
    parser.add_argument("command", choices=["status", "positions", "trades", "pnl", "import"],
                       default="status", nargs="?")
    parser.add_argument("-s", "--strategy", help="Filter by strategy")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    ledger = Ledger()

    if args.command == "status":
        summary = ledger.summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Trades: {summary['total_trades']}")
            print(f"Open positions: {summary['open_positions']}")
            print(f"Strategies: {', '.join(summary['strategies']) or 'none'}")
            if summary['positions']:
                print("\nPositions:")
                for p in summary['positions']:
                    print(f"  {p['symbol']}: {p['qty']} @ ${p['avg_entry']:.2f} ({p['strategy']})")

    elif args.command == "positions":
        positions = ledger.positions.values()
        if args.strategy:
            positions = [p for p in positions if p.strategy == args.strategy]

        if args.json:
            print(json.dumps([p.to_dict() for p in positions], indent=2))
        else:
            for p in positions:
                print(f"{p.symbol}: {p.qty} @ ${p.avg_entry:.2f} ({p.strategy})")

    elif args.command == "trades":
        trades = ledger.trades
        if args.strategy:
            trades = [t for t in trades if t.strategy == args.strategy]

        if args.json:
            print(json.dumps([t.to_dict() for t in trades[-20:]], indent=2))
        else:
            for t in trades[-20:]:
                pnl_str = f" P&L: ${t.pnl:+.2f}" if t.pnl else ""
                print(f"{t.timestamp[:10]} {t.action} {t.symbol} {t.qty} @ ${t.price:.2f} ({t.strategy}){pnl_str}")

    elif args.command == "pnl":
        if args.strategy:
            pnl = ledger.calculate_strategy_pnl(args.strategy)
            print(json.dumps(pnl, indent=2) if args.json else pnl)
        else:
            summary = ledger.summary()
            if args.json:
                print(json.dumps(summary["by_strategy"], indent=2))
            else:
                for strat, data in summary["by_strategy"].items():
                    print(f"{strat}: ${data['realized_pnl']:+.2f} ({data['win_rate']:.0f}% win rate)")

    elif args.command == "import":
        # Import existing positions from Alpaca
        print("Importing from Alpaca...")
        from trader import Trader
        t = Trader()
        positions = t.get_positions()

        for p in positions:
            if not ledger.has_position(p["symbol"]):
                # Default to momentum for existing positions
                ledger.record_buy(
                    symbol=p["symbol"],
                    qty=p["qty"],
                    price=p["avg_entry"],
                    strategy="momentum",
                    reason="imported from Alpaca",
                )
                print(f"  Imported {p['symbol']}: {p['qty']} @ ${p['avg_entry']:.2f}")

        print("Done")


if __name__ == "__main__":
    main()
