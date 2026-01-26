#!/usr/bin/env python3
"""
multi.py - Unified Multi-Strategy Trading CLI

Single interface for multi-strategy trading operations:
- scan: Show all strategy signals
- status: Per-strategy positions and P&L
- sized: Entry signals with position sizing
- config: View/edit strategy allocation
- execute: Execute a trade with strategy attribution

This is the main interface for multi-strategy trading.
"""

import sys
import json
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))


def cmd_scan(args):
    """Scan all strategies for signals"""
    from router import StrategyRouter

    router = StrategyRouter()
    print("Scanning strategies...", end=" ", flush=True)
    signals = router.scan()
    resolved = router.resolve_conflicts(signals)
    print(f"{len(signals)} raw, {len(resolved)} after conflict resolution")

    if not resolved:
        print("\nNo actionable signals")
        return

    entries = [s for s in resolved if s.strength > 0]
    exits = [s for s in resolved if s.strength < 0]

    if entries:
        print(f"\n{'='*60}")
        print("ENTRY SIGNALS")
        print(f"{'='*60}")
        print(f"{'Strategy':<12} {'Symbol':<8} {'Strength':>10} {'Reason':<30}")
        print("-" * 60)
        for s in entries:
            print(f"{s.strategy:<12} {s.symbol:<8} {s.strength:>+10.2f} {s.reason[:28]:<30}")

    if exits:
        print(f"\n{'='*60}")
        print("EXIT SIGNALS")
        print(f"{'='*60}")
        print(f"{'Strategy':<12} {'Symbol':<8} {'Strength':>10} {'Reason':<30}")
        print("-" * 60)
        for s in exits:
            print(f"{s.strategy:<12} {s.symbol:<8} {s.strength:>+10.2f} {s.reason[:28]:<30}")


def cmd_status(args):
    """Show per-strategy status"""
    from ledger import Ledger
    from trader import Trader
    from router import load_config

    ledger = Ledger()
    trader = Trader()
    config = load_config()

    account = trader.get_account()
    positions = trader.get_positions()

    print(f"{'='*60}")
    print("MULTI-STRATEGY STATUS")
    print(f"{'='*60}")
    print(f"\nPortfolio: ${account['portfolio_value']:,.2f} ({account['pl_today_pct']:+.2f}% today)")
    print(f"Cash: ${account['cash']:,.2f}")

    # Allocation config
    print(f"\n{'─'*60}")
    print("STRATEGY ALLOCATION")
    print(f"{'─'*60}")
    allocation = config.get("allocation", {})
    total_cap = config.get("total_capital", 100000)

    for strat, pct in allocation.items():
        strat_positions = ledger.get_positions_by_strategy(strat)
        used = sum(p.qty * p.avg_entry for p in strat_positions)
        alloc = total_cap * pct
        print(f"  {strat:15} {pct*100:>5.0f}% (${alloc:>8,.0f})  Used: ${used:>8,.0f}  Avail: ${alloc-used:>8,.0f}")

    # Positions by strategy
    print(f"\n{'─'*60}")
    print("POSITIONS BY STRATEGY")
    print(f"{'─'*60}")

    if not positions:
        print("  No open positions")
    else:
        for p in positions:
            strat = ledger.get_position_strategy(p["symbol"]) or "unknown"
            print(f"  [{strat:10}] {p['symbol']:6} {p['qty']:>6.0f} @ ${p['avg_entry']:>8.2f}  P&L: ${p['unrealized_pl']:>+8.2f} ({p['unrealized_pl_pct']:>+.1f}%)")

    # Per-strategy P&L
    print(f"\n{'─'*60}")
    print("REALIZED P&L BY STRATEGY")
    print(f"{'─'*60}")
    summary = ledger.summary()
    for strat, data in summary.get("by_strategy", {}).items():
        if data["closed_trades"] > 0:
            print(f"  {strat:15} ${data['realized_pnl']:>+10.2f}  Win: {data['win_rate']:>5.1f}%  Trades: {data['closed_trades']}")
        else:
            print(f"  {strat:15} No closed trades")

    print(f"{'='*60}")


def cmd_sized(args):
    """Show sized entry opportunities"""
    from router import StrategyRouter

    router = StrategyRouter()
    print("Getting sized entries...", end=" ", flush=True)
    sized = router.get_sized_entries()
    print(f"found {len(sized)}")

    if not sized:
        print("\nNo entry signals or no available capital")
        return

    print(f"\n{'='*60}")
    print("SIZED ENTRY OPPORTUNITIES")
    print(f"{'='*60}")
    print(f"{'Strategy':<12} {'Symbol':<8} {'Shares':>8} {'Price':>10} {'Notional':>12}")
    print("-" * 55)
    for entry in sized:
        sig = entry["signal"]
        sizing = entry["sizing"]
        print(f"{sig['strategy']:<12} {sig['symbol']:<8} {sizing['shares']:>8} ${entry['price']:>9.2f} ${sizing['notional']:>11,.0f}")


def cmd_config(args):
    """Show/edit configuration"""
    from router import load_config, save_config, CONFIG_FILE

    config = load_config()

    if args.set:
        # Parse key=value
        key, value = args.set.split("=", 1)
        if key in ["max_positions", "total_capital"]:
            config[key] = int(value)
        elif key == "risk_per_trade":
            config[key] = float(value)
        elif "." in key:
            # Nested, e.g., allocation.momentum=0.5
            parts = key.split(".")
            if parts[0] == "allocation":
                config["allocation"][parts[1]] = float(value)
        save_config(config)
        print(f"Set {key} = {value}")
        return

    print(f"{'='*60}")
    print(f"CONFIGURATION ({CONFIG_FILE})")
    print(f"{'='*60}")
    print(f"\nActive strategies: {', '.join(config.get('active_strategies', []))}")
    print(f"\nAllocation:")
    for strat, pct in config.get("allocation", {}).items():
        print(f"  {strat}: {pct*100:.0f}%")
    print(f"\nSymbols: {', '.join(config.get('symbols', []))}")
    print(f"Max positions: {config.get('max_positions', 4)}")
    print(f"Risk per trade: {config.get('risk_per_trade', 0.03)*100:.1f}%")
    print(f"Total capital: ${config.get('total_capital', 100000):,.0f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Strategy Trading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  multi.py scan              # Scan all strategies for signals
  multi.py status            # Per-strategy positions and P&L
  multi.py sized             # Entry signals with sizing
  multi.py config            # View configuration
  multi.py config --set allocation.momentum=0.5
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # scan
    scan_parser = subparsers.add_parser("scan", help="Scan all strategies")

    # status
    status_parser = subparsers.add_parser("status", help="Per-strategy status")

    # sized
    sized_parser = subparsers.add_parser("sized", help="Sized entry opportunities")

    # config
    config_parser = subparsers.add_parser("config", help="View/edit config")
    config_parser.add_argument("--set", help="Set config value (key=value)")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "sized":
        cmd_sized(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        # Default to status
        cmd_status(args)


if __name__ == "__main__":
    main()
