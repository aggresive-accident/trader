#!/usr/bin/env python3
"""
health.py - Strategy health monitoring

Compares live trading performance against backtest expectations.
Emits signals to ~/.organism/signals/trader_health.json for organism consumption.

Usage:
  python3 health.py              # Run health check, emit signals
  python3 health.py --json       # Output raw health data
  python3 health.py --dry-run    # Show what would be emitted without writing

Thresholds (from R036 spec):
  - warning: 15% deviation from expected
  - problem: 25% deviation from expected
  - min_trades: 20 (per strategy before alerting)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

# Output location for organism consumption
SIGNALS_DIR = Path.home() / ".organism" / "signals"
SIGNALS_FILE = SIGNALS_DIR / "trader_health.json"

# Expected values from backtests (source: analytics.py)
# These should be updated if strategies are revised
EXPECTED = {
    "overall": {
        "win_rate": 0.47,
        "profit_factor": 2.92,
    },
    # Per-strategy expectations can be added here
    # "momentum": {"win_rate": 0.52, "profit_factor": 2.1},
    # "bollinger": {"win_rate": 0.45, "profit_factor": 3.0},
}

# Thresholds from R036 spec
WARNING_THRESHOLD = 0.15  # 15% deviation
PROBLEM_THRESHOLD = 0.25  # 25% deviation
MIN_TRADES = 20  # Minimum sample size


def get_strategy_stats() -> dict:
    """
    Get per-strategy performance statistics from the ledger.

    Returns dict of {strategy: {trades, wins, win_rate, profit_factor, ...}}
    """
    from ledger import Ledger

    ledger = Ledger()

    # Group closed trades by strategy
    by_strategy = {}

    for trade in ledger.trades:
        if trade.pnl is None:  # Not closed
            continue

        strat = trade.strategy
        if strat not in by_strategy:
            by_strategy[strat] = {
                "trades": 0,
                "wins": 0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            }

        by_strategy[strat]["trades"] += 1

        if trade.pnl > 0:
            by_strategy[strat]["wins"] += 1
            by_strategy[strat]["gross_profit"] += trade.pnl
        else:
            by_strategy[strat]["gross_loss"] += abs(trade.pnl)

    # Calculate derived metrics
    for strat, data in by_strategy.items():
        if data["trades"] > 0:
            data["win_rate"] = data["wins"] / data["trades"]
        else:
            data["win_rate"] = 0.0

        if data["gross_loss"] > 0:
            data["profit_factor"] = data["gross_profit"] / data["gross_loss"]
        else:
            data["profit_factor"] = float("inf") if data["gross_profit"] > 0 else 0.0

    return by_strategy


def get_overall_stats() -> dict:
    """Get overall (all strategies combined) statistics."""
    by_strategy = get_strategy_stats()

    if not by_strategy:
        return None

    total_trades = sum(s["trades"] for s in by_strategy.values())
    total_wins = sum(s["wins"] for s in by_strategy.values())
    total_profit = sum(s["gross_profit"] for s in by_strategy.values())
    total_loss = sum(s["gross_loss"] for s in by_strategy.values())

    return {
        "trades": total_trades,
        "wins": total_wins,
        "win_rate": total_wins / total_trades if total_trades > 0 else 0.0,
        "gross_profit": total_profit,
        "gross_loss": total_loss,
        "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
    }


def check_deviation(actual: float, expected: float, metric_name: str) -> dict | None:
    """
    Check if actual deviates from expected beyond thresholds.

    Returns signal dict if deviation detected, None otherwise.
    """
    if expected == 0:
        return None

    # Handle infinity
    if actual == float("inf") or expected == float("inf"):
        return None

    deviation = (expected - actual) / expected  # Positive = underperforming

    if deviation >= PROBLEM_THRESHOLD:
        return {
            "level": "problem",
            "message": f"{metric_name}: {actual:.1%} vs {expected:.1%} expected ({deviation:+.0%} deviation)",
        }
    elif deviation >= WARNING_THRESHOLD:
        return {
            "level": "warning",
            "message": f"{metric_name}: {actual:.1%} vs {expected:.1%} expected ({deviation:+.0%} deviation)",
        }

    return None


def check_profit_factor_deviation(actual: float, expected: float, context: str) -> dict | None:
    """Check profit factor deviation."""
    if expected == 0 or actual == float("inf") or expected == float("inf"):
        return None

    deviation = (expected - actual) / expected

    if deviation >= PROBLEM_THRESHOLD:
        return {
            "level": "problem",
            "message": f"{context} profit factor: {actual:.2f} vs {expected:.2f} expected ({deviation:+.0%})",
        }
    elif deviation >= WARNING_THRESHOLD:
        return {
            "level": "warning",
            "message": f"{context} profit factor: {actual:.2f} vs {expected:.2f} expected ({deviation:+.0%})",
        }

    return None


def run_health_check() -> dict:
    """
    Run full health check and return results.

    Returns:
        {
            "timestamp": "...",
            "overall": {...},
            "by_strategy": {...},
            "signals": [...],
            "status": "healthy|warning|problem"
        }
    """
    signals = []

    # Get stats
    overall = get_overall_stats()
    by_strategy = get_strategy_stats()

    # Check overall metrics
    if overall and overall["trades"] >= MIN_TRADES:
        expected_overall = EXPECTED.get("overall", {})

        # Win rate
        if "win_rate" in expected_overall:
            sig = check_deviation(
                overall["win_rate"],
                expected_overall["win_rate"],
                "Overall win rate"
            )
            if sig:
                signals.append(sig)

        # Profit factor
        if "profit_factor" in expected_overall:
            sig = check_profit_factor_deviation(
                overall["profit_factor"],
                expected_overall["profit_factor"],
                "Overall"
            )
            if sig:
                signals.append(sig)

    # Check per-strategy metrics
    for strat, stats in by_strategy.items():
        if stats["trades"] < MIN_TRADES:
            continue

        expected_strat = EXPECTED.get(strat, {})
        if not expected_strat:
            continue

        if "win_rate" in expected_strat:
            sig = check_deviation(
                stats["win_rate"],
                expected_strat["win_rate"],
                f"{strat} win rate"
            )
            if sig:
                signals.append(sig)

        if "profit_factor" in expected_strat:
            sig = check_profit_factor_deviation(
                stats["profit_factor"],
                expected_strat["profit_factor"],
                strat
            )
            if sig:
                signals.append(sig)

    # Determine overall status
    has_problems = any(s["level"] == "problem" for s in signals)
    has_warnings = any(s["level"] == "warning" for s in signals)

    if has_problems:
        status = "problem"
    elif has_warnings:
        status = "warning"
    else:
        status = "healthy"

    return {
        "timestamp": datetime.now().isoformat(),
        "overall": overall,
        "by_strategy": by_strategy,
        "signals": signals,
        "status": status,
        "min_trades_threshold": MIN_TRADES,
    }


def emit_signals(health_data: dict) -> Path:
    """Write signals to organism signals directory."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "source": "trader_health",
        "timestamp": health_data["timestamp"],
        "status": health_data["status"],
        "signals": health_data["signals"],
        "meta": {
            "total_trades": health_data["overall"]["trades"] if health_data["overall"] else 0,
            "strategies_checked": list(health_data["by_strategy"].keys()),
            "min_trades_threshold": MIN_TRADES,
        }
    }

    SIGNALS_FILE.write_text(json.dumps(output, indent=2))
    return SIGNALS_FILE


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Strategy health check")
    parser.add_argument("--json", action="store_true", help="Output raw health data as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show signals without writing")
    args = parser.parse_args()

    health = run_health_check()

    if args.json:
        # Clean up infinity for JSON serialization
        def clean_inf(obj):
            if isinstance(obj, float) and obj == float("inf"):
                return "inf"
            elif isinstance(obj, dict):
                return {k: clean_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_inf(v) for v in obj]
            return obj

        print(json.dumps(clean_inf(health), indent=2))
        return

    # Display summary
    print("=" * 50)
    print("STRATEGY HEALTH CHECK")
    print("=" * 50)

    if health["overall"]:
        o = health["overall"]
        print(f"\nOverall ({o['trades']} trades):")
        print(f"  Win rate: {o['win_rate']:.1%}")
        pf = o['profit_factor']
        print(f"  Profit factor: {pf:.2f}" if pf != float("inf") else "  Profit factor: ∞")
    else:
        print("\nNo closed trades yet.")

    if health["by_strategy"]:
        print("\nBy Strategy:")
        for strat, stats in health["by_strategy"].items():
            pf = stats['profit_factor']
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
            print(f"  {strat}: {stats['trades']} trades, {stats['win_rate']:.0%} win, PF {pf_str}")

    print(f"\nStatus: {health['status'].upper()}")

    if health["signals"]:
        print("\nSignals:")
        for sig in health["signals"]:
            prefix = "!" if sig["level"] == "problem" else "⚠"
            print(f"  {prefix} {sig['message']}")
    else:
        if health["overall"] and health["overall"]["trades"] < MIN_TRADES:
            print(f"\n(Need {MIN_TRADES} trades minimum for alerts, have {health['overall']['trades']})")
        else:
            print("\nNo deviations detected.")

    # Emit signals unless dry-run
    if not args.dry_run:
        path = emit_signals(health)
        print(f"\nSignals written to: {path}")
    else:
        print("\n[dry-run] Would write signals to:", SIGNALS_FILE)

    print("=" * 50)


if __name__ == "__main__":
    main()
