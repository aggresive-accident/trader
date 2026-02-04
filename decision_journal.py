#!/usr/bin/env python3
"""
decision_journal.py - Trading decision logging for traceability.

Logs trading decisions (not just trades) so we can answer "why did we buy X?"

Storage: trader/decisions/{YYYY-MM-DD}.jsonl (append-only JSON Lines)

Usage:
  from decision_journal import DecisionJournal, Decision

  journal = DecisionJournal()
  journal.log(Decision(
      strategy="xs",
      action="rebalance",
      context={...},
      outcome={...},
      reasoning="...",
  ))

CLI:
  python3 decision_journal.py list [--date 2026-02-04]
  python3 decision_journal.py show <date> <index>
  python3 decision_journal.py search <symbol>
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

BASE_DIR = Path(__file__).parent
DECISIONS_DIR = BASE_DIR / "decisions"


@dataclass
class Decision:
    """A trading decision record."""

    # When and what
    strategy: str           # xs | zoo | thesis
    action: str             # rebalance | entry | exit | hold
    symbol: Optional[str]   # None for portfolio-level decisions (e.g., rebalance)

    # What was computed
    context: dict           # Strategy-specific (rankings, signals, scores)

    # What was decided
    outcome: dict           # Selected, dropped, orders proposed/executed

    # Why
    reasoning: str          # Human-readable explanation

    # Execution result
    executed: bool = True
    execution_note: str = ""

    # Auto-filled
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        return cls(**d)


class DecisionJournal:
    """
    Append-only journal for trading decisions.

    One JSONL file per day for easy date-based queries.
    """

    def __init__(self, decisions_dir: Path = DECISIONS_DIR):
        self.decisions_dir = decisions_dir
        self.decisions_dir.mkdir(exist_ok=True)

    def _get_path(self, date: str) -> Path:
        """Get path for a specific date's journal."""
        return self.decisions_dir / f"{date}.jsonl"

    def log(self, decision: Decision) -> None:
        """Append decision to today's journal."""
        today = datetime.now().strftime("%Y-%m-%d")
        path = self._get_path(today)

        with open(path, "a") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

    def get_decisions(
        self,
        date: str,
        strategy: Optional[str] = None,
        action: Optional[str] = None,
    ) -> list[Decision]:
        """Read decisions for a date, optionally filtered."""
        path = self._get_path(date)
        if not path.exists():
            return []

        decisions = []
        for line in path.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                d = Decision.from_dict(json.loads(line))
                if strategy and d.strategy != strategy:
                    continue
                if action and d.action != action:
                    continue
                decisions.append(d)
            except (json.JSONDecodeError, TypeError):
                continue

        return decisions

    def get_decision_for_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[Decision]:
        """Find all decisions involving a specific symbol."""
        start = datetime.fromisoformat(start_date) if start_date else datetime.now() - timedelta(days=30)
        end = datetime.fromisoformat(end_date) if end_date else datetime.now()

        results = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            for d in self.get_decisions(date_str):
                # Check symbol field
                if d.symbol == symbol:
                    results.append(d)
                    continue

                # Check outcome for symbol mentions
                outcome_str = json.dumps(d.outcome)
                if symbol in outcome_str:
                    results.append(d)

            current += timedelta(days=1)

        return results

    def list_dates(self, limit: int = 30) -> list[str]:
        """List available journal dates."""
        files = sorted(self.decisions_dir.glob("*.jsonl"), reverse=True)
        return [f.stem for f in files[:limit]]

    def stats(self, date: str) -> dict:
        """Get stats for a date's decisions."""
        decisions = self.get_decisions(date)

        by_strategy = {}
        by_action = {}

        for d in decisions:
            by_strategy[d.strategy] = by_strategy.get(d.strategy, 0) + 1
            by_action[d.action] = by_action.get(d.action, 0) + 1

        return {
            "date": date,
            "total": len(decisions),
            "by_strategy": by_strategy,
            "by_action": by_action,
        }


# === XS Helpers ===

def build_xs_rebalance_decision(
    rankings: list[dict],
    current_positions: list[dict],
    trades: list[dict],
    results: list[dict],
    allocation: dict,
    dry_run: bool = False,
) -> Decision:
    """
    Build a Decision record for XS rebalance.

    Args:
        rankings: Full rankings from rank_universe()
        current_positions: Positions before rebalance
        trades: Proposed trades from calculate_rebalance_trades()
        results: Execution results from execute_trades()
        allocation: Allocation info from calculate_xs_allocation()
        dry_run: Whether this was a dry run
    """
    # Build context: top 20 + summary stats
    top_20 = rankings[:20]
    all_scores = [r["score"] for r in rankings if r.get("score") is not None]

    context = {
        "universe_size": len(rankings),
        "rankings_top_20": [
            {"symbol": r["symbol"], "rank": r["rank"], "momentum_25d": r["score"]}
            for r in top_20
        ],
        "summary": {
            "median_momentum": round(sorted(all_scores)[len(all_scores)//2], 4) if all_scores else None,
            "top_10_cutoff": rankings[9]["score"] if len(rankings) > 9 else None,
            "sell_band_cutoff": rankings[14]["score"] if len(rankings) > 14 else None,
        },
        "current_positions": [
            {"symbol": p["symbol"], "shares": p["shares"], "entry_rank": p.get("entry_rank")}
            for p in current_positions
        ],
        "allocation": {
            "total_equity": allocation.get("equity"),
            "xs_allocation": allocation.get("xs_allocation"),
            "target_per_position": allocation.get("target_per_position"),
        },
    }

    # Build outcome
    buys = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]

    # Determine holds (positions not sold)
    sold_symbols = {t["symbol"] for t in sells}
    holds = [p for p in current_positions if p["symbol"] not in sold_symbols]

    outcome = {
        "buys": [
            {"symbol": t["symbol"], "shares": t["shares"], "rank": t.get("rank"), "reason": t["reason"]}
            for t in buys
        ],
        "sells": [
            {"symbol": t["symbol"], "shares": t["shares"], "pnl": t.get("pnl", 0), "reason": t["reason"]}
            for t in sells
        ],
        "holds": [
            {"symbol": p["symbol"], "shares": p["shares"]}
            for p in holds
        ],
        "orders_proposed": len(trades),
    }

    # Build execution summary
    executed = [r for r in results if r.get("status") == "executed"]
    failed = [r for r in results if r.get("status") == "failed"]

    execution_parts = []
    if dry_run:
        execution_parts.append("DRY RUN - no orders submitted")
    else:
        execution_parts.append(f"{len(executed)} executed, {len(failed)} failed")

    for r in executed:
        if "fill_price" in r:
            slip = (r["fill_price"] - r["price"]) / r["price"] * 100 if r["price"] else 0
            execution_parts.append(f"{r['symbol']}: filled at ${r['fill_price']:.2f} ({slip:+.2f}% slip)")

    for r in failed:
        execution_parts.append(f"{r['symbol']}: FAILED - {r.get('error', 'unknown')}")

    # Build reasoning
    reasoning_parts = []
    if not trades:
        reasoning_parts.append("No trades needed - all positions within persistence band.")
    else:
        if sells:
            reasoning_parts.append(f"Selling {len(sells)}: " + ", ".join(
                f"{t['symbol']} (rank {rankings_lookup(rankings, t['symbol'])})"
                for t in sells
            ))
        if buys:
            reasoning_parts.append(f"Buying {len(buys)}: " + ", ".join(
                f"{t['symbol']} (rank {t.get('rank')})"
                for t in buys
            ))
        if holds:
            reasoning_parts.append(f"Holding {len(holds)} within band.")

    return Decision(
        strategy="xs",
        action="rebalance",
        symbol=None,
        context=context,
        outcome=outcome,
        reasoning=" ".join(reasoning_parts),
        executed=not dry_run and len(failed) == 0,
        execution_note="; ".join(execution_parts),
    )


def rankings_lookup(rankings: list[dict], symbol: str) -> int:
    """Helper to find rank for a symbol."""
    for r in rankings:
        if r["symbol"] == symbol:
            return r["rank"]
    return 999


# === CLI ===

def cmd_list(date: Optional[str] = None):
    """List decisions for a date or available dates."""
    journal = DecisionJournal()

    if date:
        decisions = journal.get_decisions(date)
        if not decisions:
            print(f"No decisions for {date}")
            return

        print(f"Decisions for {date}: {len(decisions)}")
        print("-" * 60)
        for i, d in enumerate(decisions):
            ts = d.timestamp.split("T")[1][:8] if "T" in d.timestamp else ""
            print(f"  [{i}] {ts} {d.strategy}/{d.action} {d.symbol or '(portfolio)'}")
    else:
        dates = journal.list_dates()
        if not dates:
            print("No decision journals found.")
            return

        print("Available dates:")
        for date in dates:
            stats = journal.stats(date)
            print(f"  {date}: {stats['total']} decisions")


def cmd_show(date: str, index: int):
    """Show a specific decision."""
    journal = DecisionJournal()
    decisions = journal.get_decisions(date)

    if index >= len(decisions):
        print(f"Index {index} out of range (max {len(decisions) - 1})")
        return

    d = decisions[index]
    print(json.dumps(d.to_dict(), indent=2))


def cmd_search(symbol: str):
    """Search for decisions involving a symbol."""
    journal = DecisionJournal()
    decisions = journal.get_decision_for_symbol(symbol)

    if not decisions:
        print(f"No decisions found for {symbol}")
        return

    print(f"Decisions involving {symbol}: {len(decisions)}")
    print("-" * 60)
    for d in decisions:
        date = d.timestamp.split("T")[0]
        print(f"  {date} {d.strategy}/{d.action}: {d.reasoning[:60]}...")


def cmd_summary(last_xs: bool = False, today: bool = False):
    """Summary output for ceremony scripts."""
    journal = DecisionJournal()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    if last_xs:
        # Find most recent XS rebalance
        for i in range(30):  # Search last 30 days
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            decisions = journal.get_decisions(date, strategy="xs")
            xs_decisions = [d for d in decisions if d.action == "rebalance"]
            if xs_decisions:
                d = xs_decisions[-1]  # Most recent
                # Parse outcome
                buys = len(d.outcome.get("buys", []))
                sells = len(d.outcome.get("sells", []))
                holds = len(d.outcome.get("holds", []))
                ts = d.timestamp.split("T")
                date_part = ts[0]
                time_part = ts[1][:5] if len(ts) > 1 else ""
                status = "executed" if d.executed else "FAILED"
                print(f"{date_part} {time_part} | rebalance | +{buys} -{sells} ={holds} | {status}")
                return
        print("No XS rebalance found in last 30 days")

    elif today:
        decisions = journal.get_decisions(today_str)
        if not decisions:
            print(f"0 decisions today")
            return

        # Count by strategy
        by_strategy = {}
        executed = 0
        for d in decisions:
            by_strategy[d.strategy] = by_strategy.get(d.strategy, 0) + 1
            if d.executed:
                executed += 1

        strat_str = " ".join(f"{k}:{v}" for k, v in sorted(by_strategy.items()))
        print(f"{len(decisions)} decisions | {strat_str} | {executed}/{len(decisions)} executed")

    else:
        # Default: show stats for today
        cmd_list(today_str)


def cmd_health() -> int:
    """Check journal health for ceremony scripts. Returns exit code."""
    journal = DecisionJournal()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    today_file = journal.decisions_dir / f"{today_str}.jsonl"

    # Market hours check (9:35 ET to 16:00 ET, simplified)
    market_open = 9 <= now.hour < 16

    issues = []

    # Check if today's file exists after market open
    if market_open and now.hour >= 10 and not today_file.exists():
        issues.append("no journal today")

    # Check staleness
    last_entry_age = None
    if today_file.exists():
        mtime = datetime.fromtimestamp(today_file.stat().st_mtime)
        last_entry_age = (now - mtime).total_seconds() / 3600

        if market_open and last_entry_age > 2:
            issues.append(f"stale ({last_entry_age:.1f}h)")

    if issues:
        print(f"WARN: {', '.join(issues)}")
        return 1
    elif last_entry_age is not None:
        print(f"OK (last entry {last_entry_age:.1f}h ago)")
        return 0
    else:
        print("OK (no entries expected yet)")
        return 0


def get_journal_summary_for_morning() -> dict:
    """Get journal summary for morning.py integration."""
    journal = DecisionJournal()
    now = datetime.now()
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Find last XS rebalance
    last_xs = None
    last_xs_date = None
    for i in range(30):
        date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        decisions = journal.get_decisions(date, strategy="xs")
        xs_decisions = [d for d in decisions if d.action == "rebalance"]
        if xs_decisions:
            last_xs = xs_decisions[-1]
            last_xs_date = date
            break

    # Yesterday's count
    yesterday_decisions = journal.get_decisions(yesterday)
    by_strategy = {}
    for d in yesterday_decisions:
        by_strategy[d.strategy] = by_strategy.get(d.strategy, 0) + 1

    return {
        "last_xs": last_xs,
        "last_xs_date": last_xs_date,
        "yesterday_count": len(yesterday_decisions),
        "yesterday_by_strategy": by_strategy,
    }


def get_journal_summary_for_evening() -> dict:
    """Get journal summary for evening.py integration."""
    journal = DecisionJournal()
    today = datetime.now().strftime("%Y-%m-%d")
    decisions = journal.get_decisions(today)

    by_strategy = {}
    by_action = {}
    failures = []
    notable = []

    for d in decisions:
        by_strategy[d.strategy] = by_strategy.get(d.strategy, 0) + 1
        by_action[d.action] = by_action.get(d.action, 0) + 1
        if not d.executed:
            failures.append(d)
        if d.action in ("rebalance", "entry"):
            notable.append(d)

    return {
        "total": len(decisions),
        "by_strategy": by_strategy,
        "by_action": by_action,
        "failures": failures,
        "notable": notable,
        "executed": len(decisions) - len(failures),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decision journal CLI")
    subparsers = parser.add_subparsers(dest="command")

    list_p = subparsers.add_parser("list", help="List decisions")
    list_p.add_argument("--date", "-d", help="Date (YYYY-MM-DD)")

    show_p = subparsers.add_parser("show", help="Show a decision")
    show_p.add_argument("date", help="Date (YYYY-MM-DD)")
    show_p.add_argument("index", type=int, help="Decision index")

    search_p = subparsers.add_parser("search", help="Search by symbol")
    search_p.add_argument("symbol", help="Symbol to search")

    summary_p = subparsers.add_parser("summary", help="Summary for ceremonies")
    summary_p.add_argument("--last-xs", action="store_true", help="Show last XS rebalance")
    summary_p.add_argument("--today", action="store_true", help="Show today's summary")

    health_p = subparsers.add_parser("health", help="Check journal health")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args.date)
    elif args.command == "show":
        cmd_show(args.date, args.index)
    elif args.command == "search":
        cmd_search(args.symbol)
    elif args.command == "summary":
        cmd_summary(last_xs=args.last_xs, today=args.today)
    elif args.command == "health":
        exit_code = cmd_health()
        sys.exit(exit_code)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
