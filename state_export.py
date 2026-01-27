#!/usr/bin/env python3
"""
state_export.py - Generate strategic context for external advisors.

Produces a markdown file summarizing the full state of the trading system:
strategies, positions, performance, backtests, recent activity, system health.

Usage:
    python3 state_export.py                  # generate context file
    python3 state_export.py --stdout         # print to stdout instead

Module:
    from state_export import generate_context
    md = generate_context()
"""

import sys
import json
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

BASE = Path(__file__).parent
OUTPUT = BASE / "state" / "strategic_context.md"

# === Section registry ===
# Each section is a function returning (title, markdown_body).
# Add new sections by defining a function and appending to SECTIONS.

SECTIONS = []


def section(fn):
    """Decorator to register a section generator."""
    SECTIONS.append(fn)
    return fn


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _load_jsonl(path, limit=None):
    entries = []
    try:
        for line in Path(path).read_text().strip().split("\n"):
            if line:
                entries.append(json.loads(line))
    except Exception:
        pass
    if limit:
        entries = entries[-limit:]
    return entries


def _tail_file(path, lines=20):
    try:
        all_lines = Path(path).read_text().strip().split("\n")
        return all_lines[-lines:]
    except Exception:
        return []


# === Sections ===

@section
def section_header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "Strategic Context", f"Generated: {now}\n\nThis file describes the full state of the trader system for cross-session strategic context."


@section
def section_zoo():
    config = _load_json(BASE / "router_config.json")
    if not config:
        return "Strategy Zoo", "_No router_config.json found._"

    lines = []
    lines.append("### Active Strategies\n")
    lines.append("| Strategy | Allocation | Stop ATR× | Trailing | MA Exit | Giveback |")
    lines.append("|----------|-----------|-----------|----------|---------|----------|")

    defaults = config.get("exit_defaults", {})
    overrides = config.get("exit_overrides", {})
    allocation = config.get("allocation", {})

    for name in config.get("active_strategies", []):
        alloc = allocation.get(name, 0)
        exit_p = {**defaults, **overrides.get(name, {})}
        atr = exit_p.get("stop_atr_multiplier", "?")
        trail = "yes" if exit_p.get("trailing_stop_enabled") else "no"
        ma = exit_p.get("ma_exit_period")
        ma_str = str(ma) if ma else "off"
        gb = exit_p.get("profit_giveback_pct", "?")
        lines.append(f"| {name} | {alloc:.0%} | {atr} | {trail} | {ma_str} | {gb} |")

    lines.append(f"\n**Max positions:** {config.get('max_positions', '?')}")
    lines.append(f"**Risk per trade:** {config.get('risk_per_trade', '?')}")
    lines.append(f"**Universe:** {', '.join(config.get('symbols', []))}")

    return "Strategy Zoo", "\n".join(lines)


@section
def section_positions():
    try:
        from trader import Trader
        trader = Trader()
        positions = trader.get_positions()
    except Exception as e:
        return "Current Positions", f"_Failed to fetch from Alpaca: {e}_"

    ledger = _load_json(BASE / "trades_ledger.json")
    ledger_positions = ledger.get("positions", {}) if ledger else {}

    if not positions:
        return "Current Positions", "_No open positions._"

    lines = []
    lines.append("| Symbol | Qty | Entry | Current | P/L | P/L% | Strategy |")
    lines.append("|--------|-----|-------|---------|-----|------|----------|")

    total_value = 0
    total_pnl = 0
    for p in positions:
        sym = p["symbol"]
        strategy = ledger_positions.get(sym, {}).get("strategy", "unknown")
        mv = float(p["qty"]) * p["current_price"]
        pnl = float(p["qty"]) * (p["current_price"] - p["avg_entry"])
        pnl_pct = p["unrealized_pl_pct"]
        total_value += mv
        total_pnl += pnl
        lines.append(f"| {sym} | {p['qty']} | ${p['avg_entry']:.2f} | ${p['current_price']:.2f} | "
                      f"${pnl:+,.2f} | {pnl_pct:+.2f}% | {strategy} |")

    lines.append(f"\n**Total market value:** ${total_value:,.2f}")
    lines.append(f"**Unrealized P/L:** ${total_pnl:+,.2f}")

    return "Current Positions", "\n".join(lines)


@section
def section_account():
    try:
        from trader import Trader
        trader = Trader()
        acct = trader.get_account()
        clock = trader.get_clock()
    except Exception as e:
        return "Account State", f"_Failed to fetch from Alpaca: {e}_"

    market = "OPEN" if clock["is_open"] else "CLOSED"

    lines = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Equity | ${acct['portfolio_value']:,.2f} |",
        f"| Cash | ${acct['cash']:,.2f} |",
        f"| Buying Power | ${acct['buying_power']:,.2f} |",
        f"| Day P/L | ${acct['portfolio_value'] - acct.get('last_equity', acct['portfolio_value']):+,.2f} |",
        f"| Market | {market} |",
    ]

    return "Account State", "\n".join(lines)


@section
def section_performance():
    # Per-strategy P/L from ledger
    ledger = _load_json(BASE / "trades_ledger.json")
    trades = _load_jsonl(BASE / "autopilot_trades.jsonl")

    lines = []

    # Per-strategy from ledger
    if ledger:
        strat_trades = {}
        for t in ledger.get("trades", []):
            s = t.get("strategy", "unknown")
            strat_trades.setdefault(s, []).append(t)

        lines.append("### Per-Strategy P/L (from ledger)\n")
        lines.append("| Strategy | Trades | Closed | Realized P/L | Win Rate |")
        lines.append("|----------|--------|--------|-------------|----------|")

        for strat, ts in strat_trades.items():
            sells = [t for t in ts if t.get("action") == "SELL" and t.get("pnl") is not None]
            realized = sum(t["pnl"] for t in sells)
            wins = sum(1 for t in sells if t["pnl"] > 0)
            wr = wins / len(sells) * 100 if sells else 0
            lines.append(f"| {strat} | {len(ts)} | {len(sells)} | ${realized:+,.2f} | {wr:.0f}% |")

    # Cumulative from trade log
    if trades:
        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        lines.append(f"\n### Autopilot Trade Log Summary\n")
        lines.append(f"- Total entries: {len(buys)} buys, {len(sells)} sells")
        if sells:
            lines.append(f"- Strategies used: {', '.join(set(t.get('strategy', '?') for t in trades))}")

    # Equity curve
    eq = _load_json(BASE / "equity_curve.json")
    if eq and len(eq) > 1:
        lines.append(f"\n### Equity Curve\n")
        lines.append("| Date | Equity | Return | Positions |")
        lines.append("|------|--------|--------|-----------|")
        for e in eq[-10:]:
            lines.append(f"| {e['date']} | ${e['equity']:,.2f} | {e.get('return_pct', 0):+.2f}% | {e.get('positions_count', '?')} |")

    if not lines:
        return "Performance", "_No performance data available._"

    return "Performance", "\n".join(lines)


@section
def section_backtests():
    lines = []
    lines.append("### Backtest Registry\n")

    # Scan for backtest result files
    bt_files = sorted(BASE.glob("backtest_*.json"))
    if not bt_files:
        return "Backtests", "_No backtest results found._"

    for f in bt_files:
        data = _load_json(f)
        if not data:
            continue

        name = f.stem
        lines.append(f"#### {name}\n")

        if isinstance(data, list):
            # Multi-strategy result (e.g., backtest_all_strategies.json)
            lines.append("| Strategy | Return | Sharpe | MaxDD | Trades | Win% | PF |")
            lines.append("|----------|--------|--------|-------|--------|------|----|")
            for r in data:
                pf = f"{r['profit_factor']:.2f}" if r.get("profit_factor", 0) < 100 else "inf"
                lines.append(f"| {r.get('strategy', '?')} | {r.get('total_return_pct', 0):+.1f}% | "
                              f"{r.get('sharpe', 0):.2f} | {r.get('max_drawdown_pct', 0):.1f}% | "
                              f"{r.get('total_trades', 0)} | {r.get('win_rate', 0):.0f}% | {pf} |")
        elif isinstance(data, dict):
            if "blend" in data:
                # Blend result
                b = data["blend"]
                lines.append(f"- **Blend return:** {b.get('return_pct', 0):+.1f}%")
                lines.append(f"- **Blend Sharpe:** {b.get('sharpe', 0):.2f}")
                lines.append(f"- **Blend max DD:** {b.get('max_drawdown_pct', 0):.1f}%")
                lines.append(f"- **Correlation:** {b.get('correlation', 0):+.3f}")
                if "spy" in data:
                    lines.append(f"- **SPY B&H:** {data['spy'].get('return_pct', 0):+.1f}%")
            elif "total_return_pct" in data:
                # Single strategy result
                pf = f"{data['profit_factor']:.2f}" if data.get("profit_factor", 0) < 100 else "inf"
                lines.append(f"- Return: {data['total_return_pct']:+.1f}%")
                lines.append(f"- Sharpe: {data.get('sharpe', 0):.2f}")
                lines.append(f"- Max DD: {data.get('max_drawdown_pct', 0):.1f}%")
                lines.append(f"- Trades: {data.get('total_trades', 0)}, Win: {data.get('win_rate', 0):.0f}%, PF: {pf}")
                lines.append(f"- Period: {data.get('period', '?')}")

        lines.append("")

    # Verdict summary
    lines.append("### Verdict Summary\n")
    lines.append("- **ATR stop-based exits** destroy all edge across every regime (PF 0.49-0.56)")
    lines.append("- **Signal-based exits + 20d max hold** produce slight positive expectancy (PF 1.09-1.15)")
    lines.append("- **No strategy beats SPY buy-and-hold** over 2022-2025 (+44.3%)")
    lines.append("- **50/50 momentum+bollinger blend** returns +31.6% with 29.7% max DD (vs 44.6% momentum alone)")
    lines.append("- **Correlation between strategies:** +0.354 (partial diversification)")

    return "Backtests", "\n".join(lines)


@section
def section_thesis_trades():
    data = _load_json(BASE / "thesis_trades.json")
    if not data or not data.get("trades"):
        return "Thesis Trades", "_No thesis trades._"

    lines = []
    lines.append("_Discretionary trades managed separately from autopilot/strategy zoo._\n")

    for t in data["trades"]:
        status = t.get("status", "?").upper()
        sym = t.get("symbol", "?")
        entry = t.get("entry", {})
        thesis = t.get("thesis", {})
        inv = t.get("invalidation", {})
        targets = t.get("targets", [])
        outcome = t.get("outcome", {})

        lines.append(f"### {sym} [{status}]\n")
        lines.append(f"- **Entry:** {entry.get('date', '?')} | "
                     f"{entry.get('shares', '?')} shares @ ${entry.get('price') or 'pending'} | "
                     f"${entry.get('notional', '?'):,}")
        lines.append(f"- **Thesis:** {thesis.get('summary', 'none')}")
        lines.append(f"- **Invalidation:** close below ${inv.get('price_below', '?')} OR "
                     f"{inv.get('time_condition', '?')} (by {inv.get('time_deadline', '?')})")

        if targets:
            target_strs = []
            for tgt in targets:
                hit = " ✓" if tgt.get("hit") else ""
                price = f"${tgt['price']:.0f}" if tgt.get("price") else "hold"
                target_strs.append(f"{tgt['label']}: {price} (trim {tgt['trim_pct']}%){hit}")
            lines.append(f"- **Targets:** {' | '.join(target_strs)}")

        if outcome.get("realized_pnl"):
            lines.append(f"- **Realized P/L:** ${outcome['realized_pnl']:+,.2f}")
        if outcome.get("thesis_correct") is not None:
            lines.append(f"- **Thesis correct:** {outcome['thesis_correct']}")

        lines.append("")

    return "Thesis Trades", "\n".join(lines)


@section
def section_recent_activity():
    lines = []

    # Last N trades
    trades = _load_jsonl(BASE / "autopilot_trades.jsonl", limit=15)
    if trades:
        lines.append("### Recent Trades\n")
        lines.append("| Time | Action | Symbol | Qty | Price | Strategy | Reason |")
        lines.append("|------|--------|--------|-----|-------|----------|--------|")
        for t in trades:
            time_short = t.get("time", "?")[:19]
            price_str = f"${t['price']:.2f}" if t.get("price") else "mkt"
            lines.append(f"| {time_short} | {t['action']} | {t['symbol']} | {t['qty']} | "
                          f"{price_str} | {t.get('strategy', '?')} | {t.get('reason', '')[:40]} |")
    else:
        lines.append("_No recent trades._")

    # Last N log entries
    log_lines = _tail_file(BASE / "autopilot.log", 15)
    if log_lines:
        lines.append("\n### Recent Autopilot Log\n")
        lines.append("```")
        for l in log_lines:
            lines.append(l)
        lines.append("```")

    return "Recent Activity", "\n".join(lines)


@section
def section_system_status():
    lines = []

    # Autopilot state
    state = _load_json(BASE / "autopilot_state.json")
    if state:
        lines.append(f"- **Last autopilot run:** {state.get('last_run', 'never')}")
        lines.append(f"- **Trades today:** {state.get('trades_today', 0)}")
        stopped = state.get("stopped_out", [])
        if stopped:
            lines.append(f"- **Stopped out today:** {', '.join(stopped)}")
    else:
        lines.append("- _No autopilot state file found._")

    # Check for errors in recent log
    log_lines = _tail_file(BASE / "autopilot.log", 50)
    errors = [l for l in log_lines if "ERROR" in l]
    if errors:
        lines.append(f"\n### Recent Errors ({len(errors)})\n")
        lines.append("```")
        for e in errors[-5:]:
            lines.append(e)
        lines.append("```")
    else:
        lines.append("- **Errors:** none in recent log")

    # High water marks
    hwm = _load_json(BASE / "high_water_marks.json")
    if hwm:
        lines.append(f"\n### High Water Marks\n")
        for sym, price in hwm.items():
            lines.append(f"- {sym}: ${price:.2f}")

    return "System Status", "\n".join(lines)


# === Generator ===

def generate_context() -> str:
    """Generate the full strategic context markdown."""
    parts = []
    for fn in SECTIONS:
        try:
            title, body = fn()
            parts.append(f"## {title}\n\n{body}")
        except Exception as e:
            parts.append(f"## {fn.__name__}\n\n_Error generating section: {e}_")

    return "\n\n---\n\n".join(parts) + "\n"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export strategic context")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    parser.add_argument("--output", "-o", default=str(OUTPUT), help="Output file path")
    args = parser.parse_args()

    md = generate_context()

    if args.stdout:
        print(md)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"Strategic context written to {out}")
        print(f"Size: {len(md)} bytes, {md.count(chr(10))} lines")


if __name__ == "__main__":
    main()
