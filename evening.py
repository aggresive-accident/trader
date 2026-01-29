#!/usr/bin/env python3
# See CODEBASE.md for public interface documentation
"""
evening.py - End of day / market close wrapper

Runs all end-of-day checks in one shot:
1. state_export.py → capture closing state
2. Day's activity (trades, realized/unrealized P&L)
3. Compare to previous EOD (equity delta)
4. Thesis trade target/stop checks
5. Log rotation check
6. Archive today's state snapshot
7. Output: summary to stdout + state/evening_report.md + optional --json

Idempotent. No stdin. Exit 0=healthy, 1=warnings, 2=errors.

Usage:
    python3 evening.py              # full report
    python3 evening.py --quiet      # minimal output
    python3 evening.py --json       # structured JSON
"""

import sys
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

BASE = Path(__file__).parent
STATE_DIR = BASE / "state"
ARCHIVE_DIR = STATE_DIR / "archive"
REPORT_FILE = STATE_DIR / "evening_report.md"


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


def _file_size_mb(path):
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0


# === Checks ===

def run_state_export() -> dict:
    """Run state_export.py to capture closing state."""
    import subprocess
    try:
        result = subprocess.run(
            ["python3", str(BASE / "state_export.py")],
            capture_output=True, text=True, timeout=60, cwd=BASE,
        )
        return {"success": result.returncode == 0, "output": result.stdout.strip()[:200]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_days_activity() -> dict:
    """Get today's trades, realized and unrealized P&L."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Today's trades from autopilot log
    trades = _load_jsonl(BASE / "autopilot_trades.jsonl")
    todays_trades = [t for t in trades if t.get("time", "").startswith(today)]

    buys = [t for t in todays_trades if t.get("action") == "BUY"]
    sells = [t for t in todays_trades if t.get("action") == "SELL"]

    realized_pnl = 0
    for s in sells:
        if s.get("pnl") is not None:
            realized_pnl += s["pnl"]

    # Unrealized from current positions
    try:
        from trader import Trader
        t = Trader()
        account = t.get_account()
        positions = t.get_positions()

        unrealized_pnl = sum(p["unrealized_pl"] for p in positions)
        position_details = []
        for p in positions:
            position_details.append({
                "symbol": p["symbol"],
                "qty": p["qty"],
                "entry": p["avg_entry"],
                "current": p["current_price"],
                "unrealized_pl": p["unrealized_pl"],
                "unrealized_pl_pct": p["unrealized_pl_pct"],
                "market_value": p["market_value"],
            })
    except Exception as e:
        unrealized_pnl = 0
        position_details = []
        account = {}

    trade_details = []
    for t in todays_trades:
        trade_details.append({
            "time": t.get("time", "")[:19],
            "action": t.get("action"),
            "symbol": t.get("symbol"),
            "qty": t.get("qty"),
            "price": t.get("price"),
            "strategy": t.get("strategy", "?"),
            "reason": t.get("reason", "")[:60],
        })

    return {
        "trades_today": len(todays_trades),
        "buys": len(buys),
        "sells": len(sells),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": realized_pnl + unrealized_pnl,
        "equity": account.get("portfolio_value", 0),
        "cash": account.get("cash", 0),
        "positions": position_details,
        "trades": trade_details,
    }


def compare_to_previous_eod() -> dict:
    """Compare current equity to previous EOD snapshot."""
    # Check equity curve
    eq_data = _load_json(BASE / "equity_curve.json")

    # Also check previous evening report
    prev_evening = _load_json(STATE_DIR / "evening_report.json")

    current_equity = 0
    try:
        from trader import Trader
        t = Trader()
        account = t.get_account()
        current_equity = account["portfolio_value"]
    except Exception:
        pass

    previous_equity = None
    source = None

    # Try evening report first (most accurate EOD)
    if prev_evening and prev_evening.get("activity", {}).get("equity"):
        previous_equity = prev_evening["activity"]["equity"]
        source = "evening_report"
    elif eq_data and len(eq_data) >= 1:
        previous_equity = eq_data[-1].get("equity", 0)
        source = "equity_curve"

    if previous_equity and current_equity:
        delta = current_equity - previous_equity
        delta_pct = (delta / previous_equity) * 100 if previous_equity else 0
        return {
            "current_equity": current_equity,
            "previous_equity": previous_equity,
            "delta": delta,
            "delta_pct": delta_pct,
            "source": source,
        }
    else:
        return {
            "current_equity": current_equity,
            "previous_equity": previous_equity,
            "delta": None,
            "source": source,
        }


def check_thesis_targets() -> dict:
    """Check if any thesis trade targets or stops were hit today."""
    data = _load_json(BASE / "thesis_trades.json")
    if not data:
        return {"trades": []}

    results = []
    try:
        from trader import Trader
        t = Trader()
        positions = {p["symbol"]: p for p in t.get_positions()}
    except Exception:
        positions = {}

    for trade in data.get("trades", []):
        if trade.get("status", "").lower() not in ("active", "pending"):
            continue

        sym = trade["symbol"]
        pos = positions.get(sym)
        current_price = pos["current_price"] if pos else None

        inv = trade.get("invalidation", {})
        stop_price = inv.get("price_below")
        stop_hit = False
        if current_price and stop_price and current_price < stop_price:
            stop_hit = True

        targets_hit = []
        for tgt in trade.get("targets", []):
            if tgt.get("hit"):
                continue  # already hit before
            if current_price and tgt.get("price") and current_price >= tgt["price"]:
                targets_hit.append(tgt.get("label", "?"))

        results.append({
            "symbol": sym,
            "status": trade.get("status"),
            "current_price": current_price,
            "stop_price": stop_price,
            "stop_hit": stop_hit,
            "targets_hit_today": targets_hit,
        })

    return {"trades": results}


def check_log_rotation() -> dict:
    """Check autopilot.log size."""
    log_path = BASE / "autopilot.log"
    thesis_log = BASE / "thesis_trades.log"

    logs = {}
    for name, path in [("autopilot.log", log_path), ("thesis_trades.log", thesis_log)]:
        size = _file_size_mb(path)
        logs[name] = {
            "size_mb": round(size, 2),
            "needs_rotation": size > 10,
        }

    return logs


def archive_snapshot() -> dict:
    """Archive today's state snapshot."""
    today = datetime.now().strftime("%Y-%m-%d")
    archive_day_dir = ARCHIVE_DIR / today

    # Idempotent: if already archived today, just report it
    if archive_day_dir.exists():
        files = list(archive_day_dir.iterdir())
        return {"archived": True, "path": str(archive_day_dir), "files": len(files), "already_existed": True}

    archive_day_dir.mkdir(parents=True, exist_ok=True)

    files_to_archive = [
        "autopilot_state.json",
        "autopilot_xs_state.json",
        "trades_ledger.json",
        "ledger_xs.json",
        "thesis_trades.json",
        "router_config.json",
        "equity_curve.json",
        "high_water_marks.json",
    ]

    archived = 0
    for fname in files_to_archive:
        src = BASE / fname
        if src.exists():
            shutil.copy2(src, archive_day_dir / fname)
            archived += 1

    # Also archive strategic context
    ctx = STATE_DIR / "strategic_context.md"
    if ctx.exists():
        shutil.copy2(ctx, archive_day_dir / "strategic_context.md")
        archived += 1

    return {"archived": True, "path": str(archive_day_dir), "files": archived, "already_existed": False}


def check_autopilot_xs() -> dict:
    """Check cross-sectional autopilot status for evening report."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    state = _load_json(BASE / "autopilot_xs_state.json")
    ledger = _load_json(BASE / "ledger_xs.json")

    if not state and not ledger:
        return {"enabled": False}

    state = state or {}
    ledger = ledger or {"positions": [], "stats": {}}

    # Get current holdings and P&L
    positions = ledger.get("positions", [])
    holdings = []
    total_value = 0
    total_unrealized = 0

    if positions:
        try:
            from trader import Trader
            from bar_cache import load_bars
            t = Trader()
            for p in positions:
                try:
                    quote = t.get_quote(p["symbol"])
                    price = quote.get("price") or quote.get("last") or p["entry_price"]
                except Exception:
                    df = load_bars(p["symbol"])
                    price = float(df["close"].iloc[-1]) if not df.empty else p["entry_price"]

                value = p["shares"] * price
                pnl = (price - p["entry_price"]) * p["shares"]
                pnl_pct = (price / p["entry_price"] - 1) * 100

                holdings.append({
                    "symbol": p["symbol"],
                    "shares": p["shares"],
                    "entry_price": p["entry_price"],
                    "current_price": price,
                    "value": value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                })
                total_value += value
                total_unrealized += pnl
        except Exception:
            pass

    # Calculate allocation info
    try:
        from trader import Trader
        t = Trader()
        account = t.get_account()
        total_equity = float(account["equity"])
        xs_allocation_target = total_equity * 0.30
        xs_allocation_pct = (total_value / xs_allocation_target * 100) if xs_allocation_target > 0 else 0
    except Exception:
        xs_allocation_target = 0
        xs_allocation_pct = 0

    return {
        "enabled": True,
        "holdings": holdings,
        "holdings_count": len(positions),
        "total_value": total_value,
        "unrealized_pnl": total_unrealized,
        "realized_pnl": ledger.get("stats", {}).get("realized_pnl", 0),
        "total_rebalances": ledger.get("stats", {}).get("total_rebalances", 0),
        "total_trades": ledger.get("stats", {}).get("total_trades", 0),
        "last_rebalance": state.get("last_rebalance"),
        "allocation_target": xs_allocation_target,
        "allocation_used": total_value,
        "allocation_pct": xs_allocation_pct,
    }


# === Output ===

def build_report(data: dict) -> str:
    """Build markdown report from structured data."""
    lines = []
    ts = data["timestamp"]
    lines.append(f"# Evening Report - {ts[:10]}")
    lines.append(f"Generated: {ts}")
    lines.append("")

    # State export
    export = data.get("state_export", {})
    lines.append(f"State export: {'OK' if export.get('success') else 'FAILED'}")
    lines.append("")

    # Day's activity
    act = data.get("activity", {})
    lines.append("## Day's Activity")
    lines.append(f"- Equity: ${act.get('equity', 0):,.2f}")
    lines.append(f"- Cash: ${act.get('cash', 0):,.2f}")
    lines.append(f"- Trades today: {act.get('trades_today', 0)} ({act.get('buys', 0)} buys, {act.get('sells', 0)} sells)")
    lines.append(f"- Realized P&L: ${act.get('realized_pnl', 0):+,.2f}")
    lines.append(f"- Unrealized P&L: ${act.get('unrealized_pnl', 0):+,.2f}")
    lines.append(f"- Total P&L: ${act.get('total_pnl', 0):+,.2f}")
    lines.append("")

    if act.get("trades"):
        lines.append("### Trades")
        lines.append("| Time | Action | Symbol | Qty | Price | Strategy | Reason |")
        lines.append("|------|--------|--------|-----|-------|----------|--------|")
        for t in act["trades"]:
            price = f"${t['price']:.2f}" if t.get("price") else "mkt"
            lines.append(f"| {t['time']} | {t['action']} | {t['symbol']} | {t['qty']} | {price} | {t['strategy']} | {t['reason']} |")
        lines.append("")

    if act.get("positions"):
        lines.append("### Open Positions")
        lines.append("| Symbol | Qty | Entry | Current | P&L | P&L% | MktVal |")
        lines.append("|--------|-----|-------|---------|-----|------|--------|")
        for p in act["positions"]:
            lines.append(f"| {p['symbol']} | {p['qty']:.0f} | ${p['entry']:.2f} | ${p['current']:.2f} | ${p['unrealized_pl']:+,.2f} | {p['unrealized_pl_pct']:+.1f}% | ${p['market_value']:,.0f} |")
        lines.append("")

    # Equity delta
    delta = data.get("equity_delta", {})
    if delta.get("delta") is not None:
        sign = "+" if delta["delta"] >= 0 else ""
        lines.append("## Equity vs Previous EOD")
        lines.append(f"- Previous: ${delta['previous_equity']:,.2f} (from {delta.get('source', '?')})")
        lines.append(f"- Current: ${delta['current_equity']:,.2f}")
        lines.append(f"- Delta: {sign}${delta['delta']:,.2f} ({sign}{delta['delta_pct']:.2f}%)")
        lines.append("")

    # Thesis targets
    thesis = data.get("thesis_targets", {})
    thesis_trades = thesis.get("trades", [])
    if thesis_trades:
        lines.append("## Thesis Trade Status")
        for t in thesis_trades:
            price_str = f"${t['current_price']:.2f}" if t.get("current_price") else "N/A"
            lines.append(f"- **{t['symbol']}** [{t['status']}] @ {price_str}")
            if t.get("stop_hit"):
                lines.append(f"  **STOP HIT** (below ${t['stop_price']})")
            if t.get("targets_hit_today"):
                lines.append(f"  Targets hit: {', '.join(t['targets_hit_today'])}")
        lines.append("")

    # Cross-Sectional Autopilot
    xs = data.get("autopilot_xs", {})
    if xs.get("enabled"):
        lines.append("## Cross-Sectional Autopilot")
        lines.append(f"- Allocation: ${xs.get('allocation_used', 0):,.2f} / ${xs.get('allocation_target', 0):,.2f} ({xs.get('allocation_pct', 0):.0f}%)")
        lines.append(f"- Holdings: {xs.get('holdings_count', 0)}/10")
        lines.append(f"- Unrealized P&L: ${xs.get('unrealized_pnl', 0):+,.2f}")
        lines.append(f"- Realized P&L: ${xs.get('realized_pnl', 0):+,.2f}")
        lines.append(f"- Total rebalances: {xs.get('total_rebalances', 0)}")
        lines.append(f"- Last rebalance: {xs.get('last_rebalance', 'never')}")
        lines.append("")

        holdings = xs.get("holdings", [])
        if holdings:
            lines.append("### XS Holdings")
            lines.append("| Symbol | Shares | Entry | Current | P&L | P&L% | Value |")
            lines.append("|--------|--------|-------|---------|-----|------|-------|")
            for h in holdings:
                lines.append(f"| {h['symbol']} | {h['shares']:.0f} | ${h['entry_price']:.2f} | ${h['current_price']:.2f} | ${h['pnl']:+,.2f} | {h['pnl_pct']:+.1f}% | ${h['value']:,.0f} |")
            lines.append("")

    # Log rotation
    logs = data.get("logs", {})
    needs_rotation = [name for name, info in logs.items() if info.get("needs_rotation")]
    if needs_rotation:
        lines.append("## Log Rotation Needed")
        for name in needs_rotation:
            lines.append(f"- {name}: {logs[name]['size_mb']:.1f}MB")
        lines.append("")

    # Archive
    archive = data.get("archive", {})
    if archive.get("archived"):
        existed = " (already existed)" if archive.get("already_existed") else ""
        lines.append(f"## Archive: {archive['files']} files{existed}")
        lines.append(f"Path: {archive.get('path', '?')}")
        lines.append("")

    return "\n".join(lines)


def print_quiet(data: dict):
    """Minimal one-line output."""
    act = data.get("activity", {})
    delta = data.get("equity_delta", {})
    thesis = data.get("thesis_targets", {})
    archive = data.get("archive", {})
    xs = data.get("autopilot_xs", {})

    equity = f"${act.get('equity', 0):,.0f}"
    total_pl = act.get("total_pnl", 0)
    pl_str = f"{'+' if total_pl >= 0 else ''}${total_pl:,.0f}"
    trades = act.get("trades_today", 0)

    d = delta.get("delta")
    delta_str = f"{'+' if d and d >= 0 else ''}${d:,.0f}" if d is not None else "?"

    stops = sum(1 for t in thesis.get("trades", []) if t.get("stop_hit"))
    targets = sum(len(t.get("targets_hit_today", [])) for t in thesis.get("trades", []))

    archived = archive.get("files", 0)

    # XS status
    xs_str = ""
    if xs.get("enabled"):
        xs_count = xs.get("holdings_count", 0)
        xs_pnl = xs.get("unrealized_pnl", 0)
        xs_str = f" | xs:{xs_count}/10 (${xs_pnl:+,.0f})"

    print(f"evening | {equity} ({pl_str}) | trades:{trades} | vs_prev:{delta_str} | stops:{stops} tgts:{targets} | archived:{archived}{xs_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evening end-of-day wrapper")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal one-line output")
    parser.add_argument("--json", "-j", action="store_true", help="Structured JSON output")
    args = parser.parse_args()

    # Gather all data
    data = {"timestamp": datetime.now().isoformat()}

    data["state_export"] = run_state_export()
    data["activity"] = get_days_activity()
    data["equity_delta"] = compare_to_previous_eod()
    data["thesis_targets"] = check_thesis_targets()
    data["autopilot_xs"] = check_autopilot_xs()
    data["logs"] = check_log_rotation()
    data["archive"] = archive_snapshot()

    # Determine exit code
    has_errors = (
        not data["state_export"].get("success")
        or data.get("activity", {}).get("equity", 0) == 0
    )
    has_warnings = (
        any(t.get("stop_hit") for t in data.get("thesis_targets", {}).get("trades", []))
        or any(v.get("needs_rotation") for v in data.get("logs", {}).values())
    )

    exit_code = 2 if has_errors else (1 if has_warnings else 0)
    data["exit_code"] = exit_code

    # Output
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    elif args.quiet:
        print_quiet(data)
    else:
        report = build_report(data)
        print(report)

    # Always write report files
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(build_report(data))

    json_file = STATE_DIR / "evening_report.json"
    json_file.write_text(json.dumps(data, indent=2, default=str))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
