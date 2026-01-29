#!/usr/bin/env python3
# See CODEBASE.md for public interface documentation
"""
morning.py - Pre-market / market-open wrapper

Runs all start-of-day checks in one shot:
1. System memory check
2. state_export.py → fresh context
3. Alpaca connection + account status
4. Reconcile ledger vs Alpaca positions
5. Overnight changes (positions, P&L, fills)
6. Pending thesis trades
7. Autopilot timer status
8. Anomaly detection (orphans, failed runs, log errors)

Output: summary to stdout + state/morning_report.md + optional --json

Idempotent. No stdin. Exit 0=healthy, 1=warnings, 2=errors.

Usage:
    python3 morning.py              # full report
    python3 morning.py --quiet      # minimal output
    python3 morning.py --json       # structured JSON
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

BASE = Path(__file__).parent
STATE_DIR = BASE / "state"
REPORT_FILE = STATE_DIR / "morning_report.md"


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _tail_file(path, n=50):
    try:
        lines = Path(path).read_text().strip().split("\n")
        return lines[-n:]
    except Exception:
        return []


# === Checks ===

def check_memory() -> dict:
    """Check available RAM, warn if < 500MB."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])

        available_kb = meminfo.get("MemAvailable", 0)
        total_kb = meminfo.get("MemTotal", 0)
        swap_total = meminfo.get("SwapTotal", 0)
        swap_free = meminfo.get("SwapFree", 0)

        available_mb = available_kb // 1024
        total_mb = total_kb // 1024
        swap_used_mb = (swap_total - swap_free) // 1024

        warning = available_mb < 500
        return {
            "available_mb": available_mb,
            "total_mb": total_mb,
            "swap_used_mb": swap_used_mb,
            "warning": warning,
        }
    except Exception as e:
        return {"available_mb": -1, "total_mb": -1, "swap_used_mb": -1, "warning": True, "error": str(e)}


def run_state_export() -> dict:
    """Run state_export.py to refresh context."""
    try:
        result = subprocess.run(
            ["python3", str(BASE / "state_export.py")],
            capture_output=True, text=True, timeout=60, cwd=BASE,
        )
        success = result.returncode == 0
        return {"success": success, "output": result.stdout.strip()[:200]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_alpaca() -> dict:
    """Check Alpaca connection and account status."""
    try:
        from trader import Trader
        t = Trader()
        account = t.get_account()
        clock = t.get_clock()
        return {
            "connected": True,
            "equity": account["portfolio_value"],
            "cash": account["cash"],
            "buying_power": account["buying_power"],
            "last_equity": account.get("last_equity", account["portfolio_value"]),
            "pl_today": account.get("pl_today", 0),
            "pl_today_pct": account.get("pl_today_pct", 0),
            "market_open": clock["is_open"],
            "market_phase": clock.get("phase", "unknown"),
            "next_open": clock.get("next_open", ""),
            "next_close": clock.get("next_close", ""),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def reconcile_positions() -> dict:
    """Compare ledger positions vs Alpaca positions (excluding thesis trades and XS positions)."""
    try:
        from trader import Trader
        from ledger import Ledger

        t = Trader()
        alpaca_positions = {p["symbol"]: p for p in t.get_positions()}
        ledger = Ledger()
        ledger_positions = {sym: pos.to_dict() for sym, pos in ledger.positions.items()}

        # Get active thesis trade symbols (managed separately from ledger)
        thesis_data = _load_json(BASE / "thesis_trades.json")
        thesis_syms = set()
        if thesis_data:
            for trade in thesis_data.get("trades", []):
                if trade.get("status", "").lower() in ("active", "open", "pending"):
                    thesis_syms.add(trade.get("symbol"))

        # Get XS positions (managed separately from zoo ledger)
        xs_ledger = _load_json(BASE / "ledger_xs.json")
        xs_syms = set()
        if xs_ledger:
            for pos in xs_ledger.get("positions", []):
                xs_syms.add(pos.get("symbol"))

        alpaca_syms = set(alpaca_positions.keys())
        ledger_syms = set(ledger_positions.keys())

        # Exclude thesis and XS symbols from orphan detection
        excluded_syms = thesis_syms | xs_syms
        in_alpaca_only = alpaca_syms - ledger_syms - excluded_syms
        in_ledger_only = ledger_syms - alpaca_syms
        in_both = alpaca_syms & ledger_syms

        mismatches = []
        for sym in in_both:
            a_qty = alpaca_positions[sym]["qty"]
            l_qty = ledger_positions[sym]["qty"]
            if abs(a_qty - l_qty) > 0.01:
                mismatches.append({
                    "symbol": sym,
                    "alpaca_qty": a_qty,
                    "ledger_qty": l_qty,
                })

        return {
            "in_sync": len(in_alpaca_only) == 0 and len(in_ledger_only) == 0 and len(mismatches) == 0,
            "alpaca_only": list(in_alpaca_only),
            "ledger_only": list(in_ledger_only),
            "qty_mismatches": mismatches,
            "alpaca_count": len(alpaca_syms),
            "ledger_count": len(ledger_syms),
            "xs_count": len(xs_syms),
        }
    except Exception as e:
        return {"in_sync": False, "error": str(e)}


def get_overnight_changes() -> dict:
    """Show overnight position changes and P&L."""
    try:
        from trader import Trader
        t = Trader()
        account = t.get_account()
        positions = t.get_positions()

        overnight_pl = account.get("pl_today", 0)
        overnight_pl_pct = account.get("pl_today_pct", 0)

        position_details = []
        for p in positions:
            position_details.append({
                "symbol": p["symbol"],
                "qty": p["qty"],
                "entry": p["avg_entry"],
                "current": p["current_price"],
                "unrealized_pl": p["unrealized_pl"],
                "unrealized_pl_pct": p["unrealized_pl_pct"],
            })

        # Check for overnight fills
        orders = t.get_orders(status="closed")
        today = datetime.now().strftime("%Y-%m-%d")
        overnight_fills = []
        for o in orders:
            filled = o.get("filled_at", "")
            if filled and filled[:10] == today:
                overnight_fills.append({
                    "symbol": o["symbol"],
                    "side": o["side"],
                    "qty": o["qty"],
                    "filled_price": o.get("filled_avg_price"),
                })

        return {
            "equity": account["portfolio_value"],
            "last_equity": account.get("last_equity", account["portfolio_value"]),
            "overnight_pl": overnight_pl,
            "overnight_pl_pct": overnight_pl_pct,
            "positions": position_details,
            "fills_today": overnight_fills,
        }
    except Exception as e:
        return {"error": str(e)}


def get_pending_thesis() -> dict:
    """List thesis trades pending execution."""
    data = _load_json(BASE / "thesis_trades.json")
    if not data:
        return {"trades": [], "pending_count": 0}

    pending = []
    active = []
    for t in data.get("trades", []):
        status = t.get("status", "").lower()
        if status == "pending":
            pending.append({
                "symbol": t["symbol"],
                "thesis": t.get("thesis", {}).get("summary", ""),
                "target_size": t.get("entry", {}).get("notional", 0),
            })
        elif status == "active":
            active.append({
                "symbol": t["symbol"],
                "shares": t.get("entry", {}).get("shares", 0),
                "entry_price": t.get("entry", {}).get("price"),
                "targets": [tgt.get("label", "") for tgt in t.get("targets", []) if not tgt.get("hit")],
            })

    return {
        "pending": pending,
        "pending_count": len(pending),
        "active": active,
        "active_count": len(active),
    }


def check_autopilot() -> dict:
    """Check autopilot timer status and last run."""
    # Timer status
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "trader-monitor.timer"],
            capture_output=True, text=True, timeout=5,
        )
        timer_active = result.stdout.strip() == "active"
    except Exception:
        timer_active = False

    # Last run from state file
    state = _load_json(BASE / "autopilot_state.json")
    last_run = state.get("last_run", "never") if state else "never"
    trades_today = state.get("trades_today", 0) if state else 0
    stopped_out = state.get("stopped_out", []) if state else []

    return {
        "timer_active": timer_active,
        "last_run": last_run,
        "trades_today": trades_today,
        "stopped_out": stopped_out,
    }


def check_autopilot_xs() -> dict:
    """Check cross-sectional autopilot status."""
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
            t = Trader()
            for p in positions:
                try:
                    quote = t.get_quote(p["symbol"])
                    price = quote.get("price") or quote.get("last") or p["entry_price"]
                except Exception:
                    # Fallback to bar_cache
                    from bar_cache import load_bars
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
                    "rank": p.get("entry_rank", "?"),
                })
                total_value += value
                total_unrealized += pnl
        except Exception:
            pass

    # Check if rebalance is due
    now = datetime.now(ET)
    rebalance_due = now.weekday() == 0  # Monday
    last_rebalance = state.get("last_rebalance")

    if last_rebalance:
        try:
            last_dt = datetime.fromisoformat(last_rebalance)
            if hasattr(last_dt, 'date') and last_dt.date() == now.date():
                rebalance_due = False  # Already rebalanced today
        except Exception:
            pass

    # Calculate allocation info
    try:
        from trader import Trader
        t = Trader()
        account = t.get_account()
        total_equity = float(account["equity"])
        xs_allocation_target = total_equity * 0.30  # 30%
        xs_allocation_used = total_value
        xs_allocation_pct = (total_value / xs_allocation_target * 100) if xs_allocation_target > 0 else 0
    except Exception:
        xs_allocation_target = 0
        xs_allocation_used = total_value
        xs_allocation_pct = 0

    return {
        "enabled": True,
        "holdings": holdings,
        "holdings_count": len(positions),
        "total_value": total_value,
        "unrealized_pnl": total_unrealized,
        "realized_pnl": ledger.get("stats", {}).get("realized_pnl", 0),
        "total_rebalances": ledger.get("stats", {}).get("total_rebalances", 0),
        "last_rebalance": last_rebalance,
        "rebalance_due_today": rebalance_due and now.weekday() == 0,
        "next_rebalance": "Monday 9:35 ET",
        "allocation_target": xs_allocation_target,
        "allocation_used": xs_allocation_used,
        "allocation_pct": xs_allocation_pct,
    }


def detect_anomalies() -> dict:
    """Flag orphan positions, failed runs, log errors."""
    anomalies = []

    # Check for log errors
    log_lines = _tail_file(BASE / "autopilot.log", 100)
    errors = [l for l in log_lines if "ERROR" in l]
    if errors:
        for e in errors[-5:]:
            anomalies.append({"type": "log_error", "message": e.strip()[:120]})

    # Check autopilot state for issues
    state = _load_json(BASE / "autopilot_state.json")
    if state:
        last_run = state.get("last_run", "")
        if last_run and last_run != "never":
            try:
                last_dt = datetime.fromisoformat(last_run)
                hours_ago = (datetime.now() - last_dt).total_seconds() / 3600
                if hours_ago > 24:
                    anomalies.append({
                        "type": "stale_autopilot",
                        "message": f"Autopilot last ran {hours_ago:.0f}h ago",
                    })
            except Exception:
                pass

    # Check reconciliation already done above, but flag orphans here
    try:
        from trader import Trader
        from ledger import Ledger

        t = Trader()
        alpaca_syms = {p["symbol"] for p in t.get_positions()}
        ledger = Ledger()
        ledger_syms = set(ledger.positions.keys())

        # Get active thesis trade symbols (managed separately from ledger)
        thesis_data = _load_json(BASE / "thesis_trades.json")
        thesis_syms = set()
        if thesis_data:
            for trade in thesis_data.get("trades", []):
                if trade.get("status", "").lower() in ("active", "open", "pending"):
                    thesis_syms.add(trade.get("symbol"))

        # Get XS positions (managed separately from zoo ledger)
        xs_ledger = _load_json(BASE / "ledger_xs.json")
        xs_syms = set()
        if xs_ledger:
            for pos in xs_ledger.get("positions", []):
                xs_syms.add(pos.get("symbol"))

        # Positions in Alpaca but not in ledger (excluding thesis and XS) = orphans
        excluded_syms = thesis_syms | xs_syms
        for sym in alpaca_syms - ledger_syms - excluded_syms:
            anomalies.append({"type": "orphan_position", "message": f"{sym} in Alpaca but not in ledger"})

        # Positions in ledger but not in Alpaca = ghost
        for sym in ledger_syms - alpaca_syms:
            anomalies.append({"type": "ghost_position", "message": f"{sym} in ledger but not in Alpaca"})
    except Exception as e:
        anomalies.append({"type": "reconcile_error", "message": str(e)[:120]})

    return {"anomalies": anomalies, "count": len(anomalies)}


# === Output ===

def build_report(data: dict) -> str:
    """Build markdown report from structured data."""
    lines = []
    ts = data["timestamp"]
    lines.append(f"# Morning Report - {ts[:10]}")
    lines.append(f"Generated: {ts}")
    lines.append("")

    # Memory
    mem = data["memory"]
    if mem.get("warning"):
        lines.append(f"**WARNING: Low memory — {mem['available_mb']}MB available**")
        lines.append("")
    lines.append(f"System: {mem['available_mb']}MB available / {mem['total_mb']}MB total | Swap used: {mem['swap_used_mb']}MB")
    lines.append("")

    # Account
    acct = data.get("account", {})
    if acct.get("connected"):
        lines.append("## Account")
        lines.append(f"- Equity: ${acct['equity']:,.2f}")
        lines.append(f"- Cash: ${acct['cash']:,.2f}")
        lines.append(f"- Buying power: ${acct['buying_power']:,.2f}")
        lines.append(f"- Market: {'OPEN' if acct['market_open'] else 'CLOSED'} ({acct.get('market_phase', '')})")
        if not acct["market_open"]:
            lines.append(f"- Next open: {acct.get('next_open', '?')}")
        lines.append("")
    else:
        lines.append(f"## Account\n**DISCONNECTED:** {acct.get('error', 'unknown')}\n")

    # Overnight
    overnight = data.get("overnight", {})
    if not overnight.get("error"):
        pl = overnight.get("overnight_pl", 0)
        pl_pct = overnight.get("overnight_pl_pct", 0)
        sign = "+" if pl >= 0 else ""
        lines.append("## Overnight Changes")
        lines.append(f"- Previous close equity: ${overnight.get('last_equity', 0):,.2f}")
        lines.append(f"- Current equity: ${overnight.get('equity', 0):,.2f}")
        lines.append(f"- Overnight P&L: {sign}${pl:,.2f} ({sign}{pl_pct:.2f}%)")
        if overnight.get("fills_today"):
            lines.append(f"- Fills today: {len(overnight['fills_today'])}")
            for f in overnight["fills_today"]:
                lines.append(f"  - {f['side']} {f['qty']} {f['symbol']} @ ${f.get('filled_price', '?')}")
        lines.append("")

        if overnight.get("positions"):
            lines.append("### Positions")
            lines.append("| Symbol | Qty | Entry | Current | P&L | P&L% |")
            lines.append("|--------|-----|-------|---------|-----|------|")
            for p in overnight["positions"]:
                sign = "+" if p["unrealized_pl"] >= 0 else ""
                lines.append(f"| {p['symbol']} | {p['qty']:.0f} | ${p['entry']:.2f} | ${p['current']:.2f} | {sign}${p['unrealized_pl']:,.2f} | {sign}{p['unrealized_pl_pct']:.1f}% |")
            lines.append("")

    # Reconciliation
    recon = data.get("reconciliation", {})
    if recon.get("in_sync"):
        lines.append(f"## Reconciliation: IN SYNC ({recon['alpaca_count']} positions)")
    else:
        lines.append("## Reconciliation: **MISMATCH**")
        if recon.get("alpaca_only"):
            lines.append(f"- In Alpaca only: {', '.join(recon['alpaca_only'])}")
        if recon.get("ledger_only"):
            lines.append(f"- In ledger only: {', '.join(recon['ledger_only'])}")
        for m in recon.get("qty_mismatches", []):
            lines.append(f"- {m['symbol']}: Alpaca={m['alpaca_qty']}, Ledger={m['ledger_qty']}")
        if recon.get("error"):
            lines.append(f"- Error: {recon['error']}")
    lines.append("")

    # Thesis
    thesis = data.get("thesis", {})
    if thesis.get("pending_count", 0) > 0 or thesis.get("active_count", 0) > 0:
        lines.append("## Thesis Trades")
        for t in thesis.get("pending", []):
            lines.append(f"- **PENDING** {t['symbol']}: {t['thesis'][:60]} (${t['target_size']:,})")
        for t in thesis.get("active", []):
            remaining = ", ".join(t["targets"]) if t["targets"] else "none"
            lines.append(f"- **ACTIVE** {t['symbol']}: {t['shares']} shares @ ${t.get('entry_price', '?')} | targets: {remaining}")
        lines.append("")

    # Autopilot (Zoo)
    ap = data.get("autopilot", {})
    lines.append("## Autopilot (Zoo)")
    lines.append(f"- Timer: {'ACTIVE' if ap.get('timer_active') else 'INACTIVE'}")
    lines.append(f"- Last run: {ap.get('last_run', 'never')}")
    lines.append(f"- Trades today: {ap.get('trades_today', 0)}")
    if ap.get("stopped_out"):
        lines.append(f"- Stopped out: {', '.join(ap['stopped_out'])}")
    lines.append("")

    # Cross-Sectional Autopilot
    xs = data.get("autopilot_xs", {})
    if xs.get("enabled"):
        lines.append("## Autopilot (Cross-Sectional)")

        # Rebalance alert
        if xs.get("rebalance_due_today"):
            lines.append("**REBALANCE DUE TODAY (Monday)**")
            lines.append("")

        # Allocation
        lines.append(f"- Allocation: ${xs.get('allocation_used', 0):,.2f} / ${xs.get('allocation_target', 0):,.2f} ({xs.get('allocation_pct', 0):.0f}%)")
        lines.append(f"- Holdings: {xs.get('holdings_count', 0)}/10")
        lines.append(f"- Last rebalance: {xs.get('last_rebalance', 'never')}")
        lines.append(f"- Next rebalance: {xs.get('next_rebalance', 'Monday 9:35 ET')}")
        lines.append(f"- Unrealized P&L: ${xs.get('unrealized_pnl', 0):+,.2f}")
        lines.append(f"- Realized P&L: ${xs.get('realized_pnl', 0):+,.2f}")
        lines.append("")

        # Holdings table
        holdings = xs.get("holdings", [])
        if holdings:
            lines.append("### XS Holdings")
            lines.append("| Symbol | Shares | Entry | Current | P&L | P&L% |")
            lines.append("|--------|--------|-------|---------|-----|------|")
            for h in holdings:
                lines.append(f"| {h['symbol']} | {h['shares']:.0f} | ${h['entry_price']:.2f} | ${h['current_price']:.2f} | ${h['pnl']:+,.2f} | {h['pnl_pct']:+.1f}% |")
            lines.append("")
    else:
        lines.append("## Autopilot (Cross-Sectional)")
        lines.append("Not initialized (no positions)")
        lines.append("")

    # Anomalies
    anom = data.get("anomalies", {})
    if anom.get("count", 0) > 0:
        lines.append("## Anomalies")
        for a in anom["anomalies"]:
            lines.append(f"- [{a['type']}] {a['message']}")
        lines.append("")

    # State export
    export = data.get("state_export", {})
    lines.append(f"## State Export: {'OK' if export.get('success') else 'FAILED'}")
    lines.append("")

    return "\n".join(lines)


def print_quiet(data: dict):
    """Minimal one-line output."""
    acct = data.get("account", {})
    mem = data.get("memory", {})
    anom = data.get("anomalies", {})
    recon = data.get("reconciliation", {})
    ap = data.get("autopilot", {})
    xs = data.get("autopilot_xs", {})

    equity = f"${acct.get('equity', 0):,.0f}" if acct.get("connected") else "DISCONNECTED"
    pl = acct.get("pl_today", 0)
    pl_str = f"{'+' if pl >= 0 else ''}${pl:,.0f}" if acct.get("connected") else "?"
    mem_warn = " MEM!" if mem.get("warning") else ""
    sync = "sync" if recon.get("in_sync") else "MISMATCH"
    timer = "on" if ap.get("timer_active") else "OFF"
    issues = anom.get("count", 0)

    # XS status
    xs_str = ""
    if xs.get("enabled"):
        xs_count = xs.get("holdings_count", 0)
        xs_rebal = " REBAL!" if xs.get("rebalance_due_today") else ""
        xs_str = f" | xs:{xs_count}/10{xs_rebal}"

    print(f"morning | {equity} ({pl_str}) | {sync} | timer:{timer} | issues:{issues}{mem_warn}{xs_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Morning pre-market wrapper")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal one-line output")
    parser.add_argument("--json", "-j", action="store_true", help="Structured JSON output")
    args = parser.parse_args()

    # Gather all data
    data = {"timestamp": datetime.now().isoformat()}

    data["memory"] = check_memory()
    data["state_export"] = run_state_export()
    data["account"] = check_alpaca()
    data["reconciliation"] = reconcile_positions()
    data["overnight"] = get_overnight_changes()
    data["thesis"] = get_pending_thesis()
    data["autopilot"] = check_autopilot()
    data["autopilot_xs"] = check_autopilot_xs()
    data["anomalies"] = detect_anomalies()

    # Determine exit code
    has_errors = (
        not data["account"].get("connected")
        or data["anomalies"].get("count", 0) > 0
        or data.get("overnight", {}).get("error")
    )
    has_warnings = (
        data["memory"].get("warning")
        or not data["reconciliation"].get("in_sync")
        or not data["autopilot"].get("timer_active")
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

    # Also write JSON for machine consumption
    json_file = STATE_DIR / "morning_report.json"
    json_file.write_text(json.dumps(data, indent=2, default=str))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
