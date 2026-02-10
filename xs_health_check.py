#!/usr/bin/env python3
"""
xs_health_check.py - Weekly XS momentum health check (R044)

Compares live XS momentum behavior against R002 backtest distributions.
Emits signals to organism attention system when deviations detected.

Usage:
  python3 xs_health_check.py              # Console output
  python3 xs_health_check.py --emit-signal # Also write organism signal
  python3 xs_health_check.py --json        # Raw JSON output
  python3 xs_health_check.py baselines     # Show baseline distributions

Scheduling:
  Runs Monday evening via evening.py integration, or standalone anytime.

Known limitation: Baselines are from R002 backtest (weekly rebalance, 10 positions,
2022-2025). Live XS uses the same 10-position setup but with a persistence band
(buy <=10, sell >15) which may produce different turnover/duration patterns.
If z-scores look systematically off, the false-positive recalibration gate
(>4 FP in month 1) will widen thresholds automatically.
"""

import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

# === Paths ===
BASE_DIR = Path(__file__).parent
BACKTEST_TRADES = BASE_DIR / "cross_backtest_trades.jsonl"
LEDGER_FILE = BASE_DIR / "ledger_xs.json"
STATE_FILE = BASE_DIR / "autopilot_xs_state.json"
REPORT_DIR = BASE_DIR / "reports" / "xs_weekly_health"
BASELINES_CACHE = REPORT_DIR / "baselines.json"
FP_TRACKER_FILE = REPORT_DIR / "fp_tracker.json"
SIGNALS_DIR = Path.home() / ".organism" / "signals"
SIGNALS_FILE = SIGNALS_DIR / "xs_health.json"

# === Thresholds ===
SIGMA_WARNING = 2.0
SIGMA_PROBLEM = 3.0
MIN_CLOSED_TRADES = 5
MIN_WEEKS_FOR_DRAWDOWN = 4
MIN_WEEKS_FOR_SHARPE = 4
GRACE_PERIOD_WEEKS = 4
METRICS = ["win_rate", "avg_return", "avg_duration", "turnover", "max_drawdown", "sharpe_4w"]


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# =============================================================================
# Baseline Reconstruction
# =============================================================================

def load_backtest_trades():
    """Load raw BUY/SELL records from cross_backtest_trades.jsonl."""
    trades = []
    for line in BACKTEST_TRADES.read_text().strip().split("\n"):
        if line.strip():
            trades.append(json.loads(line))
    return trades


def reconstruct_round_trips(raw_trades):
    """FIFO match BUY/SELL pairs per symbol into round-trip trades."""
    open_positions = defaultdict(list)  # symbol -> [list of buys]
    round_trips = []

    for t in raw_trades:
        sym = t["symbol"]
        if t["side"] == "BUY":
            open_positions[sym].append(t)
        elif t["side"] == "SELL":
            if open_positions[sym]:
                buy = open_positions[sym].pop(0)  # FIFO
                entry_price = buy["price"]
                exit_price = t["price"]
                shares = buy["shares"]
                return_pct = (exit_price - entry_price) / entry_price
                pnl = (exit_price - entry_price) * shares
                entry_date = buy["date"]
                exit_date = t["date"]
                duration = (datetime.strptime(exit_date, "%Y-%m-%d") -
                            datetime.strptime(entry_date, "%Y-%m-%d")).days

                round_trips.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "return_pct": return_pct,
                    "pnl": pnl,
                    "duration_days": duration,
                })

    return round_trips


def compute_baselines(round_trips, raw_trades):
    """
    Compute per-window distributions from round-trip trades.

    Returns dict of {metric: {mean, std}} for each metric.
    """
    # --- Per-window metrics (grouped by exit_date) ---
    by_exit_date = defaultdict(list)
    for rt in round_trips:
        by_exit_date[rt["exit_date"]].append(rt)

    window_win_rates = []
    window_avg_returns = []
    window_avg_durations = []

    for date, trades in sorted(by_exit_date.items()):
        if len(trades) == 0:
            continue
        wins = sum(1 for t in trades if t["pnl"] > 0)
        window_win_rates.append(wins / len(trades))
        window_avg_returns.append(statistics.mean(t["return_pct"] for t in trades))
        window_avg_durations.append(statistics.mean(t["duration_days"] for t in trades))

    # --- Turnover: count all trades (buys + sells) per date ---
    trades_per_date = defaultdict(int)
    for t in raw_trades:
        trades_per_date[t["date"]] += 1

    window_turnovers = list(trades_per_date.values())

    # --- Cumulative PnL for drawdown and sharpe ---
    # Build daily PnL series from round-trips (attributed to exit_date)
    daily_pnl = defaultdict(float)
    for rt in round_trips:
        daily_pnl[rt["exit_date"]] += rt["pnl"]

    sorted_dates = sorted(daily_pnl.keys())
    cumulative = 0.0
    high_water = 0.0
    equity_series = []  # (date, cumulative_pnl)

    for d in sorted_dates:
        cumulative += daily_pnl[d]
        equity_series.append((d, cumulative))
        high_water = max(high_water, cumulative)

    # Rolling 4-week sharpe from weekly returns
    # Group into weekly buckets
    weekly_returns = []
    if equity_series:
        week_start_cum = 0.0
        week_start_idx = 0
        for i, (d, cum) in enumerate(equity_series):
            dt = datetime.strptime(d, "%Y-%m-%d")
            dt_start = datetime.strptime(equity_series[week_start_idx][0], "%Y-%m-%d")
            if (dt - dt_start).days >= 7 or i == len(equity_series) - 1:
                weekly_ret = cum - week_start_cum
                weekly_returns.append(weekly_ret)
                week_start_cum = cum
                week_start_idx = i

    rolling_sharpes = []
    for i in range(4, len(weekly_returns) + 1):
        window = weekly_returns[i - 4:i]
        if len(window) == 4:
            mean_r = statistics.mean(window)
            std_r = statistics.stdev(window) if len(set(window)) > 1 else 0.001
            rolling_sharpes.append(mean_r / std_r if std_r > 0 else 0.0)

    # Running drawdown series
    cumulative = 0.0
    high_water = 0.0
    drawdowns = []
    for d in sorted_dates:
        cumulative += daily_pnl[d]
        high_water = max(high_water, cumulative)
        dd = (high_water - cumulative) / max(high_water, 1.0) if high_water > 0 else 0.0
        drawdowns.append(dd)

    # Build baselines dict
    def safe_stats(values, min_n=3):
        if len(values) < min_n:
            return {"mean": None, "std": None, "n": len(values)}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "n": len(values),
        }

    # For win_rate, also store cumulative baseline for SE-based comparison
    all_wins = sum(1 for rt in round_trips if rt["pnl"] > 0)
    cumulative_win_rate = all_wins / len(round_trips) if round_trips else 0.5

    baselines = {
        "win_rate": {
            **safe_stats(window_win_rates),
            "cumulative_p": cumulative_win_rate,
            "total_trades": len(round_trips),
        },
        "avg_return": safe_stats(window_avg_returns),
        "avg_duration": safe_stats(window_avg_durations),
        "turnover": safe_stats(window_turnovers),
        "max_drawdown": safe_stats(drawdowns, min_n=10),
        "sharpe_4w": safe_stats(rolling_sharpes, min_n=5),
        "meta": {
            "source": str(BACKTEST_TRADES),
            "round_trips": len(round_trips),
            "windows": len(by_exit_date),
            "date_range": f"{sorted_dates[0]} to {sorted_dates[-1]}" if sorted_dates else "N/A",
        },
    }

    return baselines


def load_or_compute_baselines():
    """Load cached baselines or recompute from JSONL."""
    # Check cache freshness
    if BASELINES_CACHE.exists() and BACKTEST_TRADES.exists():
        cache_mtime = BASELINES_CACHE.stat().st_mtime
        source_mtime = BACKTEST_TRADES.stat().st_mtime
        if cache_mtime > source_mtime:
            cached = _load_json(BASELINES_CACHE)
            if cached:
                return cached

    raw = load_backtest_trades()
    round_trips = reconstruct_round_trips(raw)
    baselines = compute_baselines(round_trips, raw)

    # Cache
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINES_CACHE.write_text(json.dumps(baselines, indent=2))

    return baselines


# =============================================================================
# Live Metrics
# =============================================================================

def compute_live_metrics():
    """Compute current XS metrics from ledger and state files."""
    ledger = _load_json(LEDGER_FILE) or {"positions": [], "trades": [], "closed_trades": [], "stats": {}}
    state = _load_json(STATE_FILE) or {}

    positions = ledger.get("positions", [])
    trades = ledger.get("trades", [])
    closed = ledger.get("closed_trades", [])
    stats = ledger.get("stats", {})

    # Weeks running since first trade
    first_trade_date = None
    if trades:
        try:
            first_trade_date = datetime.fromisoformat(trades[0]["timestamp"])
        except Exception:
            pass

    weeks_running = 0
    if first_trade_date:
        delta = datetime.now().astimezone() - first_trade_date
        weeks_running = max(1, int(delta.days / 7))

    # --- Trade-level metrics from closed_trades ---
    win_rate = None
    avg_return = None
    avg_duration = None

    if len(closed) >= MIN_CLOSED_TRADES:
        wins = 0
        returns = []
        durations = []

        for ct in closed:
            pnl = ct.get("pnl", 0)
            if pnl > 0:
                wins += 1

            entry_p = ct.get("entry_price", 0)
            exit_p = ct.get("exit_price", 0)
            if entry_p > 0:
                returns.append((exit_p - entry_p) / entry_p)

            try:
                entry_dt = datetime.fromisoformat(ct["entry_date"])
                exit_dt = datetime.fromisoformat(ct["exit_date"])
                durations.append((exit_dt - entry_dt).days)
            except Exception:
                pass

        win_rate = wins / len(closed) if closed else None
        avg_return = statistics.mean(returns) if returns else None
        avg_duration = statistics.mean(durations) if durations else None

    # --- Turnover: average trades per rebalance ---
    turnover = None
    total_rebalances = stats.get("total_rebalances", 0)
    total_trades = stats.get("total_trades", 0)
    if total_rebalances > 0:
        turnover = total_trades / total_rebalances

    # --- Max drawdown from cumulative PnL ---
    max_drawdown = None
    if weeks_running >= MIN_WEEKS_FOR_DRAWDOWN and closed:
        cum_pnl = 0.0
        high_water = 0.0
        worst_dd = 0.0
        for ct in closed:
            cum_pnl += ct.get("pnl", 0)
            high_water = max(high_water, cum_pnl)
            if high_water > 0:
                dd = (high_water - cum_pnl) / high_water
                worst_dd = max(worst_dd, dd)
        max_drawdown = worst_dd

    # --- Sharpe 4w: need weekly P&L snapshots ---
    sharpe_4w = None
    # Not enough data yet — would need weekly_pnl_snapshots stored across runs
    # TODO: accumulate weekly PnL in report files for future sharpe computation

    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "avg_duration": avg_duration,
        "turnover": turnover,
        "max_drawdown": max_drawdown,
        "sharpe_4w": sharpe_4w,
        "closed_trades": len(closed),
        "open_positions": len(positions),
        "weeks_running": weeks_running,
        "total_rebalances": total_rebalances,
        "total_trades": total_trades,
        "realized_pnl": stats.get("realized_pnl", 0.0),
    }


# =============================================================================
# Check Engine
# =============================================================================

def check_metric(name, live_value, baseline, weeks_running):
    """
    Compare one live metric against its baseline distribution.

    Returns: {name, status, live, baseline_mean, baseline_std, z_score, detail}
    """
    result = {
        "name": name,
        "status": "INSUFFICIENT_DATA",
        "live": live_value,
        "baseline_mean": baseline.get("mean"),
        "baseline_std": baseline.get("std"),
        "z_score": None,
        "detail": "",
    }

    # Check if baseline exists
    if baseline.get("mean") is None or baseline.get("std") is None:
        result["detail"] = "no baseline data"
        return result

    # Check if live value exists
    if live_value is None:
        min_req = {
            "win_rate": f"{MIN_CLOSED_TRADES} closed trades",
            "avg_return": f"{MIN_CLOSED_TRADES} closed trades",
            "avg_duration": f"{MIN_CLOSED_TRADES} closed trades",
            "turnover": "1 rebalance",
            "max_drawdown": f"{MIN_WEEKS_FOR_DRAWDOWN} weeks",
            "sharpe_4w": f"{MIN_WEEKS_FOR_SHARPE} weeks",
        }
        result["detail"] = f"need {min_req.get(name, 'more data')}"
        return result

    # Compute z-score
    # For win_rate, use SE of proportion instead of per-window stdev
    if name == "win_rate" and "cumulative_p" in baseline:
        p = baseline["cumulative_p"]
        # Use live sample size for SE calculation
        n = max(1, baseline.get("_live_n", MIN_CLOSED_TRADES))
        se = math.sqrt(p * (1 - p) / n) if 0 < p < 1 else 0.001
        z = (live_value - p) / se if se > 0 else 0.0
        result["baseline_mean"] = p
        result["baseline_std"] = se
    else:
        std = baseline["std"]
        if std == 0 or std is None:
            std = 0.001
        z = (live_value - baseline["mean"]) / std

    result["z_score"] = round(z, 2)

    # Map z-score to status
    abs_z = abs(z)
    if abs_z >= SIGMA_PROBLEM:
        status = "PROBLEM"
    elif abs_z >= SIGMA_WARNING:
        status = "WARNING"
    else:
        status = "PASS"

    # Grace period: cap at WARNING for first N weeks
    if weeks_running <= GRACE_PERIOD_WEEKS and status == "PROBLEM":
        status = "WARNING"
        result["detail"] = f"z={z:+.2f} (capped: week {weeks_running}/{GRACE_PERIOD_WEEKS})"
    else:
        result["detail"] = f"z={z:+.2f}"

    result["status"] = status
    return result


def run_all_checks(baselines, live_metrics):
    """Run all metric checks. Returns list of check results."""
    checks = []
    weeks = live_metrics["weeks_running"]

    # Inject live sample size for win_rate SE calculation
    if baselines.get("win_rate"):
        baselines["win_rate"]["_live_n"] = live_metrics["closed_trades"]

    for metric in METRICS:
        bl = baselines.get(metric, {})
        live_val = live_metrics.get(metric)
        checks.append(check_metric(metric, live_val, bl, weeks))

    return checks


def worst_status(checks):
    """Return the worst status across all checks."""
    priority = {"PROBLEM": 3, "WARNING": 2, "PASS": 1, "INSUFFICIENT_DATA": 0}
    worst = max(checks, key=lambda c: priority.get(c["status"], 0))
    status = worst["status"]
    if status == "INSUFFICIENT_DATA":
        return "healthy"  # Not enough data isn't unhealthy
    return {"PASS": "healthy", "WARNING": "warning", "PROBLEM": "problem"}.get(status, "healthy")


# =============================================================================
# False-Positive Tracking
# =============================================================================

def load_fp_tracker():
    """Load false-positive tracker state."""
    return _load_json(FP_TRACKER_FILE) or {
        "alerts": [],
        "total_alerts": 0,
        "false_positives": 0,
        "recalibration_triggered": False,
        "current_thresholds": {"warning": SIGMA_WARNING, "problem": SIGMA_PROBLEM},
    }


def update_fp_tracker(checks, fp_tracker):
    """Update false-positive tracking with new check results."""
    today = datetime.now().strftime("%Y-%m-%d")

    # Find new alerts
    new_alerts = [c for c in checks if c["status"] in ("WARNING", "PROBLEM")]

    # Check if previous alerts resolved (resolved = was WARNING/PROBLEM, now PASS)
    active_metrics = {c["name"] for c in new_alerts}
    for alert in fp_tracker["alerts"]:
        if alert.get("resolved_by"):
            continue
        if alert["metric"] not in active_metrics:
            # Resolved
            alert["resolved_by"] = today
            alert["was_false_positive"] = True
            fp_tracker["false_positives"] += 1

    # Add new alerts (only if not already tracked as active)
    tracked_active = {a["metric"] for a in fp_tracker["alerts"] if not a.get("resolved_by")}
    for c in new_alerts:
        if c["name"] not in tracked_active:
            fp_tracker["alerts"].append({
                "date": today,
                "metric": c["name"],
                "level": c["status"],
                "z_score": c["z_score"],
                "resolved_by": None,
                "was_false_positive": None,
            })
            fp_tracker["total_alerts"] += 1

    # Recalibration check: >4 FP in first month
    if not fp_tracker["recalibration_triggered"] and fp_tracker["false_positives"] > 4:
        fp_tracker["recalibration_triggered"] = True
        fp_tracker["current_thresholds"] = {"warning": 3.0, "problem": 4.0}

    return fp_tracker


def save_fp_tracker(fp_tracker):
    """Save false-positive tracker state."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FP_TRACKER_FILE.write_text(json.dumps(fp_tracker, indent=2))


# =============================================================================
# Output
# =============================================================================

def emit_signal(health_data):
    """Write organism signal file."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    signals = []
    for c in health_data["checks"]:
        if c["status"] == "WARNING":
            signals.append({
                "level": "warning",
                "message": f"XS {c['name']}: {c['live']} (baseline {c['baseline_mean']:.3f} +/- {c['baseline_std']:.3f}, {c['detail']})",
            })
        elif c["status"] == "PROBLEM":
            signals.append({
                "level": "problem",
                "message": f"XS {c['name']}: {c['live']} (baseline {c['baseline_mean']:.3f} +/- {c['baseline_std']:.3f}, {c['detail']})",
            })

    output = {
        "source": "xs_health",
        "timestamp": health_data["timestamp"],
        "status": health_data["status"],
        "signals": signals,
        "meta": {
            "weeks_running": health_data["live_metrics"]["weeks_running"],
            "closed_trades": health_data["live_metrics"]["closed_trades"],
            "checks_passed": sum(1 for c in health_data["checks"] if c["status"] == "PASS"),
            "checks_warned": sum(1 for c in health_data["checks"] if c["status"] == "WARNING"),
            "checks_problem": sum(1 for c in health_data["checks"] if c["status"] == "PROBLEM"),
            "checks_insufficient": sum(1 for c in health_data["checks"] if c["status"] == "INSUFFICIENT_DATA"),
            "report_path": health_data.get("report_path", ""),
        },
    }

    SIGNALS_FILE.write_text(json.dumps(output, indent=2))
    return SIGNALS_FILE


def save_report(health_data):
    """Save weekly report JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORT_DIR / f"{today}.json"
    health_data["report_path"] = str(report_path)
    report_path.write_text(json.dumps(health_data, indent=2, default=str))
    return report_path


def format_console(health_data):
    """Format health check results for console display."""
    live = health_data["live_metrics"]
    baselines_meta = health_data["baselines_meta"]
    checks = health_data["checks"]
    fp = health_data["false_positive_tracker"]

    lines = []
    lines.append("=" * 60)
    lines.append("XS WEEKLY HEALTH CHECK (R044)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Week {live['weeks_running']} | {live['closed_trades']} closed trades | {live['open_positions']} open positions")
    lines.append(f"Baselines (from R002: {baselines_meta['round_trips']} trades, {baselines_meta['windows']} windows)")
    lines.append("")

    for c in checks:
        name = c["name"].ljust(14)
        if c["status"] == "INSUFFICIENT_DATA":
            lines.append(f"  {name}  INSUFFICIENT_DATA ({c['detail']})")
        else:
            live_val = c["live"]
            bl_mean = c["baseline_mean"]
            bl_std = c["baseline_std"]
            z = c["z_score"]

            if c["name"] in ("win_rate",):
                live_str = f"{live_val:.3f}" if live_val is not None else "N/A"
                bl_str = f"{bl_mean:.3f} +/- {bl_std:.3f}"
            elif c["name"] in ("avg_return",):
                live_str = f"{live_val:+.4f}" if live_val is not None else "N/A"
                bl_str = f"{bl_mean:+.4f} +/- {bl_std:.4f}"
            elif c["name"] in ("avg_duration", "turnover"):
                live_str = f"{live_val:.1f}" if live_val is not None else "N/A"
                bl_str = f"{bl_mean:.1f} +/- {bl_std:.1f}"
            else:
                live_str = f"{live_val:.4f}" if live_val is not None else "N/A"
                bl_str = f"{bl_mean:.4f} +/- {bl_std:.4f}"

            status = c["status"]
            lines.append(f"  {name}  {live_str.rjust(8)} (baseline {bl_str})  z={z:+.2f}  {status}")

    lines.append("")
    status_upper = health_data["status"].upper()
    lines.append(f"Status: {status_upper}")
    lines.append("")
    lines.append(f"FP tracker: {fp['total_alerts']} alerts / {fp['false_positives']} FP"
                 + (" [RECALIBRATED]" if fp["recalibration_triggered"] else ""))
    lines.append("=" * 60)

    return "\n".join(lines)


def format_baselines(baselines):
    """Format baseline distributions for console display."""
    lines = []
    lines.append("=" * 60)
    lines.append("XS BASELINE DISTRIBUTIONS (from R002)")
    lines.append("=" * 60)

    meta = baselines.get("meta", {})
    lines.append(f"Source: {meta.get('source', 'N/A')}")
    lines.append(f"Round-trips: {meta.get('round_trips', 'N/A')}")
    lines.append(f"Windows: {meta.get('windows', 'N/A')}")
    lines.append(f"Date range: {meta.get('date_range', 'N/A')}")
    lines.append("")

    for metric in METRICS:
        bl = baselines.get(metric, {})
        mean = bl.get("mean")
        std = bl.get("std")
        n = bl.get("n", 0)

        if mean is None:
            lines.append(f"  {metric.ljust(14)}  no data")
        else:
            lines.append(f"  {metric.ljust(14)}  mean={mean:.4f}  std={std:.4f}  n={n}")

            # Show 2-sigma and 3-sigma bands
            lines.append(f"  {''.ljust(14)}  2-sigma: [{mean - 2*std:.4f}, {mean + 2*std:.4f}]")
            lines.append(f"  {''.ljust(14)}  3-sigma: [{mean - 3*std:.4f}, {mean + 3*std:.4f}]")

    # Win rate special
    wr = baselines.get("win_rate", {})
    if wr.get("cumulative_p") is not None:
        p = wr["cumulative_p"]
        lines.append(f"\n  win_rate (cumulative): p={p:.4f} (SE-based comparison for live)")

    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def run_xs_health_check():
    """Main orchestrator. Returns full health check result dict."""
    baselines = load_or_compute_baselines()
    live = compute_live_metrics()
    checks = run_all_checks(baselines, live)
    status = worst_status(checks)

    fp = load_fp_tracker()
    fp = update_fp_tracker(checks, fp)
    save_fp_tracker(fp)

    return {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "checks": checks,
        "live_metrics": live,
        "baselines_meta": baselines.get("meta", {}),
        "false_positive_tracker": {
            "total_alerts": fp["total_alerts"],
            "false_positives": fp["false_positives"],
            "recalibration_triggered": fp["recalibration_triggered"],
        },
    }


def run_for_evening():
    """Minimal version for evening.py integration. Returns summary dict."""
    try:
        health = run_xs_health_check()
        report_path = save_report(health)
        return {
            "status": health["status"],
            "weeks_running": health["live_metrics"]["weeks_running"],
            "closed_trades": health["live_metrics"]["closed_trades"],
            "checks_passed": sum(1 for c in health["checks"] if c["status"] == "PASS"),
            "checks_warned": sum(1 for c in health["checks"] if c["status"] == "WARNING"),
            "checks_problem": sum(1 for c in health["checks"] if c["status"] == "PROBLEM"),
            "checks_insufficient": sum(1 for c in health["checks"] if c["status"] == "INSUFFICIENT_DATA"),
            "report_path": str(report_path),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="XS weekly health check (R044)")
    parser.add_argument("command", nargs="?", default="check", choices=["check", "baselines"],
                        help="'check' (default) or 'baselines' to show distributions")
    parser.add_argument("--emit-signal", action="store_true", help="Write organism signal")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.command == "baselines":
        baselines = load_or_compute_baselines()
        if args.json:
            print(json.dumps(baselines, indent=2))
        else:
            print(format_baselines(baselines))
        return

    # Run health check
    health = run_xs_health_check()
    report_path = save_report(health)
    health["report_path"] = str(report_path)

    if args.emit_signal:
        sig_path = emit_signal(health)
        if not args.json:
            print(f"Signal written to: {sig_path}")

    if args.json:
        print(json.dumps(health, indent=2, default=str))
    else:
        print(format_console(health))
        print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
