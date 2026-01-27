#!/usr/bin/env python3
"""One-off: backtest 50/50 momentum+bollinger blend with signal exits."""
import sys, json
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from config import load_keys
from router import get_strategy_class
from backtest_strategies import (
    fetch_all_data, run_strategy_backtest, fetch_spy_benchmark, UNIVERSE,
)

k, s = load_keys()
dc = StockHistoricalDataClient(k, s)
start = datetime(2022, 1, 1)
end = datetime(2026, 1, 24)

print("Fetching data...")
all_bars = fetch_all_data(dc, UNIVERSE, start, end)
print(f"Loaded {len(all_bars)} symbols")
spy = fetch_spy_benchmark(dc, start, end)
print(f"SPY: {spy['return_pct']:+.1f}%\n")

signal_exit = {
    "exit_mode": "signal",
    "max_hold_days": 20,
    "stop_atr_multiplier": 0,
    "trailing_stop_enabled": False,
    "profit_giveback_pct": 0,
    "ma_exit_period": None,
}

mom = get_strategy_class("momentum")()
boll = get_strategy_class("bollinger")()

# 50% allocation each
print("Running momentum (50% capital)...")
r_mom = run_strategy_backtest("momentum", mom, signal_exit, all_bars, start, allocation_pct=0.5)
print(f"  {r_mom['total_return_pct']:+.1f}%")

print("Running bollinger (50% capital)...")
r_boll = run_strategy_backtest("bollinger", boll, signal_exit, all_bars, start, allocation_pct=0.5)
print(f"  {r_boll['total_return_pct']:+.1f}%")

# 100% each for comparison
print("Running momentum (100%)...")
r_mom_full = run_strategy_backtest("mom_100", mom, signal_exit, all_bars, start, allocation_pct=1.0)
print("Running bollinger (100%)...")
r_boll_full = run_strategy_backtest("boll_100", boll, signal_exit, all_bars, start, allocation_pct=1.0)

# === Combined equity curve ===
mom_eq = {e["date"]: e["equity"] for e in r_mom["daily_equity"]}
boll_eq = {e["date"]: e["equity"] for e in r_boll["daily_equity"]}
all_dates = sorted(set(mom_eq.keys()) | set(boll_eq.keys()))

combined_equity = []
peak = 100000
max_dd = 0
max_dd_date = None

for d in all_dates:
    m = mom_eq.get(d, 50000)
    b = boll_eq.get(d, 50000)
    total = m + b
    combined_equity.append({"date": d, "equity": total, "mom": m, "boll": b})
    if total > peak:
        peak = total
    dd = (peak - total) / peak * 100
    if dd > max_dd:
        max_dd = dd
        max_dd_date = d

final = combined_equity[-1]["equity"]
total_return = (final - 100000) / 100000 * 100

# Sharpe
daily_returns = []
for i in range(1, len(combined_equity)):
    r = (combined_equity[i]["equity"] - combined_equity[i-1]["equity"]) / combined_equity[i-1]["equity"]
    daily_returns.append(r)
mean_r = sum(daily_returns) / len(daily_returns) if daily_returns else 0
std_r = (sum((r - mean_r)**2 for r in daily_returns) / len(daily_returns))**0.5 if daily_returns else 1
sharpe = (mean_r / std_r) * (252**0.5) if std_r > 0 else 0

# Correlation
mom_rets = {}
boll_rets = {}
for i in range(1, len(r_mom["daily_equity"])):
    d = r_mom["daily_equity"][i]["date"]
    mom_rets[d] = (r_mom["daily_equity"][i]["equity"] - r_mom["daily_equity"][i-1]["equity"]) / r_mom["daily_equity"][i-1]["equity"]
for i in range(1, len(r_boll["daily_equity"])):
    d = r_boll["daily_equity"][i]["date"]
    boll_rets[d] = (r_boll["daily_equity"][i]["equity"] - r_boll["daily_equity"][i-1]["equity"]) / r_boll["daily_equity"][i-1]["equity"]

common = sorted(set(mom_rets.keys()) & set(boll_rets.keys()))
if common:
    mv = [mom_rets[d] for d in common]
    bv = [boll_rets[d] for d in common]
    mm = sum(mv) / len(mv)
    bm = sum(bv) / len(bv)
    cov = sum((a - mm) * (b - bm) for a, b in zip(mv, bv)) / len(mv)
    ms = (sum((a - mm)**2 for a in mv) / len(mv))**0.5
    bs = (sum((b - bm)**2 for b in bv) / len(bv))**0.5
    correlation = cov / (ms * bs) if ms > 0 and bs > 0 else 0
else:
    correlation = 0

# Combined trade stats
combined_trades = r_mom["total_trades"] + r_boll["total_trades"]
combined_wins = (r_mom["win_rate"] * r_mom["total_trades"] + r_boll["win_rate"] * r_boll["total_trades"]) / 100
combined_win_rate = combined_wins / combined_trades * 100 if combined_trades > 0 else 0

# === Print ===
print()
print("=" * 85)
print("50/50 MOMENTUM + BOLLINGER BLEND (signal exit, 20d max hold)")
print("=" * 85)
print()

hdr = f"{'':18} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Win%':>7} {'PF':>7} {'Hold':>7}"
print(hdr)
print("-" * 85)

print(f"{'BLEND 50/50':18} {total_return:>+7.1f}% {sharpe:>7.2f} {max_dd:>7.1f}% "
      f"{combined_trades:>8} {combined_win_rate:>6.1f}%     ---     ---")

for label, r in [("Momentum 100%", r_mom_full), ("Bollinger 100%", r_boll_full)]:
    pf = f"{r['profit_factor']:.2f}" if r["profit_factor"] < 100 else "inf"
    print(f"{label:18} {r['total_return_pct']:>+7.1f}% {r['sharpe']:>7.2f} {r['max_drawdown_pct']:>7.1f}% "
          f"{r['total_trades']:>8} {r['win_rate']:>6.1f}% {pf:>7} {r['avg_hold_days']:>6.1f}d")

print(f"{'SPY Buy & Hold':18} {spy['return_pct']:>+7.1f}%     ---      ---      ---     ---     ---     ---")
print("-" * 85)
print()
print(f"Strategy correlation: {correlation:+.3f}")
print(f"  (Lower = more diversification. 0 = uncorrelated, 1 = identical)")
print()

# Equity milestones
print("Equity curve (quarterly):")
print(f"{'Date':12} {'Blend':>10} {'Mom 50%':>10} {'Boll 50%':>10}")
print("-" * 45)
quarterly = [e for i, e in enumerate(combined_equity) if i % 63 == 0]
quarterly.append(combined_equity[-1])
for e in quarterly:
    print(f"{e['date']:12} ${e['equity']:>9,.0f} ${e['mom']:>9,.0f} ${e['boll']:>9,.0f}")

print()
print(f"Max drawdown: {max_dd:.1f}% on {max_dd_date}")
print(f"  vs Momentum alone:  {r_mom_full['max_drawdown_pct']:.1f}%")
print(f"  vs Bollinger alone: {r_boll_full['max_drawdown_pct']:.1f}%")
print()

# Per-year breakdown
print("Per-year breakdown:")
print(f"{'Year':6} {'Blend':>10} {'Mom':>10} {'Boll':>10}")
print("-" * 40)

for year in [2022, 2023, 2024, 2025]:
    year_start_eq = None
    year_end_eq = None
    year_start_m = None
    year_end_m = None
    year_start_b = None
    year_end_b = None
    for e in combined_equity:
        y = int(e["date"][:4])
        if y == year:
            if year_start_eq is None:
                year_start_eq = e["equity"]
                year_start_m = e["mom"]
                year_start_b = e["boll"]
            year_end_eq = e["equity"]
            year_end_m = e["mom"]
            year_end_b = e["boll"]

    if year_start_eq and year_end_eq:
        br = (year_end_eq - year_start_eq) / year_start_eq * 100
        mr = (year_end_m - year_start_m) / year_start_m * 100 if year_start_m else 0
        bor = (year_end_b - year_start_b) / year_start_b * 100 if year_start_b else 0
        print(f"{year:6} {br:>+9.1f}% {mr:>+9.1f}% {bor:>+9.1f}%")

print()

# Save
save = {
    "blend": {
        "return_pct": total_return, "sharpe": sharpe,
        "max_drawdown_pct": max_dd, "max_drawdown_date": max_dd_date,
        "total_trades": combined_trades, "win_rate": combined_win_rate,
        "correlation": correlation,
    },
    "momentum_100": {k: v for k, v in r_mom_full.items() if k != "daily_equity"},
    "bollinger_100": {k: v for k, v in r_boll_full.items() if k != "daily_equity"},
    "spy": spy,
}
Path("backtest_blend.json").write_text(json.dumps(save, indent=2, default=str))
print("Saved to backtest_blend.json")
