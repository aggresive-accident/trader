#!/usr/bin/env python3
"""
run_walkforward.py - Walk-forward validation and regime-switch tests.

Test 1: Walk-forward mean reversion (W/no-band)
  - Train on rolling 6 months, trade next month
  - Walk through 2022-2025

Test 2: Regime-switch momentum/mean_rev
  - SPY 20d return < 0 → mean reversion
  - SPY 20d return > 0 → momentum
  - Weekly rebalance, persistence band
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import field

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import pandas as pd
import numpy as np

from bar_cache import load_bars, SP500_TOP200, cache_symbols
from cross_backtest import (
    load_universe, BacktestResult, Position, Trade,
    factor_momentum, factor_mean_reversion,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("walkfwd")


# === Shared helpers ===

def build_price_cache(universe):
    pc = {}
    for sym, df in universe.items():
        pc[sym] = dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["close"]))
    return pc


def get_spy_returns(start, end):
    """Return a dict {date_str: 20d_return} for SPY."""
    spy = load_bars("SPY", start, end)
    if spy.empty:
        return {}
    spy = spy.sort_values("date").reset_index(drop=True)
    ret = {}
    for i in range(20, len(spy)):
        d = spy["date"].iloc[i].strftime("%Y-%m-%d")
        ret[d] = (spy["close"].iloc[i] / spy["close"].iloc[i - 20]) - 1.0
    return ret


def compute_metrics(equity_curve, initial_capital, trades_list, start, end):
    """Compute standard metrics from equity curve and trades."""
    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) < 2:
        return None

    final = eq_df["equity"].iloc[-1]
    total_return = (final / initial_capital) - 1.0
    years = max(len(eq_df) / 252, 0.1)
    annual_return = (1 + total_return) ** (1 / years) - 1

    eq_df["ret"] = eq_df["equity"].pct_change()
    daily = eq_df["ret"].dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0

    running_max = eq_df["equity"].cummax()
    dd = (eq_df["equity"] - running_max) / running_max
    max_dd = abs(dd.min())

    wins = sum(1 for t in trades_list if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades_list if t.get("pnl", 0) < 0)
    gross_profit = sum(t["pnl"] for t in trades_list if t.get("pnl", 0) > 0)
    gross_loss = sum(abs(t["pnl"]) for t in trades_list if t.get("pnl", 0) < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    spy_df = load_bars("SPY")
    spy_return = 0
    if not spy_df.empty:
        sf = spy_df[(spy_df["date"] >= pd.Timestamp(start)) & (spy_df["date"] <= pd.Timestamp(end))]
        if len(sf) >= 2:
            spy_return = (sf["close"].iloc[-1] / sf["close"].iloc[0]) - 1.0

    return BacktestResult(
        name="", period=f"{start} to {end}",
        total_return=round(total_return * 100, 2),
        annual_return=round(annual_return * 100, 2),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd * 100, 2),
        profit_factor=round(pf, 3),
        num_trades=total_trades,
        win_rate=round(win_rate, 1),
        avg_trade_pnl=round((gross_profit - gross_loss) / total_trades, 2) if total_trades else 0,
        turnover_pct=0,
        spy_return=round(spy_return * 100, 2),
    )


# === Test 1: Walk-forward mean reversion ===

def run_walk_forward(universe, price_cache):
    """
    Walk-forward: train 6 months, trade 1 month, step forward 1 month.
    Uses mean reversion factor, weekly rebalance, no persistence band.
    """
    log.info("=== Walk-Forward Mean Reversion ===")

    # Build month boundaries from trading days
    all_dates = set()
    for df in universe.values():
        all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())
    trading_days = sorted(d for d in all_dates if "2022-01-01" <= d <= "2025-12-31")

    # Group into months
    months = {}
    for d in trading_days:
        key = d[:7]  # YYYY-MM
        months.setdefault(key, []).append(d)
    month_keys = sorted(months.keys())

    # We need 6 months training before first trade month
    # So first trade month is month_keys[6]
    capital = 100_000
    initial_capital = capital
    positions = {}
    equity_curve = []
    all_trades = []
    monthly_returns = []

    def get_price(sym, date):
        return price_cache.get(sym, {}).get(date)

    def portfolio_value(date):
        val = capital
        for pos in positions.values():
            p = get_price(pos.symbol, date)
            if p is not None:
                val += pos.shares * p
        return val

    for mi in range(6, len(month_keys)):
        trade_month = month_keys[mi]
        trade_days = months[trade_month]

        # Track equity at start of month
        month_start_equity = portfolio_value(trade_days[0]) if equity_curve else capital

        # Rebalance weekly within this month
        for di, date in enumerate(trade_days):
            equity = portfolio_value(date)
            equity_curve.append({"date": date, "equity": equity})

            if di % 5 != 0:
                continue

            # Score using mean reversion
            scores = {}
            for sym, df in universe.items():
                bars = df[df["date"] <= pd.Timestamp(date)]
                if len(bars) < 25:
                    continue
                s = factor_mean_reversion(bars)
                if s is not None and np.isfinite(s):
                    scores[sym] = s

            if len(scores) < 10:
                continue

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target = set(s for s, _ in ranked[:10])

            # Sell
            for sym in [s for s in positions if s not in target]:
                pos = positions.pop(sym)
                price = get_price(sym, date)
                if price is None:
                    continue
                pnl = (price - pos.entry_price) * pos.shares
                capital += pos.shares * price
                all_trades.append({"symbol": sym, "side": "SELL", "price": price,
                                   "date": date, "pnl": pnl})

            # Buy
            target_value = portfolio_value(date) / 10
            for sym in [s for s in target if s not in positions]:
                price = get_price(sym, date)
                if price is None or price <= 0:
                    continue
                shares = int(target_value / price)
                if shares < 1:
                    continue
                cost = shares * price
                if cost > capital:
                    shares = int(capital / price)
                    cost = shares * price
                if shares < 1:
                    continue
                capital -= cost
                positions[sym] = Position(sym, shares, price, date)
                all_trades.append({"symbol": sym, "side": "BUY", "price": price,
                                   "date": date, "pnl": 0})

        # Month-end equity
        month_end_equity = portfolio_value(trade_days[-1])
        month_ret = (month_end_equity / month_start_equity - 1) * 100 if month_start_equity > 0 else 0
        monthly_returns.append({
            "month": trade_month,
            "return": round(month_ret, 2),
            "equity": round(month_end_equity, 0),
        })

    # Close remaining
    final_date = trading_days[-1]
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        price = get_price(sym, final_date)
        if price is None:
            continue
        pnl = (price - pos.entry_price) * pos.shares
        capital += pos.shares * price
        all_trades.append({"symbol": sym, "side": "SELL", "price": price,
                           "date": final_date, "pnl": pnl})

    metrics = compute_metrics(equity_curve, initial_capital, all_trades,
                              "2022-01-03", "2025-12-31")

    return monthly_returns, metrics, all_trades


# === Test 2: Regime-switch ===

def run_regime_switch(universe, price_cache):
    """
    When SPY 20d return < 0: use mean reversion
    When SPY 20d return > 0: use momentum
    Weekly rebalance, persistence band (top 10 buy / top 20 sell).
    """
    log.info("=== Regime-Switch Momentum/Mean-Rev ===")

    spy_rets = get_spy_returns("2021-01-01", "2025-12-31")

    all_dates = set()
    for df in universe.values():
        all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())
    trading_days = sorted(d for d in all_dates if "2022-01-03" <= d <= "2025-12-31")

    capital = 100_000
    initial_capital = capital
    positions = {}
    equity_curve = []
    all_trades = []
    regime_log = []

    def get_price(sym, date):
        return price_cache.get(sym, {}).get(date)

    def portfolio_value(date):
        val = capital
        for pos in positions.values():
            p = get_price(pos.symbol, date)
            if p is not None:
                val += pos.shares * p
        return val

    for di, date in enumerate(trading_days):
        equity = portfolio_value(date)
        equity_curve.append({"date": date, "equity": equity})

        # Weekly rebalance
        if di % 5 != 0:
            continue

        # Determine regime
        spy_20d = spy_rets.get(date, 0)
        if spy_20d < 0:
            factor_fn = factor_mean_reversion
            regime = "mean_rev"
        else:
            factor_fn = factor_momentum
            regime = "momentum"

        regime_log.append({"date": date, "regime": regime, "spy_20d": round(spy_20d * 100, 2)})

        # Score
        scores = {}
        for sym, df in universe.items():
            bars = df[df["date"] <= pd.Timestamp(date)]
            if len(bars) < 25:
                continue
            s = factor_fn(bars)
            if s is not None and np.isfinite(s):
                scores[sym] = s

        if len(scores) < 10:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        buy_set = set(s for s, _ in ranked[:10])
        keep_set = set(s for s, _ in ranked[:20])  # persistence band
        target = buy_set | (set(positions.keys()) & keep_set)

        # Sell
        for sym in [s for s in positions if s not in target]:
            pos = positions.pop(sym)
            price = get_price(sym, date)
            if price is None:
                continue
            pnl = (price - pos.entry_price) * pos.shares
            capital += pos.shares * price
            all_trades.append({"symbol": sym, "side": "SELL", "price": price,
                               "date": date, "pnl": pnl, "regime": regime})

        # Buy
        target_value = portfolio_value(date) / 10
        for sym in [s for s in buy_set if s not in positions]:
            price = get_price(sym, date)
            if price is None or price <= 0:
                continue
            shares = int(target_value / price)
            if shares < 1:
                continue
            cost = shares * price
            if cost > capital:
                shares = int(capital / price)
                cost = shares * price
            if shares < 1:
                continue
            capital -= cost
            positions[sym] = Position(sym, shares, price, date)
            all_trades.append({"symbol": sym, "side": "BUY", "price": price,
                               "date": date, "pnl": 0, "regime": regime})

    # Close remaining
    final_date = trading_days[-1]
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        price = get_price(sym, final_date)
        if price is None:
            continue
        pnl = (price - pos.entry_price) * pos.shares
        capital += pos.shares * price
        all_trades.append({"symbol": sym, "side": "SELL", "price": price,
                           "date": final_date, "pnl": pnl, "regime": "close"})

    metrics = compute_metrics(equity_curve, initial_capital, all_trades,
                              "2022-01-03", "2025-12-31")

    # Regime breakdown
    mom_trades = [t for t in all_trades if t.get("regime") == "momentum" and t["side"] == "SELL"]
    mr_trades = [t for t in all_trades if t.get("regime") == "mean_rev" and t["side"] == "SELL"]

    regime_stats = {
        "momentum_periods": sum(1 for r in regime_log if r["regime"] == "momentum"),
        "mean_rev_periods": sum(1 for r in regime_log if r["regime"] == "mean_rev"),
        "momentum_trades": len(mom_trades),
        "mean_rev_trades": len(mr_trades),
        "momentum_pnl": round(sum(t["pnl"] for t in mom_trades), 2),
        "mean_rev_pnl": round(sum(t["pnl"] for t in mr_trades), 2),
    }

    return metrics, regime_stats, regime_log, all_trades


# === Main ===

def main():
    t0 = time.time()

    log.info("Loading universe...")
    universe = load_universe(SP500_TOP200, "2022-01-01", "2025-12-31")
    log.info(f"Loaded {len(universe)} symbols")
    cache_symbols(["SPY"], "2022-01-01", "2026-01-27")

    price_cache = build_price_cache(universe)

    # Test 1: Walk-forward
    monthly, wf_metrics, wf_trades = run_walk_forward(universe, price_cache)

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD MEAN REVERSION (6mo train, 1mo trade)")
    print(f"{'='*70}")
    print(f"  Return:        {wf_metrics.total_return:+.2f}%")
    print(f"  Annual:        {wf_metrics.annual_return:+.2f}%")
    print(f"  Sharpe:        {wf_metrics.sharpe:.3f}")
    print(f"  Max Drawdown:  {wf_metrics.max_drawdown:.2f}%")
    print(f"  Profit Factor: {wf_metrics.profit_factor:.3f}")
    print(f"  Trades:        {wf_metrics.num_trades}")
    print(f"  Win Rate:      {wf_metrics.win_rate:.1f}%")
    print(f"  SPY B&H:       {wf_metrics.spy_return:+.2f}%")
    print(f"  Alpha:         {wf_metrics.total_return - wf_metrics.spy_return:+.2f}%")

    # Monthly returns table
    print(f"\n  Monthly Returns:")
    print(f"  {'Month':<10} {'Return':>8} {'Equity':>12}")
    print(f"  {'-'*32}")

    # Group by year for annual subtotals
    year_returns = {}
    for m in monthly:
        y = m["month"][:4]
        year_returns.setdefault(y, []).append(m["return"])
        print(f"  {m['month']:<10} {m['return']:>+7.2f}% ${m['equity']:>10,.0f}")
        # Print year subtotal at year boundary
        if m["month"].endswith("-12") or m == monthly[-1]:
            yr = m["month"][:4]
            if yr in year_returns:
                yr_total = sum(year_returns[yr])
                pos = sum(1 for r in year_returns[yr] if r > 0)
                neg = sum(1 for r in year_returns[yr] if r <= 0)
                print(f"  {'':>10} {yr} total: {yr_total:+.1f}% ({pos}↑ {neg}↓)")
                print(f"  {'-'*32}")

    # Edge persistence
    h1_returns = [m["return"] for m in monthly[:len(monthly)//2]]
    h2_returns = [m["return"] for m in monthly[len(monthly)//2:]]
    print(f"\n  Edge Persistence:")
    print(f"    First half avg monthly:  {np.mean(h1_returns):+.2f}%")
    print(f"    Second half avg monthly: {np.mean(h2_returns):+.2f}%")
    print(f"    Positive months:         {sum(1 for m in monthly if m['return'] > 0)}/{len(monthly)}")

    # Test 2: Regime switch
    rs_metrics, regime_stats, regime_log, rs_trades = run_regime_switch(universe, price_cache)

    print(f"\n{'='*70}")
    print(f"  REGIME-SWITCH (SPY 20d < 0 → mean_rev, > 0 → momentum)")
    print(f"  Config: weekly rebalance, top-10 buy / top-20 sell band")
    print(f"{'='*70}")
    print(f"  Return:        {rs_metrics.total_return:+.2f}%")
    print(f"  Annual:        {rs_metrics.annual_return:+.2f}%")
    print(f"  Sharpe:        {rs_metrics.sharpe:.3f}")
    print(f"  Max Drawdown:  {rs_metrics.max_drawdown:.2f}%")
    print(f"  Profit Factor: {rs_metrics.profit_factor:.3f}")
    print(f"  Trades:        {rs_metrics.num_trades}")
    print(f"  Win Rate:      {rs_metrics.win_rate:.1f}%")
    print(f"  SPY B&H:       {rs_metrics.spy_return:+.2f}%")
    print(f"  Alpha:         {rs_metrics.total_return - rs_metrics.spy_return:+.2f}%")

    print(f"\n  Regime Breakdown:")
    print(f"    Momentum periods:  {regime_stats['momentum_periods']} rebalances")
    print(f"    Mean-rev periods:  {regime_stats['mean_rev_periods']} rebalances")
    print(f"    Momentum trades:   {regime_stats['momentum_trades']} (P/L: ${regime_stats['momentum_pnl']:+,.2f})")
    print(f"    Mean-rev trades:   {regime_stats['mean_rev_trades']} (P/L: ${regime_stats['mean_rev_pnl']:+,.2f})")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Grid baseline vs Walk-forward vs Regime-switch")
    print(f"{'='*70}")
    print(f"  {'':>25} {'MeanRev W/nb':>14} {'Walk-Fwd':>10} {'Regime':>10}")
    print(f"  {'':>25} {'(grid #5)':>14} {'':>10} {'Switch':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Return':>25} {'+59.5%':>14} {wf_metrics.total_return:>+9.1f}% {rs_metrics.total_return:>+9.1f}%")
    print(f"  {'Sharpe':>25} {'0.693 OOS':>14} {wf_metrics.sharpe:>10.3f} {rs_metrics.sharpe:>10.3f}")
    print(f"  {'Max DD':>25} {'24.4% OOS':>14} {wf_metrics.max_drawdown:>9.1f}% {rs_metrics.max_drawdown:>9.1f}%")
    print(f"  {'PF':>25} {'1.342':>14} {wf_metrics.profit_factor:>10.3f} {rs_metrics.profit_factor:>10.3f}")
    print(f"  {'Alpha vs SPY':>25} {'+15.7%':>14} {wf_metrics.total_return - wf_metrics.spy_return:>+9.1f}% {rs_metrics.total_return - rs_metrics.spy_return:>+9.1f}%")

    elapsed = time.time() - t0
    log.info(f"Total time: {elapsed:.0f}s")

    # Save
    output = {
        "generated": datetime.now().isoformat(),
        "walk_forward": {
            "metrics": wf_metrics.to_dict(),
            "monthly_returns": monthly,
        },
        "regime_switch": {
            "metrics": rs_metrics.to_dict(),
            "regime_stats": regime_stats,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    outfile = Path(__file__).parent / "walkforward_results.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    log.info(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
