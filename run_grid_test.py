#!/usr/bin/env python3
"""
run_grid_test.py - Cross-sectional factor grid search.

5 factors × 4 turnover configs = 20 combinations.
Reports ranked table sorted by out-of-sample Sharpe.
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import pandas as pd
import numpy as np

from bar_cache import load_bars, SP500_TOP200, cache_symbols
from cross_backtest import (
    load_universe, run_backtest, save_results,
    factor_momentum, factor_mean_reversion, factor_low_volatility,
    factor_volume_momentum, BacktestResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("grid")


# === Composite factor: rank-blend of momentum + low vol ===

def make_composite_factor(universe, date):
    """
    Compute composite scores for all symbols on a given date.
    Returns a dict {symbol: composite_score}.

    Composite = avg of momentum percentile rank + low vol percentile rank.
    """
    mom_scores = {}
    vol_scores = {}

    for sym, df in universe.items():
        bars = df[df["date"] <= pd.Timestamp(date)]
        if len(bars) < 25:
            continue
        m = factor_momentum(bars)
        v = factor_low_volatility(bars)
        if m is not None and np.isfinite(m):
            mom_scores[sym] = m
        if v is not None and np.isfinite(v):
            vol_scores[sym] = v

    # Percentile rank each factor
    common = set(mom_scores.keys()) & set(vol_scores.keys())
    if len(common) < 10:
        return {}

    mom_series = pd.Series({s: mom_scores[s] for s in common})
    vol_series = pd.Series({s: vol_scores[s] for s in common})

    mom_rank = mom_series.rank(pct=True)
    vol_rank = vol_series.rank(pct=True)

    composite = (mom_rank + vol_rank) / 2
    return composite.to_dict()


class CompositeFactorWrapper:
    """
    Wraps the composite factor so it can be used with run_backtest.

    Since run_backtest calls factor_fn(bars_for_one_symbol), but composite
    needs cross-sectional ranks, we pre-compute scores each rebalance date
    and cache them. The factor_fn just looks up the cached score.
    """

    def __init__(self, universe):
        self.universe = universe
        self._cache = {}  # date_str -> {symbol: score}

    def precompute(self, date: str):
        if date not in self._cache:
            self._cache[date] = make_composite_factor(self.universe, date)

    def score(self, symbol: str, date: str) -> float | None:
        scores = self._cache.get(date, {})
        return scores.get(symbol)


def run_backtest_composite(
    universe, start, end, top_n=10, sell_threshold=0,
    rebalance_days=5, initial_capital=100_000, name="composite",
) -> BacktestResult:
    """
    Custom backtest loop for composite factor (needs cross-sectional ranking).
    Mirrors run_backtest but uses CompositeFactorWrapper.
    """
    from cross_backtest import Position, Trade

    comp = CompositeFactorWrapper(universe)

    all_dates = set()
    for df in universe.values():
        all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())
    trading_days = sorted(d for d in all_dates if start <= d <= end)

    capital = initial_capital
    positions = {}
    equity_curve = []
    trades = []
    wins = losses = 0
    gross_profit = gross_loss = 0.0
    total_turnover = 0.0
    rebalance_count = 0

    price_cache = {}
    for sym, df in universe.items():
        price_cache[sym] = dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["close"]))

    def get_price(sym, date):
        return price_cache.get(sym, {}).get(date)

    def portfolio_value(date):
        val = capital
        for pos in positions.values():
            p = get_price(pos.symbol, date)
            if p is not None:
                val += pos.shares * p
        return val

    for day_idx, date in enumerate(trading_days):
        equity = portfolio_value(date)
        equity_curve.append({"date": date, "equity": equity})

        if day_idx % rebalance_days != 0:
            continue

        rebalance_count += 1
        comp.precompute(date)
        scores = comp._cache.get(date, {})

        if len(scores) < top_n:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        buy_set = set(s for s, _ in ranked[:top_n])

        if sell_threshold > 0:
            keep_set = set(s for s, _ in ranked[:sell_threshold])
        else:
            keep_set = buy_set

        target_symbols = buy_set | (set(positions.keys()) & keep_set)

        turnover_value = 0.0
        to_sell = [sym for sym in positions if sym not in target_symbols]
        for sym in to_sell:
            pos = positions.pop(sym)
            price = get_price(sym, date)
            if price is None:
                continue
            pnl = (price - pos.entry_price) * pos.shares
            capital += pos.shares * price
            turnover_value += pos.shares * price
            if pnl > 0:
                wins += 1; gross_profit += pnl
            else:
                losses += 1; gross_loss += abs(pnl)
            trades.append(Trade(sym, "SELL", pos.shares, price, date, "rebalance", pnl))

        target_value = portfolio_value(date) / top_n
        to_buy = [sym for sym in buy_set if sym not in positions]
        for sym in to_buy:
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
            turnover_value += cost
            positions[sym] = Position(sym, shares, price, date)
            trades.append(Trade(sym, "BUY", shares, price, date, f"composite={scores.get(sym, 0):.3f}"))

        total_turnover += turnover_value / max(portfolio_value(date), 1)

    # Close remaining
    final_date = trading_days[-1]
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        price = get_price(sym, final_date)
        if price is None:
            continue
        pnl = (price - pos.entry_price) * pos.shares
        capital += pos.shares * price
        if pnl > 0:
            wins += 1; gross_profit += pnl
        else:
            losses += 1; gross_loss += abs(pnl)
        trades.append(Trade(sym, "SELL", pos.shares, price, final_date, "end", pnl))

    final_equity = capital
    total_return = (final_equity / initial_capital) - 1.0
    years = max(len(trading_days) / 252, 0.1)
    annual_return = (1 + total_return) ** (1 / years) - 1

    eq_df = pd.DataFrame(equity_curve)
    eq_df["return"] = eq_df["equity"].pct_change()
    daily_returns = eq_df["return"].dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    eq_series = eq_df["equity"]
    drawdowns = (eq_series - eq_series.cummax()) / eq_series.cummax()
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0
    avg_turnover = total_turnover / rebalance_count * 100 if rebalance_count > 0 else 0

    spy_df = load_bars("SPY")
    spy_return = 0
    if not spy_df.empty:
        spy_filt = spy_df[(spy_df["date"] >= pd.Timestamp(start)) & (spy_df["date"] <= pd.Timestamp(end))]
        if len(spy_filt) >= 2:
            spy_return = (spy_filt["close"].iloc[-1] / spy_filt["close"].iloc[0]) - 1.0

    return BacktestResult(
        name=name, period=f"{start} to {end}",
        total_return=round(total_return * 100, 2),
        annual_return=round(annual_return * 100, 2),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd * 100, 2),
        profit_factor=round(pf, 3),
        num_trades=total_trades,
        win_rate=round(win_rate, 1),
        avg_trade_pnl=round(avg_pnl, 2),
        turnover_pct=round(avg_turnover, 1),
        spy_return=round(spy_return * 100, 2),
    )


# === Grid definition ===

FACTOR_FNS = {
    "momentum": factor_momentum,
    "mean_rev": factor_mean_reversion,
    "low_vol": factor_low_volatility,
    "vol_mom": factor_volume_momentum,
}

TURNOVER_CONFIGS = [
    {"label": "W/no-band",  "rebalance_days": 5,  "sell_threshold": 0},
    {"label": "W/band",     "rebalance_days": 5,  "sell_threshold": 20},
    {"label": "M/no-band",  "rebalance_days": 21, "sell_threshold": 0},
    {"label": "M/band",     "rebalance_days": 21, "sell_threshold": 20},
]

IS_START, IS_END = "2022-01-03", "2023-12-29"
OOS_START, OOS_END = "2024-01-02", "2025-12-31"
FULL_START, FULL_END = "2022-01-03", "2025-12-31"


def run_grid():
    log.info("Loading universe...")
    universe = load_universe(SP500_TOP200, "2022-01-01", "2025-12-31")
    log.info(f"Loaded {len(universe)} symbols")

    # Ensure SPY cached
    cache_symbols(["SPY"], "2022-01-01", "2026-01-27")

    rows = []
    total = (len(FACTOR_FNS) + 1) * len(TURNOVER_CONFIGS)  # +1 for composite
    done = 0
    t0 = time.time()

    # Standard factors
    for factor_name, factor_fn in FACTOR_FNS.items():
        for tc in TURNOVER_CONFIGS:
            done += 1
            label = f"{factor_name}/{tc['label']}"
            log.info(f"[{done}/{total}] {label}")

            is_r = run_backtest(universe, factor_fn, IS_START, IS_END,
                                top_n=10, sell_threshold=tc["sell_threshold"],
                                rebalance_days=tc["rebalance_days"], name=label)
            oos_r = run_backtest(universe, factor_fn, OOS_START, OOS_END,
                                 top_n=10, sell_threshold=tc["sell_threshold"],
                                 rebalance_days=tc["rebalance_days"], name=label)
            full_r = run_backtest(universe, factor_fn, FULL_START, FULL_END,
                                  top_n=10, sell_threshold=tc["sell_threshold"],
                                  rebalance_days=tc["rebalance_days"], name=label)

            rows.append({
                "factor": factor_name,
                "turnover": tc["label"],
                "is_return": is_r.total_return,
                "is_sharpe": is_r.sharpe,
                "is_maxdd": is_r.max_drawdown,
                "oos_return": oos_r.total_return,
                "oos_sharpe": oos_r.sharpe,
                "oos_maxdd": oos_r.max_drawdown,
                "full_return": full_r.total_return,
                "full_spy": full_r.spy_return,
                "full_alpha": round(full_r.total_return - full_r.spy_return, 2),
                "turnover_pct": full_r.turnover_pct,
                "pf": full_r.profit_factor,
                "trades": full_r.num_trades,
                "win_rate": full_r.win_rate,
            })

    # Composite factor
    for tc in TURNOVER_CONFIGS:
        done += 1
        label = f"composite/{tc['label']}"
        log.info(f"[{done}/{total}] {label}")

        is_r = run_backtest_composite(universe, IS_START, IS_END,
                                       top_n=10, sell_threshold=tc["sell_threshold"],
                                       rebalance_days=tc["rebalance_days"], name=label)
        oos_r = run_backtest_composite(universe, OOS_START, OOS_END,
                                        top_n=10, sell_threshold=tc["sell_threshold"],
                                        rebalance_days=tc["rebalance_days"], name=label)
        full_r = run_backtest_composite(universe, FULL_START, FULL_END,
                                         top_n=10, sell_threshold=tc["sell_threshold"],
                                         rebalance_days=tc["rebalance_days"], name=label)

        rows.append({
            "factor": "composite",
            "turnover": tc["label"],
            "is_return": is_r.total_return,
            "is_sharpe": is_r.sharpe,
            "is_maxdd": is_r.max_drawdown,
            "oos_return": oos_r.total_return,
            "oos_sharpe": oos_r.sharpe,
            "oos_maxdd": oos_r.max_drawdown,
            "full_return": full_r.total_return,
            "full_spy": full_r.spy_return,
            "full_alpha": round(full_r.total_return - full_r.spy_return, 2),
            "turnover_pct": full_r.turnover_pct,
            "pf": full_r.profit_factor,
            "trades": full_r.num_trades,
            "win_rate": full_r.win_rate,
        })

    elapsed = time.time() - t0
    log.info(f"Grid complete: {total} combos in {elapsed:.0f}s")

    # Sort by OOS Sharpe descending
    rows.sort(key=lambda r: r["oos_sharpe"], reverse=True)

    # Print table
    print(f"\n{'CROSS-SECTIONAL FACTOR GRID SEARCH':^110}")
    print(f"{'Sorted by out-of-sample Sharpe':^110}")
    print(f"{'Universe: top 200 S&P500 | Top 10 holdings | $100k initial':^110}\n")

    header = (f"{'#':>2} {'Factor':<12} {'Turnover':<12} "
              f"{'IS Ret%':>8} {'IS Shp':>7} {'IS DD%':>7} "
              f"{'OOS Ret%':>9} {'OOS Shp':>8} {'OOS DD%':>8} "
              f"{'Full%':>7} {'SPY%':>6} {'Alpha%':>7} "
              f"{'TO%':>5} {'PF':>6} {'Win%':>5}")
    print(header)
    print("-" * len(header))

    for i, r in enumerate(rows):
        print(f"{i+1:>2} {r['factor']:<12} {r['turnover']:<12} "
              f"{r['is_return']:>+7.1f}% {r['is_sharpe']:>7.3f} {r['is_maxdd']:>6.1f}% "
              f"{r['oos_return']:>+8.1f}% {r['oos_sharpe']:>8.3f} {r['oos_maxdd']:>7.1f}% "
              f"{r['full_return']:>+6.1f}% {r['full_spy']:>5.1f}% {r['full_alpha']:>+6.1f}% "
              f"{r['turnover_pct']:>5.0f} {r['pf']:>6.3f} {r['win_rate']:>4.1f}%")

    # Save
    output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "universe_size": len(universe),
            "top_n": 10,
            "factors": list(FACTOR_FNS.keys()) + ["composite"],
            "turnover_configs": TURNOVER_CONFIGS,
        },
        "results": rows,
        "elapsed_seconds": round(elapsed, 1),
    }
    outfile = Path(__file__).parent / "grid_results.json"
    outfile.write_text(json.dumps(output, indent=2))
    log.info(f"Saved to {outfile}")


if __name__ == "__main__":
    run_grid()
