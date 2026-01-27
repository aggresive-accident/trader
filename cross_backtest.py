#!/usr/bin/env python3
"""
cross_backtest.py - Cross-sectional factor backtester.

Ranks a universe of stocks by a factor, goes long top N, rebalances weekly.

Usage:
  python3 cross_backtest.py                     # run momentum factor
  python3 cross_backtest.py --top-n 20          # hold 20 stocks
  python3 cross_backtest.py --factor mean_rev   # use mean reversion factor

Factors:
  momentum  - 20-day return
  mean_rev  - inverse 20-day return (buy losers)
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import pandas as pd
import numpy as np

from bar_cache import load_bars, SP500_TOP200

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cross_bt")

RESULTS_DIR = Path(__file__).parent
TRADE_LOG_FILE = RESULTS_DIR / "cross_backtest_trades.jsonl"


# === Factor functions ===
# Each takes a DataFrame with date/open/high/low/close/volume columns
# and returns a float score. Higher = stronger signal to go long.

def factor_momentum(df: pd.DataFrame, period: int = 20) -> float | None:
    """20-day return."""
    if len(df) < period + 1:
        return None
    return (df["close"].iloc[-1] / df["close"].iloc[-period - 1]) - 1.0


def factor_mean_reversion(df: pd.DataFrame, period: int = 20) -> float | None:
    """Inverse 20-day return (buy losers)."""
    m = factor_momentum(df, period)
    return -m if m is not None else None


def factor_low_volatility(df: pd.DataFrame, period: int = 20) -> float | None:
    """Inverse 20-day standard deviation of returns. Lower vol = higher score."""
    if len(df) < period + 1:
        return None
    returns = df["close"].pct_change().iloc[-period:]
    std = returns.std()
    if std <= 0 or not np.isfinite(std):
        return None
    return -std  # negative so lower vol ranks higher


def factor_volume_momentum(df: pd.DataFrame, period: int = 20) -> float | None:
    """20-day volume change (recent avg vs prior avg)."""
    if len(df) < period * 2:
        return None
    recent = df["volume"].iloc[-period:].mean()
    prior = df["volume"].iloc[-period * 2:-period].mean()
    if prior <= 0:
        return None
    return (recent / prior) - 1.0


def factor_composite(df: pd.DataFrame, period: int = 20) -> float | None:
    """Placeholder - composite is handled via rank blending in the grid runner."""
    # This won't be called directly; the grid runner computes rank-blended composites
    return factor_momentum(df, period)


FACTORS = {
    "momentum": factor_momentum,
    "mean_rev": factor_mean_reversion,
    "low_vol": factor_low_volatility,
    "vol_mom": factor_volume_momentum,
    "composite": factor_composite,
}


# === Data loader ===

def load_universe(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load all cached bars for universe, filtered to date range with warmup."""
    universe = {}
    # Load with warmup buffer
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    for sym in symbols:
        df = load_bars(sym, warmup_start, end)
        if len(df) >= 30:  # need minimum history
            universe[sym] = df
    return universe


# === Core backtester ===

@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_date: str


@dataclass
class Trade:
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: float
    price: float
    date: str
    reason: str
    pnl: float = 0.0


@dataclass
class BacktestResult:
    name: str
    period: str
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    num_trades: int
    win_rate: float
    avg_trade_pnl: float
    turnover_pct: float  # avg weekly turnover as % of portfolio
    spy_return: float
    equity_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("equity_curve", "trades")}


def run_backtest(
    universe: dict[str, pd.DataFrame],
    factor_fn,
    start: str,
    end: str,
    top_n: int = 10,
    sell_threshold: int = 0,
    rebalance_days: int = 5,
    initial_capital: float = 100_000,
    name: str = "backtest",
) -> BacktestResult:
    """
    Run a cross-sectional factor backtest.

    On each rebalance date:
      1. Score all symbols with factor_fn
      2. Rank, pick top N
      3. Equal-weight long portfolio
      4. Sell positions not in top N, buy new ones
    """
    # Build a unified date index from any symbol
    all_dates = set()
    for df in universe.values():
        all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())
    trading_days = sorted(d for d in all_dates if start <= d <= end)

    if not trading_days:
        raise ValueError(f"No trading days between {start} and {end}")

    capital = initial_capital
    positions: dict[str, Position] = {}
    equity_curve = []
    trades: list[Trade] = []
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_turnover = 0.0
    rebalance_count = 0
    peak_equity = initial_capital

    # Price lookup: symbol -> {date_str -> close}
    price_cache: dict[str, dict[str, float]] = {}
    for sym, df in universe.items():
        price_cache[sym] = dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["close"]))

    def get_price(sym: str, date: str) -> float | None:
        return price_cache.get(sym, {}).get(date)

    def portfolio_value(date: str) -> float:
        val = capital
        for pos in positions.values():
            p = get_price(pos.symbol, date)
            if p is not None:
                val += pos.shares * p
        return val

    for day_idx, date in enumerate(trading_days):
        # Mark-to-market
        equity = portfolio_value(date)
        equity_curve.append({"date": date, "equity": equity})
        peak_equity = max(peak_equity, equity)

        # Rebalance every N trading days
        if day_idx % rebalance_days != 0:
            continue

        rebalance_count += 1

        # Score all symbols using bars up to current date
        scores = {}
        for sym, df in universe.items():
            bars_to_date = df[df["date"] <= pd.Timestamp(date)]
            if len(bars_to_date) < 25:
                continue
            score = factor_fn(bars_to_date)
            if score is not None and np.isfinite(score):
                scores[sym] = score

        if len(scores) < top_n:
            continue

        # Rank: top N by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        buy_set = set(s for s, _ in ranked[:top_n])

        # Persistence band: sell only if dropped below sell_threshold rank
        # If sell_threshold == 0, no band (sell if not in top_n)
        if sell_threshold > 0:
            keep_set = set(s for s, _ in ranked[:sell_threshold])
        else:
            keep_set = buy_set

        target_symbols = buy_set | (set(positions.keys()) & keep_set)

        # Sell positions not in target
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
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

            trades.append(Trade(sym, "SELL", pos.shares, price, date,
                                f"dropped from top {top_n}", pnl))

        # Buy new positions (equal weight)
        target_value = portfolio_value(date) / top_n
        to_buy = [sym for sym in target_symbols if sym not in positions]

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
            trades.append(Trade(sym, "BUY", shares, price, date,
                                f"entered top {top_n} (score={scores[sym]:+.4f})"))

        total_turnover += turnover_value / max(portfolio_value(date), 1)

    # Close remaining positions at final prices
    final_date = trading_days[-1]
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        price = get_price(sym, final_date)
        if price is None:
            continue
        pnl = (price - pos.entry_price) * pos.shares
        capital += pos.shares * price

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

        trades.append(Trade(sym, "SELL", pos.shares, price, final_date,
                            f"end of backtest", pnl))

    # Compute metrics
    final_equity = capital
    total_return = (final_equity / initial_capital) - 1.0
    years = max(len(trading_days) / 252, 0.1)
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Sharpe from daily returns
    eq_df = pd.DataFrame(equity_curve)
    eq_df["return"] = eq_df["equity"].pct_change()
    daily_returns = eq_df["return"].dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    # Max drawdown
    eq_series = eq_df["equity"]
    running_max = eq_series.cummax()
    drawdowns = (eq_series - running_max) / running_max
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

    # Profit factor
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0
    avg_turnover = total_turnover / rebalance_count * 100 if rebalance_count > 0 else 0

    # SPY benchmark
    spy_df = load_bars("SPY")
    if spy_df.empty:
        # Try to load from cache, otherwise fetch
        spy_return = 0
    else:
        spy_start = spy_df[spy_df["date"] >= pd.Timestamp(start)]
        spy_end = spy_start[spy_start["date"] <= pd.Timestamp(end)]
        if len(spy_end) >= 2:
            spy_return = (spy_end["close"].iloc[-1] / spy_end["close"].iloc[0]) - 1.0
        else:
            spy_return = 0

    return BacktestResult(
        name=name,
        period=f"{start} to {end}",
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
        equity_curve=equity_curve,
        trades=[t.__dict__ for t in trades],
    )


def print_result(r: BacktestResult):
    """Pretty-print a backtest result."""
    print(f"\n{'='*60}")
    print(f"  {r.name} | {r.period}")
    print(f"{'='*60}")
    print(f"  Return:        {r.total_return:+.2f}%")
    print(f"  Annual:        {r.annual_return:+.2f}%")
    print(f"  Sharpe:        {r.sharpe:.3f}")
    print(f"  Max Drawdown:  {r.max_drawdown:.2f}%")
    print(f"  Profit Factor: {r.profit_factor:.3f}")
    print(f"  Trades:        {r.num_trades}")
    print(f"  Win Rate:      {r.win_rate:.1f}%")
    print(f"  Avg Trade P/L: ${r.avg_trade_pnl:.2f}")
    print(f"  Turnover/Reb:  {r.turnover_pct:.1f}%")
    print(f"  SPY B&H:       {r.spy_return:+.2f}%")
    alpha = r.total_return - r.spy_return
    print(f"  Alpha vs SPY:  {alpha:+.2f}%")
    print(f"{'='*60}")


def save_results(results: list[BacktestResult], filename: str):
    """Save results to JSON."""
    path = RESULTS_DIR / filename
    data = {
        "generated": datetime.now().isoformat(),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(data, indent=2))
    log.info(f"Saved results to {path}")


def save_trades(trades: list[dict], filename: str = None):
    """Save trade log."""
    path = TRADE_LOG_FILE if filename is None else RESULTS_DIR / filename
    with open(path, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


# === Main ===

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-sectional factor backtester")
    parser.add_argument("--factor", default="momentum", choices=list(FACTORS.keys()))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rebalance", type=int, default=5, help="Rebalance every N trading days")
    parser.add_argument("--capital", type=float, default=100_000)
    args = parser.parse_args()

    factor_fn = FACTORS[args.factor]

    # Load universe
    log.info(f"Loading universe: {len(SP500_TOP200)} symbols...")
    universe = load_universe(SP500_TOP200, "2022-01-01", "2025-12-31")
    log.info(f"Loaded {len(universe)} symbols with sufficient data")

    # Also cache SPY for benchmark
    from bar_cache import cache_symbols
    cache_symbols(["SPY"], "2022-01-01", "2026-01-27")

    # In-sample: 2022-2023
    log.info("Running in-sample (2022-2023)...")
    is_result = run_backtest(
        universe, factor_fn,
        start="2022-01-03", end="2023-12-29",
        top_n=args.top_n, rebalance_days=args.rebalance,
        initial_capital=args.capital,
        name=f"XS {args.factor} top-{args.top_n} [IN-SAMPLE 2022-2023]",
    )

    # Out-of-sample: 2024-2025
    log.info("Running out-of-sample (2024-2025)...")
    oos_result = run_backtest(
        universe, factor_fn,
        start="2024-01-02", end="2025-12-31",
        top_n=args.top_n, rebalance_days=args.rebalance,
        initial_capital=args.capital,
        name=f"XS {args.factor} top-{args.top_n} [OUT-OF-SAMPLE 2024-2025]",
    )

    # Full period
    log.info("Running full period (2022-2025)...")
    full_result = run_backtest(
        universe, factor_fn,
        start="2022-01-03", end="2025-12-31",
        top_n=args.top_n, rebalance_days=args.rebalance,
        initial_capital=args.capital,
        name=f"XS {args.factor} top-{args.top_n} [FULL 2022-2025]",
    )

    # Print results
    print_result(is_result)
    print_result(oos_result)
    print_result(full_result)

    # Summary comparison
    print(f"\n{'VALIDATION SPLIT COMPARISON':^60}")
    print(f"{'':>25} {'In-Sample':>12} {'Out-of-Sample':>14} {'Full':>10}")
    print(f"{'Return':>25} {is_result.total_return:>+11.2f}% {oos_result.total_return:>+13.2f}% {full_result.total_return:>+9.2f}%")
    print(f"{'Sharpe':>25} {is_result.sharpe:>12.3f} {oos_result.sharpe:>14.3f} {full_result.sharpe:>10.3f}")
    print(f"{'Max DD':>25} {is_result.max_drawdown:>11.2f}% {oos_result.max_drawdown:>13.2f}% {full_result.max_drawdown:>9.2f}%")
    print(f"{'Profit Factor':>25} {is_result.profit_factor:>12.3f} {oos_result.profit_factor:>14.3f} {full_result.profit_factor:>10.3f}")
    print(f"{'Win Rate':>25} {is_result.win_rate:>11.1f}% {oos_result.win_rate:>13.1f}% {full_result.win_rate:>9.1f}%")
    print(f"{'SPY B&H':>25} {is_result.spy_return:>+11.2f}% {oos_result.spy_return:>+13.2f}% {full_result.spy_return:>+9.2f}%")
    print(f"{'Alpha':>25} {is_result.total_return - is_result.spy_return:>+11.2f}% {oos_result.total_return - oos_result.spy_return:>+13.2f}% {full_result.total_return - full_result.spy_return:>+9.2f}%")

    # Save
    save_results([is_result, oos_result, full_result], "cross_backtest_momentum.json")
    save_trades(full_result.trades)

    log.info("Done.")


if __name__ == "__main__":
    main()
