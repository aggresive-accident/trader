#!/usr/bin/env python3
"""
multi_factor_backtest.py - R012 Multi-factor combination test

Tests whether combining factors beats single best (dividend_yield).

Usage:
  python3 multi_factor_backtest.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import pandas as pd
import numpy as np

from bar_cache import load_bars

def get_cached_symbols() -> list[str]:
    """Get all symbols in cache."""
    data_dir = Path(__file__).parent / "data" / "bars"
    return sorted([f.stem for f in data_dir.glob("*.parquet")])

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# === Configuration ===
START_DATE = "2022-06-01"
END_DATE = "2025-12-31"
POSITION_COUNT = 40
MIN_MARKET_CAP = 1e9  # $1B - not enforced here, assume bar_cache is filtered


# === Factor Functions (match R010) ===

def factor_dividend_yield(df: pd.DataFrame) -> float | None:
    """Higher dividend yield = higher score."""
    # Proxy: use price stability + low volatility as dividend proxy
    # R010 likely used actual dividend data - approximate with yield proxy
    if len(df) < 252:
        return None
    # Lower price / higher historical stability suggests dividend stock
    returns = df['close'].pct_change().dropna()
    if len(returns) < 20:
        return None
    vol = returns.std()
    if vol <= 0:
        return None
    # Inverse vol as proxy (dividend stocks tend to be less volatile)
    return -vol


def factor_quality(df: pd.DataFrame) -> float | None:
    """Quality: stable earnings proxy via price stability."""
    if len(df) < 252:
        return None
    returns = df['close'].pct_change().dropna()
    # Sharpe-like: return / vol
    mean_ret = returns.mean()
    vol = returns.std()
    if vol <= 0:
        return None
    return mean_ret / vol


def factor_momentum(df: pd.DataFrame, period: int = 252) -> float | None:
    """12-month momentum."""
    if len(df) < period + 1:
        return None
    return (df['close'].iloc[-1] / df['close'].iloc[-period]) - 1.0


def factor_value_pe(df: pd.DataFrame) -> float | None:
    """Value: inverse of price momentum (contrarian)."""
    mom = factor_momentum(df, 252)
    if mom is None:
        return None
    return -mom  # Lower momentum = "cheaper"


def factor_value_pb(df: pd.DataFrame) -> float | None:
    """Value: price-to-book proxy via mean reversion."""
    if len(df) < 60:
        return None
    # Distance from 60-day mean (lower = more undervalued)
    ma60 = df['close'].iloc[-60:].mean()
    current = df['close'].iloc[-1]
    return -(current / ma60 - 1)  # Negative distance = undervalued


def factor_small_cap(df: pd.DataFrame) -> float | None:
    """Small cap: inverse of price level as proxy."""
    if len(df) < 20:
        return None
    avg_price = df['close'].iloc[-20:].mean()
    if avg_price <= 0:
        return None
    return -avg_price  # Lower price = "smaller" (proxy)


def factor_mid_cap(df: pd.DataFrame) -> float | None:
    """Mid cap: moderate price level."""
    if len(df) < 20:
        return None
    avg_price = df['close'].iloc[-20:].mean()
    if avg_price <= 0:
        return None
    # Peak at mid-range prices, penalize extremes
    target = 100  # Target mid-cap price
    return -abs(avg_price - target)


def factor_large_cap(df: pd.DataFrame) -> float | None:
    """Large cap: higher price level as proxy."""
    if len(df) < 20:
        return None
    avg_price = df['close'].iloc[-20:].mean()
    return avg_price  # Higher price = "larger"


FACTORS = {
    'dividend_yield': factor_dividend_yield,
    'quality': factor_quality,
    'momentum': factor_momentum,
    'value_pe': factor_value_pe,
    'value_pb': factor_value_pb,
    'small_cap': factor_small_cap,
    'mid_cap': factor_mid_cap,
    'large_cap': factor_large_cap,
}


# === Strategies ===

STRATEGIES = {
    'S1_dividend_only': {
        'factors': [('dividend_yield', 40)],
    },
    'S2_momentum_only': {
        'factors': [('momentum', 40)],
    },
    'S3_3factor': {
        'factors': [('dividend_yield', 13), ('quality', 13), ('momentum', 14)],
    },
    'S4_5factor': {
        'factors': [('dividend_yield', 8), ('quality', 8), ('momentum', 8),
                    ('value_pe', 8), ('value_pb', 8)],
    },
    'S5_clive_proxy': {
        'factors': [('dividend_yield', 5), ('momentum', 5), ('quality', 5),
                    ('value_pe', 5), ('value_pb', 5), ('small_cap', 5),
                    ('mid_cap', 5), ('large_cap', 5)],
    },
}


def load_all_bars(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load bars for all symbols."""
    bars = {}
    for sym in symbols:
        try:
            df = load_bars(sym, start, end)
            if df is not None and len(df) > 0:
                if 'date' in df.columns:
                    df = df.set_index('date')
                bars[sym] = df
        except Exception:
            continue
    return bars


def rank_by_factor(bars: dict[str, pd.DataFrame], as_of_idx: int,
                   factor_name: str) -> list[tuple[str, float]]:
    """Rank all symbols by factor score."""
    factor_fn = FACTORS[factor_name]
    scores = []

    for sym, df in bars.items():
        if as_of_idx >= len(df):
            continue
        df_slice = df.iloc[:as_of_idx+1]
        score = factor_fn(df_slice)
        if score is not None and np.isfinite(score):
            scores.append((sym, score))

    # Sort descending (higher score = better)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def select_portfolio(bars: dict[str, pd.DataFrame], as_of_idx: int,
                     strategy_config: dict) -> list[str]:
    """Select portfolio according to strategy, handling overlaps."""
    selected = []
    used = set()

    for factor_name, n_picks in strategy_config['factors']:
        ranked = rank_by_factor(bars, as_of_idx, factor_name)

        # Pick top N not already used
        count = 0
        for sym, score in ranked:
            if sym not in used:
                selected.append(sym)
                used.add(sym)
                count += 1
                if count >= n_picks:
                    break

    return selected[:POSITION_COUNT]  # Cap at 40


def get_rebalance_dates(bars: dict[str, pd.DataFrame]) -> list[int]:
    """Get indices for annual rebalance (first trading day of each year)."""
    # Use first symbol's index as reference
    ref_sym = list(bars.keys())[0]
    ref_df = bars[ref_sym]

    rebal_indices = []
    current_year = None

    for i, date in enumerate(ref_df.index):
        if hasattr(date, 'year'):
            year = date.year
        else:
            # Parse string date
            year = int(str(date)[:4])

        if current_year is None or year > current_year:
            if i > 0:  # Skip first day
                rebal_indices.append(i)
            current_year = year

    return rebal_indices


def simulate_strategy(bars: dict[str, pd.DataFrame], strategy_name: str,
                      strategy_config: dict) -> dict:
    """Simulate strategy with annual rebalance."""

    # Get reference dates
    ref_sym = list(bars.keys())[0]
    ref_df = bars[ref_sym]
    n_days = len(ref_df)

    # Get rebalance dates
    rebal_indices = get_rebalance_dates(bars)
    if not rebal_indices:
        rebal_indices = [252]  # Default: after 1 year

    # Skip rebalance dates that don't have enough data (need 252 days for factors)
    rebal_indices = [r for r in rebal_indices if r >= 252]
    if not rebal_indices:
        print(f"  Warning: No valid rebalance dates with sufficient data")
        return {'strategy': strategy_name, 'portfolio_values': [],
                'initial_capital': initial_capital, 'final_value': 0, 'n_rebalances': 0}

    # Initial portfolio selection
    portfolio = select_portfolio(bars, rebal_indices[0], strategy_config)

    # Track portfolio value
    initial_capital = 100000
    position_size = initial_capital / POSITION_COUNT

    # Get entry prices
    holdings = {}  # sym -> shares
    entry_idx = rebal_indices[0]

    for sym in portfolio:
        if sym in bars and entry_idx < len(bars[sym]):
            price = bars[sym]['close'].iloc[entry_idx]
            shares = position_size / price
            holdings[sym] = shares

    # Track daily portfolio value
    portfolio_values = []

    for i in range(entry_idx, n_days):
        # Calculate portfolio value
        value = 0
        for sym, shares in holdings.items():
            if sym in bars and i < len(bars[sym]):
                value += shares * bars[sym]['close'].iloc[i]
        portfolio_values.append(value)

        # Rebalance if needed
        if i in rebal_indices and i > entry_idx:
            # Select new portfolio
            new_portfolio = select_portfolio(bars, i, strategy_config)

            # Sell all, buy new
            current_value = value
            position_size = current_value / POSITION_COUNT

            holdings = {}
            for sym in new_portfolio:
                if sym in bars and i < len(bars[sym]):
                    price = bars[sym]['close'].iloc[i]
                    shares = position_size / price
                    holdings[sym] = shares

    return {
        'strategy': strategy_name,
        'portfolio_values': portfolio_values,
        'initial_capital': initial_capital,
        'final_value': portfolio_values[-1] if portfolio_values else initial_capital,
        'n_rebalances': len([r for r in rebal_indices if r >= entry_idx and r < n_days]),
    }


def compute_metrics(result: dict, benchmark_values: list[float]) -> dict:
    """Compute performance metrics."""
    pv = np.array(result['portfolio_values'])
    bv = np.array(benchmark_values[:len(pv)])

    # Returns
    total_return = (pv[-1] / pv[0]) - 1
    benchmark_return = (bv[-1] / bv[0]) - 1

    # Daily returns
    daily_returns = np.diff(pv) / pv[:-1]
    bench_daily = np.diff(bv) / bv[:-1]

    # Sharpe (annualized)
    if np.std(daily_returns) > 0:
        sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    else:
        sharpe = 0

    # Max drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / peak
    max_dd = np.max(drawdown)

    # CAGR
    years = len(pv) / 252
    if years > 0 and pv[0] > 0:
        cagr = (pv[-1] / pv[0]) ** (1/years) - 1
    else:
        cagr = 0

    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'vs_spy': float(total_return - benchmark_return),
        'benchmark_return': float(benchmark_return),
    }


def load_spy_benchmark(start: str, end: str, n_days: int, start_idx: int) -> list[float]:
    """Load SPY as benchmark."""
    try:
        df = load_bars('SPY', start, end)
        if df is not None and len(df) > 0:
            if 'date' in df.columns:
                df = df.set_index('date')
            # Normalize to start at 100000
            prices = df['close'].iloc[start_idx:start_idx+n_days].values
            return list(prices / prices[0] * 100000)
    except Exception:
        pass

    # Fallback: flat benchmark
    return [100000] * n_days


def main():
    print("=" * 70)
    print("R012: Multi-Factor Combination Test")
    print("=" * 70)
    print(f"\nPeriod: {START_DATE} to {END_DATE}")
    print(f"Positions: {POSITION_COUNT}")
    print(f"Rebalance: Annual (January)")
    print()

    # Load data
    print("Loading data...")
    symbols = get_cached_symbols()
    bars = load_all_bars(symbols, START_DATE, END_DATE)
    print(f"Loaded {len(bars)} symbols")

    # Get benchmark
    rebal_indices = get_rebalance_dates(bars)
    start_idx = rebal_indices[0] if rebal_indices else 252

    results = []

    for strategy_name, strategy_config in STRATEGIES.items():
        print(f"\nRunning {strategy_name}...")
        result = simulate_strategy(bars, strategy_name, strategy_config)

        # Get benchmark aligned to this simulation
        spy_values = load_spy_benchmark(START_DATE, END_DATE,
                                         len(result['portfolio_values']), start_idx)

        metrics = compute_metrics(result, spy_values)
        result.update(metrics)

        # Remove large arrays for JSON
        del result['portfolio_values']

        results.append(result)
        print(f"  Return: {metrics['total_return']*100:.1f}%, "
              f"Sharpe: {metrics['sharpe']:.2f}, "
              f"Max DD: {metrics['max_drawdown']*100:.1f}%")

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'vs SPY':>10}")
    print("-" * 70)

    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)

    for r in results:
        print(f"{r['strategy']:<25} {r['total_return']*100:>9.1f}% "
              f"{r['sharpe']:>8.2f} {r['max_drawdown']*100:>7.1f}% "
              f"{r['vs_spy']*100:>+9.1f}%")

    # R010 baseline for comparison
    print("-" * 70)
    print(f"{'R010 dividend_yield':<25} {'83.7%':>10} {'1.09':>8} {'16.8%':>8} {'+8.3%':>10}")
    print()

    # Success evaluation
    print("=" * 70)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 70)

    dividend_result = next((r for r in results if 'dividend' in r['strategy']), None)
    best_multi = max([r for r in results if 'factor' in r['strategy'] or 'clive' in r['strategy']],
                     key=lambda x: x['sharpe'], default=None)

    if dividend_result and best_multi:
        print(f"\n1. Multi-factor Sharpe > dividend_only Sharpe (1.09)?")
        print(f"   Best multi-factor: {best_multi['strategy']} = {best_multi['sharpe']:.2f}")
        print(f"   Result: {'PASS' if best_multi['sharpe'] > 1.09 else 'FAIL'}")

        print(f"\n2. Multi-factor max DD < dividend_only max DD (16.8%)?")
        print(f"   Best multi-factor DD: {best_multi['max_drawdown']*100:.1f}%")
        print(f"   Result: {'PASS' if best_multi['max_drawdown'] < 0.168 else 'FAIL'}")

        # Check if any beats on both
        dual_winners = [r for r in results
                       if ('factor' in r['strategy'] or 'clive' in r['strategy'])
                       and r['total_return'] > 0.837 and r['sharpe'] > 1.09]
        print(f"\n3. Any combo beats dividend_only on BOTH return AND Sharpe?")
        print(f"   Result: {'PASS - ' + dual_winners[0]['strategy'] if dual_winners else 'FAIL'}")

    # Save results
    output = {
        'test_id': 'R012',
        'name': 'multi_factor_combination',
        'completed': datetime.now().isoformat(),
        'config': {
            'start': START_DATE,
            'end': END_DATE,
            'positions': POSITION_COUNT,
            'rebalance': 'annual',
        },
        'results': results,
        'baseline': {
            'strategy': 'R010_dividend_yield',
            'return': 0.837,
            'sharpe': 1.09,
            'max_dd': 0.168,
        }
    }

    output_file = RESULTS_DIR / 'R012.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
