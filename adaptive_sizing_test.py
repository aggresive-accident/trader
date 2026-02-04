#!/usr/bin/env python3
"""
adaptive_sizing_test.py - R035 Adaptive Regime Sizing Test

Tests whether reducing position size by 50% when RWEC regime detection
triggers improves risk-adjusted returns vs static full-size positions.

Hypothesis: Even with 2-5 day detection lag, rapid response to regime
shifts captures value if instability persists for 1-2 weeks.

Usage:
  python3 adaptive_sizing_test.py           # Run full test
  python3 adaptive_sizing_test.py --quick   # Faster with fewer symbols
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import numpy as np
import pandas as pd
from scipy import linalg

from bar_cache import load_bars, SP500_TOP200

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Config from R035 spec
RWEC_WINDOW = 60
RWEC_THRESHOLD = 15.0  # degrees
TRIGGER_DELAY = 3  # days after detection before reducing
RESTORATION_DAYS = 10  # days to hold reduced position
BASELINE_SIZE = 1.0
ADAPTIVE_SIZE = 0.5
REBALANCE_DAYS = 5  # weekly

# Known regime events for per-event analysis
REGIME_EVENTS = [
    ("Bear_2022_start", "2022-01-03"),
    ("SVB_Crisis", "2023-03-10"),
    ("Oct_2023_selloff", "2023-10-26"),
    ("Liberation_Day", "2025-04-02"),
]


# === Data Loading ===

def load_returns(symbols: list[str], start: str = None, end: str = None) -> pd.DataFrame:
    """Load daily returns for symbols."""
    frames = []
    for sym in symbols:
        try:
            df = load_bars(sym, start, end)
            if df is not None and len(df) > 0:
                if 'date' in df.columns:
                    df = df.set_index(pd.to_datetime(df['date']))
                returns = df['close'].pct_change()
                returns.name = sym
                frames.append(returns)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.8))
    return combined


def load_prices(symbols: list[str], start: str = None, end: str = None) -> pd.DataFrame:
    """Load daily close prices for symbols."""
    frames = []
    for sym in symbols:
        try:
            df = load_bars(sym, start, end)
            if df is not None and len(df) > 0:
                if 'date' in df.columns:
                    df = df.set_index(pd.to_datetime(df['date']))
                prices = df['close']
                prices.name = sym
                frames.append(prices)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.8))
    return combined


# === RWEC Detection (from regime_monitor.py) ===

def compute_correlation_matrix(returns: pd.DataFrame, end_idx: int, window: int) -> np.ndarray:
    """Compute correlation matrix for window ending at end_idx."""
    start_idx = max(0, end_idx - window)
    window_returns = returns.iloc[start_idx:end_idx]

    if len(window_returns) < window * 0.8:
        return None

    return window_returns.corr().values


def principal_eigenvector(corr_matrix: np.ndarray) -> np.ndarray:
    """Extract principal eigenvector."""
    if corr_matrix is None:
        return None

    corr_matrix = np.nan_to_num(corr_matrix, nan=0, posinf=1, neginf=-1)

    try:
        eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
        principal = eigenvectors[:, -1]
        principal = principal / np.linalg.norm(principal)
        return principal
    except Exception:
        return None


def rwec_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute angle between eigenvectors in degrees."""
    if vec1 is None or vec2 is None:
        return np.nan

    cos_sim = np.abs(np.dot(vec1, vec2))
    cos_sim = np.clip(cos_sim, -1, 1)
    angle = np.degrees(np.arccos(cos_sim))
    return angle


def compute_rwec_series(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Compute RWEC angle time series."""
    dates = []
    angles = []
    prev_vec = None

    for i in range(window, len(returns)):
        date = returns.index[i]
        corr = compute_correlation_matrix(returns, i, window)
        vec = principal_eigenvector(corr)

        if prev_vec is not None and vec is not None:
            angle = rwec_angle(prev_vec, vec)
        else:
            angle = 0.0

        dates.append(date)
        angles.append(angle)
        prev_vec = vec

    return pd.Series(angles, index=dates, name='rwec_angle')


# === Momentum Factor ===

def compute_momentum(prices: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Compute momentum (period-day return) for each symbol."""
    return prices.pct_change(period)


# === Backtest Engine ===

@dataclass
class BacktestResult:
    variant: str
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    max_drawdown_date: str
    recovery_days: int
    volatility: float
    num_trades: int
    triggers: int
    trigger_dates: list
    daily_returns: list  # For per-event analysis


def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rwec_angles: pd.Series,
    variant: str = "BASELINE",
    top_n: int = 10,
    rebalance_days: int = 5,
    verbose: bool = False
) -> BacktestResult:
    """
    Run cross-sectional momentum backtest.

    variant: "BASELINE" (always full size) or "ADAPTIVE" (reduce on regime trigger)
    """
    # Align dates
    common_dates = prices.index.intersection(rwec_angles.index)
    prices = prices.loc[common_dates]
    returns = returns.loc[common_dates]
    rwec_angles = rwec_angles.loc[common_dates]

    # Track state
    portfolio_values = [1.0]
    daily_rets = []
    positions = {}  # symbol -> weight
    last_rebalance = None

    # Adaptive sizing state
    regime_triggered = False
    trigger_date = None
    days_since_trigger = 0
    trigger_dates = []

    # Momentum lookback
    momentum_period = 20

    for i in range(momentum_period + 1, len(prices)):
        date = prices.index[i]
        prev_date = prices.index[i - 1]

        # Check regime trigger (for adaptive variant)
        if variant == "ADAPTIVE":
            angle = rwec_angles.iloc[i] if i < len(rwec_angles) else 0

            if angle > RWEC_THRESHOLD and not regime_triggered:
                regime_triggered = True
                trigger_date = date
                days_since_trigger = 0
                trigger_dates.append(date.strftime("%Y-%m-%d"))
                if verbose:
                    print(f"  {date.date()}: REGIME TRIGGER (angle={angle:.1f})")

            elif regime_triggered:
                days_since_trigger += 1
                # Only restore after minimum hold period (trigger_delay + some buffer)
                # Must have: angle normalized AND held for at least trigger_delay days
                # OR: restoration period fully elapsed regardless of angle
                min_hold = TRIGGER_DELAY + 2  # At least trigger delay + 2 days
                if days_since_trigger >= RESTORATION_DAYS:
                    regime_triggered = False
                    if verbose:
                        print(f"  {date.date()}: REGIME RESTORED (max days reached)")
                elif days_since_trigger >= min_hold and angle < RWEC_THRESHOLD:
                    regime_triggered = False
                    if verbose:
                        print(f"  {date.date()}: REGIME RESTORED (normalized after {days_since_trigger}d)")

        # Determine position size multiplier
        if variant == "ADAPTIVE" and regime_triggered and days_since_trigger >= TRIGGER_DELAY:
            size_mult = ADAPTIVE_SIZE
        else:
            size_mult = BASELINE_SIZE

        # Rebalance check
        should_rebalance = (
            last_rebalance is None or
            (date - last_rebalance).days >= rebalance_days
        )

        if should_rebalance:
            last_rebalance = date

            # Compute momentum scores
            momentum = compute_momentum(prices.iloc[:i], momentum_period)
            if momentum.empty or i - 1 >= len(momentum):
                continue

            scores = momentum.iloc[-1].dropna()

            if len(scores) < top_n:
                continue

            # Select top N
            top_symbols = scores.nlargest(top_n).index.tolist()

            # Equal weight with size multiplier
            weight = size_mult / top_n
            positions = {sym: weight for sym in top_symbols}

        # Calculate daily return
        if positions:
            port_return = 0.0
            for sym, weight in positions.items():
                if sym in returns.columns and i < len(returns):
                    sym_return = returns[sym].iloc[i]
                    if not np.isnan(sym_return):
                        port_return += weight * sym_return

            # Adjust for cash position when size_mult < 1
            # (remaining capital earns 0 - simplified, no interest)

            daily_rets.append(port_return)
            portfolio_values.append(portfolio_values[-1] * (1 + port_return))
        else:
            daily_rets.append(0.0)
            portfolio_values.append(portfolio_values[-1])

    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    daily_rets = np.array(daily_rets)

    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    years = len(daily_rets) / 252
    cagr = ((portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    volatility = np.std(daily_rets) * np.sqrt(252) * 100
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_dd = np.min(drawdown) * 100
    max_dd_idx = np.argmin(drawdown)
    max_dd_date = prices.index[min(max_dd_idx, len(prices) - 1)].strftime("%Y-%m-%d")

    # Recovery days (simplified - days from max DD to new high)
    recovery_days = 0
    if max_dd_idx < len(portfolio_values) - 1:
        post_dd = portfolio_values[max_dd_idx:]
        recovery_idx = np.where(post_dd >= peak[max_dd_idx])[0]
        if len(recovery_idx) > 0:
            recovery_days = recovery_idx[0]

    return BacktestResult(
        variant=variant,
        total_return=round(total_return, 2),
        cagr=round(cagr, 2),
        sharpe=round(sharpe, 3),
        max_drawdown=round(max_dd, 2),
        max_drawdown_date=max_dd_date,
        recovery_days=recovery_days,
        volatility=round(volatility, 2),
        num_trades=len(prices) // rebalance_days,
        triggers=len(trigger_dates),
        trigger_dates=trigger_dates,
        daily_returns=daily_rets.tolist()
    )


def analyze_event_performance(
    baseline_returns: list,
    adaptive_returns: list,
    dates: pd.DatetimeIndex,
    event_date: str,
    event_name: str,
    trigger_dates: list,
    window_before: int = 10,
    window_after: int = 30
) -> dict:
    """Analyze performance around a specific event."""
    event_dt = pd.to_datetime(event_date)

    # Find event index
    event_idx = None
    for i, d in enumerate(dates):
        if d >= event_dt:
            event_idx = i
            break

    if event_idx is None or event_idx >= len(baseline_returns):
        return {
            "event": event_name,
            "date": event_date,
            "error": "Event outside data range"
        }

    # Extract window
    start_idx = max(0, event_idx - window_before)
    end_idx = min(len(baseline_returns), event_idx + window_after)

    baseline_window = np.array(baseline_returns[start_idx:end_idx])
    adaptive_window = np.array(adaptive_returns[start_idx:end_idx])

    # Cumulative returns
    baseline_cum = np.cumprod(1 + baseline_window) - 1
    adaptive_cum = np.cumprod(1 + adaptive_window) - 1

    # Metrics during event window
    baseline_total = baseline_cum[-1] * 100 if len(baseline_cum) > 0 else 0
    adaptive_total = adaptive_cum[-1] * 100 if len(adaptive_cum) > 0 else 0

    # Max drawdown in window
    baseline_peak = np.maximum.accumulate(np.cumprod(1 + baseline_window))
    baseline_dd = np.min((np.cumprod(1 + baseline_window) - baseline_peak) / baseline_peak) * 100

    adaptive_peak = np.maximum.accumulate(np.cumprod(1 + adaptive_window))
    adaptive_dd = np.min((np.cumprod(1 + adaptive_window) - adaptive_peak) / adaptive_peak) * 100

    # Find nearest trigger to this event
    event_dt = pd.to_datetime(event_date)
    nearest_trigger = None
    trigger_lag = None
    for t in trigger_dates:
        t_dt = pd.to_datetime(t)
        lag = (t_dt - event_dt).days
        if -10 <= lag <= 30:  # Within analysis window
            if nearest_trigger is None or abs(lag) < abs(trigger_lag):
                nearest_trigger = t
                trigger_lag = lag

    return {
        "event": event_name,
        "date": event_date,
        "baseline_return": round(baseline_total, 2),
        "adaptive_return": round(adaptive_total, 2),
        "return_diff": round(adaptive_total - baseline_total, 2),
        "baseline_max_dd": round(baseline_dd, 2),
        "adaptive_max_dd": round(adaptive_dd, 2),
        "dd_improvement": round(baseline_dd - adaptive_dd, 2),  # Positive = adaptive better
        "adaptive_better": adaptive_dd > baseline_dd,  # Less negative = better
        "nearest_trigger": nearest_trigger,
        "trigger_lag_days": trigger_lag,  # Positive = trigger after event
        "effective_response_day": trigger_lag + TRIGGER_DELAY if trigger_lag is not None else None
    }


def run_test(quick: bool = False, verbose: bool = True) -> dict:
    """Run full R035 test."""

    if verbose:
        print("=" * 70)
        print("R035: ADAPTIVE REGIME SIZING TEST")
        print("=" * 70)
        print(f"\nConfig:")
        print(f"  RWEC window: {RWEC_WINDOW} days")
        print(f"  Threshold: {RWEC_THRESHOLD} degrees")
        print(f"  Trigger delay: {TRIGGER_DELAY} days")
        print(f"  Restoration: {RESTORATION_DAYS} days")
        print(f"  Baseline size: {BASELINE_SIZE}")
        print(f"  Adaptive size: {ADAPTIVE_SIZE}")

    # Load data
    symbols = SP500_TOP200[:50] if quick else SP500_TOP200

    if verbose:
        print(f"\nLoading data for {len(symbols)} symbols...")

    prices = load_prices(symbols)
    returns = load_returns(symbols)

    if prices.empty:
        print("ERROR: No data loaded")
        return {"error": "No data"}

    if verbose:
        print(f"  Loaded {len(prices.columns)} symbols, {len(prices)} days")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Compute RWEC angles
    if verbose:
        print("\nComputing RWEC angles...")

    rwec_angles = compute_rwec_series(returns, RWEC_WINDOW)

    if verbose:
        print(f"  Computed {len(rwec_angles)} angles")
        print(f"  Mean angle: {rwec_angles.mean():.2f} degrees")
        print(f"  Max angle: {rwec_angles.max():.2f} degrees")

    # Run backtests
    if verbose:
        print("\nRunning BASELINE backtest...")

    baseline = run_backtest(
        prices, returns, rwec_angles,
        variant="BASELINE",
        verbose=verbose
    )

    if verbose:
        print("\nRunning ADAPTIVE backtest...")

    adaptive = run_backtest(
        prices, returns, rwec_angles,
        variant="ADAPTIVE",
        verbose=verbose
    )

    # Per-event analysis
    if verbose:
        print("\nAnalyzing per-event performance...")

    # Build date index for daily returns
    start_idx = RWEC_WINDOW + 21  # After momentum lookback + RWEC window
    return_dates = prices.index[start_idx:start_idx + len(baseline.daily_returns)]

    event_results = []
    for event_name, event_date in REGIME_EVENTS:
        event_dt = pd.to_datetime(event_date)
        if event_dt < return_dates[0] or event_dt > return_dates[-1]:
            if verbose:
                print(f"  Skipping {event_name} ({event_date}) - outside data range")
            continue

        result = analyze_event_performance(
            baseline.daily_returns,
            adaptive.daily_returns,
            return_dates,
            event_date,
            event_name,
            adaptive.trigger_dates
        )
        event_results.append(result)

    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)

        print(f"\n{'METRIC':<25} {'BASELINE':>12} {'ADAPTIVE':>12} {'DIFF':>12}")
        print("-" * 65)
        print(f"{'Total Return':<25} {baseline.total_return:>+11.2f}% {adaptive.total_return:>+11.2f}% {adaptive.total_return - baseline.total_return:>+11.2f}%")
        print(f"{'CAGR':<25} {baseline.cagr:>+11.2f}% {adaptive.cagr:>+11.2f}% {adaptive.cagr - baseline.cagr:>+11.2f}%")
        print(f"{'Sharpe Ratio':<25} {baseline.sharpe:>12.3f} {adaptive.sharpe:>12.3f} {adaptive.sharpe - baseline.sharpe:>+12.3f}")
        print(f"{'Max Drawdown':<25} {baseline.max_drawdown:>11.2f}% {adaptive.max_drawdown:>11.2f}% {adaptive.max_drawdown - baseline.max_drawdown:>+11.2f}%")
        print(f"{'Volatility':<25} {baseline.volatility:>11.2f}% {adaptive.volatility:>11.2f}% {adaptive.volatility - baseline.volatility:>+11.2f}%")
        print(f"{'Recovery Days':<25} {baseline.recovery_days:>12d} {adaptive.recovery_days:>12d} {adaptive.recovery_days - baseline.recovery_days:>+12d}")
        print(f"{'Regime Triggers':<25} {'n/a':>12} {adaptive.triggers:>12d}")

        if adaptive.trigger_dates:
            print(f"\nTrigger dates: {', '.join(adaptive.trigger_dates[:10])}")

        print("\n" + "=" * 70)
        print("PER-EVENT BREAKDOWN")
        print("=" * 70)

        print(f"\n{'EVENT':<20} {'DATE':<12} {'TRIGGER':>12} {'DETECT':>8} {'RESPOND':>8} {'DD_DIFF':>10}")
        print("-" * 80)

        for e in event_results:
            if "error" in e:
                print(f"{e['event']:<20} {e['date']:<12} {'n/a':>12} {'n/a':>8} {'n/a':>8} {e['error']}")
            else:
                trigger = e.get("nearest_trigger") or "none"
                detect = f"+{e['trigger_lag_days']}d" if e.get("trigger_lag_days") is not None else "n/a"
                respond = f"+{e['effective_response_day']}d" if e.get("effective_response_day") is not None else "n/a"
                dd_diff = e["dd_improvement"]
                dd_str = f"{dd_diff:+.2f}%" if dd_diff != 0 else "0"
                print(f"{e['event']:<20} {e['date']:<12} {trigger:>12} {detect:>8} {respond:>8} {dd_str:>10}")

        # Success criteria
        print("\n" + "=" * 70)
        print("SUCCESS CRITERIA")
        print("=" * 70)

        criteria_met = 0
        total_criteria = 5

        # 1. Sharpe improvement > 0.1
        sharpe_diff = adaptive.sharpe - baseline.sharpe
        c1 = sharpe_diff > 0.1
        print(f"\n1. Sharpe improvement > 0.1: {sharpe_diff:+.3f} {'PASS' if c1 else 'FAIL'}")
        criteria_met += c1

        # 2. Max DD improvement > 5%
        dd_diff = baseline.max_drawdown - adaptive.max_drawdown  # Positive = adaptive better
        c2 = dd_diff > 5
        print(f"2. Max DD improvement > 5%: {dd_diff:+.2f}% {'PASS' if c2 else 'FAIL'}")
        criteria_met += c2

        # 3. Recovery time improvement
        recovery_diff = baseline.recovery_days - adaptive.recovery_days
        c3 = recovery_diff > 0
        print(f"3. Recovery time improvement: {recovery_diff:+d} days {'PASS' if c3 else 'FAIL'}")
        criteria_met += c3

        # 4. FP cost < 2% annual drag
        # Estimate: CAGR difference when triggers are false positives
        # If adaptive CAGR is within 2% of baseline, FP cost is acceptable
        cagr_drag = baseline.cagr - adaptive.cagr
        c4 = cagr_drag < 2.0
        print(f"4. FP cost < 2% annual drag: {cagr_drag:+.2f}% {'PASS' if c4 else 'FAIL'}")
        criteria_met += c4

        # 5. Trigger frequency < 4/year
        years = len(baseline.daily_returns) / 252
        triggers_per_year = adaptive.triggers / years if years > 0 else 0
        c5 = triggers_per_year < 4
        print(f"5. Triggers < 4/year: {triggers_per_year:.2f}/year {'PASS' if c5 else 'FAIL'}")
        criteria_met += c5

        print(f"\nCriteria met: {criteria_met}/{total_criteria}")

        if criteria_met >= 4:
            print("\nVERDICT: PASS - Adaptive sizing improves risk-adjusted returns")
        elif criteria_met >= 2:
            print("\nVERDICT: PARTIAL - Mixed results, analyze per-event for guidance")
        else:
            print("\nVERDICT: FAIL - Adaptive sizing does not improve performance")

    # Build output
    output = {
        "test_id": "R035",
        "name": "adaptive_regime_sizing",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "rwec_window": RWEC_WINDOW,
            "threshold": RWEC_THRESHOLD,
            "trigger_delay": TRIGGER_DELAY,
            "restoration_days": RESTORATION_DAYS,
            "baseline_size": BASELINE_SIZE,
            "adaptive_size": ADAPTIVE_SIZE,
            "symbols_count": len(prices.columns),
            "date_range": {
                "start": prices.index[0].strftime("%Y-%m-%d"),
                "end": prices.index[-1].strftime("%Y-%m-%d"),
            }
        },
        "results": {
            "baseline": asdict(baseline),
            "adaptive": asdict(adaptive),
        },
        "event_analysis": event_results,
        "criteria_assessment": {
            "sharpe_improvement": sharpe_diff,
            "dd_improvement": dd_diff,
            "recovery_improvement": recovery_diff,
            "cagr_drag": cagr_drag,
            "triggers_per_year": triggers_per_year,
            "criteria_met": criteria_met,
            "total_criteria": total_criteria,
        }
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="R035 Adaptive Regime Sizing Test")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick test with fewer symbols")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to JSON")
    args = parser.parse_args()

    results = run_test(quick=args.quick, verbose=True)

    if "error" in results:
        sys.exit(1)

    # Save results
    if not args.no_save:
        output_file = RESULTS_DIR / "R035.json"
        with open(output_file, "w") as f:
            # Don't save daily returns to keep file size reasonable
            save_results = results.copy()
            save_results["results"]["baseline"]["daily_returns"] = f"[{len(results['results']['baseline']['daily_returns'])} values]"
            save_results["results"]["adaptive"]["daily_returns"] = f"[{len(results['results']['adaptive']['daily_returns'])} values]"
            json.dump(save_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
