#!/usr/bin/env python3
"""
vol_regime_test.py - R034 Volatility Regime Precursor Test

Tests whether volatility percentile rank detects regime shifts
BEFORE correlation structure breaks (addressing R030's RWEC lag).

Hypothesis: Vol expansion precedes correlation shifts by 3-7 days.

Usage:
  python3 vol_regime_test.py              # Run full test
  python3 vol_regime_test.py --threshold 85  # Single threshold
  python3 vol_regime_test.py --cap-weighted  # Cap-weighted vol
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import numpy as np
import pandas as pd

from bar_cache import load_bars, SP500_TOP200

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Config from R034 spec
WINDOW_ROLLING_VOL = 20   # 20-day rolling volatility
WINDOW_PERCENTILE = 252   # 252-day lookback for percentile
THRESHOLDS = [85, 90, 95]  # Percentile thresholds to test

# Test events from R034 spec
# Format: (label, break_date, rwec_lag_days)
# rwec_lag from R030: detected 2-5 days AFTER break
TEST_EVENTS = [
    ("COVID_crash", "2020-02-20", 3),      # R030: 2-5 day lag (use midpoint)
    ("Bear_2022", "2022-01-03", 4),        # R030: 2-5 day lag
    ("SVB_Crisis", "2023-03-10", 3),       # Not fully tested in R030, estimate
    ("Oct_2023_selloff", "2023-10-26", 4), # Not fully tested in R030, estimate
    ("Liberation_Day", "2025-04-02", 3),   # R030: 2-5 day lag
]


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


def compute_rolling_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling volatility (std dev) for each symbol."""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized


def compute_market_volatility(vol_df: pd.DataFrame, cap_weighted: bool = False) -> pd.Series:
    """Aggregate to single market volatility series."""
    if cap_weighted:
        # Simple proxy: weight by inverse of volatility (more stable = higher weight)
        # This is a rough approximation - proper cap weighting needs market cap data
        weights = 1 / vol_df.mean()
        weights = weights / weights.sum()
        return (vol_df * weights).sum(axis=1)
    else:
        # Equal weighted
        return vol_df.mean(axis=1)


def compute_percentile_rank(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Compute rolling percentile rank vs trailing distribution."""
    def pct_rank(window):
        if len(window) < lookback * 0.8:
            return np.nan
        current = window.iloc[-1]
        historical = window.iloc[:-1]
        return (historical < current).sum() / len(historical) * 100

    return series.rolling(window=lookback).apply(pct_rank, raw=False)


def find_signal_dates(pct_rank: pd.Series, threshold: float) -> list:
    """Find dates where percentile rank crosses above threshold."""
    above = pct_rank > threshold
    # Find first day of each "above threshold" period
    signals = above & ~above.shift(1).fillna(False)
    return pct_rank[signals].index.tolist()


def evaluate_event(
    pct_rank: pd.Series,
    threshold: float,
    event_date: str,
    event_label: str,
    search_window_days: int = 30
) -> dict:
    """
    Evaluate if vol signal fired before event.

    Returns dict with:
    - detected: bool
    - signal_date: first signal in search window (if any)
    - lead_time: days before event (positive = before, negative = after)
    """
    event_dt = pd.to_datetime(event_date)

    # Search window: 30 days before to 10 days after event
    search_start = event_dt - timedelta(days=search_window_days)
    search_end = event_dt + timedelta(days=10)

    # Filter to search window
    mask = (pct_rank.index >= search_start) & (pct_rank.index <= search_end)
    window_data = pct_rank[mask]

    if window_data.empty:
        return {
            "event": event_label,
            "event_date": event_date,
            "detected": False,
            "signal_date": None,
            "lead_time": None,
            "threshold": threshold,
            "note": "No data in search window"
        }

    # Find first signal in window
    above = window_data > threshold
    signal_dates = window_data[above].index.tolist()

    if not signal_dates:
        return {
            "event": event_label,
            "event_date": event_date,
            "detected": False,
            "signal_date": None,
            "lead_time": None,
            "threshold": threshold,
            "note": f"No signal above {threshold}th percentile"
        }

    first_signal = signal_dates[0]
    lead_time = (event_dt - first_signal).days

    # Success criteria: 3-7 days lead time (positive = before event)
    success = 3 <= lead_time <= 7

    return {
        "event": event_label,
        "event_date": event_date,
        "detected": True,
        "signal_date": first_signal.strftime("%Y-%m-%d"),
        "lead_time": lead_time,
        "threshold": threshold,
        "success": success,
        "note": f"{'PASS' if success else 'FAIL'}: {lead_time:+d} days"
    }


def count_false_positives(
    pct_rank: pd.Series,
    threshold: float,
    event_dates: list[str],
    exclusion_window_days: int = 30
) -> dict:
    """
    Count signals that don't correspond to known events.

    Excludes +/- exclusion_window_days around each known event.
    """
    # Build exclusion mask
    exclude_mask = pd.Series(False, index=pct_rank.index)
    for event_date in event_dates:
        event_dt = pd.to_datetime(event_date)
        start = event_dt - timedelta(days=exclusion_window_days)
        end = event_dt + timedelta(days=exclusion_window_days)
        exclude_mask |= (pct_rank.index >= start) & (pct_rank.index <= end)

    # Find signals outside exclusion zones
    above = pct_rank > threshold
    signals = above & ~above.shift(1).fillna(False)
    fp_signals = signals & ~exclude_mask

    fp_dates = pct_rank[fp_signals].index.tolist()

    # Calculate annualized rate
    total_days = len(pct_rank.dropna())
    years = total_days / 252
    fp_per_year = len(fp_dates) / years if years > 0 else 0

    return {
        "threshold": threshold,
        "false_positives": len(fp_dates),
        "fp_per_year": round(fp_per_year, 2),
        "years_analyzed": round(years, 2),
        "fp_dates": [d.strftime("%Y-%m-%d") for d in fp_dates[:10]]  # First 10
    }


def run_test(cap_weighted: bool = False, verbose: bool = True) -> dict:
    """Run full R034 test."""

    if verbose:
        print("=" * 70)
        print("R034: VOLATILITY REGIME PRECURSOR TEST")
        print("=" * 70)
        print(f"\nConfig:")
        print(f"  Rolling vol window: {WINDOW_ROLLING_VOL} days")
        print(f"  Percentile lookback: {WINDOW_PERCENTILE} days")
        print(f"  Thresholds: {THRESHOLDS}")
        print(f"  Aggregation: {'cap-weighted' if cap_weighted else 'equal-weighted'}")

    # Load data
    if verbose:
        print(f"\nLoading data for {len(SP500_TOP200)} symbols...")

    returns = load_returns(SP500_TOP200)

    if returns.empty:
        print("ERROR: No data loaded")
        return {"error": "No data"}

    if verbose:
        print(f"  Loaded {len(returns.columns)} symbols, {len(returns)} days")
        print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # Compute volatility metrics
    if verbose:
        print("\nComputing volatility metrics...")

    rolling_vol = compute_rolling_volatility(returns, WINDOW_ROLLING_VOL)
    market_vol = compute_market_volatility(rolling_vol, cap_weighted)
    pct_rank = compute_percentile_rank(market_vol, WINDOW_PERCENTILE)

    # Filter events to those within data range
    data_start = returns.index[0]
    data_end = returns.index[-1]
    valid_events = []

    for label, date, rwec_lag in TEST_EVENTS:
        event_dt = pd.to_datetime(date)
        if event_dt >= data_start and event_dt <= data_end:
            valid_events.append((label, date, rwec_lag))
        elif verbose:
            print(f"  Skipping {label} ({date}) - outside data range")

    if verbose:
        print(f"\nTesting {len(valid_events)} events across {len(THRESHOLDS)} thresholds...")

    # Run tests for each threshold
    results_by_threshold = {}

    for threshold in THRESHOLDS:
        event_results = []

        for label, date, rwec_lag in valid_events:
            result = evaluate_event(pct_rank, threshold, date, label)
            result["rwec_lag"] = rwec_lag

            # Calculate improvement over RWEC
            if result["lead_time"] is not None:
                # RWEC lag is negative (after event), our lead time positive is before
                rwec_detection_day = -rwec_lag  # e.g., -3 means 3 days after
                vol_detection_day = result["lead_time"]  # positive = before
                improvement = vol_detection_day - rwec_detection_day
                result["improvement_vs_rwec"] = improvement
            else:
                result["improvement_vs_rwec"] = None

            event_results.append(result)

        # Count false positives
        event_dates = [date for _, date, _ in valid_events]
        fp_result = count_false_positives(pct_rank, threshold, event_dates)

        # Aggregate stats
        detected = [r for r in event_results if r["detected"]]
        passed = [r for r in event_results if r.get("success")]

        results_by_threshold[threshold] = {
            "threshold": threshold,
            "events": event_results,
            "summary": {
                "events_tested": len(event_results),
                "events_detected": len(detected),
                "events_passed": len(passed),
                "detection_rate": len(detected) / len(event_results) if event_results else 0,
                "success_rate": len(passed) / len(event_results) if event_results else 0,
            },
            "false_positives": fp_result
        }

    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS BY THRESHOLD")
        print("=" * 70)

        for threshold, data in results_by_threshold.items():
            print(f"\n--- Threshold: {threshold}th percentile ---")
            print(f"\n{'EVENT':<20} {'BREAK':<12} {'SIGNAL':<12} {'LEAD':<8} {'RWEC':<8} {'IMPROVE':<10} {'RESULT'}")
            print("-" * 90)

            for e in data["events"]:
                signal = e["signal_date"] or "none"
                lead = f"{e['lead_time']:+d}d" if e["lead_time"] is not None else "n/a"
                rwec = f"{-e['rwec_lag']:+d}d"  # Show as negative (lag after event)
                improve = f"{e['improvement_vs_rwec']:+d}d" if e.get("improvement_vs_rwec") is not None else "n/a"
                result = "PASS" if e.get("success") else ("MISS" if not e["detected"] else "FAIL")
                print(f"{e['event']:<20} {e['event_date']:<12} {signal:<12} {lead:<8} {rwec:<8} {improve:<10} {result}")

            s = data["summary"]
            fp = data["false_positives"]
            print(f"\nSummary:")
            print(f"  Detection: {s['events_detected']}/{s['events_tested']} ({s['detection_rate']:.0%})")
            print(f"  Success (3-7d lead): {s['events_passed']}/{s['events_tested']} ({s['success_rate']:.0%})")
            print(f"  False positives: {fp['fp_per_year']}/year over {fp['years_analyzed']} years")

    # Overall assessment
    if verbose:
        print("\n" + "=" * 70)
        print("OVERALL ASSESSMENT")
        print("=" * 70)

        # Find best threshold
        best_threshold = max(
            results_by_threshold.keys(),
            key=lambda t: (
                results_by_threshold[t]["summary"]["success_rate"],
                -results_by_threshold[t]["false_positives"]["fp_per_year"]
            )
        )
        best = results_by_threshold[best_threshold]

        print(f"\nBest threshold: {best_threshold}th percentile")
        print(f"  Success rate: {best['summary']['success_rate']:.0%}")
        print(f"  False positive rate: {best['false_positives']['fp_per_year']}/year")

        # Check against success criteria
        print("\n--- vs Success Criteria ---")
        criteria_met = 0
        total_criteria = 4

        # Criterion 1: Detect 3+ events with 3-7 day lead
        passed_events = best["summary"]["events_passed"]
        c1 = passed_events >= 3
        print(f"1. Detect 3+ events with 3-7d lead: {passed_events} events {'PASS' if c1 else 'FAIL'}")
        criteria_met += c1

        # Criterion 2: FP rate < 2/year
        fp_rate = best["false_positives"]["fp_per_year"]
        c2 = fp_rate < 2
        print(f"2. False positives < 2/year: {fp_rate}/year {'PASS' if c2 else 'FAIL'}")
        criteria_met += c2

        # Criterion 3: Detection precision > 60%
        precision = best["summary"]["detection_rate"]
        c3 = precision > 0.6
        print(f"3. Detection precision > 60%: {precision:.0%} {'PASS' if c3 else 'FAIL'}")
        criteria_met += c3

        # Criterion 4: Outperform RWEC
        improvements = [e["improvement_vs_rwec"] for e in best["events"]
                       if e.get("improvement_vs_rwec") is not None]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        c4 = avg_improvement > 0
        print(f"4. Outperform RWEC: avg improvement {avg_improvement:+.1f}d {'PASS' if c4 else 'FAIL'}")
        criteria_met += c4

        print(f"\nCriteria met: {criteria_met}/{total_criteria}")

        if criteria_met >= 3:
            print("\nVERDICT: PASS - Vol regime precursor shows promise")
        elif criteria_met >= 2:
            print("\nVERDICT: PARTIAL - Mixed results, consider hybrid approach")
        else:
            print("\nVERDICT: FAIL - Vol does not reliably lead regime shifts")

    # Build output
    output = {
        "test_id": "R034",
        "name": "vol_regime_precursor",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "rolling_vol_window": WINDOW_ROLLING_VOL,
            "percentile_lookback": WINDOW_PERCENTILE,
            "thresholds": THRESHOLDS,
            "aggregation": "cap_weighted" if cap_weighted else "equal_weighted",
            "symbols_count": len(returns.columns),
            "date_range": {
                "start": returns.index[0].strftime("%Y-%m-%d"),
                "end": returns.index[-1].strftime("%Y-%m-%d"),
            }
        },
        "results": results_by_threshold,
        "best_threshold": best_threshold,
        "criteria_assessment": {
            "events_with_lead": passed_events,
            "fp_per_year": fp_rate,
            "detection_precision": precision,
            "avg_improvement_vs_rwec": avg_improvement,
            "criteria_met": criteria_met,
            "total_criteria": total_criteria,
        }
    }

    return output


def main():
    parser = argparse.ArgumentParser(description="R034 Vol Regime Precursor Test")
    parser.add_argument("--cap-weighted", action="store_true",
                       help="Use cap-weighted vol aggregation")
    parser.add_argument("--threshold", type=int, choices=[85, 90, 95],
                       help="Test single threshold only")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal output")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to JSON")
    args = parser.parse_args()

    global THRESHOLDS
    if args.threshold:
        THRESHOLDS = [args.threshold]

    results = run_test(
        cap_weighted=args.cap_weighted,
        verbose=not args.quiet
    )

    if "error" in results:
        sys.exit(1)

    # Save results
    if not args.no_save:
        output_file = RESULTS_DIR / "R034.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
