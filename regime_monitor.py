#!/usr/bin/env python3
"""
regime_monitor.py - RWEC regime detection for R030 validation

Detects regime shifts via Rolling Window Eigenvector Comparison.
Measures stability of correlation structure over time.

Usage:
  python3 regime_monitor.py baseline          # Phase 1: full time series
  python3 regime_monitor.py detect            # Phase 2: known break detection
  python3 regime_monitor.py analyze           # Phase 3: false positive analysis
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import numpy as np
import pandas as pd
from scipy import linalg

from bar_cache import load_bars

DATA_DIR = Path(__file__).parent / "data" / "bars"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_cached_symbols() -> list[str]:
    """Get all symbols in cache."""
    return sorted([f.stem for f in DATA_DIR.glob("*.parquet")])


def load_returns(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Load daily returns for symbols."""
    frames = []
    for sym in symbols:
        try:
            df = load_bars(sym, start, end)
            if df is not None and len(df) > 0:
                # Set date as index
                if 'date' in df.columns:
                    df = df.set_index('date')
                returns = df['close'].pct_change().dropna()
                returns.name = sym
                frames.append(returns)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    # Drop symbols with too many NaNs
    combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.9))
    combined = combined.fillna(0)
    return combined


def compute_correlation_matrix(returns: pd.DataFrame, end_idx: int, window: int) -> np.ndarray:
    """Compute correlation matrix for window ending at end_idx."""
    start_idx = max(0, end_idx - window)
    window_returns = returns.iloc[start_idx:end_idx]

    if len(window_returns) < window * 0.8:  # Need at least 80% of window
        return None

    return window_returns.corr().values


def principal_eigenvector(corr_matrix: np.ndarray) -> np.ndarray:
    """Extract principal eigenvector (first PC)."""
    if corr_matrix is None:
        return None

    # Handle NaN/Inf
    corr_matrix = np.nan_to_num(corr_matrix, nan=0, posinf=1, neginf=-1)

    try:
        eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
        # Largest eigenvalue is last (ascending order)
        principal = eigenvectors[:, -1]
        # Normalize
        principal = principal / np.linalg.norm(principal)
        return principal
    except Exception:
        return None


def rwec_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute angle between eigenvectors in degrees."""
    if vec1 is None or vec2 is None:
        return np.nan

    # Cosine similarity (handle sign ambiguity in eigenvectors)
    cos_sim = np.abs(np.dot(vec1, vec2))
    cos_sim = np.clip(cos_sim, -1, 1)

    # Convert to angle in degrees
    angle = np.degrees(np.arccos(cos_sim))
    return angle


def run_baseline(windows: list[int] = [60, 252], threshold: float = 30.0) -> dict:
    """
    Phase 1: Compute eigenvector angle time series.

    Returns dict with results for each window size.
    """
    print("Loading symbols...")
    symbols = get_cached_symbols()
    print(f"Found {len(symbols)} symbols")

    print("Loading returns (2022-01-03 to 2026-01-26)...")
    returns = load_returns(symbols, "2022-01-01", "2026-01-27")
    print(f"Returns shape: {returns.shape}")

    results = {}

    for window in windows:
        print(f"\nProcessing window={window}...")

        dates = []
        angles = []
        prev_vec = None

        # Start after first full window
        for i in range(window, len(returns)):
            date = returns.index[i]

            # Handle different index types
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)

            corr = compute_correlation_matrix(returns, i, window)
            vec = principal_eigenvector(corr)

            if prev_vec is not None and vec is not None:
                angle = rwec_angle(prev_vec, vec)
                dates.append(date)
                angles.append(angle)

            prev_vec = vec

            if i % 100 == 0:
                print(f"  {i}/{len(returns)} ({date_str})")

        # Find peaks above threshold
        angles_arr = np.array(angles)
        peaks = [(dates[i], angles[i]) for i in range(len(angles))
                 if angles[i] > threshold]

        def fmt_date(d):
            return d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10]

        results[window] = {
            'dates': [fmt_date(d) for d in dates],
            'angles': [float(a) if not np.isnan(a) else None for a in angles],
            'threshold': threshold,
            'peaks_above_threshold': len(peaks),
            'peak_dates': [fmt_date(p[0]) for p in peaks],
            'stats': {
                'mean': float(np.nanmean(angles_arr)),
                'std': float(np.nanstd(angles_arr)),
                'max': float(np.nanmax(angles_arr)),
                'pct_above_threshold': float(np.sum(angles_arr > threshold) / len(angles_arr) * 100)
            }
        }

        print(f"  Window {window}: mean={results[window]['stats']['mean']:.2f}°, "
              f"max={results[window]['stats']['max']:.2f}°, "
              f"peaks>{threshold}°: {len(peaks)}")

    return results


def save_results(results: dict, filename: str):
    """Save results to JSON."""
    output_path = RESULTS_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_ascii_plot(results: dict, window: int, width: int = 70, height: int = 15):
    """Print ASCII time series plot."""
    data = results[window]
    angles = [a if a is not None else 0 for a in data['angles']]
    dates = data['dates']
    threshold = data['threshold']

    if not angles:
        print("No data to plot")
        return

    # Downsample for display
    step = max(1, len(angles) // width)
    sampled_angles = [angles[i] for i in range(0, len(angles), step)][:width]
    sampled_dates = [dates[i] for i in range(0, len(dates), step)][:width]

    max_val = max(max(sampled_angles), threshold + 5)
    min_val = 0

    print(f"\n{'='*width}")
    print(f"RWEC Angle Time Series (window={window}d, threshold={threshold}°)")
    print(f"{'='*width}")

    # Plot rows
    for row in range(height, -1, -1):
        val = min_val + (max_val - min_val) * row / height

        if row == height:
            label = f"{max_val:5.1f}°|"
        elif row == 0:
            label = f"{min_val:5.1f}°|"
        elif abs(val - threshold) < (max_val - min_val) / height:
            label = f"{threshold:5.1f}°|"
        else:
            label = "      |"

        line = ""
        for i, a in enumerate(sampled_angles):
            a_row = int((a - min_val) / (max_val - min_val) * height)
            threshold_row = int((threshold - min_val) / (max_val - min_val) * height)

            if a_row == row:
                line += "*" if a > threshold else "."
            elif row == threshold_row:
                line += "-"
            else:
                line += " "

        print(label + line)

    # X-axis
    print("      +" + "-" * width)
    print(f"       {sampled_dates[0][:7]}{'':>{width-20}}{sampled_dates[-1][:7]}")

    # Stats
    print(f"\nStats: mean={data['stats']['mean']:.1f}°, max={data['stats']['max']:.1f}°, "
          f">{threshold}°: {data['stats']['pct_above_threshold']:.1f}%")
    print(f"Peaks above threshold: {data['peaks_above_threshold']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "baseline":
        results = run_baseline(windows=[60, 252])
        save_results(results, "R030_baseline.json")

        print("\n" + "="*70)
        print("PHASE 1 RESULTS")
        print("="*70)

        for window in [60, 252]:
            print_ascii_plot(results, window)

    elif cmd == "detect":
        print("Phase 2: Not yet implemented")

    elif cmd == "analyze":
        print("Phase 3: Not yet implemented")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
