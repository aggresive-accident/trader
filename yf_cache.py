#!/usr/bin/env python3
# See CODEBASE.md for public interface documentation
"""
yf_cache.py - yfinance data layer with parquet caching

Public interface:
- fetch_bars(symbol, period='5y') -> pd.DataFrame
- fetch_info(symbol) -> dict
- get_sp500_symbols() -> list[str]
- get_cached_symbols() -> list[str]
- clear_cache(symbol=None)

Cache location: data/yf_cache/
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Lazy import yfinance
_yf = None
def get_yf():
    global _yf
    if _yf is None:
        import yfinance as yf
        _yf = yf
    return _yf

# Cache directories
CACHE_DIR = Path(__file__).parent / "data" / "yf_cache"
BARS_DIR = CACHE_DIR / "bars"
INFO_DIR = CACHE_DIR / "info"

# Ensure directories exist
BARS_DIR.mkdir(parents=True, exist_ok=True)
INFO_DIR.mkdir(parents=True, exist_ok=True)

# Cache settings
BARS_CACHE_DAYS = 1  # Re-fetch bars if older than 1 day
INFO_CACHE_DAYS = 7  # Re-fetch info if older than 7 days
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def _retry_on_failure(func, *args, retries=MAX_RETRIES, **kwargs):
    """Retry a function on failure with exponential backoff."""
    last_error = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                log.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    log.error(f"All {retries} attempts failed: {last_error}")
    raise last_error


def _is_cache_fresh(cache_path: Path, max_age_days: int) -> bool:
    """Check if cache file exists and is fresh."""
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age < timedelta(days=max_age_days)


def fetch_bars(symbol: str, period: str = '5y', force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a symbol.

    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        period: yfinance period string ('1y', '2y', '5y', 'max')
        force_refresh: Bypass cache

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    cache_path = BARS_DIR / f"{symbol}_{period}.parquet"

    # Check cache
    if not force_refresh and _is_cache_fresh(cache_path, BARS_CACHE_DAYS):
        try:
            df = pd.read_parquet(cache_path)
            log.debug(f"Loaded {symbol} from cache ({len(df)} bars)")
            return df
        except Exception as e:
            log.warning(f"Failed to read cache for {symbol}: {e}")

    # Fetch from yfinance
    log.info(f"Fetching {symbol} bars from yfinance (period={period})")

    def _fetch():
        yf = get_yf()
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        return df

    try:
        df = _retry_on_failure(_fetch)
    except Exception as e:
        log.error(f"Failed to fetch {symbol}: {e}")
        # Return empty DataFrame on failure
        return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    # Normalize column names and format
    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # Rename 'date' column if needed (yfinance uses 'Date' as index name)
    if 'date' not in df.columns and 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})

    # Keep only OHLCV columns
    keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep_cols if c in df.columns]]

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        log.debug(f"Cached {symbol} ({len(df)} bars)")
    except Exception as e:
        log.warning(f"Failed to cache {symbol}: {e}")

    return df


def fetch_info(symbol: str, force_refresh: bool = False) -> dict:
    """
    Fetch fundamental info for a symbol.

    Args:
        symbol: Stock ticker
        force_refresh: Bypass cache

    Returns:
        Dict with keys like: marketCap, trailingPE, priceToBook, dividendYield, etc.
    """
    cache_path = INFO_DIR / f"{symbol}.json"

    # Check cache
    if not force_refresh and _is_cache_fresh(cache_path, INFO_CACHE_DAYS):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to read info cache for {symbol}: {e}")

    # Fetch from yfinance
    log.info(f"Fetching {symbol} info from yfinance")

    def _fetch():
        yf = get_yf()
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or info.get('regularMarketPrice') is None:
            raise ValueError(f"No info returned for {symbol}")
        return info

    try:
        info = _retry_on_failure(_fetch)
    except Exception as e:
        log.error(f"Failed to fetch info for {symbol}: {e}")
        return {}

    # Extract relevant fields
    relevant_keys = [
        'marketCap', 'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
        'dividendYield', 'trailingEps', 'forwardEps', 'returnOnEquity', 'returnOnAssets',
        'debtToEquity', 'currentRatio', 'quickRatio', 'revenueGrowth', 'earningsGrowth',
        'profitMargins', 'operatingMargins', 'grossMargins', 'beta', 'fiftyTwoWeekHigh',
        'fiftyTwoWeekLow', 'sector', 'industry', 'shortName', 'longName'
    ]

    filtered = {k: info.get(k) for k in relevant_keys if k in info}
    filtered['_fetched_at'] = datetime.now().isoformat()
    filtered['symbol'] = symbol

    # Save to cache
    try:
        with open(cache_path, 'w') as f:
            json.dump(filtered, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"Failed to cache info for {symbol}: {e}")

    return filtered


def get_sp500_symbols() -> list[str]:
    """
    Get current S&P 500 constituents.

    Returns:
        List of ticker symbols
    """
    cache_path = CACHE_DIR / "sp500_symbols.json"

    # Check cache (refresh weekly)
    if _is_cache_fresh(cache_path, 7):
        try:
            with open(cache_path) as f:
                data = json.load(f)
                return data['symbols']
        except Exception as e:
            log.warning(f"Failed to read S&P 500 cache: {e}")

    # Fetch from Wikipedia
    log.info("Fetching S&P 500 symbols from Wikipedia")

    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Symbol'].tolist()

        # Clean up symbols (some have dots that need to be dashes for yfinance)
        symbols = [s.replace('.', '-') for s in symbols]

        # Cache
        with open(cache_path, 'w') as f:
            json.dump({'symbols': symbols, 'fetched_at': datetime.now().isoformat()}, f)

        log.info(f"Found {len(symbols)} S&P 500 symbols")
        return symbols

    except Exception as e:
        log.error(f"Failed to fetch S&P 500 symbols: {e}")
        # Fallback to a basic list
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'V', 'XOM', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV',
            'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ABT', 'DHR'
        ]


def get_cached_symbols() -> list[str]:
    """Get list of symbols with cached bar data."""
    symbols = set()
    for f in BARS_DIR.glob("*.parquet"):
        # Extract symbol from filename (format: SYMBOL_period.parquet)
        parts = f.stem.rsplit('_', 1)
        if parts:
            symbols.add(parts[0])
    return sorted(symbols)


def clear_cache(symbol: str = None):
    """
    Clear cache for a symbol or all symbols.

    Args:
        symbol: If None, clear all cache
    """
    if symbol:
        for f in BARS_DIR.glob(f"{symbol}_*.parquet"):
            f.unlink()
        info_file = INFO_DIR / f"{symbol}.json"
        if info_file.exists():
            info_file.unlink()
        log.info(f"Cleared cache for {symbol}")
    else:
        import shutil
        shutil.rmtree(BARS_DIR, ignore_errors=True)
        shutil.rmtree(INFO_DIR, ignore_errors=True)
        BARS_DIR.mkdir(parents=True, exist_ok=True)
        INFO_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Cleared all cache")


def seed_cache(symbols: list[str] = None, period: str = '5y'):
    """
    Pre-fetch bars for multiple symbols.

    Args:
        symbols: List of symbols (defaults to S&P 500)
        period: yfinance period
    """
    if symbols is None:
        symbols = get_sp500_symbols()

    log.info(f"Seeding cache for {len(symbols)} symbols")
    success = 0
    failed = []

    for i, symbol in enumerate(symbols):
        try:
            df = fetch_bars(symbol, period)
            if not df.empty:
                success += 1
            else:
                failed.append(symbol)
        except Exception as e:
            failed.append(symbol)
            log.error(f"Failed {symbol}: {e}")

        if (i + 1) % 50 == 0:
            log.info(f"Progress: {i + 1}/{len(symbols)} ({success} success, {len(failed)} failed)")

        # Rate limiting
        time.sleep(0.2)

    log.info(f"Seeding complete: {success} success, {len(failed)} failed")
    if failed:
        log.warning(f"Failed symbols: {failed[:20]}...")


def cache_status() -> dict:
    """Get cache statistics."""
    bars_files = list(BARS_DIR.glob("*.parquet"))
    info_files = list(INFO_DIR.glob("*.json"))

    total_size = sum(f.stat().st_size for f in bars_files + info_files)

    return {
        'bars_count': len(bars_files),
        'info_count': len(info_files),
        'total_size_mb': round(total_size / 1024 / 1024, 2),
        'cache_dir': str(CACHE_DIR)
    }


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 yf_cache.py <command> [args]")
        print("Commands:")
        print("  status              - Show cache statistics")
        print("  seed [period]       - Seed cache with S&P 500 (default: 5y)")
        print("  fetch <symbol>      - Fetch and display bars")
        print("  info <symbol>       - Fetch and display info")
        print("  sp500               - List S&P 500 symbols")
        print("  clear [symbol]      - Clear cache")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        status = cache_status()
        print(f"Bars cached: {status['bars_count']}")
        print(f"Info cached: {status['info_count']}")
        print(f"Total size: {status['total_size_mb']} MB")
        print(f"Cache dir: {status['cache_dir']}")

    elif cmd == "seed":
        period = sys.argv[2] if len(sys.argv) > 2 else '5y'
        seed_cache(period=period)

    elif cmd == "fetch":
        if len(sys.argv) < 3:
            print("Usage: python3 yf_cache.py fetch <symbol>")
            sys.exit(1)
        symbol = sys.argv[2].upper()
        df = fetch_bars(symbol)
        print(df.tail(20).to_string())
        print(f"\nTotal: {len(df)} bars")

    elif cmd == "info":
        if len(sys.argv) < 3:
            print("Usage: python3 yf_cache.py info <symbol>")
            sys.exit(1)
        symbol = sys.argv[2].upper()
        info = fetch_info(symbol)
        for k, v in info.items():
            print(f"{k}: {v}")

    elif cmd == "sp500":
        symbols = get_sp500_symbols()
        print(f"S&P 500 ({len(symbols)} symbols):")
        for i in range(0, len(symbols), 10):
            print("  " + ", ".join(symbols[i:i+10]))

    elif cmd == "clear":
        symbol = sys.argv[2].upper() if len(sys.argv) > 2 else None
        clear_cache(symbol)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
