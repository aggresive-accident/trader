#!/usr/bin/env python3
"""
bar_cache.py - Local parquet cache for historical daily bars.

Cache structure:
  data/bars/{SYMBOL}.parquet - one file per symbol
  Schema: date (datetime64), open, high, low, close, volume

Usage:
  from bar_cache import cache_symbols, load_bars, update_cache

  # Seed cache for 200 symbols
  cache_symbols(SP500_TOP200, "2022-01-01", "2026-01-27")

  # Load from cache (never hits Alpaca)
  df = load_bars("AAPL", "2023-01-01", "2024-01-01")

  # Append new bars since last cached date
  update_cache()

CLI:
  python3 bar_cache.py seed          # seed top 200
  python3 bar_cache.py update        # append new bars
  python3 bar_cache.py status        # cache stats
  python3 bar_cache.py load AAPL     # print bars for symbol
"""

import sys
import time
import gc
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import load_keys

CACHE_DIR = Path(__file__).parent / "data" / "bars"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

META_FILE = CACHE_DIR / "_meta.json"

log = logging.getLogger("bar_cache")

# Top ~200 S&P 500 constituents by market cap (approximate, stable list)
SP500_TOP200 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "LLY", "AVGO", "JPM", "UNH",
    "XOM", "V", "TSLA", "MA", "PG", "JNJ", "COST", "HD", "ABBV", "MRK",
    "WMT", "NFLX", "AMD", "CRM", "BAC", "ORCL", "PEP", "KO", "CVX", "TMO",
    "ACN", "LIN", "MCD", "CSCO", "ABT", "ADBE", "WFC", "DHR", "TXN", "PM",
    "INTC", "NEE", "CMCSA", "INTU", "DIS", "BMY", "VZ", "QCOM", "RTX", "AMGN",
    "SPGI", "HON", "GE", "ISRG", "PFE", "SYK", "NOW", "BKNG", "ELV", "BLK",
    "AXP", "T", "AMAT", "MDLZ", "VRTX", "GILD", "LRCX", "PANW", "ADI", "DE",
    "MU", "REGN", "SLB", "TMUS", "ETN", "BSX", "ADP", "CB", "ZTS", "SCHW",
    "PLD", "FI", "SBUX", "SO", "CI", "MO", "BDX", "KLAC", "MMC", "DUK",
    "ICE", "CME", "CL", "EQIX", "SHW", "CMG", "MCK", "PNC", "TGT", "NOC",
    "SNPS", "PYPL", "APD", "USB", "HCA", "CDNS", "ITW", "TT", "AON", "EMR",
    "MSI", "GD", "ECL", "CEG", "FDX", "ABNB", "ORLY", "MPC", "MAR", "COF",
    "WM", "CTAS", "AJG", "MCO", "AZO", "ROP", "TDG", "SPG", "SRE", "AFL",
    "OKE", "PCAR", "AEP", "CCI", "CARR", "PSA", "NSC", "HLT", "FCX", "TFC",
    "AIG", "D", "JCI", "BK", "KMB", "PAYX", "GIS", "COR", "FAST", "LHX",
    "WELL", "ALL", "RSG", "HES", "PRU", "AME", "DLR", "CTVA", "KHC", "A",
    "MSCI", "YUM", "OTIS", "VRSK", "EW", "KR", "IDXX", "MTD", "GEHC", "EA",
    "URI", "DOW", "HPQ", "KEYS", "RMD", "MCHP", "STZ", "DD", "IFF", "GPC",
    "ON", "ANSS", "CDW", "DHI", "WAB", "ROK", "EXC", "XEL", "ED", "AWK",
    "WEC", "ES", "PPL", "FE", "TSCO", "BAX", "BRO", "WRB", "TER", "CINF",
]

PARQUET_SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
])


def _get_data_client() -> StockHistoricalDataClient:
    k, s = load_keys()
    return StockHistoricalDataClient(k, s)


def _symbol_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}.parquet"


def _alpaca_bars_to_df(bars: list) -> pd.DataFrame:
    """Convert alpaca bar objects to a DataFrame."""
    rows = []
    for b in bars:
        rows.append({
            "date": b.timestamp,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return df


def _fetch_batch(client: StockHistoricalDataClient, symbols: list,
                 start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch bars for a batch of symbols from Alpaca. Returns {symbol: DataFrame}."""
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
    )
    result = client.get_stock_bars(request)
    out = {}
    for sym in symbols:
        if sym in result.data:
            out[sym] = _alpaca_bars_to_df(result.data[sym])
    return out


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pq.read_table(path).to_pandas()


def _write_parquet(df: pd.DataFrame, path: Path):
    """Write a DataFrame to parquet."""
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _date_range_cached(path: Path) -> tuple[str, str] | None:
    """Return (min_date, max_date) from a cached parquet, or None."""
    if not path.exists():
        return None
    df = _read_parquet(path)
    if df.empty:
        return None
    return (df["date"].min().isoformat()[:10], df["date"].max().isoformat()[:10])


# === Public API ===

def cache_symbols(symbols: list[str], start: str, end: str,
                  batch_size: int = 100, force: bool = False) -> dict:
    """
    Fetch and cache bars for a list of symbols.

    Skips symbols that already cover the requested date range unless force=True.
    Batches requests to stay under memory limits.

    Returns stats dict.
    """
    client = _get_data_client()
    stats = {"fetched": 0, "skipped": 0, "failed": [], "bars": 0, "time": 0}
    t0 = time.time()

    # Filter to symbols that need fetching
    to_fetch = []
    for sym in symbols:
        path = _symbol_path(sym)
        if not force:
            cached = _date_range_cached(path)
            if cached and cached[0] <= start and cached[1] >= end[:10]:
                stats["skipped"] += 1
                continue
        to_fetch.append(sym)

    log.info(f"Cache: {len(to_fetch)} to fetch, {stats['skipped']} already cached")

    # Batch fetch
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(to_fetch) + batch_size - 1) // batch_size
        log.info(f"Batch {batch_num}/{total_batches}: {len(batch)} symbols")

        try:
            results = _fetch_batch(client, batch, start, end)
            for sym, df in results.items():
                path = _symbol_path(sym)

                # Merge with existing cache if present
                if path.exists() and not force:
                    existing = _read_parquet(path)
                    df = pd.concat([existing, df]).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

                _write_parquet(df, path)
                stats["fetched"] += 1
                stats["bars"] += len(df)

            # Flag symbols that returned no data
            for sym in batch:
                if sym not in results:
                    stats["failed"].append(sym)

        except Exception as e:
            log.error(f"Batch {batch_num} failed: {e}")
            stats["failed"].extend(batch)

        # Free memory between batches
        del results
        gc.collect()

    stats["time"] = round(time.time() - t0, 1)
    log.info(f"Cache complete: {stats['fetched']} symbols, {stats['bars']:,} bars in {stats['time']}s")
    _save_meta(stats)
    return stats


def load_bars(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Load bars from local cache. Never hits Alpaca.

    Returns empty DataFrame if symbol not cached.
    """
    path = _symbol_path(symbol)
    if not path.exists():
        return pd.DataFrame()

    df = _read_parquet(path)
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def update_cache(symbols: list[str] = None) -> dict:
    """
    Append new bars since last cached date for all cached symbols.

    If symbols is None, updates all symbols in the cache directory.
    """
    if symbols is None:
        symbols = [p.stem for p in CACHE_DIR.glob("*.parquet")]

    if not symbols:
        log.info("No cached symbols to update")
        return {"updated": 0}

    client = _get_data_client()
    end = datetime.now().strftime("%Y-%m-%d")
    stats = {"updated": 0, "bars_added": 0, "time": 0}
    t0 = time.time()

    # Group by last cached date to batch efficiently
    to_fetch = []
    for sym in symbols:
        cached = _date_range_cached(_symbol_path(sym))
        if cached:
            # Fetch from day after last cached date
            last = datetime.fromisoformat(cached[1]) + timedelta(days=1)
            if last.strftime("%Y-%m-%d") < end:
                to_fetch.append((sym, last.strftime("%Y-%m-%d")))
        else:
            to_fetch.append((sym, "2022-01-01"))

    if not to_fetch:
        log.info("All symbols up to date")
        return stats

    # Batch by start date (most will share the same start)
    by_start = {}
    for sym, start in to_fetch:
        by_start.setdefault(start, []).append(sym)

    for start_date, syms in by_start.items():
        for i in range(0, len(syms), 100):
            batch = syms[i:i + 100]
            try:
                results = _fetch_batch(client, batch, start_date, end)
                for sym, df in results.items():
                    if df.empty:
                        continue
                    path = _symbol_path(sym)
                    if path.exists():
                        existing = _read_parquet(path)
                        new_bars = len(df)
                        df = pd.concat([existing, df]).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
                    else:
                        new_bars = len(df)
                    _write_parquet(df, path)
                    stats["updated"] += 1
                    stats["bars_added"] += new_bars
                del results
                gc.collect()
            except Exception as e:
                log.error(f"Update batch failed: {e}")

    stats["time"] = round(time.time() - t0, 1)
    log.info(f"Update complete: {stats['updated']} symbols, {stats['bars_added']} new bars in {stats['time']}s")
    return stats


def cache_status() -> dict:
    """Return cache statistics."""
    files = list(CACHE_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)
    total_bars = 0
    date_ranges = {}

    for f in files:
        df = _read_parquet(f)
        total_bars += len(df)
        if not df.empty:
            date_ranges[f.stem] = {
                "bars": len(df),
                "start": df["date"].min().isoformat()[:10],
                "end": df["date"].max().isoformat()[:10],
            }

    return {
        "symbols": len(files),
        "total_bars": total_bars,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "avg_bars_per_symbol": round(total_bars / len(files), 0) if files else 0,
        "date_ranges_sample": dict(list(date_ranges.items())[:5]),
    }


def _save_meta(stats: dict):
    """Save last operation metadata."""
    meta = {}
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
    meta["last_operation"] = {
        "time": datetime.now().isoformat(),
        **stats,
    }
    META_FILE.write_text(json.dumps(meta, indent=2, default=str))


# === CLI ===

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Bar data cache manager")
    parser.add_argument("command", choices=["seed", "update", "status", "load"],
                        help="Operation to perform")
    parser.add_argument("symbol", nargs="?", help="Symbol for load command")
    parser.add_argument("--start", default="2022-01-01", help="Start date (default: 2022-01-01)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--symbols", help="Comma-separated symbol list (overrides default)")
    parser.add_argument("--batch-size", type=int, default=100, help="Symbols per API batch")
    args = parser.parse_args()

    if args.command == "seed":
        symbols = args.symbols.split(",") if args.symbols else SP500_TOP200
        print(f"Seeding cache: {len(symbols)} symbols, {args.start} to {args.end}")
        stats = cache_symbols(symbols, args.start, args.end,
                              batch_size=args.batch_size, force=args.force)
        print(f"\nResults:")
        print(f"  Fetched: {stats['fetched']} symbols")
        print(f"  Skipped: {stats['skipped']} (already cached)")
        print(f"  Failed:  {len(stats['failed'])} {stats['failed'][:10] if stats['failed'] else ''}")
        print(f"  Bars:    {stats['bars']:,}")
        print(f"  Time:    {stats['time']}s")

    elif args.command == "update":
        symbols = args.symbols.split(",") if args.symbols else None
        stats = update_cache(symbols)
        print(f"Updated: {stats['updated']} symbols, {stats['bars_added']} new bars in {stats['time']}s")

    elif args.command == "status":
        status = cache_status()
        print(f"Cache: {status['symbols']} symbols, {status['total_bars']:,} bars")
        print(f"Size:  {status['total_size_mb']} MB")
        print(f"Avg:   {status['avg_bars_per_symbol']:.0f} bars/symbol")
        if status["date_ranges_sample"]:
            print(f"\nSample:")
            for sym, info in status["date_ranges_sample"].items():
                print(f"  {sym}: {info['bars']} bars ({info['start']} to {info['end']})")

    elif args.command == "load":
        if not args.symbol:
            print("Usage: bar_cache.py load SYMBOL [--start DATE] [--end DATE]")
            return
        df = load_bars(args.symbol, args.start, args.end)
        if df.empty:
            print(f"No cached data for {args.symbol}")
        else:
            print(f"{args.symbol}: {len(df)} bars ({df['date'].min().date()} to {df['date'].max().date()})")
            print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
