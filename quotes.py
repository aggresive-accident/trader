#!/usr/bin/env python3
"""
quotes.py - price quote tracking

Saves price history for symbols.
Useful for charting and analysis.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from trader import Trader

QUOTES_DIR = Path(__file__).parent / "quotes"
QUOTES_DIR.mkdir(exist_ok=True)


def get_quote_file(symbol: str) -> Path:
    """Get path to quote history file for symbol"""
    return QUOTES_DIR / f"{symbol.upper()}.jsonl"


def record_quote(symbol: str) -> dict:
    """Record current quote for symbol"""
    trader = Trader()
    quote = trader.get_quote(symbol)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "symbol": quote["symbol"],
        "bid": quote["bid"],
        "ask": quote["ask"],
        "mid": quote["mid"],
        "spread": quote["spread"],
    }

    # Append to file
    quote_file = get_quote_file(symbol)
    with open(quote_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def record_quotes(symbols: list[str]) -> list[dict]:
    """Record quotes for multiple symbols"""
    results = []
    for symbol in symbols:
        try:
            entry = record_quote(symbol)
            results.append(entry)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    return results


def load_history(symbol: str, n: int = 100) -> list[dict]:
    """Load quote history for symbol"""
    quote_file = get_quote_file(symbol)
    if not quote_file.exists():
        return []

    entries = []
    for line in quote_file.read_text().strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except:
                pass

    return entries[-n:] if len(entries) > n else entries


def get_latest(symbol: str) -> dict:
    """Get latest quote for symbol (from history or live)"""
    history = load_history(symbol, 1)
    if history:
        return history[-1]
    return record_quote(symbol)


def list_tracked() -> list[str]:
    """List tracked symbols"""
    symbols = []
    for f in QUOTES_DIR.glob("*.jsonl"):
        symbols.append(f.stem)
    return sorted(symbols)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quote tracking")
    parser.add_argument("command", nargs="?", default="status",
                       choices=["status", "record", "history", "list", "track"])
    parser.add_argument("symbols", nargs="*", help="Symbols to operate on")
    parser.add_argument("-n", type=int, default=10, help="Number of entries")
    args = parser.parse_args()

    if args.command == "list":
        symbols = list_tracked()
        if symbols:
            print(f"Tracked Symbols ({len(symbols)}):")
            for s in symbols:
                history = load_history(s, 1)
                latest = history[-1] if history else None
                if latest:
                    print(f"  {s:6} ${latest['mid']:8.2f}  ({latest['timestamp'][:19]})")
                else:
                    print(f"  {s:6} (no data)")
        else:
            print("No symbols tracked")
        return

    if args.command == "record":
        symbols = args.symbols or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        print(f"Recording quotes for {len(symbols)} symbols...")
        results = record_quotes(symbols)
        for r in results:
            if "error" in r:
                print(f"  {r['symbol']:6} ERROR: {r['error']}")
            else:
                print(f"  {r['symbol']:6} ${r['mid']:8.2f}")
        return

    if args.command == "track":
        if not args.symbols:
            print("Usage: quotes track SYMBOL [SYMBOL...]")
            return
        for symbol in args.symbols:
            entry = record_quote(symbol)
            print(f"Tracking {entry['symbol']}: ${entry['mid']:.2f}")
        return

    if args.command == "history":
        if not args.symbols:
            print("Usage: quotes history SYMBOL")
            return
        symbol = args.symbols[0].upper()
        history = load_history(symbol, args.n)
        if not history:
            print(f"No history for {symbol}")
            return

        print(f"Quote History: {symbol} (last {len(history)})")
        print("-" * 50)
        for h in history:
            ts = h["timestamp"][:19]
            print(f"{ts}  ${h['mid']:8.2f}  spread: ${h['spread']:.2f}")
        return

    # Default: status
    symbols = list_tracked()
    print(f"Quote Tracking Status")
    print("-" * 40)
    print(f"Tracked: {len(symbols)} symbols")
    if symbols:
        print(f"Symbols: {', '.join(symbols[:10])}")
        if len(symbols) > 10:
            print(f"         ...and {len(symbols) - 10} more")


if __name__ == "__main__":
    main()
