#!/usr/bin/env python3
"""
watchlist.py - manage symbols of interest

Track symbols I'm watching.
Separate from the scanner's default list.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"

# Default watchlist
DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
]


def load_watchlist() -> dict:
    """Load watchlist"""
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text())
        except:
            pass
    return {
        "symbols": DEFAULT_WATCHLIST.copy(),
        "notes": {},
        "updated": datetime.now().isoformat(),
    }


def save_watchlist(data: dict):
    """Save watchlist"""
    data["updated"] = datetime.now().isoformat()
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))


def add_symbol(symbol: str, note: str = ""):
    """Add symbol to watchlist"""
    data = load_watchlist()
    symbol = symbol.upper()
    if symbol not in data["symbols"]:
        data["symbols"].append(symbol)
    if note:
        data["notes"][symbol] = note
    save_watchlist(data)
    return data


def remove_symbol(symbol: str):
    """Remove symbol from watchlist"""
    data = load_watchlist()
    symbol = symbol.upper()
    if symbol in data["symbols"]:
        data["symbols"].remove(symbol)
    if symbol in data["notes"]:
        del data["notes"][symbol]
    save_watchlist(data)
    return data


def get_symbols() -> list[str]:
    """Get watchlist symbols"""
    data = load_watchlist()
    return data["symbols"]


def get_notes() -> dict:
    """Get notes for symbols"""
    data = load_watchlist()
    return data.get("notes", {})


def scan_watchlist():
    """Scan watchlist symbols"""
    # Add venv to path
    VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
    if VENV_PATH.exists():
        sys.path.insert(0, str(VENV_PATH))

    from scanner import Scanner

    symbols = get_symbols()
    notes = get_notes()

    scanner = Scanner()
    results = scanner.calculate_momentum(symbols)

    return results, notes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Watchlist management")
    parser.add_argument("command", nargs="?", default="list",
                       choices=["list", "add", "remove", "scan", "reset"])
    parser.add_argument("symbols", nargs="*", help="Symbols")
    parser.add_argument("-n", "--note", default="", help="Note for symbol")
    args = parser.parse_args()

    if args.command == "list":
        data = load_watchlist()
        symbols = data["symbols"]
        notes = data.get("notes", {})

        print(f"Watchlist ({len(symbols)} symbols)")
        print("-" * 40)
        for s in symbols:
            note = notes.get(s, "")
            if note:
                print(f"  {s:6} - {note}")
            else:
                print(f"  {s}")

    elif args.command == "add":
        if not args.symbols:
            print("Usage: watchlist add SYMBOL [SYMBOL...] [-n NOTE]")
            return
        for symbol in args.symbols:
            data = add_symbol(symbol, args.note)
            print(f"Added {symbol.upper()}")

    elif args.command == "remove":
        if not args.symbols:
            print("Usage: watchlist remove SYMBOL [SYMBOL...]")
            return
        for symbol in args.symbols:
            data = remove_symbol(symbol)
            print(f"Removed {symbol.upper()}")

    elif args.command == "scan":
        results, notes = scan_watchlist()
        if not results:
            print("No results")
            return

        print(f"Watchlist Scan ({len(results)} symbols)")
        print("-" * 55)
        for r in results:
            signal_icon = {
                "STRONG BUY": "🟢",
                "BUY": "🟡",
                "HOLD": "⚪",
                "WEAK": "🟠",
                "AVOID": "🔴",
            }.get(r["signal"], "")

            note = notes.get(r["symbol"], "")
            note_str = f"  [{note[:20]}]" if note else ""

            print(f"{signal_icon} {r['symbol']:6} ${r['price']:8.2f}  {r['momentum']:+6.2f}%  {r['signal']}{note_str}")

    elif args.command == "reset":
        data = {
            "symbols": DEFAULT_WATCHLIST.copy(),
            "notes": {},
            "updated": datetime.now().isoformat(),
        }
        save_watchlist(data)
        print(f"Reset to default ({len(DEFAULT_WATCHLIST)} symbols)")


if __name__ == "__main__":
    main()
