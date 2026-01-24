#!/usr/bin/env python3
"""
journal.py - trade journal

Record reasoning for trades.
Track what worked and what didn't.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

JOURNAL_FILE = Path(__file__).parent / "trade_journal.jsonl"


def add_entry(
    action: str,
    symbol: str,
    qty: float = None,
    price: float = None,
    reason: str = "",
    signals: dict = None,
) -> dict:
    """Add a journal entry"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action.upper(),
        "symbol": symbol.upper(),
        "qty": qty,
        "price": price,
        "reason": reason,
        "signals": signals or {},
    }

    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def load_entries(symbol: str = None, n: int = 50) -> list[dict]:
    """Load journal entries"""
    if not JOURNAL_FILE.exists():
        return []

    entries = []
    for line in JOURNAL_FILE.read_text().strip().split("\n"):
        if line:
            try:
                entry = json.loads(line)
                if symbol is None or entry.get("symbol") == symbol.upper():
                    entries.append(entry)
            except:
                pass

    return entries[-n:] if len(entries) > n else entries


def buy(symbol: str, qty: float, price: float, reason: str, signals: dict = None):
    """Record a buy"""
    return add_entry("BUY", symbol, qty, price, reason, signals)


def sell(symbol: str, qty: float, price: float, reason: str):
    """Record a sell"""
    return add_entry("SELL", symbol, qty, price, reason)


def note(symbol: str, note: str):
    """Add a note about a symbol"""
    return add_entry("NOTE", symbol, reason=note)


def review(symbol: str) -> dict:
    """Review history for a symbol"""
    entries = load_entries(symbol)

    if not entries:
        return {"symbol": symbol, "trades": 0}

    buys = [e for e in entries if e["action"] == "BUY"]
    sells = [e for e in entries if e["action"] == "SELL"]
    notes = [e for e in entries if e["action"] == "NOTE"]

    # Calculate P&L if we have matching buys and sells
    total_pl = 0
    if buys and sells:
        for sell in sells:
            # Find corresponding buy
            for buy in buys:
                if buy["timestamp"] < sell["timestamp"]:
                    if buy["price"] and sell["price"] and buy["qty"] and sell["qty"]:
                        pl = (sell["price"] - buy["price"]) * min(buy["qty"], sell["qty"])
                        total_pl += pl
                    break

    return {
        "symbol": symbol,
        "buys": len(buys),
        "sells": len(sells),
        "notes": len(notes),
        "estimated_pl": total_pl,
        "entries": entries,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trade journal")
    parser.add_argument("command", nargs="?", default="list",
                       choices=["list", "buy", "sell", "note", "review", "clear"])
    parser.add_argument("symbol", nargs="?", help="Symbol")
    parser.add_argument("-q", "--qty", type=float, help="Quantity")
    parser.add_argument("-p", "--price", type=float, help="Price")
    parser.add_argument("-r", "--reason", default="", help="Reason")
    parser.add_argument("-n", type=int, default=10, help="Number of entries")
    args = parser.parse_args()

    if args.command == "list":
        entries = load_entries(args.symbol, args.n)
        if not entries:
            print("No journal entries")
            return

        print(f"Trade Journal (last {len(entries)})")
        print("-" * 60)
        for e in entries:
            ts = e["timestamp"][:10]
            action = e["action"]
            symbol = e["symbol"]
            reason = e.get("reason", "")[:40]

            if action == "NOTE":
                print(f"{ts}  {action:4}  {symbol:6}  {reason}")
            else:
                qty = e.get("qty", 0) or 0
                price = e.get("price", 0) or 0
                print(f"{ts}  {action:4}  {symbol:6}  {qty:5.0f} @ ${price:8.2f}  {reason}")

    elif args.command == "buy":
        if not args.symbol:
            print("Usage: journal buy SYMBOL -q QTY -p PRICE -r REASON")
            return
        entry = buy(args.symbol, args.qty, args.price, args.reason)
        print(f"Recorded: BUY {entry['symbol']} x{entry['qty']}")

    elif args.command == "sell":
        if not args.symbol:
            print("Usage: journal sell SYMBOL -q QTY -p PRICE -r REASON")
            return
        entry = sell(args.symbol, args.qty, args.price, args.reason)
        print(f"Recorded: SELL {entry['symbol']} x{entry['qty']}")

    elif args.command == "note":
        if not args.symbol:
            print("Usage: journal note SYMBOL -r NOTE")
            return
        entry = note(args.symbol, args.reason)
        print(f"Recorded note for {entry['symbol']}")

    elif args.command == "review":
        if not args.symbol:
            print("Usage: journal review SYMBOL")
            return
        result = review(args.symbol)
        print(f"Review: {result['symbol']}")
        print("-" * 40)
        print(f"Buys:  {result['buys']}")
        print(f"Sells: {result['sells']}")
        print(f"Notes: {result['notes']}")
        if result['estimated_pl']:
            print(f"Est. P&L: ${result['estimated_pl']:+,.2f}")

        if result["entries"]:
            print("\nHistory:")
            for e in result["entries"][-5:]:
                ts = e["timestamp"][:10]
                action = e["action"]
                reason = e.get("reason", "")[:50]
                print(f"  {ts} {action}: {reason}")

    elif args.command == "clear":
        if JOURNAL_FILE.exists():
            JOURNAL_FILE.unlink()
        print("Journal cleared")


if __name__ == "__main__":
    main()
