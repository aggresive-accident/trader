#!/usr/bin/env python3
"""
thesis_execute.py - Manual execution and tracking for thesis trades.

Completely separate from autopilot/strategy zoo. Uses thesis_trades.json
for state and Alpaca API for execution.

Usage:
  python3 thesis_execute.py buy GME 10000        # buy $10k worth at market
  python3 thesis_execute.py sell GME 25%          # trim 25% of position
  python3 thesis_execute.py sell GME 100           # sell 100 shares
  python3 thesis_execute.py status                # show all thesis positions
  python3 thesis_execute.py check GME             # detailed check for one symbol
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import load_keys
from router import load_config

THESIS_FILE = Path(__file__).parent / "thesis_trades.json"
LOG_FILE = Path(__file__).parent / "thesis_trades.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("thesis")


# === Data ===

def load_thesis() -> dict:
    if THESIS_FILE.exists():
        return json.loads(THESIS_FILE.read_text())
    return {"meta": {}, "trades": []}


def save_thesis(data: dict):
    THESIS_FILE.write_text(json.dumps(data, indent=2))


def find_trade(data: dict, symbol: str) -> dict | None:
    for t in data["trades"]:
        if t["symbol"].upper() == symbol.upper():
            return t
    return None


# === Validation ===

def validate_symbol(symbol: str) -> bool:
    """Confirm symbol is in exclusions list and not managed by autopilot."""
    config = load_config()
    exclusions = set(config.get("exclusions", []))

    if symbol.upper() not in exclusions:
        print(f"BLOCKED: {symbol} is not in router_config.json exclusions list.")
        print(f"Current exclusions: {exclusions or 'none'}")
        print(f"Add it to exclusions before executing thesis trades.")
        return False

    # Check it's not in the autopilot's strategy zoo
    zoo_symbols = set(config.get("symbols", []))
    if symbol.upper() in zoo_symbols:
        print(f"BLOCKED: {symbol} is in the autopilot symbol universe.")
        print(f"Remove it from router_config.json symbols before trading it as thesis.")
        return False

    return True


def get_client() -> TradingClient:
    k, s = load_keys()
    return TradingClient(k, s, paper=True)


def get_alpaca_position(client: TradingClient, symbol: str) -> dict | None:
    """Get current Alpaca position for a symbol, or None."""
    try:
        pos = client.get_open_position(symbol)
        return {
            "symbol": pos.symbol,
            "qty": float(pos.qty),
            "avg_entry": float(pos.avg_entry_price),
            "current_price": float(pos.current_price),
            "market_value": float(pos.market_value),
            "unrealized_pnl": float(pos.unrealized_pl),
            "unrealized_pnl_pct": float(pos.unrealized_plpc) * 100,
        }
    except Exception:
        return None


def get_quote_price(client: TradingClient, symbol: str) -> float | None:
    """Get latest quote price. Uses bid/ask midpoint with sanity check, falls back to cached close."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        k, s = load_keys()
        dc = StockHistoricalDataClient(k, s)
        from alpaca.data.requests import StockLatestQuoteRequest
        quote = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        if symbol in quote:
            ask = float(quote[symbol].ask_price)
            bid = float(quote[symbol].bid_price)
            # Sanity: if spread > 50% of bid, quote is stale/unreliable
            if bid > 0 and ask > 0 and (ask - bid) / bid < 0.5:
                return round((ask + bid) / 2, 2)
            # Wide spread — use bid if available
            if bid > 0:
                return round(bid, 2)
    except Exception:
        pass
    # Fall back to cached last close
    try:
        from bar_cache import load_bars
        df = load_bars(symbol)
        if not df.empty:
            return round(float(df["close"].iloc[-1]), 2)
    except Exception:
        pass
    return None


# === Commands ===

def cmd_buy(symbol: str, dollar_amount: float, auto_confirm: bool = False):
    """Buy $X worth of a symbol at market."""
    symbol = symbol.upper()

    if not validate_symbol(symbol):
        return

    data = load_thesis()
    trade = find_trade(data, symbol)
    if not trade:
        print(f"No thesis trade entry for {symbol} in thesis_trades.json.")
        print(f"Add it first, then execute.")
        return

    if trade["status"] == "open":
        print(f"{symbol} already has an open position.")
        pos = get_alpaca_position(get_client(), symbol)
        if pos:
            print(f"  Current: {pos['qty']} shares @ ${pos['current_price']:.2f}")
        print(f"Use 'sell' to trim, or add a new trade entry for averaging in.")
        return

    client = get_client()

    # Get current price for share calculation
    price = get_quote_price(client, symbol)
    if not price:
        print(f"Could not get quote for {symbol}. Market may be closed.")
        return

    shares = int(dollar_amount / price)
    if shares < 1:
        print(f"${dollar_amount:.0f} / ${price:.2f} = {dollar_amount/price:.1f} shares. Need at least 1.")
        return

    actual_notional = shares * price
    print(f"\n  THESIS BUY: {symbol}")
    print(f"  Shares:    {shares}")
    print(f"  Price:     ~${price:.2f}")
    print(f"  Notional:  ~${actual_notional:,.2f}")
    print(f"  Thesis:    {trade['thesis']['summary']}")
    print()

    if not auto_confirm:
        print("  [Preview only - pass --confirm to execute]")
        return

    # Execute
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)
        log.info(f"BUY {shares} {symbol} @ ~${price:.2f} (order {result.id})")

        # Update thesis_trades.json
        trade["status"] = "open"
        trade["entry"]["date"] = datetime.now().strftime("%Y-%m-%d")
        trade["entry"]["price"] = round(price, 2)
        trade["entry"]["shares"] = shares
        trade["entry"]["notional"] = round(actual_notional, 2)
        trade["entry"]["order_id"] = str(result.id)
        trade["entry"]["timestamp"] = datetime.now().isoformat()
        save_thesis(data)

        print(f"\n  EXECUTED: BUY {shares} {symbol} @ ~${price:.2f}")
        print(f"  Order ID: {result.id}")
        print(f"  thesis_trades.json updated.")
        print(f"\n  Note: fill price may differ. Run 'status' after fill to verify.")

    except Exception as e:
        log.error(f"Failed to buy {symbol}: {e}")
        print(f"  FAILED: {e}")


def cmd_sell(symbol: str, amount_str: str, auto_confirm: bool = False):
    """Sell shares - either percentage ('25%') or share count ('100')."""
    symbol = symbol.upper()

    if not validate_symbol(symbol):
        return

    data = load_thesis()
    trade = find_trade(data, symbol)
    if not trade:
        print(f"No thesis trade for {symbol}.")
        return

    if trade["status"] not in ("open", "partial"):
        print(f"{symbol} status is '{trade['status']}' - nothing to sell.")
        return

    client = get_client()
    pos = get_alpaca_position(client, symbol)
    if not pos:
        print(f"No Alpaca position for {symbol}.")
        return

    current_qty = pos["qty"]
    entry_price = trade["entry"].get("price", pos["avg_entry"])

    # Parse amount
    if amount_str.endswith("%"):
        pct = float(amount_str.rstrip("%")) / 100
        sell_qty = int(current_qty * pct)
        reason = f"trim {amount_str}"
    else:
        sell_qty = int(float(amount_str))
        reason = f"sell {sell_qty} shares"

    if sell_qty < 1:
        print(f"Sell quantity too small: {sell_qty}")
        return
    if sell_qty > current_qty:
        sell_qty = int(current_qty)
        reason = "full exit"

    remaining = int(current_qty - sell_qty)
    est_pnl = (pos["current_price"] - entry_price) * sell_qty

    print(f"\n  THESIS SELL: {symbol}")
    print(f"  Selling:     {sell_qty} of {int(current_qty)} shares ({reason})")
    print(f"  Price:       ~${pos['current_price']:.2f}")
    print(f"  Est P/L:     ${est_pnl:+,.2f}")
    print(f"  Remaining:   {remaining} shares")
    print()

    if not auto_confirm:
        print("  [Preview only - pass --confirm to execute]")
        return

    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=sell_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)
        log.info(f"SELL {sell_qty} {symbol} @ ~${pos['current_price']:.2f} ({reason}) (order {result.id})")

        # Update thesis_trades.json
        trade["outcome"]["realized_pnl"] = round(
            trade["outcome"].get("realized_pnl", 0) + est_pnl, 2
        )

        if remaining == 0:
            trade["status"] = "closed"
            trade["outcome"]["closed_date"] = datetime.now().strftime("%Y-%m-%d")
        else:
            trade["status"] = "partial"
            trade["entry"]["shares"] = remaining

        # Check if this matches a target
        for tgt in trade.get("targets", []):
            if (tgt.get("price") and not tgt["hit"]
                    and pos["current_price"] >= tgt["price"]):
                tgt["hit"] = True
                tgt["hit_date"] = datetime.now().strftime("%Y-%m-%d")
                log.info(f"Target {tgt['label']} hit at ${pos['current_price']:.2f}")

        # Log execution
        trade.setdefault("executions", []).append({
            "action": "SELL",
            "qty": sell_qty,
            "price": round(pos["current_price"], 2),
            "reason": reason,
            "pnl": round(est_pnl, 2),
            "order_id": str(result.id),
            "timestamp": datetime.now().isoformat(),
        })

        save_thesis(data)

        print(f"\n  EXECUTED: SELL {sell_qty} {symbol} @ ~${pos['current_price']:.2f}")
        print(f"  Est P/L: ${est_pnl:+,.2f}")
        print(f"  Remaining: {remaining} shares")
        print(f"  thesis_trades.json updated.")

    except Exception as e:
        log.error(f"Failed to sell {symbol}: {e}")
        print(f"  FAILED: {e}")


def cmd_status():
    """Show all thesis trades with live pricing."""
    data = load_thesis()
    if not data["trades"]:
        print("No thesis trades.")
        return

    client = get_client()
    now = datetime.now()

    print(f"\n{'THESIS TRADES':^70}")
    print(f"{'='*70}")

    for trade in data["trades"]:
        sym = trade["symbol"]
        status = trade["status"].upper()
        entry = trade["entry"]
        inv = trade["invalidation"]
        targets = trade.get("targets", [])
        outcome = trade.get("outcome", {})

        # Live data
        pos = get_alpaca_position(client, sym)
        price = pos["current_price"] if pos else get_quote_price(client, sym)

        print(f"\n  {sym} [{status}]")
        print(f"  Thesis: {trade['thesis']['summary']}")

        if entry.get("price"):
            print(f"  Entry:  {entry['date']} | {entry.get('shares', '?')} shares @ ${entry['price']:.2f} "
                  f"(${entry.get('notional', 0):,.2f})")
        else:
            print(f"  Entry:  PENDING (planned {entry.get('date', '?')} | ${entry.get('notional', 0):,})")

        if pos and price:
            pnl = pos["unrealized_pnl"]
            pnl_pct = pos["unrealized_pnl_pct"]
            print(f"  Live:   ${price:.2f} | P/L: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | "
                  f"MV: ${pos['market_value']:,.2f}")
        elif price:
            print(f"  Quote:  ${price:.2f}")

        # Invalidation
        inv_price = inv.get("price_below")
        deadline = inv.get("time_deadline")
        if price and inv_price:
            distance = ((price - inv_price) / price) * 100
            print(f"  Stop:   ${inv_price:.2f} ({distance:+.1f}% away)")
        if deadline:
            days_left = (datetime.fromisoformat(deadline) - now).days
            print(f"  Deadline: {deadline} ({days_left}d left) - {inv.get('time_condition', '')}")

        # Targets
        if targets and price:
            for tgt in targets:
                tgt_price = tgt.get("price")
                if tgt_price:
                    dist = ((tgt_price - price) / price) * 100
                    hit_mark = " HIT" if tgt["hit"] else ""
                    print(f"  {tgt['label']}:     ${tgt_price:.0f} (trim {tgt['trim_pct']}%) "
                          f"- {dist:+.1f}% away{hit_mark}")
                else:
                    print(f"  {tgt['label']}:     hold remainder (trim {tgt['trim_pct']}%)")

        # Realized P/L
        realized = outcome.get("realized_pnl", 0)
        if realized:
            print(f"  Realized: ${realized:+,.2f}")

        # Executions log
        execs = trade.get("executions", [])
        if execs:
            print(f"  History: {len(execs)} executions")
            for ex in execs[-3:]:
                print(f"    {ex['timestamp'][:10]} {ex['action']} {ex['qty']} @ ${ex['price']:.2f} "
                      f"({ex['reason']}) P/L: ${ex.get('pnl', 0):+,.2f}")

    print(f"\n{'='*70}")


def cmd_check(symbol: str):
    """Detailed single-symbol check."""
    symbol = symbol.upper()
    data = load_thesis()
    trade = find_trade(data, symbol)
    if not trade:
        print(f"No thesis trade for {symbol}.")
        return

    # Just show status for the one symbol
    # Reuse status but filter
    original_trades = data["trades"]
    data["trades"] = [trade]
    cmd_status()
    data["trades"] = original_trades


# === CLI ===

def main():
    # Parse --confirm flag from anywhere in argv
    auto_confirm = "--confirm" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--confirm"]

    if not args:
        print("Usage:")
        print("  thesis_execute.py buy SYMBOL DOLLARS              # preview")
        print("  thesis_execute.py buy SYMBOL DOLLARS --confirm    # execute")
        print("  thesis_execute.py sell SYMBOL AMOUNT              # preview")
        print("  thesis_execute.py sell SYMBOL AMOUNT --confirm    # execute")
        print("  thesis_execute.py status                          # show positions")
        print("  thesis_execute.py check SYMBOL                    # check one")
        return

    cmd = args[0].lower()

    if cmd == "buy":
        if len(args) != 3:
            print("Usage: thesis_execute.py buy SYMBOL DOLLARS [--confirm]")
            return
        cmd_buy(args[1], float(args[2]), auto_confirm)

    elif cmd == "sell":
        if len(args) != 3:
            print("Usage: thesis_execute.py sell SYMBOL AMOUNT [--confirm]")
            return
        cmd_sell(args[1], args[2], auto_confirm)

    elif cmd == "status":
        cmd_status()

    elif cmd == "check":
        if len(args) != 2:
            print("Usage: thesis_execute.py check SYMBOL")
            return
        cmd_check(args[1])

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: buy, sell, status, check")


if __name__ == "__main__":
    main()
