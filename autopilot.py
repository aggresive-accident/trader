#!/usr/bin/env python3
"""
autopilot.py - fully automated trading

Runs the strategy end-to-end:
1. Check positions for exit signals -> sell
2. Cancel stale stop orders, place fresh ones
3. If slots open, scan for entries -> buy
4. Log everything

Run via systemd timer every 5 minutes during market hours.
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
from alpaca.trading.requests import (
    MarketOrderRequest, StopOrderRequest,
    GetOrdersRequest, QueryOrderStatus,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

from trader import Trader
from edge import EdgeTrader, MAX_POSITIONS, ATR_STOP_MULT
from monitor import check_position, load_high_water_marks, clear_high_water
from config import load_keys

LOG_FILE = Path(__file__).parent / "autopilot.log"
STATE_FILE = Path(__file__).parent / "autopilot_state.json"
TRADE_LOG = Path(__file__).parent / "autopilot_trades.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("autopilot")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"trades_today": 0, "last_run": None, "stopped_out": []}


def save_state(state: dict):
    state["last_run"] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_trade(action: str, symbol: str, qty: float, price: float, reason: str):
    entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "reason": reason,
    }
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"TRADE: {action} {qty} {symbol} @ ${price:.2f} - {reason}")


def get_trading_client() -> TradingClient:
    k, s = load_keys()
    return TradingClient(k, s, paper=True)


def cancel_existing_stops(client: TradingClient, symbol: str):
    """Cancel all open stop orders for a symbol."""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    orders = client.get_orders(request)
    for order in orders:
        if order.symbol == symbol and order.type == OrderType.STOP:
            client.cancel_order_by_id(order.id)
            log.info(f"Cancelled stop order {order.id} for {symbol}")


def place_stop(client: TradingClient, symbol: str, qty: float, stop_price: float):
    """Place a GTC stop-loss order."""
    order = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=round(stop_price, 2),
    )
    result = client.submit_order(order)
    log.info(f"Stop placed: SELL {qty} {symbol} @ ${stop_price:.2f} id={result.id}")
    return result


def sell_position(client: TradingClient, symbol: str, qty: float, reason: str) -> bool:
    """Market sell a position."""
    try:
        cancel_existing_stops(client, symbol)
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)
        log_trade("SELL", symbol, qty, 0, reason)  # price filled async
        clear_high_water(symbol)
        log.info(f"SOLD {qty} {symbol} - {reason} (order {result.id})")
        return True
    except Exception as e:
        log.error(f"Failed to sell {symbol}: {e}")
        return False


def buy_position(client: TradingClient, symbol: str, qty: int, stop_price: float, reason: str) -> bool:
    """Market buy and place stop."""
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)
        log_trade("BUY", symbol, qty, 0, reason)
        log.info(f"BOUGHT {qty} {symbol} (order {result.id})")

        # Place stop
        place_stop(client, symbol, qty, stop_price)
        return True
    except Exception as e:
        log.error(f"Failed to buy {symbol}: {e}")
        return False


def run():
    """Main autopilot loop - one pass."""
    log.info("=" * 50)
    log.info("AUTOPILOT RUN")
    log.info("=" * 50)

    state = load_state()
    edge = EdgeTrader()
    trader = Trader()
    client = get_trading_client()

    # Check market
    clock = trader.get_clock()
    if not clock["is_open"]:
        log.info("Market closed. Exiting.")
        save_state(state)
        return

    # === PHASE 1: CHECK EXITS ===
    log.info("--- Phase 1: Exit checks ---")
    positions = trader.get_positions()
    current_symbols = [p["symbol"] for p in positions]

    for p in positions:
        status = check_position(edge, p)

        if status["exit_signal"]:
            log.warning(f"EXIT SIGNAL for {p['symbol']}: {status['exit_reason']}")
            sold = sell_position(client, p["symbol"], float(p["qty"]), status["exit_reason"])
            if sold:
                state.setdefault("stopped_out", []).append(p["symbol"])
                state["trades_today"] = state.get("trades_today", 0) + 1
        else:
            stop_str = f"${status['stop']:.2f}" if status.get("stop") else "N/A"
            log.info(f"{p['symbol']}: OK (P/L: {status['pnl_pct']:+.1f}%, stop: {stop_str})")

            # Refresh stop orders with latest calculated stop
            if status.get("stop"):
                cancel_existing_stops(client, p["symbol"])
                place_stop(client, p["symbol"], float(p["qty"]), status["stop"])

    # === PHASE 2: CHECK ENTRIES ===
    # Refresh positions after potential sells
    positions = trader.get_positions()
    current_symbols = [p["symbol"] for p in positions]
    available_slots = MAX_POSITIONS - len(positions)

    if available_slots > 0:
        log.info(f"--- Phase 2: Entry scan ({available_slots} slots) ---")

        # Don't re-enter stocks we got stopped out of today
        blocked = set(state.get("stopped_out", []))

        setups = edge.scan_for_setups(edge_watchlist())
        new_setups = [s for s in setups if s["symbol"] not in current_symbols and s["symbol"] not in blocked]

        if new_setups:
            for setup in new_setups[:available_slots]:
                pos = edge.calculate_position(setup)
                if pos["shares"] < 1:
                    log.info(f"Skip {setup['symbol']}: position too small")
                    continue

                log.info(f"ENTRY: {setup['symbol']} score={setup['score']} | {pos['shares']} shares @ ${setup['price']:.2f} | stop ${pos['stop_price']:.2f}")
                bought = buy_position(
                    client,
                    setup["symbol"],
                    pos["shares"],
                    pos["stop_price"],
                    f"Autopilot entry: score {setup['score']}, {', '.join(setup['reasons'])}",
                )
                if bought:
                    state["trades_today"] = state.get("trades_today", 0) + 1
                    available_slots -= 1
        else:
            log.info("No new setups meeting criteria.")
    else:
        log.info(f"--- Phase 2: Skip (fully loaded {MAX_POSITIONS}/{MAX_POSITIONS}) ---")

    # === SUMMARY ===
    positions = trader.get_positions()
    account = trader.get_account()
    log.info("--- Summary ---")
    log.info(f"Equity: ${account['portfolio_value']:,.2f} | Cash: ${account['cash']:,.2f}")
    for p in positions:
        log.info(f"  {p['symbol']}: {p['qty']} @ ${p['current_price']:.2f} ({p['unrealized_pl_pct']:+.1f}%)")
    log.info(f"Trades today: {state.get('trades_today', 0)}")

    save_state(state)
    log.info("Autopilot run complete.")


def edge_watchlist() -> list:
    """The universe we scan."""
    return [
        "NVDA", "META", "TSLA", "AMD", "AVGO", "NFLX", "AMZN", "GOOGL", "AAPL", "MSFT",
        "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "QCOM", "INTC", "ARM", "SMCI",
        "CRM", "ORCL", "ADBE", "PLTR", "COIN", "MSTR", "SNOW", "CRWD", "NET",
        "XOM", "CVX", "OXY", "SLB", "HAL",
    ]


def reset_daily():
    """Reset daily state (run at market open)."""
    state = load_state()
    state["trades_today"] = 0
    state["stopped_out"] = []
    save_state(state)
    log.info("Daily state reset.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autopilot trading")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "reset", "status"])
    args = parser.parse_args()

    if args.command == "run":
        run()
    elif args.command == "reset":
        reset_daily()
    elif args.command == "status":
        state = load_state()
        print(json.dumps(state, indent=2))
