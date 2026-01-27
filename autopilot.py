#!/usr/bin/env python3
"""
autopilot.py - fully automated multi-strategy trading

Runs the strategy router end-to-end:
1. Check positions for exit signals (strategy-keyed exit params)
2. Cancel stale stop orders, place fresh ones
3. If slots open, scan via StrategyRouter for entries
4. Record all trades in ledger with strategy attribution
5. Log everything

Run via systemd timer every 5 minutes during market hours.
"""

import sys
import json
import logging
from datetime import datetime, timedelta
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
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trader import Trader
from router import StrategyRouter, load_config
from ledger import Ledger
from monitor import update_high_water, clear_high_water
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


# === Exit config resolution ===

DEFAULT_EXIT_PARAMS = {
    "stop_atr_multiplier": 1.5,
    "trailing_stop_enabled": True,
    "profit_giveback_pct": 0.4,
    "ma_exit_period": 10,
}


def get_exit_params(config: dict, strategy: str) -> dict:
    """Resolve exit params for a strategy: override > default > hardcoded."""
    defaults = config.get("exit_defaults", DEFAULT_EXIT_PARAMS)
    overrides = config.get("exit_overrides", {}).get(strategy, {})
    params = {**DEFAULT_EXIT_PARAMS, **defaults, **overrides}
    return params


# === Technical helpers ===

def calculate_atr(bars: list, period: int = 14) -> float:
    """Calculate Average True Range from bar objects."""
    if len(bars) < period + 1:
        return 0
    trs = []
    for i in range(-period, 0):
        high = float(bars[i].high)
        low = float(bars[i].low)
        prev_close = float(bars[i - 1].close)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)


def calculate_ma(bars: list, period: int) -> float:
    """Calculate simple moving average."""
    if len(bars) < period:
        return 0
    return sum(float(bars[i].close) for i in range(-period, 0)) / period


def fetch_bars(data_client: StockHistoricalDataClient, symbol: str, days: int = 60) -> list:
    """Fetch historical bars for a symbol."""
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=days)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    result = data_client.get_stock_bars(request)
    if hasattr(result, 'data') and symbol in result.data:
        return list(result.data[symbol])
    return []


# === Parameterized exit check ===

def check_exit(bars: list, entry_price: float, high_water: float, exit_params: dict) -> dict:
    """
    Check if a position should be exited using strategy-specific parameters.

    Args:
        bars: Historical bar data for the symbol
        entry_price: Average entry price
        high_water: Peak price since entry
        exit_params: Strategy-keyed exit configuration:
            stop_atr_multiplier: ATR multiplier for initial stop
            trailing_stop_enabled: Whether to use trailing stop
            profit_giveback_pct: Max fraction of gains to give back (if trailing)
            ma_exit_period: MA period for trend exit (None to disable)

    Returns:
        dict with exit signal, stop price, and diagnostics
    """
    if not bars or len(bars) < 20:
        return {"exit": False, "reason": "insufficient data", "stop": None, "ma": None}

    current = float(bars[-1].close)
    atr = calculate_atr(bars)
    if atr <= 0:
        return {"exit": False, "reason": "no ATR", "stop": None, "ma": None}

    atr_mult = exit_params["stop_atr_multiplier"]
    trailing = exit_params["trailing_stop_enabled"]
    giveback = exit_params["profit_giveback_pct"]
    ma_period = exit_params["ma_exit_period"]

    # Initial stop: N x ATR below entry
    initial_stop = entry_price - (atr * atr_mult)

    # Trailing stop: don't give back more than X% of gains
    stop_price = initial_stop
    if trailing and high_water > entry_price:
        gain = high_water - entry_price
        trail_stop = high_water - (gain * giveback)
        stop_price = max(initial_stop, trail_stop)

    # Exit: stop hit
    if current < stop_price:
        return {
            "exit": True,
            "reason": f"stop hit at ${stop_price:.2f} (ATR×{atr_mult})",
            "price": current,
        }

    # Exit: MA cross
    ma_value = None
    if ma_period:
        ma_value = calculate_ma(bars, ma_period)
        if ma_value and current < ma_value:
            return {
                "exit": True,
                "reason": f"closed below {ma_period} MA (${ma_value:.2f})",
                "price": current,
            }

    return {
        "exit": False,
        "current": current,
        "stop": stop_price,
        "ma": ma_value,
        "atr": atr,
        "gain_pct": (current - entry_price) / entry_price * 100,
    }


# === State management ===

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"trades_today": 0, "last_run": None, "stopped_out": []}


def save_state(state: dict):
    state["last_run"] = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# === Trade logging ===

def log_trade_entry(action: str, symbol: str, qty: float, price: float,
                    reason: str, strategy: str = "unknown"):
    entry = {
        "time": datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "qty": qty,
        "price": price,
        "reason": reason,
        "strategy": strategy,
    }
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"TRADE: {action} {qty} {symbol} @ ${price:.2f} [{strategy}] - {reason}")


# === Broker operations ===

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


def sell_position(client: TradingClient, ledger: Ledger, symbol: str,
                  qty: float, price: float, reason: str, strategy: str) -> bool:
    """Market sell a position and record in ledger."""
    try:
        cancel_existing_stops(client, symbol)
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)

        # Record in ledger (strategy inferred from position)
        try:
            ledger.record_sell(symbol, qty, price, reason=reason, order_id=str(result.id))
        except ValueError as e:
            log.warning(f"Ledger sell failed for {symbol}: {e}")

        log_trade_entry("SELL", symbol, qty, price, reason, strategy)
        clear_high_water(symbol)
        log.info(f"SOLD {qty} {symbol} [{strategy}] - {reason} (order {result.id})")
        return True
    except Exception as e:
        log.error(f"Failed to sell {symbol}: {e}")
        return False


def buy_position(client: TradingClient, ledger: Ledger, symbol: str,
                 qty: int, price: float, stop_price: float,
                 reason: str, strategy: str) -> bool:
    """Market buy, record in ledger, and place stop."""
    try:
        # Cancel any existing stops for this symbol first (prevents wash trade rejection)
        cancel_existing_stops(client, symbol)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(order)

        # Record in ledger with strategy attribution
        try:
            ledger.record_buy(symbol, qty, price, strategy=strategy,
                              reason=reason, order_id=str(result.id))
        except ValueError as e:
            log.warning(f"Ledger buy failed for {symbol}: {e}")

        log_trade_entry("BUY", symbol, qty, price, reason, strategy)
        log.info(f"BOUGHT {qty} {symbol} [{strategy}] (order {result.id})")

        # Place stop
        place_stop(client, symbol, qty, stop_price)
        return True
    except Exception as e:
        log.error(f"Failed to buy {symbol}: {e}")
        return False


# === Reconciliation ===

def reconcile_ledger(trader: Trader, ledger: Ledger):
    """
    Ensure ledger matches Alpaca's actual positions.

    Handles:
    - Positions in Alpaca but not in ledger (orphans from crashes/manual trades)
    - Positions in ledger but not in Alpaca (stale from external sells)
    """
    alpaca_positions = trader.get_positions()
    alpaca_symbols = {p["symbol"] for p in alpaca_positions}
    ledger_symbols = set(ledger.positions.keys())

    # Orphans: in Alpaca but not in ledger
    for p in alpaca_positions:
        sym = p["symbol"]
        if sym not in ledger_symbols:
            log.warning(f"RECONCILE: {sym} in Alpaca but not in ledger. Importing as 'unknown'.")
            try:
                ledger.record_buy(
                    sym, float(p["qty"]), p["avg_entry"],
                    strategy="unknown",
                    reason="reconciled from Alpaca (orphan)",
                )
            except ValueError as e:
                log.error(f"RECONCILE: Failed to import {sym}: {e}")

    # Stale: in ledger but not in Alpaca
    for sym in ledger_symbols - alpaca_symbols:
        pos = ledger.positions[sym]
        log.warning(f"RECONCILE: {sym} in ledger [{pos.strategy}] but not in Alpaca. Closing.")
        try:
            # Use entry price as close price (we don't know the real exit price)
            ledger.record_sell(
                sym, pos.qty, pos.avg_entry,
                reason="reconciled: position not found in Alpaca",
            )
        except ValueError as e:
            log.error(f"RECONCILE: Failed to close {sym}: {e}")


# === Main autopilot ===

def run():
    """Main autopilot loop - one pass."""
    log.info("=" * 50)
    log.info("AUTOPILOT RUN (multi-strategy)")
    log.info("=" * 50)

    state = load_state()
    config = load_config()
    trader = Trader()
    client = get_trading_client()
    ledger = Ledger()
    router = StrategyRouter(config)

    k, s = load_keys()
    data_client = StockHistoricalDataClient(k, s)

    max_positions = config.get("max_positions", 4)

    # Check market
    clock = trader.get_clock()
    if not clock["is_open"]:
        log.info("Market closed. Exiting.")
        save_state(state)
        return

    # Reconcile ledger against Alpaca before doing anything
    reconcile_ledger(trader, ledger)
    # Reload ledger after reconciliation
    ledger = Ledger()

    # === PHASE 1: CHECK EXITS ===
    log.info("--- Phase 1: Exit checks ---")
    positions = trader.get_positions()

    for p in positions:
        symbol = p["symbol"]
        entry_price = p["avg_entry"]
        current_price = p["current_price"]
        qty = float(p["qty"])

        # Determine owning strategy from ledger
        strategy = ledger.get_position_strategy(symbol) or "unknown"

        # Get strategy-specific exit params
        exit_params = get_exit_params(config, strategy)

        # Update high water mark
        high_water = update_high_water(symbol, current_price)

        # Fetch bars and check exit
        bars = fetch_bars(data_client, symbol)
        result = check_exit(bars, entry_price, high_water, exit_params)

        if result["exit"]:
            log.warning(f"EXIT SIGNAL for {symbol} [{strategy}]: {result['reason']}")
            sold = sell_position(client, ledger, symbol, qty, current_price,
                                result["reason"], strategy)
            if sold:
                state.setdefault("stopped_out", []).append(symbol)
                state["trades_today"] = state.get("trades_today", 0) + 1
        else:
            stop_str = f"${result['stop']:.2f}" if result.get("stop") else "N/A"
            ma_str = f"${result['ma']:.2f}" if result.get("ma") else "off"
            log.info(f"{symbol} [{strategy}]: OK (P/L: {p['unrealized_pl_pct']:+.1f}%, "
                     f"stop: {stop_str}, MA: {ma_str}, "
                     f"ATR×{exit_params['stop_atr_multiplier']})")

            # Refresh broker stop orders
            if result.get("stop"):
                cancel_existing_stops(client, symbol)
                place_stop(client, symbol, qty, result["stop"])

    # === PHASE 2: CHECK ENTRIES ===
    positions = trader.get_positions()
    current_symbols = [p["symbol"] for p in positions]
    available_slots = max_positions - len(positions)

    if available_slots > 0:
        log.info(f"--- Phase 2: Entry scan ({available_slots} slots) ---")

        # Don't re-enter stocks we got stopped out of today
        blocked = set(state.get("stopped_out", []))

        # Scan via router - returns strategy-attributed signals
        signals = router.get_entry_signals()
        new_signals = [s for s in signals
                       if s.symbol not in current_symbols and s.symbol not in blocked]

        if new_signals:
            for sig in new_signals[:available_slots]:
                # Size position using router (respects per-strategy allocation)
                try:
                    quote = trader.get_quote(sig.symbol)
                    price = quote["ask"]
                except Exception:
                    log.warning(f"Could not get quote for {sig.symbol}")
                    continue

                sizing = router.calculate_position_size(sig, price)
                shares = sizing["shares"]

                if shares < 1:
                    log.info(f"Skip {sig.symbol} [{sig.strategy}]: position too small "
                             f"(avail: ${sizing['available_capital']:,.0f})")
                    continue

                # Calculate stop using this strategy's exit params
                exit_params = get_exit_params(config, sig.strategy)
                bars = fetch_bars(data_client, sig.symbol)
                atr = calculate_atr(bars) if bars and len(bars) > 15 else 0
                stop_price = price - (atr * exit_params["stop_atr_multiplier"]) if atr > 0 else price * 0.95

                log.info(f"ENTRY: {sig.symbol} [{sig.strategy}] str={sig.strength:+.2f} | "
                         f"{shares} shares @ ${price:.2f} | stop ${stop_price:.2f} "
                         f"(ATR×{exit_params['stop_atr_multiplier']})")

                bought = buy_position(
                    client, ledger, sig.symbol, shares, price, stop_price,
                    f"Router entry: {sig.strategy} str={sig.strength:+.2f}, {sig.reason}",
                    sig.strategy,
                )
                if bought:
                    state["trades_today"] = state.get("trades_today", 0) + 1
                    available_slots -= 1
        else:
            log.info("No new entry signals from router.")
    else:
        log.info(f"--- Phase 2: Skip (fully loaded {max_positions}/{max_positions}) ---")

    # === SUMMARY ===
    positions = trader.get_positions()
    account = trader.get_account()
    log.info("--- Summary ---")
    log.info(f"Equity: ${account['portfolio_value']:,.2f} | Cash: ${account['cash']:,.2f}")
    for p in positions:
        strategy = ledger.get_position_strategy(p["symbol"]) or "?"
        log.info(f"  {p['symbol']} [{strategy}]: {p['qty']} @ ${p['current_price']:.2f} "
                 f"({p['unrealized_pl_pct']:+.1f}%)")
    log.info(f"Trades today: {state.get('trades_today', 0)}")

    # Log per-strategy P&L
    summary = ledger.summary()
    for strat, data in summary.get("by_strategy", {}).items():
        log.info(f"  Strategy {strat}: {data['closed_trades']} closed, "
                 f"${data['realized_pnl']:+.2f} realized, {data['win_rate']:.0f}% win")

    save_state(state)
    log.info("Autopilot run complete.")


def reset_daily():
    """Reset daily state (run at market open)."""
    state = load_state()
    state["trades_today"] = 0
    state["stopped_out"] = []
    save_state(state)
    log.info("Daily state reset.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Autopilot trading (multi-strategy)")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "reset", "status"])
    args = parser.parse_args()

    if args.command == "run":
        run()
    elif args.command == "reset":
        reset_daily()
    elif args.command == "status":
        state = load_state()
        print(json.dumps(state, indent=2))
