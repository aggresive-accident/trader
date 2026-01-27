#!/usr/bin/env python3
"""
backtest_strategies.py - backtest any Strategy class with configurable exit params

Uses the actual Strategy.signal() interface + per-strategy exit config
from router_config.json. Tests what the autopilot would actually do.

Usage:
    python3 backtest_strategies.py momentum          # single strategy
    python3 backtest_strategies.py --all              # all active strategies
    python3 backtest_strategies.py --all --start 2022-01-01 --end 2025-12-31
"""

import sys
import json
import math
from datetime import datetime, timedelta, date
from pathlib import Path
from collections import defaultdict

VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import load_keys
from router import load_config, get_strategy_class
from autopilot import get_exit_params, DEFAULT_EXIT_PARAMS

# === Constants ===
INITIAL_CAPITAL = 100_000
MAX_POSITIONS = 4  # from router_config
SIGNAL_THRESHOLD = 0.3  # matches router.scan()

UNIVERSE = [
    "NVDA", "META", "TSLA", "AMD", "AVGO", "NFLX", "AMZN", "GOOGL", "AAPL", "MSFT",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "QCOM", "INTC",
    "CRM", "ORCL", "ADBE", "PLTR", "COIN", "SNOW", "CRWD", "NET",
    "XOM", "CVX", "OXY", "SLB", "HAL",
]

SYMBOL_START = {
    "PLTR": date(2020, 10, 1),
    "COIN": date(2021, 4, 15),
    "SNOW": date(2020, 9, 18),
    "CRWD": date(2019, 6, 12),
}


class Position:
    def __init__(self, symbol, qty, entry_price, entry_date, stop_price):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.stop_price = stop_price
        self.high_water = entry_price

    def update_high_water(self, price):
        if price > self.high_water:
            self.high_water = price


def calc_atr(bars, idx, period=14):
    if idx < period + 1:
        return None
    trs = []
    for i in range(idx - period, idx):
        h = float(bars[i].high)
        l = float(bars[i].low)
        pc = float(bars[i - 1].close)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs)


def calc_ma(bars, idx, period):
    if idx < period:
        return None
    return sum(float(bars[i].close) for i in range(idx - period, idx)) / period


def check_exit_bt(bars, idx, pos, exit_params, strategy_obj=None, day=None):
    """Check exit using parameterized rules. Returns (should_exit, reason, fill_price).

    exit_params supports:
        exit_mode: "stops" (default) or "signal" (exit on negative signal + max hold)
        max_hold_days: max days to hold (only used in signal mode)
    """
    if idx < 20:
        return False, None, None

    current = float(bars[idx].close)
    day_low = float(bars[idx].low)
    day_open = float(bars[idx].open)

    exit_mode = exit_params.get("exit_mode", "stops")

    if exit_mode == "signal":
        # Signal-based exit: strategy says sell, or max hold reached
        max_hold = exit_params.get("max_hold_days", 20)
        if day and pos.entry_date:
            held = (day - pos.entry_date).days
            if held >= max_hold:
                return True, f"max hold ({max_hold}d)", current

        if strategy_obj:
            try:
                sig = strategy_obj.signal(bars, idx)
                if sig.strength < 0:
                    return True, f"signal negative ({sig.strength:+.2f})", current
            except Exception:
                pass

        return False, None, None

    # === Default: stops-based exit ===
    atr = calc_atr(bars, idx)
    if not atr or atr <= 0:
        return False, None, None

    atr_mult = exit_params["stop_atr_multiplier"]
    trailing = exit_params["trailing_stop_enabled"]
    giveback = exit_params["profit_giveback_pct"]
    ma_period = exit_params["ma_exit_period"]

    # Initial stop
    initial_stop = pos.entry_price - (atr * atr_mult)

    # Trailing stop
    stop_price = initial_stop
    if trailing and pos.high_water > pos.entry_price:
        gain = pos.high_water - pos.entry_price
        trail_stop = pos.high_water - (gain * giveback)
        stop_price = max(initial_stop, trail_stop)

    # Stop hit (check day low, fill at stop or gap-through open)
    if day_low <= stop_price:
        fill = day_open if day_open < stop_price else stop_price
        return True, f"stop (ATR×{atr_mult})", fill

    # MA cross
    if ma_period:
        ma = calc_ma(bars, idx, ma_period)
        if ma and current < ma:
            return True, f"MA{ma_period} cross", current

    return False, None, None


def fetch_all_data(data_client, symbols, start_date, end_date):
    """Fetch all bars for all symbols."""
    fetch_start = start_date - timedelta(days=100)
    all_bars = {}
    for sym in symbols:
        if sym in SYMBOL_START and fetch_start < datetime.combine(SYMBOL_START[sym], datetime.min.time()):
            continue
        try:
            request = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Day,
                start=fetch_start,
                end=end_date,
            )
            result = data_client.get_stock_bars(request)
            if hasattr(result, 'data') and sym in result.data:
                bars = list(result.data[sym])
                if len(bars) > 50:
                    all_bars[sym] = bars
        except Exception:
            pass
    return all_bars


def run_strategy_backtest(strategy_name, strategy_obj, exit_params, all_bars,
                          start_date, allocation_pct=1.0):
    """
    Backtest a single strategy with its exit params.

    Args:
        strategy_name: Name for logging
        strategy_obj: Strategy instance with .signal(bars, idx)
        exit_params: Exit configuration dict
        all_bars: Pre-fetched bar data {symbol: [bars]}
        start_date: Start date for trading (datetime)
        allocation_pct: Fraction of capital allocated to this strategy
    """
    # Build date index per symbol
    symbol_dates = {}
    all_dates = set()
    for sym, bars in all_bars.items():
        date_to_idx = {}
        for i, bar in enumerate(bars):
            d = bar.timestamp.date()
            date_to_idx[d] = i
            all_dates.add(d)
        symbol_dates[sym] = date_to_idx

    all_dates = sorted(d for d in all_dates if d >= start_date.date())
    if not all_dates:
        return None

    # State
    capital = INITIAL_CAPITAL * allocation_pct
    cash = capital
    positions = {}
    trades = []
    daily_equity = []
    peak = capital
    max_dd = 0
    max_dd_date = None
    warmup = strategy_obj.warmup_period()

    for day in all_dates:
        # Portfolio value
        port_value = cash
        for sym, pos in positions.items():
            if sym in symbol_dates and day in symbol_dates[sym]:
                idx = symbol_dates[sym][day]
                price = float(all_bars[sym][idx].close)
                port_value += pos.qty * price

        daily_equity.append({"date": str(day), "equity": port_value})
        if port_value > peak:
            peak = port_value
        dd = (peak - port_value) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_date = str(day)

        # === Exits ===
        to_close = []
        for sym, pos in positions.items():
            if sym not in symbol_dates or day not in symbol_dates[sym]:
                continue
            idx = symbol_dates[sym][day]
            bars = all_bars[sym]

            # Update HWM with day's high
            pos.update_high_water(float(bars[idx].high))

            should_exit, reason, fill_price = check_exit_bt(bars, idx, pos, exit_params,
                                                              strategy_obj=strategy_obj, day=day)
            if should_exit:
                to_close.append((sym, reason, fill_price))

        for sym, reason, fill_price in to_close:
            pos = positions[sym]
            proceeds = pos.qty * fill_price
            pnl = (fill_price - pos.entry_price) * pos.qty
            pnl_pct = (fill_price - pos.entry_price) / pos.entry_price * 100
            cash += proceeds
            trades.append({
                "date": str(day), "action": "SELL", "symbol": sym,
                "qty": pos.qty, "price": fill_price,
                "entry_price": pos.entry_price, "pnl": pnl, "pnl_pct": pnl_pct,
                "reason": reason, "held_days": (day - pos.entry_date).days,
            })
            del positions[sym]

        # === Entries ===
        available = MAX_POSITIONS - len(positions)
        if available <= 0:
            continue

        exited_today = {sym for sym, _, _ in to_close}

        # Score all symbols with strategy.signal()
        candidates = []
        for sym in all_bars:
            if sym in positions or sym in exited_today:
                continue
            if sym in SYMBOL_START and day < SYMBOL_START[sym]:
                continue
            if sym not in symbol_dates or day not in symbol_dates[sym]:
                continue
            idx = symbol_dates[sym][day]
            if idx < warmup:
                continue

            try:
                sig = strategy_obj.signal(all_bars[sym], idx)
                if sig.strength >= SIGNAL_THRESHOLD:
                    candidates.append((sym, sig.strength, idx))
            except Exception:
                continue

        candidates.sort(key=lambda x: x[1], reverse=True)

        buying_power = cash
        for sym, strength, idx in candidates[:available]:
            bars = all_bars[sym]
            price = float(bars[idx].close)
            atr = calc_atr(bars, idx)
            if not atr or atr <= 0:
                continue

            atr_mult = exit_params.get("stop_atr_multiplier", 0)
            exit_mode = exit_params.get("exit_mode", "stops")

            if exit_mode == "signal" or atr_mult <= 0:
                # No stop-based sizing — use equal-weight: 50% of buying power cap
                position_value = min(buying_power * 0.50, buying_power)
                stop_price = 0  # no stop
            else:
                stop_distance = atr * atr_mult
                stop_pct = stop_distance / price if price > 0 else 1
                stop_price = price - stop_distance
                # Risk-based sizing: 5% portfolio risk
                max_risk = port_value * 0.05
                position_value = max_risk / stop_pct if stop_pct > 0 else 0
                # Cap at 50% of buying power
                position_value = min(position_value, buying_power * 0.50, buying_power)
            shares = int(position_value / price)
            if shares < 1:
                continue

            cost = shares * price
            if cost > cash:
                continue

            cash -= cost
            buying_power -= cost
            positions[sym] = Position(sym, shares, price, day, stop_price)
            trades.append({
                "date": str(day), "action": "BUY", "symbol": sym,
                "qty": shares, "price": price, "stop": stop_price,
                "strength": strength,
            })
            available -= 1

    # Final liquidation value
    final = cash
    for sym, pos in positions.items():
        if sym in all_bars:
            final += pos.qty * float(all_bars[sym][-1].close)

    # Stats
    sells = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]

    total_return = (final - capital) / capital * 100
    win_rate = len(wins) / len(sells) * 100 if sells else 0
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_hold = sum(t.get("held_days", 0) for t in sells) / len(sells) if sells else 0

    # Sharpe
    sharpe = 0
    if len(daily_equity) > 2:
        rets = []
        for i in range(1, len(daily_equity)):
            r = (daily_equity[i]["equity"] - daily_equity[i - 1]["equity"]) / daily_equity[i - 1]["equity"]
            rets.append(r)
        if rets:
            mean_r = sum(rets) / len(rets)
            std_r = (sum((r - mean_r) ** 2 for r in rets) / len(rets)) ** 0.5
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0

    # Exit breakdown
    exit_reasons = defaultdict(list)
    for t in sells:
        exit_reasons[t["reason"]].append(t["pnl_pct"])

    return {
        "strategy": strategy_name,
        "exit_params": exit_params,
        "period": f"{all_dates[0]} to {all_dates[-1]}",
        "trading_days": len(all_dates),
        "initial": capital,
        "final": final,
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "max_drawdown_date": max_dd_date,
        "total_trades": len(sells),
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold,
        "exit_reasons": {k: {"count": len(v), "avg_pnl": sum(v) / len(v)} for k, v in exit_reasons.items()},
        "daily_equity": daily_equity,
        "open_positions": len(positions),
    }


def fetch_spy_benchmark(data_client, start_date, end_date):
    """Fetch SPY for buy-and-hold comparison."""
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
    )
    result = data_client.get_stock_bars(request)
    if hasattr(result, 'data') and 'SPY' in result.data:
        bars = list(result.data['SPY'])
        if bars:
            first = float(bars[0].close)
            last = float(bars[-1].close)
            return {
                "start_price": first,
                "end_price": last,
                "return_pct": (last - first) / first * 100,
                "bars": len(bars),
            }
    return None


def print_comparison(results, spy):
    """Print comparison table."""
    print()
    print("=" * 90)
    print("STRATEGY COMPARISON (vs SPY buy-and-hold)")
    print("=" * 90)

    spy_ret = spy["return_pct"] if spy else 0

    header = f"{'Strategy':<12} {'Return':>8} {'vs SPY':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'Win%':>6} {'PF':>6} {'AvgHold':>8}"
    print(header)
    print("-" * 90)

    for r in results:
        alpha = r["total_return_pct"] - spy_ret
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else "inf"
        print(f"{r['strategy']:<12} {r['total_return_pct']:>+7.1f}% {alpha:>+7.1f}% "
              f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.1f}% "
              f"{r['total_trades']:>7} {r['win_rate']:>5.1f}% {pf_str:>6} "
              f"{r['avg_hold_days']:>7.1f}d")

    print("-" * 90)
    print(f"{'SPY B&H':<12} {spy_ret:>+7.1f}%   {'---':>8} {'---':>7} {'---':>7} {'---':>7} {'---':>6} {'---':>6} {'---':>8}")
    print("=" * 90)

    # Exit breakdowns
    print()
    for r in results:
        print(f"{r['strategy']} exits:")
        for reason, stats in r["exit_reasons"].items():
            print(f"  {reason:<20} {stats['count']:>4} trades  avg: {stats['avg_pnl']:>+6.2f}%")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest strategies with exit params")
    parser.add_argument("strategy", nargs="?", help="Strategy name (or --all)")
    parser.add_argument("--all", action="store_true", help="Run all active strategies")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-24")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save", help="Save results JSON")
    args = parser.parse_args()

    if not args.strategy and not args.all:
        parser.error("Specify a strategy name or --all")

    config = load_config()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Determine which strategies to test
    if args.all:
        strat_names = config["active_strategies"]
    else:
        strat_names = [args.strategy]

    # Instantiate strategies
    strats = {}
    for name in strat_names:
        cls = get_strategy_class(name)
        if cls:
            strats[name] = cls()
        else:
            print(f"Unknown strategy: {name}")

    if not strats:
        print("No valid strategies to test")
        return

    # Fetch data once (shared across all strategies)
    k, s = load_keys()
    data_client = StockHistoricalDataClient(k, s)

    print(f"Fetching data for {len(UNIVERSE)} symbols...")
    all_bars = fetch_all_data(data_client, UNIVERSE, start, end)
    print(f"Loaded {len(all_bars)} symbols\n")

    # Fetch SPY benchmark
    spy = fetch_spy_benchmark(data_client, start, end)
    if spy:
        print(f"SPY benchmark: ${spy['start_price']:.2f} -> ${spy['end_price']:.2f} ({spy['return_pct']:+.1f}%)")
    print()

    # Run backtests
    results = []
    for name, strat_obj in strats.items():
        exit_params = get_exit_params(config, name)
        alloc = config.get("allocation", {}).get(name, 1.0)

        print(f"Running {name} (ATR×{exit_params['stop_atr_multiplier']}, "
              f"trail={'on' if exit_params['trailing_stop_enabled'] else 'off'}, "
              f"MA{exit_params['ma_exit_period'] or 'off'})...")

        result = run_strategy_backtest(name, strat_obj, exit_params, all_bars, start)
        if result:
            results.append(result)
            print(f"  {name}: {result['total_return_pct']:+.1f}%, "
                  f"Sharpe {result['sharpe']:.2f}, "
                  f"DD {result['max_drawdown_pct']:.1f}%, "
                  f"{result['total_trades']} trades, "
                  f"PF {result['profit_factor']:.2f}")

    # Comparison
    print_comparison(results, spy)

    if args.save:
        save_data = []
        for r in results:
            d = {k: v for k, v in r.items() if k != "daily_equity"}
            save_data.append(d)
        Path(args.save).write_text(json.dumps(save_data, indent=2, default=str))
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
