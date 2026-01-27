#!/usr/bin/env python3
"""
backtest_edge.py - backtest the ACTUAL edge.py strategy

Tests the real rules across multiple market regimes:
- Entry: weekly momentum 3-20%, above 20 MA, scored
- Exit: 1x ATR stop, 10 MA cross, 40% giveback trailing stop
- Sizing: 5% portfolio risk, 50% buying power cap, max 3 positions

Usage:
    python3 backtest_edge.py                     # default: 2022-2025
    python3 backtest_edge.py --start 2022-01-01  # custom start
    python3 backtest_edge.py --regime 2022       # specific year
"""

import sys
import math
import json
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

# === STRATEGY PARAMETERS (must match edge.py exactly) ===
MAX_POSITIONS = 3
MAX_POSITION_PCT = 0.50
MAX_RISK_PCT = 0.05
ATR_STOP_MULT = 1.0
TRAIL_GIVEBACK = 0.40
WEEK_MIN = 3
WEEK_MAX = 20
INITIAL_CAPITAL = 100_000

UNIVERSE = [
    "NVDA", "META", "TSLA", "AMD", "AVGO", "NFLX", "AMZN", "GOOGL", "AAPL", "MSFT",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "QCOM", "INTC",
    "CRM", "ORCL", "ADBE", "PLTR", "COIN", "SNOW", "CRWD", "NET",
    "XOM", "CVX", "OXY", "SLB", "HAL",
]

# Symbols that didn't exist or weren't liquid before certain dates
SYMBOL_START = {
    "PLTR": date(2020, 10, 1),
    "COIN": date(2021, 4, 15),
    "SNOW": date(2020, 9, 18),
    "CRWD": date(2019, 6, 12),
    "ARM": date(2023, 9, 14),
    "SMCI": date(2022, 1, 1),
    "MSTR": date(2020, 8, 1),
}


class Position:
    def __init__(self, symbol, qty, entry_price, entry_date, stop_price):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.stop_price = stop_price
        self.high_water = entry_price

    def update(self, current_price):
        if current_price > self.high_water:
            self.high_water = current_price

    def market_value(self, price):
        return self.qty * price


class BacktestEdge:
    def __init__(self, start_date: str, end_date: str, symbols: list = None):
        k, s = load_keys()
        self.data_client = StockHistoricalDataClient(k, s)
        self.symbols = symbols or UNIVERSE
        self.start = datetime.strptime(start_date, "%Y-%m-%d")
        self.end = datetime.strptime(end_date, "%Y-%m-%d")
        self.all_bars = {}  # symbol -> list of bars

    def fetch_data(self):
        """Fetch all historical data upfront."""
        # Need extra lookback for indicators
        fetch_start = self.start - timedelta(days=90)
        print(f"Fetching {len(self.symbols)} symbols from {fetch_start.date()} to {self.end.date()}...")

        for sym in self.symbols:
            # Skip symbols that didn't exist yet
            if sym in SYMBOL_START and fetch_start.date() < SYMBOL_START[sym]:
                continue
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=TimeFrame.Day,
                    start=fetch_start,
                    end=self.end,
                )
                result = self.data_client.get_stock_bars(request)
                if hasattr(result, 'data') and sym in result.data:
                    bars = list(result.data[sym])
                    if len(bars) > 30:
                        self.all_bars[sym] = bars
                        print(f"  {sym}: {len(bars)} bars")
                    else:
                        print(f"  {sym}: skipped (only {len(bars)} bars)")
            except Exception as e:
                print(f"  {sym}: ERROR {e}")

        print(f"Loaded {len(self.all_bars)} symbols\n")

    def _get_bars_up_to(self, symbol, target_date):
        """Get all bars up to and including target_date."""
        if symbol not in self.all_bars:
            return []
        return [b for b in self.all_bars[symbol] if b.timestamp.date() <= target_date]

    def _calc_atr(self, bars, period=14):
        if len(bars) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            h = float(bars[i].high)
            l = float(bars[i].low)
            pc = float(bars[i-1].close)
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return sum(trs) / len(trs)

    def _calc_ma(self, bars, period):
        if len(bars) < period:
            return None
        return sum(float(bars[i].close) for i in range(-period, 0)) / period

    def _calc_week_return(self, bars):
        if len(bars) < 6:
            return None
        return (float(bars[-1].close) - float(bars[-6].close)) / float(bars[-6].close) * 100

    def _calc_month_return(self, bars):
        if len(bars) < 22:
            return None
        return (float(bars[-1].close) - float(bars[-22].close)) / float(bars[-22].close) * 100

    def _score_setup(self, bars):
        """Score a symbol exactly like edge.py scan_for_setups."""
        week_ret = self._calc_week_return(bars)
        if week_ret is None:
            return None

        if week_ret < WEEK_MIN or week_ret > WEEK_MAX:
            return None

        ma20 = self._calc_ma(bars, 20)
        if ma20 is None:
            return None

        price = float(bars[-1].close)
        if price <= ma20:
            return None

        score = 2  # base for being in week range
        reasons = [f"+{week_ret:.1f}% week"]

        month_ret = self._calc_month_return(bars)
        if month_ret and month_ret > 15:
            score += 2
            reasons.append(f"+{month_ret:.1f}% month")
        elif month_ret and month_ret > 8:
            score += 1

        ma50 = self._calc_ma(bars, 50)
        if ma50 and price > ma50:
            score += 1
            reasons.append("above MAs")

        # Volume - use last bar vs 20d avg
        avg_vol = sum(float(bars[i].volume) for i in range(-20, 0)) / 20
        cur_vol = float(bars[-1].volume)
        if avg_vol > 0 and cur_vol / avg_vol > 1.5:
            score += 1
            reasons.append(f"{cur_vol/avg_vol:.1f}x vol")

        if score < 2:
            return None

        atr = self._calc_atr(bars)
        if not atr or atr <= 0:
            return None

        return {
            "score": score,
            "price": price,
            "atr": atr,
            "reasons": reasons,
            "week_return": week_ret,
        }

    def _check_exit(self, pos: Position, bars) -> dict:
        """Check exit conditions exactly like edge.py check_exit."""
        if len(bars) < 20:
            return {"exit": False}

        current = float(bars[-1].close)
        atr = self._calc_atr(bars)
        if not atr:
            return {"exit": False}

        ma10 = self._calc_ma(bars, 10)

        # Initial stop
        initial_stop = pos.entry_price - (atr * ATR_STOP_MULT)

        # Trailing stop
        if pos.high_water > pos.entry_price:
            gain = pos.high_water - pos.entry_price
            trail_stop = pos.high_water - (gain * TRAIL_GIVEBACK)
            stop_price = max(initial_stop, trail_stop)
        else:
            stop_price = initial_stop

        # Check low of day against stop (more realistic than close)
        day_low = float(bars[-1].low)
        if day_low <= stop_price:
            # Stopped out - use stop price or open if gapped through
            exit_price = max(stop_price, float(bars[-1].open)) if float(bars[-1].open) < stop_price else stop_price
            # Actually: if gapped below stop, we get filled at open
            day_open = float(bars[-1].open)
            fill_price = day_open if day_open < stop_price else stop_price
            return {"exit": True, "reason": "stop", "price": fill_price}

        # MA10 cross (on close)
        if ma10 and current < ma10:
            return {"exit": True, "reason": "ma10_cross", "price": current}

        return {"exit": False, "stop": stop_price}

    def _calc_position_size(self, setup, portfolio_value, buying_power):
        """Size position exactly like edge.py."""
        price = setup["price"]
        atr = setup["atr"]
        stop_distance = atr * ATR_STOP_MULT
        stop_pct = stop_distance / price

        max_risk = portfolio_value * MAX_RISK_PCT
        position_value = max_risk / stop_pct if stop_pct > 0 else 0
        max_pos = buying_power * MAX_POSITION_PCT
        position_value = min(position_value, max_pos, buying_power)

        shares = int(position_value / price)
        stop_price = price - stop_distance
        return shares, stop_price

    def run(self) -> dict:
        """Run the full backtest."""
        if not self.all_bars:
            self.fetch_data()

        # Build unified date list from all symbols
        all_dates = set()
        for sym, bars in self.all_bars.items():
            for b in bars:
                all_dates.add(b.timestamp.date())
        all_dates = sorted(all_dates)

        # Filter to our test window
        start_d = self.start.date()
        all_dates = [d for d in all_dates if d >= start_d]

        if not all_dates:
            return {"error": "No dates in range"}

        # State
        cash = float(INITIAL_CAPITAL)
        positions = {}  # symbol -> Position
        trades = []
        daily_equity = []
        peak_equity = INITIAL_CAPITAL
        max_drawdown = 0
        max_drawdown_date = None

        for day in all_dates:
            # === Portfolio value ===
            port_value = cash
            for sym, pos in positions.items():
                bars = self._get_bars_up_to(sym, day)
                if bars:
                    port_value += pos.qty * float(bars[-1].close)

            daily_equity.append({"date": str(day), "equity": port_value})

            # Track drawdown
            if port_value > peak_equity:
                peak_equity = port_value
            dd = (peak_equity - port_value) / peak_equity * 100
            if dd > max_drawdown:
                max_drawdown = dd
                max_drawdown_date = str(day)

            # === Check exits ===
            to_close = []
            for sym, pos in positions.items():
                bars = self._get_bars_up_to(sym, day)
                if not bars:
                    continue
                pos.update(float(bars[-1].high))  # update HWM with day's high
                result = self._check_exit(pos, bars)
                if result["exit"]:
                    to_close.append((sym, result))

            for sym, result in to_close:
                pos = positions[sym]
                exit_price = result["price"]
                proceeds = pos.qty * exit_price
                pnl = (exit_price - pos.entry_price) * pos.qty
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                cash += proceeds
                trades.append({
                    "date": str(day),
                    "action": "SELL",
                    "symbol": sym,
                    "qty": pos.qty,
                    "price": exit_price,
                    "entry_price": pos.entry_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": result["reason"],
                    "held_days": (day - pos.entry_date).days,
                })
                del positions[sym]

            # === Check entries ===
            available_slots = MAX_POSITIONS - len(positions)
            if available_slots <= 0:
                continue

            # Don't enter on same day we exited (avoid churn)
            exited_today = {sym for sym, _ in to_close}

            # Score all symbols
            setups = []
            for sym in self.symbols:
                if sym in positions or sym in exited_today:
                    continue
                if sym in SYMBOL_START and day < SYMBOL_START[sym]:
                    continue
                bars = self._get_bars_up_to(sym, day)
                if len(bars) < 50:
                    continue
                setup = self._score_setup(bars)
                if setup:
                    setup["symbol"] = sym
                    setups.append(setup)

            setups.sort(key=lambda x: x["score"], reverse=True)

            # Buy top setups
            buying_power = cash  # no margin in backtest for conservatism
            for setup in setups[:available_slots]:
                shares, stop_price = self._calc_position_size(setup, port_value, buying_power)
                if shares < 1:
                    continue
                cost = shares * setup["price"]
                if cost > cash:
                    continue

                cash -= cost
                buying_power -= cost
                positions[setup["symbol"]] = Position(
                    setup["symbol"], shares, setup["price"], day, stop_price
                )
                trades.append({
                    "date": str(day),
                    "action": "BUY",
                    "symbol": setup["symbol"],
                    "qty": shares,
                    "price": setup["price"],
                    "score": setup["score"],
                    "reasons": ", ".join(setup["reasons"]),
                    "stop": stop_price,
                })
                available_slots -= 1

        # === Final liquidation ===
        final_value = cash
        for sym, pos in positions.items():
            bars = self._get_bars_up_to(sym, all_dates[-1])
            if bars:
                price = float(bars[-1].close)
                final_value += pos.qty * price

        # === Compute stats ===
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        wins = [t for t in sell_trades if t["pnl"] > 0]
        losses = [t for t in sell_trades if t["pnl"] <= 0]

        total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_hold = sum(t.get("held_days", 0) for t in sell_trades) / len(sell_trades) if sell_trades else 0

        # Sharpe (annualized from daily returns)
        if len(daily_equity) > 2:
            daily_returns = []
            for i in range(1, len(daily_equity)):
                r = (daily_equity[i]["equity"] - daily_equity[i-1]["equity"]) / daily_equity[i-1]["equity"]
                daily_returns.append(r)
            if daily_returns:
                mean_r = sum(daily_returns) / len(daily_returns)
                std_r = (sum((r - mean_r)**2 for r in daily_returns) / len(daily_returns)) ** 0.5
                sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Exit reason breakdown
        exit_reasons = defaultdict(list)
        for t in sell_trades:
            exit_reasons[t["reason"]].append(t["pnl_pct"])

        return {
            "period": f"{all_dates[0]} to {all_dates[-1]}",
            "trading_days": len(all_dates),
            "initial": INITIAL_CAPITAL,
            "final": final_value,
            "total_return_pct": total_return,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown,
            "max_drawdown_date": max_drawdown_date,
            "total_trades": len(sell_trades),
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "profit_factor": profit_factor,
            "avg_hold_days": avg_hold,
            "exit_reasons": {k: {"count": len(v), "avg_pnl": sum(v)/len(v)} for k, v in exit_reasons.items()},
            "trades": trades,
            "daily_equity": daily_equity,
            "open_positions": {s: {"qty": p.qty, "entry": p.entry_price} for s, p in positions.items()},
        }


def print_results(result: dict, verbose: bool = False):
    print("=" * 60)
    print("EDGE STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Period:          {result['period']}")
    print(f"Trading days:    {result['trading_days']}")
    print()
    print(f"Initial:         ${result['initial']:>12,.2f}")
    print(f"Final:           ${result['final']:>12,.2f}")
    print(f"Return:          {result['total_return_pct']:>+11.2f}%")
    print(f"Sharpe:          {result['sharpe']:>11.2f}")
    print(f"Max Drawdown:    {result['max_drawdown_pct']:>11.2f}%  ({result['max_drawdown_date']})")
    print()
    print(f"Completed trades:{result['total_trades']:>7}")
    print(f"Win rate:        {result['win_rate']:>10.1f}%")
    print(f"Avg win:         {result['avg_win_pct']:>+10.2f}%")
    print(f"Avg loss:        {result['avg_loss_pct']:>+10.2f}%")
    print(f"Profit factor:   {result['profit_factor']:>10.2f}")
    print(f"Avg hold:        {result['avg_hold_days']:>10.1f} days")

    print(f"\nExit reasons:")
    for reason, stats in result["exit_reasons"].items():
        print(f"  {reason:15} {stats['count']:>4} trades  avg P/L: {stats['avg_pnl']:+.2f}%")

    if result["open_positions"]:
        print(f"\nOpen positions at end:")
        for sym, pos in result["open_positions"].items():
            print(f"  {sym}: {pos['qty']} @ ${pos['entry']:.2f}")

    if verbose:
        print(f"\nAll trades:")
        for t in result["trades"]:
            if t["action"] == "BUY":
                print(f"  {t['date']} BUY  {t['symbol']:>5} x{t['qty']:>4} @ ${t['price']:>8.2f}  (score:{t['score']}, {t['reasons']})")
            else:
                print(f"  {t['date']} SELL {t['symbol']:>5} x{t['qty']:>4} @ ${t['price']:>8.2f}  P/L:{t['pnl_pct']:>+6.2f}%  [{t['reason']}, {t['held_days']}d]")

    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest edge.py strategy")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-24", help="End date YYYY-MM-DD")
    parser.add_argument("--regime", choices=["2022", "2023", "2024", "2025", "all"],
                        help="Test specific regime")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--save", help="Save results to JSON file")
    args = parser.parse_args()

    if args.regime:
        regimes = {
            "2022": ("2022-01-01", "2022-12-31"),
            "2023": ("2023-01-01", "2023-12-31"),
            "2024": ("2024-01-01", "2024-12-31"),
            "2025": ("2025-01-01", "2026-01-24"),
            "all":  ("2022-01-01", "2026-01-24"),
        }
        args.start, args.end = regimes[args.regime]

    bt = BacktestEdge(args.start, args.end)
    result = bt.run()
    print_results(result, verbose=args.verbose)

    if args.save:
        # Don't save daily_equity to keep file small
        save_data = {k: v for k, v in result.items() if k != "daily_equity"}
        Path(args.save).write_text(json.dumps(save_data, indent=2, default=str))
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
