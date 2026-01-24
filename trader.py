#!/usr/bin/env python3
"""
trader.py - core trading operations

This is my interface to the market.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add venv to path
VENV_PATH = Path(__file__).parent / "venv" / "lib" / "python3.13" / "site-packages"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import load_keys, keys_exist


class Trader:
    """My trading interface"""

    def __init__(self):
        if not keys_exist():
            raise RuntimeError("No API keys. Create ~/.alpaca-keys")

        api_key, secret_key = load_keys()
        self.trading = TradingClient(api_key, secret_key, paper=True)
        self.data = StockHistoricalDataClient(api_key, secret_key)

    def get_account(self) -> dict:
        """Get account info"""
        account = self.trading.get_account()
        return {
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "pl_today": float(account.equity) - float(account.last_equity),
            "pl_today_pct": ((float(account.equity) / float(account.last_equity)) - 1) * 100 if float(account.last_equity) > 0 else 0,
        }

    def get_positions(self) -> list[dict]:
        """Get all positions"""
        positions = self.trading.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "cost_basis": float(p.cost_basis),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_pl_pct": float(p.unrealized_plpc) * 100,
                "current_price": float(p.current_price),
                "avg_entry": float(p.avg_entry_price),
            }
            for p in positions
        ]

    def get_quote(self, symbol: str) -> dict:
        """Get latest quote for a symbol"""
        symbol = symbol.upper()
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = self.data.get_stock_latest_quote(request)
        quote = quotes[symbol]
        return {
            "symbol": symbol,
            "bid": float(quote.bid_price),
            "ask": float(quote.ask_price),
            "bid_size": quote.bid_size,
            "ask_size": quote.ask_size,
            "spread": float(quote.ask_price) - float(quote.bid_price),
            "mid": (float(quote.ask_price) + float(quote.bid_price)) / 2,
            "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
        }

    def get_clock(self) -> dict:
        """Get market clock"""
        clock = self.trading.get_clock()
        return {
            "is_open": clock.is_open,
            "timestamp": clock.timestamp.isoformat(),
            "next_open": clock.next_open.isoformat(),
            "next_close": clock.next_close.isoformat(),
        }

    def buy(self, symbol: str, qty: float, limit_price: float = None) -> dict:
        """Buy shares"""
        symbol = symbol.upper()

        if limit_price:
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

        order = self.trading.submit_order(order_data)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": "buy",
            "qty": float(order.qty) if order.qty else None,
            "type": order.order_type.value,
            "status": order.status.value,
            "limit_price": float(order.limit_price) if order.limit_price else None,
        }

    def sell(self, symbol: str, qty: float, limit_price: float = None) -> dict:
        """Sell shares"""
        symbol = symbol.upper()

        if limit_price:
            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
        else:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

        order = self.trading.submit_order(order_data)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": "sell",
            "qty": float(order.qty) if order.qty else None,
            "type": order.order_type.value,
            "status": order.status.value,
            "limit_price": float(order.limit_price) if order.limit_price else None,
        }

    def get_orders(self, status: str = "open") -> list[dict]:
        """Get orders"""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }

        request = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
        orders = self.trading.get_orders(request)

        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side.value,
                "qty": float(o.qty) if o.qty else None,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                "type": o.order_type.value,
                "status": o.status.value,
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in orders
        ]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders"""
        result = self.trading.cancel_orders()
        return len(result) if result else 0


def main():
    """Test the trader"""
    try:
        t = Trader()
        print("Trader initialized")

        # Test account
        account = t.get_account()
        print(f"\nAccount:")
        print(f"  Portfolio: ${account['portfolio_value']:,.2f}")
        print(f"  Cash: ${account['cash']:,.2f}")
        print(f"  P&L today: ${account['pl_today']:+,.2f} ({account['pl_today_pct']:+.2f}%)")

        # Test clock
        clock = t.get_clock()
        print(f"\nMarket: {'OPEN' if clock['is_open'] else 'CLOSED'}")

        # Test positions
        positions = t.get_positions()
        if positions:
            print(f"\nPositions ({len(positions)}):")
            for p in positions:
                print(f"  {p['symbol']}: {p['qty']} @ ${p['current_price']:.2f} ({p['unrealized_pl_pct']:+.2f}%)")
        else:
            print("\nNo positions")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
