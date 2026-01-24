# trader

my interface to the market

paper trading via Alpaca, integrated with the organism

## setup

1. create `~/.alpaca-keys`:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

2. get keys from https://app.alpaca.markets (paper trading account)

## programs

### trader.py

core trading operations

```bash
python3 trader.py  # test account status
```

API:
```python
from trader import Trader
t = Trader()

t.get_account()      # portfolio, cash, P&L
t.get_positions()    # current holdings
t.get_quote("AAPL")  # latest bid/ask
t.get_clock()        # market open/closed

t.buy("AAPL", 10)              # market order
t.buy("AAPL", 10, limit=150)   # limit order
t.sell("AAPL", 10)             # sell

t.get_orders()       # open orders
t.cancel_order(id)   # cancel one
t.cancel_all_orders() # cancel all
```

### market_eye.py

the trading organ - integrates with organism pulse

when pulsed, reports:
- portfolio status
- concerning positions
- market state

```bash
python3 market_eye.py  # standalone pulse
```

without API keys, reports offline status

### config.py

loads credentials from `~/.alpaca-keys`

## organism integration

market_eye is registered as an organ in `~/workspace/pulse/organs/`

when you run `~/bin/organism pulse` or the health check, it reports market status alongside other organs

## notes

- paper trading only (safe to experiment)
- no real money at risk
- strategies/ dir for future trading logic
