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

### scanner.py

momentum scanner - finds buy signals

```bash
python3 scanner.py scan     # full scan with signals
python3 scanner.py top      # top 5 movers
python3 scanner.py buy      # buy candidates only
python3 scanner.py json     # output as JSON
```

signals: STRONG BUY, BUY, HOLD, WEAK, AVOID

### strategy.py

trading strategy executor

```bash
python3 strategy.py status  # portfolio state
python3 strategy.py signals # current buy candidates
python3 strategy.py dry     # dry run (no trades)
python3 strategy.py run     # LIVE TRADES
```

parameters:
- max 10% per position
- max 5 positions
- 20% cash reserve
- stop loss at -5%
- take profit at +10%

### run.py

runner with logging

```bash
python3 run.py check  # dry run, log result
python3 run.py run    # live run, log result
python3 run.py log    # show recent runs
```

### market_eye.py

the trading organ - integrates with organism pulse

```bash
python3 market_eye.py pulse   # quick status
python3 market_eye.py status  # detailed state
```

### config.py

loads credentials from `~/.alpaca-keys`

## quick commands

```bash
# check signals
python3 scanner.py buy

# dry run strategy
python3 strategy.py dry

# live trade (when market open)
python3 run.py run

# see what happened
python3 run.py log
```

## notes

- paper trading only ($100k account)
- no real money at risk
- uses IEX free data feed
