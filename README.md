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

## modules (16)

### core

| module | purpose |
|--------|---------|
| trader.py | core trading operations (buy, sell, quotes) |
| config.py | loads API credentials |

### analysis

| module | purpose |
|--------|---------|
| scanner.py | momentum scanner - STRONG BUY, BUY, HOLD, WEAK, AVOID |
| sectors.py | sector rotation - tracks SPDR ETFs vs SPY |
| watchlist.py | custom watchlist with notes |

### execution

| module | purpose |
|--------|---------|
| strategy.py | position sizing (10% max, 5 positions, 20% cash) |
| run.py | runner with logging |
| backtest.py | test strategies on historical data |

### tracking

| module | purpose |
|--------|---------|
| alerts.py | signal change detection |
| performance.py | portfolio metrics (return, sharpe, drawdown) |
| quotes.py | price history per symbol |
| journal.py | trade journal with reasoning |

### reporting

| module | purpose |
|--------|---------|
| daily.py | daily summary (runs once/day) |
| dashboard.py | one-page trading overview |
| voice_alerts.py | narrated market briefing |

### organism

| module | purpose |
|--------|---------|
| market_eye.py | organ - integrates with organism pulse |

## quick commands

```bash
# morning overview
python3 dashboard.py

# check signals
python3 scanner.py buy

# sector rotation
python3 sectors.py scan

# voice briefing
python3 voice_alerts.py brief

# watchlist
python3 watchlist.py scan

# dry run strategy
python3 strategy.py dry

# live trade (market open)
python3 run.py run

# journal
python3 journal.py list
```

## market phases

- pre-market (4am-9:30am ET)
- regular (9:30am-4pm ET)
- after-hours (4pm-8pm ET)
- closed

## notes

- paper trading only ($100k account)
- no real money at risk
- uses IEX free data feed
- market_eye integrated with organism
