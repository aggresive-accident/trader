# trader

My interface to the market. Paper trading via Alpaca.

## The Edge

**Buy the strongest momentum stock on a pullback, stop below ATR, let it run.**

Backtested over 1 year:

| Metric | Result |
|--------|--------|
| Return | +42% |
| Max Drawdown | 6.8% |
| Trades | 74 |
| Win Rate | 47% |
| Profit Factor | 2.92 |
| Alpha vs Buy-Hold | +5.3% |

## Entry Criteria

- Weekly return 5-10% (momentum, not overextended)
- Price above 20 MA
- Volume confirmation (>1.3x average)

## Exit Criteria

- Stop hit (1.5x ATR below entry)
- Close below 20 MA
- Trailing stop (gave back 50% of gains)

## Position Sizing

- Max 4 concurrent positions
- Max 20% per position
- Max 3% portfolio risk per trade

## Morning Routine

```bash
python3 morning.py  # Full pre-market checklist
```

Runs in sequence:
1. Gap scanner - overnight moves
2. Market regime - trend/volatility
3. Edge signals - entry opportunities
4. Position monitor - exit signals

## Core Scripts

| Script | Purpose |
|--------|---------|
| `edge.py` | Core system - scan and calculate positions |
| `execute.py` | Trade execution with journaling |
| `monitor.py` | Position monitoring with trailing stops |
| `premarket.py` | Gap scanner |
| `alerts.py` | Signal change notifications |
| `analytics.py` | Performance vs backtest |
| `market_report.py` | Market regime analysis |
| `morning.py` | Pre-market checklist |

## Trade Execution

```bash
# Dry run (default)
python3 execute.py

# Live execution
python3 execute.py --execute

# Manual trade
python3 execute.py --execute --symbol META --action buy --shares 30
```

## Monitoring

```bash
python3 monitor.py           # Check positions
python3 alerts.py check      # New signals
python3 alerts.py history    # Alert log
python3 analytics.py         # Performance review
```

## Setup

API keys in `~/.alpaca-keys` (chmod 600):
```
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
```

## Files

- `trades.json` - Trade journal
- `alerts_log.json` - Alert history
- `high_water_marks.json` - Trailing stop tracking

## Rules

1. One trade per slot (max 4)
2. No averaging down
3. If stopped out, done for the day
4. Journal every decision

## Strategy Zoo

`strategies/` contains 21 strategies for research/validation. The actual edge uses only momentum concentration in `edge.py`.

## Notes

- Paper trading only ($100k account)
- Uses IEX data feed
- Integrated with organism (market_eye.py)
- Trades logged to relay for continuity
