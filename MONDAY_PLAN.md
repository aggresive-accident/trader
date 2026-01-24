# Monday Execution Plan

## Pre-Market (Before 9:30 AM ET)

1. Run `python3 edge.py` - check overnight changes
2. Run `python3 market_report.py` - regime check
3. Check top setup from edge.py output

## At Open (9:30 AM ET)

### Current Signal: META (score 4)
```bash
python3 execute.py --execute --symbol META --action buy --shares 30
```
- Entry: ~$658
- Stop: $632 (1.5x ATR below entry)
- Risk: 0.8% of portfolio

### Alternative: Wait for pullback
- If META gaps up >3%: WAIT
- Check NVDA, AMD for secondary setups

## During Day

Monitor with:
```bash
python3 execute.py  # Checks exit signals
```

Exit if:
1. Stop hit ($242.78 from Friday close)
2. Closes below 20 MA (~$240)
3. Gives back 50% of gains from entry

## Rules

1. ONE trade only
2. NO averaging down
3. If stopped out, done for the day
4. Journal every decision

## The Edge

Momentum concentration. Not 24 strategies. Not diversification.
Strong stocks get stronger. Ride with discipline. Cut losses fast.

---

*"If you can't describe your edge in one sentence, you don't have one."*

Edge: Buy the strongest momentum stock on a pullback, stop below ATR, let it run.
