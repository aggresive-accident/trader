# Monday Execution Plan

## Pre-Market (Before 9:30 AM ET)

1. Run `python3 edge.py` - check overnight changes
2. Run `python3 market_report.py` - regime check
3. Check AMD pre-market price:
   - If gapped down to $247-250: PREPARE TO BUY
   - If gapped up: WAIT, don't chase
   - If flat: WATCH for first 30 min

## At Open (9:30 AM ET)

### If AMD < $250:
```bash
python3 execute.py --execute --symbol AMD --action buy --shares 77
```
- Entry: ~$247-250
- Stop: $235 (below ATR support)
- Target: Let it run, trail stop

### If AMD > $255:
- DO NOT BUY
- Wait for pullback or find alternative setup
- Check INTC, META as backup plays

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
