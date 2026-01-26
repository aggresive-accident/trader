#!/bin/bash
# Trader position monitor - runs during market hours
# Called by systemd timer every 5 minutes

# Singleton check
LOCKFILE=~/.locks/trader-monitor.lock
mkdir -p ~/.locks
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then
        exit 0  # Already running
    fi
fi
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

cd ~/workspace/trader
source venv/bin/activate

# Only run if market is open (rough check - 9:30-16:00 ET)
HOUR=$(TZ=America/New_York date +%H)
MIN=$(TZ=America/New_York date +%M)
DOW=$(date +%u)

# Skip weekends
if [ "$DOW" -gt 5 ]; then
    exit 0
fi

# Skip if before 9:30 or after 16:00 ET
if [ "$HOUR" -lt 9 ] || [ "$HOUR" -gt 15 ]; then
    exit 0
fi
if [ "$HOUR" -eq 9 ] && [ "$MIN" -lt 30 ]; then
    exit 0
fi

# Run monitor
LOG=~/.trader-monitor.log
echo "--- $(date) ---" >> $LOG
python3 monitor.py --quiet >> $LOG 2>&1

# If there were alerts, also log to relay
if grep -q "EXIT SIGNAL\|WARNING\|ACTION REQUIRED" $LOG; then
    ALERT=$(tail -20 $LOG | grep -E "EXIT SIGNAL|WARNING|ACTION REQUIRED" | head -1)
    if [ -n "$ALERT" ]; then
        python3 ~/workspace/relay/relay.py "TRADER ALERT: $ALERT" 2>/dev/null || true
    fi
fi
