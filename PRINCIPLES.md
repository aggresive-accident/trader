# Architectural Principles

These principles govern design decisions in the trader system.

## 1. No AI where determinism suffices
If a rule can be expressed precisely, code it. Reserve AI for judgment calls.

## 2. Prove edge before deploying capital
Backtest, paper trade, verify. No strategy goes live on intuition alone.

## 3. Signal-based exits > rule-based stops
ATR stops destroyed edge across all regimes. Exit when the signal that got you in reverses.

## 4. Separation of concerns with explicit interfaces
Thesis trades don't touch autopilot. Strategies don't know about execution. Clean boundaries.

## 5. Self-healing state
System should auto-recover from stale state, missed runs, and reboots without manual intervention.

## 6. Paper trade everything, then deploy
Real money is the last step, not the first. Validate behavior before risking capital.

## 7. Thesis trades separate from systematic
Discretionary positions live outside the zoo. Different logic, different risk, explicit isolation.

## 8. Log everything, rotate nothing (for now)
Debug-ability trumps disk space. Can't diagnose what wasn't recorded.

## 9. Fail loud, not silent
Errors surface immediately. Silent failures compound into disasters.

## 10. Context is sacred
State export, morning/evening wrappers, session context. An agent resuming work should know exactly where things stand.

## 11. Validate before trusting
Reconcile ledger vs broker. Check fills. Verify assumptions. Trust but verify.

## 12. Minimum viable deployment, then iterate
Ship the simplest thing that works. Complexity earns its way in through demonstrated need.

## 13. Decisions should be traceable
Every trade has a reason logged. Every exit has a cause. Audit trail is non-negotiable.
