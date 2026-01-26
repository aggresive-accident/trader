# Zoo Backtest Results - 2026-01-26

90-day backtest on SPY,QQQ,AAPL,MSFT,GOOGL,AMZN,NVDA,META,AMD,TSLA

## Top Performers

| Strategy | Return | Sharpe | MaxDD | Win% |
|----------|--------|--------|-------|------|
| bollinger | +5.44% | 3.68 | 1.9% | 100% |
| momentum | +4.89% | 2.00 | 2.7% | 60% |
| adaptive | +3.76% | 3.10 | 1.9% | 50% |
| regime | +2.83% | 1.30 | 4.4% | 38% |
| vwap | +2.06% | 1.10 | 2.6% | 69% |

## Breakout Strategies (scope requirement)

| Strategy | Return | Sharpe | MaxDD | Win% |
|----------|--------|--------|-------|------|
| donchian | +0.74% | 0.48 | 2.5% | 20% |
| range_breakout | -0.30% | -1.15 | 0.6% | 0% |
| highlow | -0.78% | -1.26 | 1.9% | 0% |
| gap | +1.52% | 0.51 | 5.7% | 46% |

## Recommendations for Multi-Strategy

1. **Momentum** (edge.py) - currently in use, proven
2. **Bollinger** - highest Sharpe, mean reversion (diversification)
3. **Adaptive** - market-aware, good Sharpe

Do NOT include:
- Donchian/breakout: underperformed, low win rates
- MA crossover: -1.59%, worst performer
- Ensembles: surprisingly poor performance

## Notes

- Bollinger reversion is negatively correlated with momentum (good for diversification)
- Adaptive strategy adjusts to market regime
- Current positions (META, XOM) entered via momentum/donchian - holding
