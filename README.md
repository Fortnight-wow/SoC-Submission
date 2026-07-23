# Statistical Arbitrage — Endterm Submission

Pairs trading + trend following with automatic regime detection via Hurst Exponent.

## How it works

1. **Hurst Exponent** classifies the market:
   - H < 0.45 → mean-reverting → pairs trading
   - H > 0.55 → trending → trend following
   - 0.45–0.55 → random → stay flat

2. **Pairs Trading**: OLS hedge ratio → OU half-life → Z-score signals (enter at ±2σ, exit at 0)

3. **Trend Following**: EMA crossover + ADX filter (>25) to avoid fakeouts + ATR trailing stop

4. **Risk Management**: Portfolio Beta hedging to zero, VaR, Sharpe ratio

5. **Backtesting**: Kelly criterion position sizing, no look-ahead bias (beta computed only from past data)

6. **Walk-Forward**: Re-estimates all parameters on rolling windows to eliminate overfitting

## Usage

```bash
python main.py your_data.csv auto           # auto-detect regime
python main.py your_data.csv mean_revert    # force pairs trading
python main.py your_data.csv trend          # force trend following
python main.py your_data.csv auto --walk-forward  # walk-forward backtest
```

## CSV Format

For pairs: `Date,Asset_Y,Asset_X`
For trend: `Date,High,Low,Close`

## Files

| File | What it does |
|------|-------------|
| `main.py` | Pipeline orchestrator |
| `hurst_exponent.py` | R/S + DFA Hurst exponent |
| `indicators.py` | Spread, Z-score, half-life |
| `filters.py` | ADF stationarity test |
| `trend_following.py` | EMA + ADX + ATR signals |
| `risk_management.py` | Beta hedging, VaR, Kelly |
| `backtest.py` | Backtesting engine |
| `walk_forward.py` | Rolling parameter re-estimation |
| `signal_generator.py` | Auto strategy selection |

## Dependencies

```bash
pip install pandas numpy statsmodels
```
