import pandas as pd
import numpy as np
import statsmodels.api as sm
from risk_management import calculate_sharpe_ratio, calculate_max_drawdown, kelly_criterion


def backtest_mean_reversion(df, initial_capital=100000, commission=0.001, fractional_kelly=0.5):
    """Pairs trading backtest with Kelly sizing. No look-ahead bias."""
    df = df.copy()
    n = len(df)
    equity, pnl, positions, pos_cap = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    equity[0] = initial_capital
    kelly_arr = np.zeros(n)
    wins, losses, win_amt, loss_amt = 0, 0, [], []
    pos, ey, ex, eb = 0, 0, 0, 0

    for i in range(1, n):
        sig, ay, ax = df.loc[i, 'Signal'], df.loc[i, 'Asset_Y'], df.loc[i, 'Asset_X']
        kf = risk_per_trade(wins, losses, win_amt, loss_amt, fractional_kelly) if wins + losses > 10 else 0.02
        kelly_arr[i] = kf

        # Close position
        if pos != 0 and (sig == "HOLD" or (pos == 1 and sig == "SHORT SPREAD") or (pos == -1 and sig == "LONG SPREAD")):
            p = pos * ((ay - ey) - eb * (ax - ex))
            p -= abs(pos_cap[i - 1]) * commission
            pnl[i] = p
            if p > 0: wins += 1; win_amt.append(p)
            else: losses += 1; loss_amt.append(p)
            pos = 0
        elif pos != 0:
            pnl[i] = pos * ((ay - ey) - eb * (ax - ex))
            positions[i] = pos

        # Open position
        if pos == 0 and sig in ("LONG SPREAD", "SHORT SPREAD"):
            yc, xc = df['Asset_Y'].iloc[:i + 1], df['Asset_X'].iloc[:i + 1]
            beta = sm.OLS(yc, sm.add_constant(xc)).fit().params.iloc[1] if len(yc) > 10 else 1.0
            cap = equity[i - 1] * kf
            pos = 1 if sig == "LONG SPREAD" else -1
            pos_cap[i], ey, ex, eb = cap, ay, ax, beta
            positions[i] = pos
            pnl[i] -= cap * commission

        equity[i] = equity[i - 1] + pnl[i]

    results = df.copy()
    results['Daily_PnL'], results['Equity'], results['Position'] = pnl, equity, positions
    results['Kelly_Fraction'] = kelly_arr
    ret = pd.Series(pnl[1:] / np.where(equity[:-1] != 0, equity[:-1], 1), index=df.index[1:])
    cum = pd.Series(equity, index=df.index)
    return results, _build_metrics(equity, ret, cum, wins, losses, kelly_arr)


def risk_per_trade(wins, losses, win_amt, loss_amt, frac):
    """Kelly position size from trade history."""
    wr = wins / (wins + losses)
    aw = np.mean(win_amt) if win_amt else 1.0
    al = abs(np.mean(loss_amt)) if loss_amt else 1.0
    return kelly_criterion(wr, aw / al if al > 0 else 1.0) * frac


def backtest_trend_following(df, initial_capital=100000, commission=0.001, risk_per_trade_val=0.02):
    """Trend-following backtest with ATR trailing stop."""
    df = df.copy()
    n = len(df)
    equity, pnl, positions = np.zeros(n), np.zeros(n), np.zeros(n)
    equity[0] = initial_capital
    pos, entry, stop, units = 0, 0, np.nan, 0

    for i in range(1, n):
        sig, close = df.loc[i, 'Signal'], df.loc[i, 'Close']
        atr = df.loc[i, 'ATR'] if 'ATR' in df.columns else np.nan

        # Trailing stop exit
        if pos != 0 and not pd.isna(stop):
            if (pos == 1 and close <= stop) or (pos == -1 and close >= stop):
                pnl[i] = units * (close - entry) - abs(units * close) * commission
                pos = 0; equity[i] = equity[i - 1] + pnl[i]; continue

        if pos != 0:
            pnl[i] = units * (close - entry)

        if sig == "LONG TREND" and pos != 1:
            if pos == -1: pnl[i] = units * (close - entry) - abs(units * close) * commission
            cap = equity[i - 1] * risk_per_trade_val
            units, entry, pos = cap / close if close > 0 else 0, close, 1
            stop = close - 2 * atr if not pd.isna(atr) else np.nan
            pnl[i] -= cap * commission
        elif sig == "SHORT TREND" and pos != -1:
            if pos == 1: pnl[i] = units * (close - entry) - abs(units * close) * commission
            cap = equity[i - 1] * risk_per_trade_val
            units, entry, pos = -cap / close if close > 0 else 0, close, -1
            stop = close + 2 * atr if not pd.isna(atr) else np.nan
            pnl[i] -= cap * commission

        # Update trailing stop
        if pos == 1 and not pd.isna(atr):
            s = close - 2 * atr
            if pd.isna(stop) or s > stop: stop = s
        elif pos == -1 and not pd.isna(atr):
            s = close + 2 * atr
            if pd.isna(stop) or s < stop: stop = s

        equity[i] = equity[i - 1] + pnl[i]
        positions[i] = pos

    results = df.copy()
    results['Daily_PnL'], results['Equity'], results['Position'] = pnl, equity, positions
    ret = pd.Series(pnl[1:] / np.where(equity[:-1] != 0, equity[:-1], 1), index=df.index[1:])
    cum = pd.Series(equity, index=df.index)
    return results, _build_metrics(equity, ret, cum)


def _build_metrics(equity, ret, cum, wins=0, losses=0, kelly_arr=None):
    """Build standard metrics dict."""
    return {
        'total_return': equity[-1] / equity[0] - 1 if equity[0] else 0,
        'annualized_return': ret.mean() * 252 if len(ret) else 0,
        'annualized_volatility': ret.std() * np.sqrt(252) if len(ret) else 0,
        'sharpe_ratio': calculate_sharpe_ratio(ret) if len(ret) else 0,
        'max_drawdown': calculate_max_drawdown(cum)[0],
        'total_trades': wins + losses,
        'win_rate': wins / (wins + losses) if (wins + losses) else 0,
        'avg_kelly_fraction': np.mean(kelly_arr[kelly_arr > 0]) if kelly_arr is not None and np.any(kelly_arr > 0) else 0,
        'final_equity': equity[-1],
    }


def print_backtest_results(metrics, name="Strategy"):
    print(f"\n{'=' * 50}\n  {name} Backtest Results\n{'=' * 50}")
    for k in ['total_return', 'annualized_return', 'annualized_volatility', 'max_drawdown']:
        print(f"  {k:20s} {metrics[k]:.2%}")
    print(f"  {'sharpe_ratio':20s} {metrics['sharpe_ratio']:.2f}")
    print(f"  {'final_equity':20s} ${metrics['final_equity']:,.2f}")
    if 'total_trades' in metrics:
        print(f"  {'total_trades':20s} {metrics['total_trades']}")
        print(f"  {'win_rate':20s} {metrics['win_rate']:.2%}")
    print(f"{'=' * 50}\n")
