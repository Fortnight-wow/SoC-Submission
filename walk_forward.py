import pandas as pd
import numpy as np
import statsmodels.api as sm
from hurst_exponent import calculate_hurst_exponent, classify_regime
from risk_management import kelly_criterion, calculate_sharpe_ratio, calculate_max_drawdown


def walk_forward_parameters(df, train_window=120, step_size=20):
    """Re-estimate Hurst, beta, half-life, Kelly on rolling windows."""
    n = len(df)
    cols = ['Regime', 'Hurst', 'Hedge_Ratio', 'Half_Life', 'Kelly_Fraction']
    params = pd.DataFrame(index=df.index, columns=cols)
    params['Kelly_Fraction'] = 0.02

    has_pair = 'Asset_Y' in df.columns and 'Asset_X' in df.columns
    y = df['Asset_Y'] if has_pair else df.get('Close', pd.Series(0, index=df.index))
    x = df['Asset_X'] if has_pair else pd.Series(0, index=df.index)

    for start in range(0, n - train_window, step_size):
        end = start + train_window
        oos_end = min(end + step_size, n)
        ty, tx = y.iloc[start:end].dropna(), x.iloc[start:end].dropna()
        idx = ty.index.intersection(tx.index)
        ty, tx = ty.loc[idx], tx.loc[idx]
        if len(ty) < 30:
            continue

        # Hurst + regime
        spread = (ty - tx) if has_pair else ty
        hr = calculate_hurst_exponent(spread.dropna())
        H = hr['hurst_combined']
        regime = classify_regime(H)[0] if not np.isnan(H) else 'random'

        # Beta via OLS
        beta = sm.OLS(ty, sm.add_constant(tx)).fit().params.iloc[1] if len(ty) > 10 else np.nan

        # Half-life from OU
        half_life = np.nan
        if has_pair and not np.isnan(beta):
            sp = ty - beta * tx
            lag, diff = sp.shift(1).dropna(), sp.diff().dropna()
            lag, diff = lag.align(diff, join='inner')
            if len(lag) > 10:
                theta = -sm.OLS(diff, sm.add_constant(lag)).fit().params.iloc[1]
                half_life = np.log(2) / theta if theta > 0 else np.nan

        # Kelly from return distribution
        ret = ty.pct_change().dropna()
        kf = 0.02
        if len(ret) > 20 and ret.std() > 0:
            wr = np.clip(np.sign(ret).diff().abs().sum() / len(ret) * 0.5 + 0.3, 0.3, 0.7)
            w, l = ret[ret > 0], ret[ret < 0]
            wl = (w.mean() if len(w) else 0.01) / abs(l.mean()) if len(l) else 1.0
            kf = kelly_criterion(wr, wl) * 0.5

        oos = df.index[end:oos_end]
        params.loc[oos, 'Regime'] = regime
        params.loc[oos, 'Hurst'] = H
        params.loc[oos, 'Hedge_Ratio'] = beta
        params.loc[oos, 'Half_Life'] = half_life
        params.loc[oos, 'Kelly_Fraction'] = kf

    return params.ffill().fillna({'Regime': 'unknown', 'Kelly_Fraction': 0.02})


def walk_forward_backtest(df, initial_capital=100000, commission=0.001,
                          train_window=120, step_size=20,
                          momentum_entry=0.02, momentum_exit=0.005):
    """Walk-forward backtest: re-estimate all params on rolling windows."""
    from indicators import calculate_z_score, calculate_spread

    df = df.copy()
    n = len(df)
    wf = walk_forward_parameters(df, train_window, step_size)
    for col in ['WF_Regime', 'WF_Hurst', 'WF_Beta', 'WF_Half_Life', 'WF_Kelly']:
        df[col] = wf[col.replace('WF_', '')]

    if 'Spread' not in df.columns:
        df = calculate_spread(df)
    df = calculate_z_score(df)
    df['Momentum'] = df.get('Asset_Y', df.get('Close', pd.Series(0, index=df.index))).pct_change(20)

    # Generate signals
    signals, pos = [], 0
    for i in range(n):
        if i == 0 or pd.isna(df.loc[i, 'Z_Score']):
            signals.append("HOLD"); continue
        z, regime = df.loc[i, 'Z_Score'], df.loc[i, 'WF_Regime']
        mom = df.loc[i, 'Momentum'] if not pd.isna(df.loc[i, 'Momentum']) else 0

        if regime == 'mean_revert':
            if z < -2.0: pos = 1
            elif z > 2.0: pos = -1
            elif (pos == 1 and z >= 0) or (pos == -1 and z <= 0): pos = 0
        elif regime == 'trend':
            if mom > momentum_entry: pos = 1
            elif mom < -momentum_entry: pos = -1
            elif pos == 1 and mom < -momentum_exit: pos = 0
            elif pos == -1 and mom > momentum_exit: pos = 0
        else:
            pos = 0
        signals.append("LONG SPREAD" if pos == 1 else "SHORT SPREAD" if pos == -1 else "HOLD")

    df['Signal'] = signals

    # Run backtest
    equity, pnl, positions, kelly_used = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    equity[0] = initial_capital
    wins, losses, win_amt, loss_amt = 0, 0, [], []
    pos, ey, ex, eb = 0, 0, 0, 0

    for i in range(1, n):
        sig, ay, ax = df.loc[i, 'Signal'], df.loc[i, 'Asset_Y'], df.loc[i, 'Asset_X']
        kf = df.loc[i, 'WF_Kelly']
        kelly_used[i] = kf

        if pos != 0 and (sig == "HOLD" or (pos == 1 and sig == "SHORT SPREAD") or (pos == -1 and sig == "LONG SPREAD")):
            p = pos * ((ay - ey) - eb * (ax - ex)) if pos == 1 else -((ay - ey) - eb * (ax - ex))
            p -= abs(positions[i - 1] if i > 0 else 0) * commission
            pnl[i] = p
            if p > 0: wins += 1; win_amt.append(p)
            else: losses += 1; loss_amt.append(p)
            pos = 0
        elif pos != 0:
            pnl[i] = pos * ((ay - ey) - eb * (ax - ex)) if pos == 1 else -((ay - ey) - eb * (ax - ex))
            positions[i] = pos

        if pos == 0 and sig in ("LONG SPREAD", "SHORT SPREAD"):
            beta = df.loc[i, 'WF_Beta'] if not pd.isna(df.loc[i, 'WF_Beta']) else 1.0
            cap = equity[i - 1] * kf
            pos = 1 if sig == "LONG SPREAD" else -1
            positions[i], ey, ex, eb = cap, ay, ax, beta
            pnl[i] -= cap * commission

        equity[i] = equity[i - 1] + pnl[i]

    results = df.copy()
    results['Daily_PnL'], results['Equity'], results['Position'] = pnl, equity, positions
    results['Kelly_Fraction'] = kelly_used
    ret = pd.Series(pnl[1:] / np.where(equity[:-1] != 0, equity[:-1], 1), index=df.index[1:])
    cum = pd.Series(equity, index=df.index)
    return results, {
        'total_return': equity[-1] / equity[0] - 1, 'annualized_return': ret.mean() * 252,
        'annualized_volatility': ret.std() * np.sqrt(252),
        'sharpe_ratio': calculate_sharpe_ratio(ret) if len(ret) else 0,
        'max_drawdown': calculate_max_drawdown(cum)[0],
        'total_trades': wins + losses, 'win_rate': wins / (wins + losses) if (wins + losses) else 0,
        'avg_kelly_fraction': np.mean(kelly_used[kelly_used > 0]) if np.any(kelly_used > 0) else 0,
        'final_equity': equity[-1], 'walk_forward': True,
    }
