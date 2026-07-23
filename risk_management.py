import pandas as pd
import numpy as np
import statsmodels.api as sm


def calculate_portfolio_beta(port_returns, bench_returns, window=None):
    """Beta via OLS or rolling covariance."""
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    aligned.columns = ['portfolio', 'benchmark']
    if window:
        cov = aligned['portfolio'].rolling(window, min_periods=10).cov(aligned['benchmark'])
        var = aligned['benchmark'].rolling(window, min_periods=10).var()
        return cov / var.replace(0, np.nan)
    if len(aligned) < 10:
        return np.nan
    model = sm.OLS(aligned['portfolio'], sm.add_constant(aligned['benchmark'])).fit()
    return model.params.iloc[1]


def apply_beta_hedge(port_returns, bench_returns, window=None):
    """Hedge portfolio to zero beta. Returns hedged returns + beta series."""
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    aligned.columns = ['portfolio', 'benchmark']
    if window:
        beta = calculate_portfolio_beta(port_returns, bench_returns, window).reindex(aligned.index)
    else:
        cov = aligned['portfolio'].expanding(min_periods=10).cov(aligned['benchmark'])
        var = aligned['benchmark'].expanding(min_periods=10).var()
        beta = cov / var.replace(0, np.nan)
    return aligned['portfolio'] - beta * aligned['benchmark'], beta


def calculate_portfolio_var(returns, confidence=0.95):
    """Historical VaR."""
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year) if excess.std() > 0 else np.nan


def calculate_max_drawdown(cumulative_returns):
    """Max drawdown from equity curve."""
    peak = cumulative_returns.expanding(min_periods=1).max()
    dd = (cumulative_returns - peak) / peak
    max_dd = dd.min()
    if max_dd >= 0:
        return 0.0, cumulative_returns.index[0], cumulative_returns.index[0]
    trough = dd.idxmin()
    peak_idx = cumulative_returns.loc[:trough].idxmax() if len(cumulative_returns.loc[:trough]) > 0 else cumulative_returns.index[0]
    return max_dd, peak_idx, trough


def risk_summary(port_returns, bench_returns=None, risk_free_rate=0.0):
    """Portfolio risk metrics."""
    s = {}
    s['total_return'] = (1 + port_returns).prod() - 1
    s['annualized_return'] = port_returns.mean() * 252
    s['annualized_volatility'] = port_returns.std() * np.sqrt(252)
    s['sharpe_ratio'] = calculate_sharpe_ratio(port_returns, risk_free_rate)
    s['var_95'] = calculate_portfolio_var(port_returns, 0.95)
    s['var_99'] = calculate_portfolio_var(port_returns, 0.99)
    s['max_drawdown'] = calculate_max_drawdown((1 + port_returns).cumprod())[0]
    if bench_returns is not None:
        aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
        if len(aligned) > 10:
            b = calculate_portfolio_beta(aligned.iloc[:, 0], aligned.iloc[:, 1])
            s['portfolio_beta'], s['hedge_ratio'] = b, -b
            hedged, _ = apply_beta_hedge(aligned.iloc[:, 0], aligned.iloc[:, 1])
            s['hedged_sharpe'] = calculate_sharpe_ratio(hedged, risk_free_rate)
            s['hedged_volatility'] = hedged.std() * np.sqrt(252)
    return s


def kelly_criterion(win_rate, win_loss_ratio):
    """Kelly fraction: f* = (bp - q) / b, clamped to [0, 1]."""
    p, b = win_rate, win_loss_ratio
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    return max(0.0, min((b * p - (1 - p)) / b, 1.0))
