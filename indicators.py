import pandas as pd
import numpy as np
import statsmodels.api as sm


def calculate_spread(df, y_col='Asset_Y', x_col='Asset_X'):
    """OLS hedge ratio and spread isolation."""
    df = df.copy()
    y, x = df[y_col], df[x_col]
    model = sm.OLS(y, sm.add_constant(x)).fit()
    beta, c = model.params.iloc[1], model.params.iloc[0]
    df['Spread'] = y - beta * x - c
    return df


def calculate_z_score(df, spread_col='Spread'):
    """OU half-life + rolling Z-score."""
    df = df.copy()
    spread = df[spread_col]
    lag = spread.shift(1).dropna()
    diff = spread.diff().dropna()
    lag, diff = lag.align(diff, join='inner')

    model = sm.OLS(diff, sm.add_constant(lag)).fit()
    theta = -model.params.iloc[1]
    half_life = np.log(2) / theta if theta > 0 else np.nan

    if pd.isna(half_life):
        df['Z_Score'] = np.nan
        return df

    window = max(int(round(half_life)), 5)
    df['Rolling_Mean'] = spread.rolling(window).mean()
    df['Rolling_Std'] = spread.rolling(window).std()
    df['Z_Score'] = (spread - df['Rolling_Mean']) / df['Rolling_Std']
    return df
