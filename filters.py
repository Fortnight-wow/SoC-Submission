import pandas as pd
from statsmodels.tsa.stattools import adfuller


def stationarity_filter(df, spread_col='Spread'):
    """ADF test on spread. Returns boolean filter."""
    df = df.copy()
    valid = df[spread_col].dropna()
    df['Stationary_Filter'] = adfuller(valid)[1] < 0.05 if len(valid) > 0 else False
    return df
