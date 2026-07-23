import pandas as pd
import numpy as np


def calculate_adx(df, period=14):
    """Average Directional Index for trend strength."""
    df = df.copy()
    h, l, c = df['High'], df['Low'], df['Close']

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    up, down = h - h.shift(1), l.shift(1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = tr.rolling(period, min_periods=1).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=1).sum() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=1).sum() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df['Plus_DI'], df['Minus_DI'], df['ADX'] = plus_di, minus_di, dx.rolling(period, min_periods=1).mean()
    return df


def calculate_atr(df, period=14):
    """Average True Range for trailing stop."""
    df = df.copy()
    tr = pd.concat([df['High'] - df['Low'],
                     (df['High'] - df['Close'].shift(1)).abs(),
                     (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(period, min_periods=1).mean()
    return df


def trend_following_signals(df, fast_period=20, slow_period=50, adx_threshold=25, atr_multiplier=2.0):
    """EMA crossover + ADX filter + ATR trailing stop."""
    df = df.copy()
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df = calculate_adx(df)
    df = calculate_atr(df)

    signals, pos, stop = [], 0, np.nan
    for i in range(len(df)):
        if i < slow_period or pd.isna(df.loc[i, 'ADX']):
            signals.append("HOLD"); continue

        fast, slow, adx, atr, close = (df.loc[i, 'EMA_Fast'], df.loc[i, 'EMA_Slow'],
                                        df.loc[i, 'ADX'], df.loc[i, 'ATR'], df.loc[i, 'Close'])
        pf, ps = df.loc[i - 1, 'EMA_Fast'], df.loc[i - 1, 'EMA_Slow']

        # Trailing stop exit
        if pos == 1 and not pd.isna(stop) and close <= stop:
            pos, stop = 0, np.nan; signals.append("HOLD"); continue
        elif pos == -1 and not pd.isna(stop) and close >= stop:
            pos, stop = 0, np.nan; signals.append("HOLD"); continue

        # Entry on crossover + ADX confirmation
        if adx > adx_threshold:
            if pf <= ps and fast > slow:
                pos, stop = 1, close - atr_multiplier * atr
            elif pf >= ps and fast < slow:
                pos, stop = -1, close + atr_multiplier * atr

        # Update trailing stop
        if pos == 1 and not pd.isna(atr):
            s = close - atr_multiplier * atr
            if pd.isna(stop) or s > stop: stop = s
        elif pos == -1 and not pd.isna(atr):
            s = close + atr_multiplier * atr
            if pd.isna(stop) or s < stop: stop = s

        # Exit on opposite crossover
        if (pos == 1 and fast < slow) or (pos == -1 and fast > slow):
            pos, stop = 0, np.nan

        signals.append("LONG TREND" if pos == 1 else "SHORT TREND" if pos == -1 else "HOLD")

    df['Signal'] = signals
    return df
