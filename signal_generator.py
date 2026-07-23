import pandas as pd
import numpy as np
from indicators import calculate_spread, calculate_z_score
from filters import stationarity_filter
from hurst_exponent import calculate_hurst_exponent


def generate_signals(csv_file, strategy='auto'):
    """Generate signals with auto regime detection via Hurst Exponent."""
    df = pd.read_csv(csv_file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Hurst analysis on spread or close
    if 'Asset_Y' in df.columns and 'Asset_X' in df.columns:
        df = calculate_spread(df)
        series = df['Spread'].dropna()
    elif 'Close' in df.columns:
        series = df['Close']
    else:
        raise ValueError("CSV needs ('Asset_Y','Asset_X') or 'Close'")

    regime_info = calculate_hurst_exponent(series)
    regime = strategy if strategy != 'auto' else regime_info['regime']
    print(f"  Hurst: {regime_info['hurst_combined']:.4f} | Regime: {regime}")

    if regime == 'mean_revert' or strategy == 'mean_revert':
        df = calculate_z_score(df)
        df = stationarity_filter(df)
        return _mean_revert_signals(df), regime_info

    elif regime == 'trend' or strategy == 'trend':
        from trend_following import trend_following_signals
        if not all(c in df.columns for c in ['High', 'Low', 'Close']):
            df['Close'] = df.get('Asset_Y', df.get('Close'))
            df['High'] = df['Close'] * 1.005
            df['Low'] = df['Close'] * 0.995
        df = trend_following_signals(df)
        df['Strategy'] = 'Trend Following'
        return df, regime_info

    df['Signal'] = 'HOLD'
    df['Strategy'] = 'No Trade'
    return df, regime_info


def _mean_revert_signals(df):
    """Z-score entry/exit logic."""
    signals, pos = [], 0
    for i in range(len(df)):
        if i == 0 or pd.isna(df.loc[i, 'Z_Score']):
            signals.append("HOLD"); continue
        z, stat = df.loc[i, 'Z_Score'], df.loc[i, 'Stationary_Filter']
        if z < -2.0 and stat: pos = 1
        elif z > 2.0 and stat: pos = -1
        elif (pos == 1 and z >= 0) or (pos == -1 and z <= 0): pos = 0
        signals.append("LONG SPREAD" if pos == 1 else "SHORT SPREAD" if pos == -1 else "HOLD")
    df['Signal'] = signals
    df['Strategy'] = 'Mean Reversion'
    return df


def generate_mean_reversion_signals(csv_file):
    """Standalone mean-reversion (backward-compatible with midterm)."""
    df = pd.read_csv(csv_file)
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'])
    df = calculate_spread(df)
    df = calculate_z_score(df)
    df = stationarity_filter(df)
    return _mean_revert_signals(df)


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else input("CSV filename: ")
    s = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    result, info = generate_signals(f, s)
    print(f"\nH (Combined): {info['hurst_combined']:.4f} | Regime: {info['regime']}")
    cols = [c for c in ['Date', 'Signal', 'Strategy', 'Z_Score', 'Spread', 'ADX'] if c in result.columns]
    print(result[cols].tail(20))
    result.to_csv("signals_output.csv", index=False)
    print("Saved to signals_output.csv")
