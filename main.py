"""Statistical Arbitrage — Endterm Pipeline.

Usage:
    python main.py <csv_file> [strategy] [--walk-forward]
    strategy: auto (default), mean_revert, trend
"""
import sys, os
import pandas as pd
import numpy as np
from hurst_exponent import calculate_hurst_exponent
from indicators import calculate_spread, calculate_z_score
from signal_generator import generate_signals
from risk_management import risk_summary, apply_beta_hedge
from backtest import (backtest_mean_reversion, backtest_trend_following, print_backtest_results)
from walk_forward import walk_forward_backtest


def run_full_pipeline(csv_file, strategy='auto', use_walk_forward=False):
    print("=" * 70 + "\n  STATISTICAL ARBITRAGE — ENDTERM PIPELINE\n" + "=" * 70)

    # Load
    df = pd.read_csv(csv_file)
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'])
    print(f"\n[1/6] Loaded {len(df)} rows from {csv_file}")

    # Hurst
    spread_df = calculate_spread(df.copy()) if 'Asset_Y' in df.columns else df
    series = spread_df['Spread'].dropna() if 'Spread' in spread_df.columns else df['Close']
    regime_info = calculate_hurst_exponent(series)
    detected = strategy if strategy != 'auto' else regime_info['regime']
    print(f"\n[2/6] Hurst: {regime_info['hurst_combined']:.4f} | Regime: {detected}")

    # Signals
    signals_df, _ = generate_signals(csv_file, strategy)
    sc = signals_df['Signal'].value_counts()
    print(f"\n[3/6] Signals: {dict(sc)}")

    # Backtest
    if use_walk_forward:
        print(f"\n[4/6] Walk-Forward Backtest...")
        bt_results, bt_metrics = walk_forward_backtest(signals_df)
        print_backtest_results(bt_metrics, f"Walk-Forward {detected.upper()}")
    elif detected == 'mean_revert':
        bt_results, bt_metrics = backtest_mean_reversion(signals_df)
        print_backtest_results(bt_metrics, "Mean-Reversion")
    elif detected == 'trend':
        bt_results, bt_metrics = backtest_trend_following(signals_df)
        print_backtest_results(bt_metrics, "Trend-Following")
    else:
        bt_metrics = {'total_return': 0, 'annualized_return': 0, 'annualized_volatility': 0,
                      'sharpe_ratio': 0, 'max_drawdown': 0, 'final_equity': 100000,
                      'total_trades': 0, 'win_rate': 0, 'avg_kelly_fraction': 0}
        bt_results = signals_df.copy()
        bt_results['Equity'] = 100000
        print("  Random market — no trades.")

    # Risk
    print(f"\n[5/6] Risk Analysis...")
    if 'Asset_Y' in signals_df.columns and 'Asset_X' in signals_df.columns:
        pr = signals_df['Asset_Y'].pct_change().dropna()
        br = signals_df['Asset_X'].pct_change().dropna()
        rm = risk_summary(pr, br)
        for k in ['portfolio_beta', 'hedge_ratio', 'hedged_sharpe', 'var_95', 'var_99']:
            if k in rm and isinstance(rm[k], float):
                print(f"  {k}: {rm[k]:.4f}")
        hedged, _ = apply_beta_hedge(pr, br)
        print(f"  Hedged Vol: {hedged.std() * np.sqrt(252):.4f}")

    # Save
    prefix = os.path.splitext(csv_file)[0].replace(' ', '_')
    bt_results.to_csv(f"{prefix}_backtest_results.csv", index=False)
    print(f"\n[6/6] Saved {prefix}_backtest_results.csv")

    print(f"\n{'=' * 70}\n  RETURN: {bt_metrics['total_return']:.2%} | SHARPE: {bt_metrics['sharpe_ratio']:.2f} | DD: {bt_metrics['max_drawdown']:.2%}\n{'=' * 70}")
    return {'regime_info': regime_info, 'backtest_metrics': bt_metrics, 'detected_regime': detected}


def generate_sample_data():
    """Generate sample cointegrated pair for demo."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    common = np.cumsum(np.random.randn(n) * 0.01)
    ax = 100 + common + np.cumsum(np.random.randn(n) * 0.005)
    ay = 1.5 * ax + 10 + np.cumsum(np.random.randn(n) * 0.02)
    pd.DataFrame({'Date': dates, 'Asset_Y': ay, 'Asset_X': ax}).to_csv('sample_cointegrated_data.csv', index=False)
    return 'sample_cointegrated_data.csv'


if __name__ == "__main__":
    use_wf = '--walk-forward' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--walk-forward']
    if not args:
        print("Usage: python main.py <csv> [strategy] [--walk-forward]")
        csv_file = generate_sample_data()
        strategy = 'auto'
    else:
        csv_file = args[0]
        strategy = args[1] if len(args) > 1 else 'auto'
        if not os.path.exists(csv_file):
            csv_file = generate_sample_data()
    run_full_pipeline(csv_file, strategy, use_wf)
