import pandas as pd


def calculate_momentum(close_prices: pd.Series, periods: int) -> pd.Series:
    """Calculate percentage price change over a given number of periods."""
    return close_prices.pct_change(periods=periods)


def calculate_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling standard deviation of returns over a given window."""
    return returns.rolling(window=window).std()