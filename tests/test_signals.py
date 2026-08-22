import pandas as pd
from pytest import approx
from Src.signals.utils import calculate_momentum


def test_calculate_momentum_basic():
    prices = pd.Series([100, 110, 121])
    result = calculate_momentum(prices, periods=1)

    assert result.iloc[1] == approx(0.10)
    assert result.iloc[2] == approx(0.10)
    
from Src.signals.utils import calculate_volatility


def test_calculate_volatility_basic():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
    result = calculate_volatility(returns, window=5)

    assert result.iloc[-1] == approx(0.01, rel=0.1)
    
def test_calculate_momentum_empty_series():
    prices = pd.Series([], dtype=float)
    result = calculate_momentum(prices, periods=1)

    assert len(result) == 0