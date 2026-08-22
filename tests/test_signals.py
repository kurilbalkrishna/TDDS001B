import pandas as pd
from pytest import approx
from Src.signals.utils import calculate_momentum


def test_calculate_momentum_basic():
    prices = pd.Series([100, 110, 121])
    result = calculate_momentum(prices, periods=1)

    assert result.iloc[1] == approx(0.10)
    assert result.iloc[2] == approx(0.10)