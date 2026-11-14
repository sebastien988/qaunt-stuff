"""Shared helpers for the tests.

Every test runs on made-up prices, so the suite needs no internet connection
and gives the same answer every time.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the scripts live one folder up


def random_walk(days: int, seed: int, start: str = "2010-01-04", drift: float = 0.0003,
                vol: float = 0.012, freq: str = "B") -> pd.Series:
    """A made-up price series: a geometric random walk on business days."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq=freq)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, days))), index=index)


def random_panel(tickers, days: int, seed: int, **kwargs) -> pd.DataFrame:
    """Made-up prices for several tickers, loosely correlated through a shared market move."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(kwargs.pop("start", "2010-01-04"), periods=days, freq=kwargs.pop("freq", "B"))
    market = rng.normal(0.0003, 0.008, days)
    columns = {}
    for ticker in tickers:
        own = rng.normal(0.0, rng.uniform(0.005, 0.02), days)
        columns[ticker] = 100 * np.exp(np.cumsum(rng.uniform(0.3, 1.3) * market + own))
    return pd.DataFrame(columns, index=index)


def scramble_after(prices, cut: int, seed: int = 99, shock: float = 1.0):
    """A copy of the prices in which everything after row `cut` is replaced.

    A backtest that only uses the past must give the same answers up to `cut`
    on the original and on the scrambled copy. If it does not, it is peeking
    at the future. `shock` also multiplies every later price, so 0.4 is a
    crash the day after the cut-off and 2.5 a surge.
    """
    rng = np.random.default_rng(seed)
    changed = prices.copy()
    shape = changed.iloc[cut + 1:].shape
    changed.iloc[cut + 1:] = changed.iloc[cut + 1:].to_numpy() * shock * np.exp(rng.normal(0, 0.2, shape))
    return changed


def futures(prices, cut: int):
    """Two different futures after row `cut`: a crash and a surge.

    A rule that compares a price with an average flips under one or the
    other, so a peek ahead cannot slip through by luck.
    """
    return [scramble_after(prices, cut, seed=99, shock=0.4), scramble_after(prices, cut, seed=98, shock=2.5)]


@pytest.fixture
def spy() -> pd.Series:
    return random_walk(3000, seed=1).rename("SPY")
