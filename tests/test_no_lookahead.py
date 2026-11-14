"""No backtest may use prices from the future.

Each test runs a backtest twice: once on made-up prices, and once on a copy in
which every price after a cut-off day has been replaced. Whatever the backtest
decided or forecast up to the cut-off must be identical in both runs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import futures, random_panel, random_walk, scramble_after

import combined_algo
import crypto_scan
import industrial_prediction
import nvidia_learning
import nvidia_prediction
import quant_research
import trading_algo
import train_models

CUT = 2000


# ----------------------------------------------------------------- trading_algo

def test_crossover_position_uses_only_earlier_prices(spy):
    before = trading_algo.crossover_position(spy, 50, 200)
    after = trading_algo.crossover_position(scramble_after(spy, CUT), 50, 200)
    # The position held on day CUT+1 was decided at the close of day CUT.
    pd.testing.assert_series_equal(before.iloc[:CUT + 2], after.iloc[:CUT + 2])


def test_crossover_position_trades_the_day_after_the_signal():
    prices = pd.Series([1.0] * 5 + [2.0] * 5, index=pd.bdate_range("2020-01-01", periods=10))
    position = trading_algo.crossover_position(prices, 1, 3)
    # The 1-day average first exceeds the 3-day average at row 5; the trade is at row 6.
    assert position.iloc[5] == 0.0
    assert position.iloc[6] == 1.0


def test_walk_forward_picks_settings_without_seeing_the_year_traded(spy):
    position, chosen, _ = trading_algo.walk_forward(spy, 1, 5.0)
    for cut in (1500, 1760, 2020, 2280):  # each a few months into a different year
        year = spy.index[cut].year
        assert min(chosen) < year
        for changed in futures(spy, cut):
            position2, chosen2, _ = trading_algo.walk_forward(changed, 1, 5.0)
            # Settings for every year up to and including the cut-off year were trained on earlier years.
            assert {y: p for y, p in chosen.items() if y <= year} == {y: p for y, p in chosen2.items() if y <= year}
            last = spy.index[cut + 1]
            pd.testing.assert_series_equal(position[:last], position2[:last])


# ----------------------------------------------------------------- combined_algo

@pytest.fixture
def etfs() -> pd.DataFrame:
    return random_panel(list(combined_algo.UNIVERSE), 2600, seed=2)


def test_combined_weights_use_only_earlier_prices(etfs):
    changed = scramble_after(etfs, CUT)
    a, _ = combined_algo.build_strategies(etfs, etfs.pct_change().fillna(0.0))
    b, _ = combined_algo.build_strategies(changed, changed.pct_change().fillna(0.0))
    for name in a:
        pd.testing.assert_frame_equal(a[name].iloc[:CUT + 1], b[name].iloc[:CUT + 1], obj=name)


def test_dip_strategy_decides_on_the_day_without_tomorrows_price(etfs):
    """Cut the data on days the dip strategy buys: what follows must not change the purchase."""
    tradable = etfs.notna().cumsum() > combined_algo.HISTORY_NEEDED
    weights = combined_algo.dip_weights(etfs, tradable)
    bought = weights.gt(0) & weights.shift(1).eq(0)
    entry_rows = np.flatnonzero(bought.any(axis=1).to_numpy())
    entry_rows = entry_rows[entry_rows > 1000][:6]
    assert len(entry_rows) == 6
    for cut in entry_rows:
        for changed in futures(etfs, int(cut)):
            again = combined_algo.dip_weights(changed, tradable)
            pd.testing.assert_frame_equal(weights.iloc[:cut + 1], again.iloc[:cut + 1])


def test_combined_simulation_uses_only_earlier_prices(etfs):
    cash = pd.Series(0.0001, index=etfs.index)
    results = []
    for prices in (etfs, scramble_after(etfs, CUT)):
        returns = prices.pct_change().fillna(0.0)
        strategies, period = combined_algo.build_strategies(prices, returns)
        weights = sum(strategies.values()) / len(strategies)
        results.append(combined_algo.simulate(weights.loc[period], returns.loc[period], cash[period], 5.0))
    day = etfs.index[CUT]
    pd.testing.assert_frame_equal(results[0][:day], results[1][:day])


def test_monthly_strategies_only_change_after_a_month_end(etfs):
    strategies, _ = combined_algo.build_strategies(etfs, etfs.pct_change().fillna(0.0))
    months = etfs.index.to_period("M")
    month_end = np.append(months[:-1] != months[1:], False)
    for name in ("Trend", "Momentum", "Leaders"):
        changed = strategies[name].diff().abs().sum(axis=1).to_numpy() > 1e-12
        assert not (changed & ~month_end).any(), name


# ----------------------------------------------------------------- crypto_scan

@pytest.fixture
def coins() -> pd.DataFrame:
    names = [crypto_scan.MARKET] + [f"C{i}" for i in range(11)]
    return random_panel(names, 1500, seed=3, start="2018-06-01", freq="D")


def test_crypto_picks_use_only_earlier_prices(coins):
    a = crypto_scan.backtest(coins, top=3, cost_bps=20)
    for cut in (900, 1000, 1100):
        day = coins.index[cut]
        # A period that started 14 days before the cut-off ended on the cut-off day.
        done = day - pd.Timedelta(days=crypto_scan.HOLD_DAYS)
        assert len(a[:done]) > 20
        for changed in futures(coins, cut):
            pd.testing.assert_frame_equal(crypto_scan.scores(coins)[:day], crypto_scan.scores(changed)[:day])
            pd.testing.assert_series_equal(crypto_scan.market_rising(coins)[:day],
                                           crypto_scan.market_rising(changed)[:day])
            b = crypto_scan.backtest(changed, top=3, cost_bps=20)
            pd.testing.assert_frame_equal(a[:done], b[:done])
            pd.testing.assert_series_equal(a["in market"][:day], b["in market"][:day])


# ----------------------------------------------------------------- quant_research

def test_portfolio_walk_forward_uses_only_earlier_returns():
    prices = random_panel(list(quant_research.UNIVERSE), 700, seed=4)
    months = prices.index.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    cut = int(month_ends[month_ends > 500][0])  # a rebalance day: the weights are set at this close
    cash = pd.Series(0.0001, index=prices.index[1:])
    names = list(quant_research.PORTFOLIOS)
    day = prices.index[cut]

    results, _, weights, _ = quant_research.walk_forward(prices.pct_change().iloc[1:], cash, 5.0)
    assert day in weights["Min variance"].index
    for changed in futures(prices, cut):
        results2, _, weights2, _ = quant_research.walk_forward(changed.pct_change().iloc[1:], cash, 5.0)
        pd.testing.assert_frame_equal(results.loc[:day, names], results2.loc[:day, names])
        for name in names:
            pd.testing.assert_frame_equal(weights[name][:day], weights2[name][:day], obj=name)


def test_garch_variance_forecast_uses_only_earlier_returns():
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 400)
    changed = e.copy()
    changed[300:] *= 5
    a = quant_research.garch_filter(e, 0.05, 0.1, 0.85, 1.0)
    b = quant_research.garch_filter(changed, 0.05, 0.1, 0.85, 1.0)
    np.testing.assert_array_equal(a[:301], b[:301])  # h[300] is the forecast for e[300]
    assert a[301] != b[301]


# ----------------------------------------------------------------- nvidia_prediction

def nvidia_inputs(days: int = 1400):
    prices = random_walk(days, seed=6, start="2017-01-02")
    rng = np.random.default_rng(7)
    calendar = pd.date_range(prices.index[0] - pd.Timedelta(days=120), prices.index[-1], freq="D")
    tone = pd.Series(rng.normal(0, 1, len(calendar)), index=calendar)
    return prices, tone


def test_news_signal_only_sees_news_from_earlier_days():
    prices, tone = nvidia_inputs()
    table = nvidia_prediction.build_table(prices, tone, 20)
    day = prices.index[1000]
    changed = tone.copy()
    changed[changed.index >= day] += 50  # news published on the day itself, or later
    table2 = nvidia_prediction.build_table(prices, changed, 20)
    assert table.loc[day, "signal"] == table2.loc[day, "signal"]
    assert table.loc[prices.index[1001], "signal"] != table2.loc[prices.index[1001], "signal"]


def test_news_prediction_uses_only_outcomes_already_known():
    prices, tone = nvidia_inputs()
    days, row_day = 20, prices.index[1100]
    table = nvidia_prediction.build_table(prices, tone, days)
    row = table.index.get_loc(row_day)
    changed = scramble_after(prices, prices.index.get_loc(row_day))
    table2 = nvidia_prediction.build_table(changed, tone, days)

    a, b = nvidia_prediction.predict(table, row, days), nvidia_prediction.predict(table2, row, days)
    assert a is not None and a == b


# ----------------------------------------------------------------- nvidia_learning

def learning_data(prices: pd.Series, tone: pd.Series, days: int):
    rng = np.random.default_rng(8)
    market = pd.DataFrame({
        nvidia_learning.TICKER: prices,
        "QQQ": random_walk(len(prices), seed=9, start="2017-01-02"),
        "^VIX": 20 * np.exp(rng.normal(0, 0.2, len(prices))),
    })
    features, sigma = nvidia_learning.build_features(market, tone)
    return nvidia_learning.Data(prices, features, sigma, days)


CONFIG = {"features": ["news tone 30d", "return 21d", "VIX level"], "lam": 1.0, "window": 1000,
          "half_life": 500, "vol_scale": True}


def test_learning_forecasts_use_only_earlier_data():
    prices, tone = nvidia_inputs(1800)
    days, cut_day = 20, prices.index[1500]
    data = learning_data(prices, tone, days)
    data2 = learning_data(scramble_after(prices, 1500), tone, days)

    assert (data.index == data2.index).all()
    row = data.index.get_loc(cut_day)
    np.testing.assert_array_equal(data.X[:row + 1], data2.X[:row + 1])  # features known at each close
    for config in (CONFIG, nvidia_learning.BASELINE):
        a, b = nvidia_learning.walk_forward(data, config), nvidia_learning.walk_forward(data2, config)
        assert np.isfinite(a[row]).all()
        np.testing.assert_array_equal(a[:row + 1], b[:row + 1])


def test_learning_features_only_see_news_from_earlier_days():
    prices, tone = nvidia_inputs(1800)
    day = prices.index[1500]
    changed = tone.copy()
    changed[changed.index >= day] += 50  # news published on the day itself, or later
    data, data2 = learning_data(prices, tone, 20), learning_data(prices, changed, 20)
    row = data.index.get_loc(day)
    np.testing.assert_array_equal(data.X[:row + 1], data2.X[:row + 1])
    assert not np.array_equal(data.X[row + 1], data2.X[row + 1])


def test_learning_fit_ignores_outcomes_not_yet_known():
    prices, tone = nvidia_inputs(1800)
    data = learning_data(prices, tone, 20)
    as_of = 1200
    rows = np.array([as_of])
    expected = nvidia_learning.forecast_rows(data, CONFIG, as_of, rows)
    data.y = data.y.copy()
    data.y[as_of - data.days + 1:] = 123.0  # outcomes that end after `as_of`
    np.testing.assert_array_equal(nvidia_learning.forecast_rows(data, CONFIG, as_of, rows), expected)
    data.y[as_of - data.days] = 123.0  # the last outcome that IS known must matter
    assert not np.array_equal(nvidia_learning.forecast_rows(data, CONFIG, as_of, rows), expected)


# ----------------------------------------------------------------- industrial_prediction

def industrial_prices(days: int = 1700) -> pd.DataFrame:
    tickers = [industrial_prediction.SECTOR, industrial_prediction.MARKET] + list(industrial_prediction.COMPANIES)
    prices = random_panel(tickers, days, seed=10)
    rng = np.random.default_rng(11)
    prices[industrial_prediction.VIX] = 20 * np.exp(rng.normal(0, 0.2, days))
    return prices


def test_sector_forecasts_use_only_earlier_prices():
    prices, cut = industrial_prices(), 1500
    a = industrial_prediction.sector_study(prices, 20)
    b = industrial_prediction.sector_study(scramble_after(prices, cut), 20)
    row = a["index"].get_loc(prices.index[cut])
    assert row > industrial_prediction.MIN_HISTORY
    for method in industrial_prediction.METHODS:
        assert np.isfinite(a["forecasts"][method][row]).all()
        np.testing.assert_array_equal(a["forecasts"][method][:row + 1], b["forecasts"][method][:row + 1])


def test_sector_fit_ignores_outcomes_not_yet_known():
    rng = np.random.default_rng(12)
    n, end = 600, 400
    X, y, scale = rng.normal(size=(n, 3)), rng.normal(0, 0.05, n), rng.uniform(0.03, 0.06, n)
    rows = np.array([end + 19])
    expected, _, _ = industrial_prediction.sector_fit(X, y, scale, end, rows)
    y2 = y.copy()
    y2[end:] = 9.0
    result, _, _ = industrial_prediction.sector_fit(X, y2, scale, end, rows)
    for method in industrial_prediction.METHODS:
        np.testing.assert_array_equal(result[method], expected[method])


def test_company_scores_and_ranges_use_only_earlier_prices():
    prices, days = industrial_prices(), 20
    a = industrial_prediction.company_study(prices, days)
    first = prices.index.get_loc(a["start"])
    # Cut on a day the model is refitted, and a few days later: neither may see beyond the cut-off.
    for cut in (first + 10 * industrial_prediction.REFIT_EVERY, first + 10 * industrial_prediction.REFIT_EVERY + 9):
        for changed in futures(prices, cut):
            b = industrial_prediction.company_study(changed, days)
            assert np.isfinite(a["score"][cut]).all()
            np.testing.assert_array_equal(a["score"][:cut + 1], b["score"][:cut + 1])
            np.testing.assert_array_equal(a["ranges"][:cut + 1], b["ranges"][:cut + 1])
            # Each check is dated at the END of its holding period, so those dated up to the cut-off are complete.
            done = sum(d <= prices.index[cut] for d in a["dates"])
            assert done >= 5
            assert a["dates"][:done] == b["dates"][:done]
            np.testing.assert_allclose(a["ic"][:done], b["ic"][:done], rtol=0, atol=1e-12)
            np.testing.assert_allclose(a["spread"][:done], b["spread"][:done], rtol=0, atol=1e-12)


# ----------------------------------------------------------------- train_models

def stock_panel(days: int = 900, stocks: int = 60):
    px = random_panel([f"S{i}" for i in range(stocks)], days, seed=13)
    rng = np.random.default_rng(14)
    dollar_volume = pd.DataFrame(np.exp(rng.normal(16, 1, px.shape)), index=px.index, columns=px.columns)
    return px, dollar_volume, random_walk(days, seed=15).rename("SPY"), px.copy()


def test_stock_features_use_only_earlier_prices():
    px, dv, spy, raw = stock_panel()
    dates, _, X, y, fwd, valid, _ = train_models.build_features(px, dv, spy, raw)
    cut = px.index.get_loc(dates[20])  # a day the features are sampled on
    known = np.flatnonzero(dates <= px.index[cut])
    ended = np.flatnonzero(dates <= px.index[cut - train_models.HORIZON])
    assert len(known) == 21 and len(ended) == 20

    for shock, seed in ((0.4, 99), (2.5, 98)):
        changed = [scramble_after(frame, cut, seed, shock) for frame in (px, dv, spy, raw)]
        _, _, X2, y2, fwd2, valid2, _ = train_models.build_features(*changed)
        np.testing.assert_array_equal(X[known], X2[known])
        np.testing.assert_array_equal(valid[known], valid2[known])
        # Targets look 21 days ahead, so only those ending by the cut-off are unchanged.
        np.testing.assert_array_equal(fwd[ended], fwd2[ended])
        np.testing.assert_array_equal(y[ended], y2[ended])
        assert not np.array_equal(fwd[known[-1]], fwd2[known[-1]])


def test_stock_targets_do_not_overlap():
    px, dv, spy, raw = stock_panel()
    dates, *_ = train_models.build_features(px, dv, spy, raw)
    gaps = np.diff([px.index.get_loc(d) for d in dates])
    assert (gaps == train_models.HORIZON).all()
    assert math.isclose(train_models.HORIZON, 21)
