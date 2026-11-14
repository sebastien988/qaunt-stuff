"""The backtest bookkeeping: returns, costs, cash and rebalancing, on cases small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from conftest import random_panel, random_walk, scramble_after

import combined_algo
import compare_chart
import crypto_scan
import nvidia_prediction
import trading_algo
import trained_algo


# ----------------------------------------------------------------- trading_algo

def test_strategy_returns_charge_for_every_trade():
    index = pd.bdate_range("2020-01-01", periods=5)
    market = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=index)
    position = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0], index=index)
    returns = trading_algo.strategy_returns(position, market, cost_bps=10)
    # In from day 1 (a trade), out on day 3 (a trade), back in on day 5 (a trade).
    assert returns.tolist() == pytest.approx([0.01 - 0.001, 0.02, -0.001, 0.0, 0.01 - 0.001])


def test_sharpe_and_summary():
    index = pd.bdate_range("2020-01-01", periods=504)
    returns = pd.Series(np.tile([0.02, -0.01], 252), index=index)
    assert trading_algo.sharpe(returns) == pytest.approx(returns.mean() / returns.std() * np.sqrt(252))
    assert trading_algo.sharpe(pd.Series(0.0, index=index)) == float("-inf")  # never traded

    summary = trading_algo.summarize(returns, pd.Series(1.0, index=index))
    total = (1.02 * 0.99) ** 252 - 1
    assert summary["Total return"] == pytest.approx(total)
    assert summary["Annual return"] == pytest.approx((1 + total) ** 0.5 - 1)  # two years
    assert summary["Max drawdown"] == pytest.approx(-0.01)
    assert summary["Time in market"] == 1.0 and summary["Trades"] == 0.0


def test_walk_forward_trades_each_year_with_the_pair_chosen_for_it(spy):
    position, chosen, next_pair = trading_algo.walk_forward(spy, train_years=3, cost_bps=5.0)
    assert sorted(chosen) == list(range(min(chosen), spy.index[-1].year + 1))
    for year, pair in chosen.items():
        expected = trading_algo.crossover_position(spy, *pair)
        in_year = position.index.year == year
        pd.testing.assert_series_equal(position[in_year], expected[position.index][in_year], check_names=False)
        assert pair[0] < pair[1]
    assert next_pair[0] < next_pair[1]


def test_best_pair_only_scores_the_window_it_is_given():
    index = pd.bdate_range("2019-01-01", "2021-12-31")
    good_early = pd.Series(np.where(index.year == 2019, 0.01, -0.01), index=index) + np.resize([0.001, -0.001], len(index))
    good_late = -good_early
    pairs = {(5, 50): good_early, (10, 100): good_late}
    assert trading_algo.best_pair(pairs, pd.Timestamp(2019, 1, 1), pd.Timestamp(2020, 1, 1)) == (5, 50)
    assert trading_algo.best_pair(pairs, pd.Timestamp(2020, 1, 1), pd.Timestamp(2022, 1, 1)) == (10, 100)


# ----------------------------------------------------------------- combined_algo

def two_assets(days: int = 6):
    index = pd.bdate_range("2020-01-01", periods=days)
    returns = pd.DataFrame({"A": [0.0, 0.10, -0.05, 0.02, 0.0, 0.01][:days],
                            "B": [0.0, -0.02, 0.03, 0.00, 0.01, 0.0][:days]}, index=index)
    return returns, pd.Series(0.0002, index=index)


def test_simulate_holds_yesterdays_weights_and_pays_cash_interest():
    returns, cash = two_assets()
    target = pd.DataFrame({"A": 0.5, "B": 0.0}, index=returns.index)
    result = combined_algo.simulate(target, returns, cash, cost_bps=0.0)
    assert result["return"].iloc[0] == pytest.approx(0.0002)  # nothing is held on the first day: all cash
    assert result["invested"].iloc[0] == 0.0
    # Day 2: half in A (+10%), half in cash.
    assert result["return"].iloc[1] == pytest.approx(0.5 * 0.10 + 0.5 * 0.0002)
    assert result["invested"].iloc[1:].tolist() == pytest.approx([0.5] * 5)


def test_simulate_charges_cost_on_what_is_actually_traded():
    returns, cash = two_assets()
    target = pd.DataFrame({"A": 0.5, "B": 0.5}, index=returns.index)
    result = combined_algo.simulate(target, returns, cash * 0, cost_bps=10.0)
    assert result["turnover"].iloc[0] == pytest.approx(1.0)  # buying the whole portfolio
    assert result["return"].iloc[0] == pytest.approx(-0.001)
    # Day 2: A +10%, B -2% -> weights drift to 0.55/1.04 and 0.49/1.04; rebalancing trades the gap.
    gross = 0.5 * 0.10 + 0.5 * -0.02
    drift_a, drift_b = 0.5 * 1.10 / (1 + gross), 0.5 * 0.98 / (1 + gross)
    assert result["turnover"].iloc[1] == pytest.approx(abs(0.5 - drift_a) + abs(0.5 - drift_b))
    assert result["return"].iloc[1] == pytest.approx(gross - result["turnover"].iloc[1] * 0.001)
    # Days when nothing moves need no trade.
    flat = combined_algo.simulate(target, returns * 0, cash * 0, cost_bps=10.0)
    assert flat["turnover"].iloc[1:].abs().max() < 1e-12


def test_simulate_with_leverage_pays_interest_on_what_is_borrowed():
    returns, cash = two_assets()
    target = pd.DataFrame({"A": 1.5, "B": 0.0}, index=returns.index)
    result = combined_algo.simulate(target, returns, cash, cost_bps=0.0)
    assert result["return"].iloc[1] == pytest.approx(1.5 * 0.10 - 0.5 * 0.0002)
    levered = trained_algo.leveraged_returns(target, returns, cash, cost_bps=0.0)
    spread = 0.5 * trained_algo.BORROW_SPREAD / trained_algo.TRADING_DAYS
    assert levered.iloc[1] == pytest.approx(result["return"].iloc[1] - spread)
    assert levered.iloc[0] == pytest.approx(result["return"].iloc[0])  # nothing borrowed yet


def test_weight_rules_never_exceed_full_investment():
    prices = random_panel(list(combined_algo.UNIVERSE), 1500, seed=31)
    prices.iloc[:400, 3] = np.nan  # one asset starts trading later
    returns = prices.pct_change().fillna(0.0)
    strategies, period = combined_algo.build_strategies(prices, returns)
    tradable = prices.notna().cumsum() > combined_algo.HISTORY_NEEDED
    for name, weights in strategies.items():
        assert (weights >= 0).all().all(), name
        assert weights.sum(axis=1).max() <= 1 + 1e-9, name
        assert (weights.where(~tradable, 0.0) == 0).all().all(), name  # nothing held before it has a history
    assert strategies["Leaders"].gt(0).sum(axis=1).max() <= combined_algo.LEADERS_HELD
    assert period[0] == prices.index[combined_algo.HISTORY_NEEDED]


def test_inverse_volatility_weights():
    prices = random_panel(["calm", "wild"], 300, seed=32)
    returns = prices.pct_change().fillna(0.0)
    returns["wild"] = returns["calm"] * 3
    weights = combined_algo.inverse_volatility(returns, pd.DataFrame(True, index=returns.index, columns=returns.columns))
    assert weights.iloc[-1].tolist() == pytest.approx([0.75, 0.25])
    assert weights.iloc[:combined_algo.VOL_WINDOW - 1].sum().sum() == 0  # no weights before there is a volatility


def test_rsi_stays_between_0_and_100_and_reads_the_direction():
    index = pd.bdate_range("2020-01-01", periods=60)
    rising = pd.DataFrame({"x": np.linspace(100, 160, 60)}, index=index)
    assert combined_algo.rsi(rising, 2)["x"].iloc[-1] == pytest.approx(100)
    assert combined_algo.rsi(rising.iloc[::-1].set_index(index), 2)["x"].iloc[-1] == pytest.approx(0)
    mixed = combined_algo.rsi(random_panel(["x"], 500, seed=33), 2).dropna()
    assert mixed["x"].between(0, 100).all()


def test_hold_through_month_keeps_the_month_end_weights():
    index = pd.bdate_range("2020-01-01", "2020-03-31")
    weights = pd.DataFrame({"A": np.arange(len(index), dtype=float)}, index=index)
    held = combined_algo.hold_through_month(weights)["A"]
    january_end = weights.loc["2020-01-31", "A"]
    assert held["2020-01-30"] == 0.0  # nothing before the first month end
    assert (held["2020-01-31":"2020-02-27"] == january_end).all()
    assert held["2020-02-28"] == weights.loc["2020-02-28", "A"]
    # The last day of the data is not treated as a month end: the month may not be over.
    assert held.iloc[-1] == weights.loc["2020-02-28", "A"]


def test_dip_strategy_buys_a_drop_in_an_uptrend_and_sells_the_bounce():
    index = pd.bdate_range("2019-01-01", periods=320)
    path = np.concatenate([np.linspace(100, 150, 310), [146, 142, 139], [143, 148, 152, 155, 158, 160, 161]])
    prices = pd.DataFrame({"A": path}, index=index)
    weights = combined_algo.dip_weights(prices, pd.DataFrame(True, index=index, columns=["A"]))["A"]
    assert weights.iloc[309] == 0.0  # steady rise: no dip
    assert weights.iloc[312] == combined_algo.DIP_POSITION  # three falls in a row, still above the 200-day average
    assert weights.iloc[-1] == 0.0  # sold once it bounced


# ----------------------------------------------------------------- trained_algo

def test_training_never_sees_the_test_period():
    prices = random_panel(list(combined_algo.UNIVERSE), 2600, seed=34)
    cash = pd.Series(0.0001, index=prices.index)
    cutoff = prices.index[1800]
    cut_row = 1799  # the last training day
    _, daily, mix, leverage, period = trained_algo.train(prices, cash, 5.0, str(cutoff.date()))
    _, daily2, mix2, leverage2, _ = trained_algo.train(scramble_after(prices, cut_row), cash, 5.0, str(cutoff.date()))
    assert (mix, leverage) == (mix2, leverage2)
    pd.testing.assert_series_equal(daily[:prices.index[cut_row]], daily2[:prices.index[cut_row]])
    assert sum(mix.values()) == pytest.approx(1.0)
    assert leverage in trained_algo.LEVERAGE_CHOICES

    in_training = period < cutoff
    market = prices[trained_algo.MARKET].pct_change().fillna(0.0)[period]
    assert daily[in_training].std() <= market[in_training].std()  # the rule it was chosen under


def test_training_needs_enough_data_on_both_sides():
    prices = random_panel(list(combined_algo.UNIVERSE), 1200, seed=35)
    cash = pd.Series(0.0, index=prices.index)
    with pytest.raises(SystemExit):
        trained_algo.train(prices, cash, 5.0, str(prices.index[-50].date()))


# ----------------------------------------------------------------- compare_chart

def test_crossover_returns_earn_interest_while_in_cash():
    index = pd.bdate_range("2020-01-01", periods=4)
    position = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)
    market = pd.Series([0.05, 0.01, -0.02, 0.03], index=index)
    cash = pd.Series(0.0002, index=index)
    returns = compare_chart.crossover_returns(position, market, cash, cost_bps=10)
    assert returns.tolist() == pytest.approx([0.0002, 0.01 - 0.001, -0.02, 0.0002 - 0.001])


# ----------------------------------------------------------------- crypto_scan

def flat_coins(days: int = 200) -> pd.DataFrame:
    """Coins that never move, starting long enough before the test for every coin to be rankable."""
    index = pd.date_range("2018-08-01", periods=days + 153, freq="D")
    names = [crypto_scan.MARKET, "UP", "DOWN", "FLAT"]
    return pd.DataFrame(100.0, index=index, columns=names)


def test_crypto_backtest_matches_a_hand_calculation():
    prices = flat_coins()
    start = prices.index.get_loc(pd.Timestamp(crypto_scan.TEST_START))
    step = np.arange(len(prices) - start)
    prices.iloc[start:, prices.columns.get_loc("UP")] = 100 * 1.01 ** step
    prices.iloc[start:, prices.columns.get_loc("DOWN")] = 100 * 0.99 ** step
    prices[crypto_scan.MARKET] = 100 * 1.001 ** np.arange(len(prices))  # always above its 50-day average

    result = crypto_scan.backtest(prices, top=1, cost_bps=20)
    assert result.index[0] == pd.Timestamp(crypto_scan.TEST_START)
    assert (result.index[1:] - result.index[:-1] == pd.Timedelta(days=crypto_scan.HOLD_DAYS)).all()
    assert result.index[-1] + pd.Timedelta(days=crypto_scan.HOLD_DAYS) <= prices.index[-1]
    assert result["in market"].all()

    up = 1.01 ** 14 - 1
    # The first pick is a tie at zero; after that UP has the best score and is simply held.
    assert result["Strategy"].iloc[2:].to_numpy() == pytest.approx(up)
    assert result["Bitcoin"].to_numpy() == pytest.approx(1.001 ** 14 - 1)
    everyone = (up + 0.99 ** 14 - 1 + 0.0 + 1.001 ** 14 - 1) / 4
    assert result["All coins equally"].to_numpy() == pytest.approx(everyone)

    free = crypto_scan.backtest(prices, top=1, cost_bps=0)
    paid = free["Strategy"] - result["Strategy"]
    assert paid.iloc[0] == pytest.approx(0.002)  # buying the first coin
    assert paid.iloc[2:].abs().max() < 1e-12  # holding costs nothing


def test_crypto_backtest_goes_to_cash_when_bitcoin_is_falling():
    prices = flat_coins()
    prices[crypto_scan.MARKET] = 100 * 0.999 ** np.arange(len(prices))
    result = crypto_scan.backtest(prices, top=2, cost_bps=20)
    assert not result["in market"].any()
    assert (result["Strategy"] == 0).all()  # no holdings, no trades, no costs


def test_crypto_scores_wait_for_enough_history():
    prices = flat_coins()
    prices.iloc[:100, prices.columns.get_loc("UP")] = np.nan  # listed 100 days late
    score = crypto_scan.scores(prices)["UP"]
    first = score.first_valid_index()
    assert prices["UP"][:first].count() == crypto_scan.HISTORY_NEEDED + 1


def test_crypto_summary():
    returns = pd.Series([0.10, -0.05, 0.02, 0.0])
    summary = crypto_scan.summarize(returns)
    growth = 1.10 * 0.95 * 1.02
    assert summary["Annual return"] == pytest.approx(growth ** (crypto_scan.PERIODS_PER_YEAR / 4) - 1)
    assert summary["Max drawdown"] == pytest.approx(-0.05)
    assert summary["Worst 2 weeks"] == -0.05
    assert summary["Winning periods"] == 0.5


# ----------------------------------------------------------------- nvidia_prediction

def test_news_prediction_matches_a_hand_calculation():
    """High-tone days were followed by +2%, the rest by -1%: a high-tone day must predict +2%."""
    days, n = 5, 1200
    index = pd.bdate_range("2017-01-02", periods=n)
    rng = np.random.default_rng(37)
    signal = np.resize([-1.0, 0.0, 1.0], n) + rng.uniform(-0.1, 0.1, n)  # three clear groups
    table = pd.DataFrame({"price": 100.0, "signal": signal, "outcome": np.where(signal > 0.5, 0.02, -0.01)}, index=index)
    row = int(np.flatnonzero(signal > 0.5)[-1])
    forecast = nvidia_prediction.predict(table, row, days)
    assert forecast["group"] == 2
    assert forecast["expected"] == pytest.approx(0.02, abs=5e-4)
    assert forecast["chance up"] > 0.98
    assert forecast["expected ignoring news"] == pytest.approx(0.0, abs=1e-3)
    assert forecast["similar days"] == pytest.approx((row - days + 1) / 3, abs=3)

    assert nvidia_prediction.predict(table, nvidia_prediction.MIN_HISTORY + days - 2, days) is None  # too little history


def test_news_check_scores_a_useful_signal_as_helpful_and_noise_as_not():
    rng = np.random.default_rng(36)
    n, days = 1600, 5
    index = pd.bdate_range("2017-01-02", periods=n)
    signal = rng.normal(size=n)
    useful = pd.DataFrame({"price": 100.0, "signal": signal,
                           "outcome": 0.03 * np.sign(signal) + rng.normal(0, 0.01, n)}, index=index)
    assert nvidia_prediction.score_check(nvidia_prediction.check_method(useful, days))["verdict"] == "helped"

    checked = nvidia_prediction.check_method(useful, days)
    assert checked.index[0] == index[nvidia_prediction.MIN_HISTORY + days - 1]
    assert set(checked["group"]) == {0, 1, 2}
