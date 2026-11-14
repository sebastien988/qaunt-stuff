"""The formulas, checked against cases with a known answer."""

import math

import numpy as np
import pandas as pd
import pytest

from conftest import random_panel, random_walk

import industrial_prediction
import nvidia_learning
import quant_research as qr
import train_models


# ----------------------------------------------------------------- statistics helpers

@pytest.mark.parametrize("p, z", [(0.5, 0.0), (0.975, 1.959964), (0.05, -1.644854), (0.01, -2.326348)])
def test_normal_quantiles(p, z):
    assert qr.norm_ppf(p) == pytest.approx(z, abs=1e-6)
    assert qr.norm_cdf(z) == pytest.approx(p, abs=1e-6)


def test_chi_square_p_value():
    assert qr.chi2_1_pvalue(3.841459) == pytest.approx(0.05, abs=1e-6)
    assert qr.chi2_1_pvalue(0.0) == 1.0


def test_skew_and_kurtosis_of_a_normal_sample_are_near_zero():
    skew, kurt = qr.skew_kurt(np.random.default_rng(0).normal(size=200_000))
    assert abs(skew) < 0.03 and abs(kurt) < 0.06


def test_nelder_mead_finds_the_minimum_of_a_bowl():
    best, value = qr.nelder_mead(lambda x: (x[0] - 3) ** 2 + 2 * (x[1] + 1) ** 2 + 5, [0.0, 0.0])
    assert best == pytest.approx([3, -1], abs=1e-3)
    assert value == pytest.approx(5, abs=1e-6)


def test_ols_matches_numpy_and_newey_west_reduces_to_white_without_lags():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 0.2 + 1.5 * x + rng.normal(size=500) * (1 + np.abs(x))
    coef, se = qr.ols_newey_west(y, x, lags=0)
    slope, intercept = np.polyfit(x, y, 1)
    assert coef == pytest.approx([intercept, slope])

    X = np.column_stack([np.ones_like(x), x])
    u = y - X @ coef
    bread = np.linalg.inv(X.T @ X)
    white = np.sqrt(np.diag(bread @ (X.T * u ** 2) @ X @ bread))
    assert se == pytest.approx(white)


def test_newey_west_widens_errors_when_residuals_are_autocorrelated():
    rng = np.random.default_rng(2)
    x = np.ones(3000) + rng.normal(0, 1e-6, 3000)
    noise = np.zeros(3000)
    for t in range(1, 3000):
        noise[t] = 0.8 * noise[t - 1] + rng.normal()
    x = rng.normal(size=3000)
    _, plain = qr.ols_newey_west(noise, x, lags=0)
    _, hac = qr.ols_newey_west(noise, x, lags=20)
    assert hac[0] > 1.5 * plain[0]  # the intercept's error is badly understated without the correction


# ----------------------------------------------------------------- risk metrics

def test_drawdown():
    returns = pd.Series([0.10, -0.50, 1.00, 0.10])
    assert qr.drawdown(returns).tolist() == pytest.approx([0.0, -0.5, 0.0, 0.0])


def test_probabilistic_sharpe():
    rng = np.random.default_rng(3)
    centred = rng.normal(0, 0.01, 2000)
    centred -= centred.mean()
    assert qr.probabilistic_sharpe(centred) == pytest.approx(0.5)
    good = centred + 0.001
    assert qr.probabilistic_sharpe(good) > 0.99
    assert qr.probabilistic_sharpe(good[:200]) < qr.probabilistic_sharpe(good)  # less data, less certainty
    assert qr.probabilistic_sharpe(-good) < 0.01


def test_deflated_sharpe_is_stricter_than_probabilistic_sharpe():
    rng = np.random.default_rng(4)
    excess = rng.normal(0.0005, 0.01, 2500)
    assert qr.deflated_sharpe(excess, [0.01, 0.03, 0.05, 0.02, 0.04]) < qr.probabilistic_sharpe(excess)
    # One try, or tries that all agree, leave nothing to deflate.
    assert qr.deflated_sharpe(excess, [0.05]) == pytest.approx(qr.probabilistic_sharpe(excess))


def test_risk_table_on_a_known_series():
    index = pd.bdate_range("2020-01-01", periods=504)
    rng = np.random.default_rng(5)
    market = pd.Series(rng.normal(0.0004, 0.01, 504), index=index)
    cash = pd.Series(0.0001, index=index)
    doubled = cash + 2 * (market - cash) + 0.0002  # beta 2, alpha 0.0002 a day, no noise
    table = qr.risk_table(pd.DataFrame({qr.MARKET: market, "X": doubled}), cash)
    assert table.loc["X", "Beta"] == pytest.approx(2.0)
    assert table.loc["X", "Alpha"] == pytest.approx(0.0002 * 252)
    assert table.loc[qr.MARKET, "Beta"] == pytest.approx(1.0)
    assert table.loc[qr.MARKET, "Vol"] == pytest.approx(market.std() * math.sqrt(252))
    assert table.loc[qr.MARKET, "CAGR"] == pytest.approx((1 + market).prod() ** (252 / 504) - 1)
    assert table.loc[qr.MARKET, "CVaR95"] >= table.loc[qr.MARKET, "VaR95"] > 0


# ----------------------------------------------------------------- GARCH and VaR tests

def test_kupiec():
    breaches = np.zeros(1000, dtype=bool)
    breaches[:10] = True
    assert qr.kupiec(breaches, 0.01) == (10, 1000, pytest.approx(1.0))  # exactly the expected rate
    breaches[:40] = True
    assert qr.kupiec(breaches, 0.01)[2] < 0.001  # four times too many
    assert qr.kupiec(np.zeros(1000, dtype=bool), 0.01)[2] < 0.001  # none at all is also wrong


def test_garch_fit_recovers_the_parameters_of_simulated_data():
    rng = np.random.default_rng(6)
    omega, alpha, beta, n = 0.05, 0.10, 0.85, 6000
    e, h = np.empty(n), omega / (1 - alpha - beta)
    for t in range(n):
        e[t] = math.sqrt(h) * rng.normal()
        h = omega + alpha * e[t] ** 2 + beta * h
    fit = qr.fit_garch(e)
    assert fit["alpha"] == pytest.approx(alpha, abs=0.03)
    assert fit["beta"] == pytest.approx(beta, abs=0.05)
    assert fit["persistence"] < 1
    assert fit["omega"] == pytest.approx(e.var() * (1 - fit["persistence"]))  # variance targeting


# ----------------------------------------------------------------- portfolio construction

def random_covariance(n: int, seed: int) -> np.ndarray:
    A = np.random.default_rng(seed).normal(size=(n, n))
    return A @ A.T / n + 0.1 * np.eye(n)


def test_project_simplex():
    assert qr.project_simplex(np.array([0.2, 0.3, 0.5])) == pytest.approx([0.2, 0.3, 0.5])
    assert qr.project_simplex(np.array([5.0, 0.0, 0.0])) == pytest.approx([1, 0, 0])
    w = qr.project_simplex(np.random.default_rng(7).normal(size=12))
    assert w.sum() == pytest.approx(1) and (w >= 0).all()


def test_ledoit_wolf_shrinks_noise_but_keeps_real_structure():
    rng = np.random.default_rng(8)
    noise = rng.normal(size=(60, 20))  # true covariance is the identity: shrink almost fully
    _, shrink = qr.ledoit_wolf(noise)
    assert shrink > 0.8

    true = random_covariance(5, seed=9)
    sample = rng.multivariate_normal(np.zeros(5), true, size=20_000)
    estimate, shrink = qr.ledoit_wolf(sample)
    assert shrink < 0.05
    assert estimate == pytest.approx(true, abs=0.08)
    centred = sample - sample.mean(axis=0)
    assert np.trace(estimate) == pytest.approx(np.trace(centred.T @ centred / len(sample)))  # total variance kept


def test_minimum_variance_matches_the_closed_form():
    variances = np.array([0.01, 0.04, 0.09, 0.02])
    w = qr.min_variance(np.diag(variances))
    assert w == pytest.approx((1 / variances) / (1 / variances).sum(), abs=1e-6)

    Sigma = random_covariance(6, seed=10)
    w = qr.min_variance(Sigma)
    assert w.sum() == pytest.approx(1) and (w >= -1e-12).all()
    for other in np.random.default_rng(11).dirichlet(np.ones(6), size=200):
        assert w @ Sigma @ w <= other @ Sigma @ other + 1e-9


def test_risk_parity_equalises_risk_contributions():
    Sigma = random_covariance(8, seed=12)
    w = qr.risk_parity(Sigma)
    assert w.sum() == pytest.approx(1) and (w > 0).all()
    assert qr.risk_contributions(w, Sigma) == pytest.approx(np.full(8, 1 / 8), abs=1e-8)


def test_max_sharpe_on_uncorrelated_assets_is_mean_over_variance():
    variances = np.array([0.01, 0.04, 0.02])
    mu = np.array([0.02, 0.06, 0.01])
    w = qr.max_sharpe(np.diag(variances), mu)
    sharpe = lambda v: v @ mu / math.sqrt(v @ np.diag(variances) @ v)
    exact = (mu / variances) / (mu / variances).sum()
    assert sharpe(w) == pytest.approx(sharpe(exact), rel=2e-3)  # found on a grid of risk aversions
    # With nothing expected to beat cash it falls back to the minimum-variance portfolio.
    assert qr.max_sharpe(np.diag(variances), -mu) == pytest.approx(qr.min_variance(np.diag(variances)))


def test_walk_forward_equal_weight_matches_a_hand_calculation():
    prices = random_panel(list(qr.UNIVERSE), 330, seed=13)
    returns = prices.pct_change().iloc[1:]
    cash = pd.Series(0.0, index=returns.index)
    results, turnover, weights, _ = qr.walk_forward(returns, cash, cost_bps=0.0)

    first_rebalance = weights["Equal weight"].index[0]
    start = returns.index.get_loc(first_rebalance) + 1
    assert results.index[0] == returns.index[start]
    # Until the next rebalance, an equal-weight portfolio grows like the average of the assets' growth.
    second_rebalance = weights["Equal weight"].index[1]
    span = returns.loc[results.index[0]:second_rebalance]
    expected = (1 + span).prod().mean()
    assert (1 + results.loc[span.index, "Equal weight"]).prod() == pytest.approx(expected)
    assert turnover["Equal weight"] < turnover["Max Sharpe"]

    costly, *_ = qr.walk_forward(returns, cash, cost_bps=50.0)
    assert (1 + costly[list(qr.PORTFOLIOS)]).prod().lt((1 + results[list(qr.PORTFOLIOS)]).prod()).all()


# ----------------------------------------------------------------- quantile forecasts

def test_pinball_loss():
    y = np.array([1.0])
    forecast = np.array([[0.0, 1.0, 3.0, 0.0, 0.0]])  # q10, q50, q90 (+ two unused columns)
    # q10: under by 1 -> 0.1; q50: exact -> 0; q90: over by 2 -> 0.1 * 2
    assert nvidia_learning.pinball(y, forecast) == pytest.approx([(0.1 + 0 + 0.2) / 3])
    assert industrial_prediction.pinball(y, forecast[:, :3]) == pytest.approx([(0.1 + 0 + 0.2) / 3])


def test_pinball_loss_is_lowest_at_the_true_quantiles():
    y = np.random.default_rng(14).normal(size=20_000)
    loss = lambda q: nvidia_learning.pinball(y, np.tile(q, (len(y), 1))).mean()
    true = np.quantile(y, nvidia_learning.QUANTILES)
    assert loss(true) < loss(true + 0.1) and loss(true) < loss(true - 0.1) and loss(true) < loss(true * 1.2)


def test_skill_is_zero_for_the_baseline_and_positive_for_a_better_forecast():
    base = np.array([1.0, 2.0, 3.0])
    everything = np.ones(3, dtype=bool)
    assert nvidia_learning.skill(base, base, everything) == 0
    assert nvidia_learning.skill(base / 2, base, everything) == pytest.approx(0.5)
    assert math.isnan(nvidia_learning.skill(base, base, np.zeros(3, dtype=bool)))


def test_ridge_forecast_finds_a_real_signal():
    """If tomorrow's move really depends on a feature, the fit has to pick it up."""
    rng = np.random.default_rng(15)
    n, days = 1500, 5
    index = pd.bdate_range("2015-01-01", periods=n)
    signal = rng.normal(size=n)
    features = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}, index=index)
    data = nvidia_learning.Data(random_walk(n, seed=16, start="2015-01-01"), features,
                                pd.Series(0.01, index=index), days)
    data.y = 0.02 * signal + rng.normal(0, 0.01, n)
    data.y[-days:] = np.nan

    config = {"features": ["signal", "noise"], "lam": 0.1, "window": None, "half_life": None, "vol_scale": False}
    forecasts = nvidia_learning.walk_forward(data, config)
    base = nvidia_learning.walk_forward(data, nvidia_learning.BASELINE)
    known = ~np.isnan(data.y) & ~np.isnan(forecasts[:, 0])
    y = np.nan_to_num(data.y)
    skill = nvidia_learning.skill(nvidia_learning.pinball(y, forecasts), nvidia_learning.pinball(y, base), known)
    assert skill > 0.4
    inside = ((data.y >= forecasts[:, 0]) & (data.y <= forecasts[:, 2]))[known].mean()
    assert inside == pytest.approx(0.8, abs=0.05)  # the 8-in-10 range holds 8 in 10


# ----------------------------------------------------------------- train_models

def test_ridge_matches_the_normal_equations():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(400, 4))
    y = X @ np.array([0.5, -1.0, 0.0, 2.0]) + 0.3 + rng.normal(0, 0.1, 400)
    plain = train_models.fit_ridge(X, y, lam=0.0)
    assert plain[1:] == pytest.approx(np.linalg.lstsq(np.column_stack([np.ones(400), X]), y, rcond=None)[0][1:])
    assert train_models.predict_ridge(plain, X) == pytest.approx(plain[0] + X @ plain[1:])

    shrunk = train_models.fit_ridge(X, y, lam=5.0)
    assert np.abs(shrunk[1:]).sum() < np.abs(plain[1:]).sum()  # the penalty pulls weights towards zero
    assert shrunk[0] == pytest.approx(y.mean() - X.mean(axis=0) @ shrunk[1:])  # the intercept is not penalised


def test_cross_sectional_rank():
    values = np.array([[3.0, 1.0, 2.0, np.nan], [1.0, 1.0, 5.0, 9.0]])
    mask = np.array([[True, True, True, True], [True, True, True, False]])
    ranks = train_models.cs_rank(values, mask)
    assert ranks[0, :3] == pytest.approx([0.5, 1 / 3 - 0.5, 2 / 3 - 0.5])
    assert np.isnan(ranks[0, 3]) and np.isnan(ranks[1, 3])
    assert ranks[1, 0] == ranks[1, 1] < ranks[1, 2]  # ties share a rank


def test_spearman():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert train_models.spearman(a, a ** 3) == pytest.approx(1.0)
    assert train_models.spearman(a, -a) == pytest.approx(-1.0)
    assert industrial_prediction.spearman(a, np.exp(a)) == pytest.approx(1.0)


def test_evaluate_rewards_a_perfect_ranking_and_charges_for_turnover():
    rng = np.random.default_rng(18)
    fwd = rng.normal(0, 0.05, size=(12, 100))
    mask = np.ones_like(fwd, dtype=bool)
    free, deciles = train_models.evaluate(fwd, fwd, mask, cost_bps=0.0)
    assert free["ic"].to_numpy() == pytest.approx(1.0)
    top, bottom = np.sort(fwd[0])[-10:].mean(), np.sort(fwd[0])[:10].mean()
    assert free["ls"].iloc[0] == pytest.approx(top - bottom)
    assert free["long_excess"].iloc[0] == pytest.approx(top - fwd[0].mean())
    assert (np.diff(deciles) > 0).all()  # each decile beats the one below

    costly, _ = train_models.evaluate(fwd, fwd, mask, cost_bps=10.0)
    # Period 1 buys both books from nothing: 100% turnover a side, bought and later sold.
    assert free["ls"].iloc[0] - costly["ls"].iloc[0] == pytest.approx(0.0010 * 2 * 2)
    spent = (free["ls"] - costly["ls"]).iloc[1:]
    assert ((spent > 0) & (spent <= 0.0010 * 2 * 2 + 1e-12)).all()

    backwards, _ = train_models.evaluate(-fwd, fwd, mask, cost_bps=0.0)
    assert backwards["ic"].to_numpy() == pytest.approx(-1.0)
    assert (backwards["ls"] < 0).all()


def test_evaluate_skips_dates_with_too_few_stocks():
    fwd = np.random.default_rng(19).normal(size=(3, 40))
    mask = np.ones_like(fwd, dtype=bool)
    mask[1, 20:] = False
    result, _ = train_models.evaluate(fwd, fwd, mask, cost_bps=0.0)
    assert result["ic"].isna().tolist() == [False, True, False]


def test_neural_net_learns_a_simple_pattern():
    rng = np.random.default_rng(20)
    X = rng.uniform(-0.5, 0.5, size=(4000, 3))
    y = 0.8 * X[:, 0] - 0.4 * np.abs(X[:, 1])
    net = train_models.MLP(3, seed=0)
    before = np.mean((net.predict(X[3000:]) - y[3000:]) ** 2)
    net.fit(X[:3000], y[:3000], X[3000:], y[3000:], epochs=40)
    after = np.mean((net.predict(X[3000:]) - y[3000:]) ** 2)
    assert after < 0.1 * before
    assert after < 0.25 * y.var()

    again = train_models.MLP(3, seed=0).fit(X[:3000], y[:3000], X[3000:], y[3000:], epochs=40)
    np.testing.assert_array_equal(net.predict(X[:5]), again.predict(X[:5]))  # same seed, same model


def test_neural_net_validates_on_the_latest_dates_only():
    """Early stopping must be judged on the most recent training dates, never on test dates."""
    X3 = np.random.default_rng(21).normal(size=(40, 30, 3))
    y2 = X3[..., 0] * 0.3
    mask = np.ones((40, 30), dtype=bool)
    mask[30:] = False  # dates 30+ are not available for training
    seen = []

    class Spy(train_models.MLP):
        def fit(self, X, y, Xv, yv, **kwargs):
            seen.append((len(X), len(Xv)))
            return self

    original, train_models.MLP = train_models.MLP, Spy
    try:
        train_models.fit_mlp_ensemble(X3, y2, mask, seeds=(0,))
    finally:
        train_models.MLP = original
    cut = int(30 * 0.85)
    assert seen == [(cut * 30, (30 - cut) * 30)]
