"""Quant research with standard textbook formulas, numpy only.

Four studies on twelve ETFs (stocks, bonds, credit, gold, commodities, property):

 1. Risk metrics per asset
      Sharpe        SR = mean(r - rf) / std(r - rf) * sqrt(252)
      Sortino       mean(r - rf) / downside deviation * sqrt(252)
      Max drawdown  min_t (V_t / max_{s<=t} V_s - 1); Calmar = CAGR / |MaxDD|
      VaR / CVaR    historical 5% quantile and the mean loss beyond it,
                    and Cornish-Fisher VaR, which corrects the normal
                    quantile for skew S and excess kurtosis K:
                    z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
      CAPM          r_i - rf = alpha + beta (r_m - rf) + e, by OLS, with
                    Newey-West (HAC) standard errors for alpha
      PSR           Probabilistic Sharpe Ratio (Bailey & Lopez de Prado):
                    P(true SR > 0) = Phi( SR sqrt(T-1) / sqrt(1 - S*SR + (K+2)/4 * SR^2) )

 2. GARCH(1,1) volatility on the S&P 500
      h_t = omega + alpha e_{t-1}^2 + beta h_{t-1}, fitted by maximum likelihood
      on the first half of the data only, then tested on the second half against
      EWMA (RiskMetrics, lambda = 0.94) and a 21-day rolling variance, using the
      QLIKE loss, and with a Kupiec test of its 99% Value at Risk.

 3. Portfolio construction, walk-forward
      Every month end, using only the past 252 days, with a Ledoit-Wolf shrunk
      covariance matrix, five long-only portfolios are built:
        equal weight, inverse volatility, minimum variance,
        risk parity (equal risk contributions w_i (Sigma w)_i),
        maximum Sharpe (Markowitz tangency, with the past year's mean returns).
      Trades pay a cost on turnover. The results get a Deflated Sharpe Ratio,
      which discounts the best Sharpe for the number of portfolios tried.

 4. The efficient frontier (in-sample, for illustration only).

Saves quant_research.png and quant_research.md.

Usage:
    python quant_research.py
    python quant_research.py --start 2010-01-01 --cost-bps 10
"""

import argparse
import math
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import yfinance as yf

FOLDER = Path(__file__).resolve().parent
TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649

UNIVERSE = {
    "SPY": "US large companies", "QQQ": "US technology", "IWM": "US small companies",
    "EFA": "Developed ex-US", "EEM": "Emerging markets", "VNQ": "US property",
    "TLT": "Long Treasuries", "IEF": "Medium Treasuries", "LQD": "Corporate bonds",
    "HYG": "High-yield bonds", "GLD": "Gold", "DBC": "Commodities",
}
MARKET = "SPY"
CASH_RATE_TICKER = "^IRX"
LOOKBACK = 252
VAR_LEVEL = 0.05


# ----------------------------------------------------------------- maths helpers

def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF by bisection (accurate to ~1e-12)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_1_pvalue(stat: float) -> float:
    """Upper-tail p-value of a chi-square with one degree of freedom."""
    return math.erfc(math.sqrt(max(stat, 0.0) / 2))


def skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Sample skewness and excess kurtosis."""
    z = (x - x.mean()) / x.std()
    return float((z ** 3).mean()), float((z ** 4).mean() - 3)


def nelder_mead(f, x0, step=0.5, iters=400, tol=1e-9):
    """Minimise f with the Nelder-Mead simplex method."""
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centre = np.mean(pts[:-1], axis=0)
        refl = centre + (centre - pts[-1])
        fr = f(refl)
        if fr < vals[0]:
            exp = centre + 2 * (centre - pts[-1])
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            contr = centre + 0.5 * (pts[-1] - centre)
            fc = f(contr)
            if fc < vals[-1]:
                pts[-1], vals[-1] = contr, fc
            else:  # shrink towards the best point
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    best = int(np.argmin(vals))
    return pts[best], vals[best]


def ols_newey_west(y: np.ndarray, x: np.ndarray, lags: int = 5):
    """OLS of y on [1, x]; returns coefficients and Newey-West standard errors."""
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    u = y - X @ coef
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)  # Bartlett kernel
        G = Xu[lag:].T @ Xu[:-lag]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return coef, np.sqrt(np.diag(cov))


# ----------------------------------------------------------------- risk metrics

def drawdown(returns: pd.Series) -> pd.Series:
    value = (1 + returns).cumprod()
    return value / value.cummax() - 1


def probabilistic_sharpe(excess: np.ndarray, benchmark_sr: float = 0.0) -> float:
    """PSR: probability that the true per-period Sharpe exceeds benchmark_sr."""
    sr = excess.mean() / excess.std(ddof=1)
    s, k = skew_kurt(excess)
    denom = math.sqrt(max(1 - s * sr + (k + 2) / 4 * sr ** 2, 1e-12))  # (k+2)/4 = (raw kurt - 1)/4
    return norm_cdf((sr - benchmark_sr) * math.sqrt(len(excess) - 1) / denom)


def deflated_sharpe(excess: np.ndarray, trial_srs: list[float]) -> float:
    """DSR: PSR against the Sharpe expected from the best of N tries by luck alone."""
    n = len(trial_srs)
    sd = float(np.std(trial_srs, ddof=1)) if n > 1 else 0.0
    expected_max = sd * ((1 - EULER_GAMMA) * norm_ppf(1 - 1 / n) + EULER_GAMMA * norm_ppf(1 - 1 / (n * math.e)))
    return probabilistic_sharpe(excess, expected_max)


def risk_table(returns: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    rows = {}
    z = norm_ppf(VAR_LEVEL)
    market_ex = (returns[MARKET] - cash).to_numpy()
    for name in returns:
        r = returns[name].dropna()
        rf = cash.loc[r.index]
        ex = (r - rf).to_numpy()
        years = len(r) / TRADING_DAYS
        cagr = (1 + r).prod() ** (1 / years) - 1
        downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
        mdd = drawdown(r).min()
        s, k = skew_kurt(r.to_numpy())
        q = np.quantile(r, VAR_LEVEL)
        z_cf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
        mx = pd.Series(market_ex, index=returns.index).loc[r.index].to_numpy()
        (a, b), (se_a, _) = ols_newey_west(ex, mx)
        rows[name] = {
            "CAGR": cagr, "Vol": r.std() * math.sqrt(TRADING_DAYS),
            "Sharpe": ex.mean() / ex.std(ddof=1) * math.sqrt(TRADING_DAYS),
            "Sortino": ex.mean() / downside * math.sqrt(TRADING_DAYS),
            "MaxDD": mdd, "Calmar": cagr / abs(mdd),
            "Skew": s, "ExKurt": k,
            "VaR95": -q, "CVaR95": -r[r <= q].mean(),
            "CF-VaR95": -(r.mean() + z_cf * r.std()),
            "Beta": b, "Alpha": a * TRADING_DAYS, "Alpha t": a / se_a,
            "PSR": probabilistic_sharpe(ex),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------- GARCH(1,1)

def garch_filter(e: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """Conditional variances h_t, each using only e up to t-1."""
    h = np.empty(len(e) + 1)
    h[0] = h0
    for t in range(len(e)):
        h[t + 1] = omega + alpha * e[t] ** 2 + beta * h[t]
    return h  # h[t] is the forecast for e[t]; h[-1] is for tomorrow


def fit_garch(e: np.ndarray) -> dict:
    """Gaussian MLE with variance targeting: omega = var * (1 - alpha - beta)."""
    var = e.var()
    sig = lambda v: 1 / (1 + math.exp(-v))

    def unpack(x):
        persistence = 0.999 * sig(x[0])
        alpha = persistence * sig(x[1])
        return var * (1 - persistence), alpha, persistence - alpha

    def nll(x):
        omega, alpha, beta = unpack(x)
        h = garch_filter(e, omega, alpha, beta, var)[:-1]
        return 0.5 * np.sum(np.log(h) + e ** 2 / h)

    x, value = nelder_mead(nll, [3.0, -2.0])
    omega, alpha, beta = unpack(x)
    return {"omega": omega, "alpha": alpha, "beta": beta, "persistence": alpha + beta,
            "long_run_var": var, "half_life": math.log(0.5) / math.log(alpha + beta), "loglik": -value}


def kupiec(exceed: np.ndarray, p: float) -> tuple[int, int, float]:
    """Kupiec proportion-of-failures test; returns (breaches, days, p-value)."""
    n, x = len(exceed), int(exceed.sum())
    if x in (0, n):
        lr = -2 * n * math.log(1 - p) if x == 0 else -2 * n * math.log(p)
    else:
        pi = x / n
        lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - (n - x) * math.log(1 - pi) - x * math.log(pi))
    return x, n, chi2_1_pvalue(lr)


def garch_study(r: pd.Series) -> dict:
    pct = r * 100  # percent returns keep the numbers well scaled
    e_all = (pct - pct.mean()).to_numpy()
    split = len(e_all) // 2
    train, test = e_all[:split], e_all[split:]

    fit = fit_garch(train)
    h_all = garch_filter(e_all, fit["omega"], fit["alpha"], fit["beta"], train.var())
    h_test = h_all[split:-1]

    ewma = np.empty(len(e_all) + 1)
    ewma[0] = train.var()
    for t in range(len(e_all)):
        ewma[t + 1] = 0.94 * ewma[t] + 0.06 * e_all[t] ** 2
    rolling = pd.Series(e_all ** 2).rolling(21).mean().shift(1).to_numpy()  # through yesterday

    qlike = lambda h: float(np.mean(np.log(h) + test ** 2 / h))
    losses = {"GARCH(1,1)": qlike(h_test), "EWMA 0.94": qlike(ewma[split:-1]),
              "21-day rolling": qlike(rolling[split:])}

    # 99% one-day VaR: normal quantile vs filtered historical simulation (residual quantile from training)
    z_norm = norm_ppf(0.01)
    z_fhs = float(np.quantile(train / np.sqrt(h_all[:split]), 0.01))
    var_tests = {
        "GARCH-normal": kupiec(test < z_norm * np.sqrt(h_test), 0.01),
        "GARCH-FHS": kupiec(test < z_fhs * np.sqrt(h_test), 0.01),
    }

    # Term structure: E[h_{t+k}] = V + (a+b)^(k-1) (h_{t+1} - V), refitted on all data for the live forecast
    live = fit_garch(e_all)
    h_live = garch_filter(e_all, live["omega"], live["alpha"], live["beta"], e_all.var())
    V, p = live["long_run_var"], live["persistence"]
    horizon = np.arange(1, TRADING_DAYS + 1)
    path = V + p ** (horizon - 1) * (h_live[-1] - V)
    term = {k: math.sqrt(path[:k].sum() * TRADING_DAYS / k) / 100 for k in (1, 5, 21, 63, 252)}

    return {
        "fit": fit, "live": live, "split_date": r.index[split], "losses": losses, "var_tests": var_tests,
        "z_fhs": z_fhs, "term": term,
        "cond_vol": pd.Series(np.sqrt(h_live[:-1] * TRADING_DAYS) / 100, index=r.index),
        "realised": (r.rolling(21).std() * math.sqrt(TRADING_DAYS)).shift(-20),  # next 21 days
    }


# ----------------------------------------------------------------- portfolio maths

def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance towards a scaled identity."""
    n, p = x.shape
    x = x - x.mean(axis=0)
    S = x.T @ x / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.sum((S - F) ** 2)
    b2 = sum(np.sum((np.outer(row, row) - S) ** 2) for row in x) / n ** 2
    shrink = min(b2, d2) / d2 if d2 > 0 else 1.0
    return shrink * F + (1 - shrink) * S, shrink


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0)


def mean_variance(Sigma: np.ndarray, mu: np.ndarray, risk_aversion: float, iters: int = 300) -> np.ndarray:
    """Long-only argmax mu'w - (lambda/2) w'Sigma w, by accelerated projected gradient (FISTA)."""
    n = len(mu)
    step = 1 / (risk_aversion * np.linalg.eigvalsh(Sigma)[-1] + 1e-12)
    w = y = np.full(n, 1 / n)
    t = 1.0
    for _ in range(iters):
        w_new = project_simplex(y + step * (mu - risk_aversion * Sigma @ y))
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        w, t = w_new, t_new
    return w


def min_variance(Sigma):
    return mean_variance(Sigma, np.zeros(len(Sigma)), 1.0, iters=500)


def max_sharpe(Sigma, mu):
    """Tangency portfolio: the frontier point with the highest mu / sigma."""
    if mu.max() <= 0:
        return min_variance(Sigma)  # no asset expected to beat cash
    best, best_sr = None, -np.inf
    for lam in np.geomspace(0.5, 500, 25):
        w = mean_variance(Sigma, mu, lam)
        sr = w @ mu / math.sqrt(w @ Sigma @ w)
        if sr > best_sr:
            best, best_sr = w, sr
    return best


def risk_parity(Sigma: np.ndarray, iters: int = 200) -> np.ndarray:
    """Equal risk contributions by cyclical coordinate descent (Griveau-Billion et al. 2013).

    Minimises 0.5 y'Sigma y - sum(b log y); each coordinate has a closed-form root.
    """
    n = len(Sigma)
    b = np.full(n, 1 / n)
    y = 1 / np.sqrt(np.diag(Sigma))
    for _ in range(iters):
        for i in range(n):
            c = Sigma[i] @ y - Sigma[i, i] * y[i]
            y[i] = (-c + math.sqrt(c * c + 4 * Sigma[i, i] * b[i])) / (2 * Sigma[i, i])
    return y / y.sum()


def risk_contributions(w, Sigma):
    rc = w * (Sigma @ w)
    return rc / rc.sum()


PORTFOLIOS = {
    "Equal weight": lambda S, mu: np.full(len(S), 1 / len(S)),
    "Inverse vol": lambda S, mu: (1 / np.sqrt(np.diag(S))) / (1 / np.sqrt(np.diag(S))).sum(),
    "Min variance": lambda S, mu: min_variance(S),
    "Risk parity": lambda S, mu: risk_parity(S),
    "Max Sharpe": max_sharpe,
}


def walk_forward(returns: pd.DataFrame, cash: pd.Series, cost_bps: float):
    """Monthly rebalanced, drifting-weight backtest. Weights use data up to the rebalance close only."""
    R = returns.to_numpy()
    rf = cash.to_numpy()
    months = returns.index.to_period("M")
    is_month_end = np.append(months[:-1] != months[1:], False)
    start = LOOKBACK
    n = R.shape[1]

    out = {name: np.zeros(len(R)) for name in PORTFOLIOS}
    turnover = {name: 0.0 for name in PORTFOLIOS}
    weights_log = {name: {} for name in PORTFOLIOS}
    held = {name: np.zeros(n) for name in PORTFOLIOS}
    shrinks = []

    for t in range(start, len(R)):
        # day t's return is earned on weights set at an earlier close
        for name in PORTFOLIOS:
            w = held[name]
            invested = w.sum()
            day = w @ R[t] + (1 - invested) * rf[t]
            out[name][t] += day
            if invested > 0:
                held[name] = w * (1 + R[t]) / (1 + day)
        if is_month_end[t]:
            window = R[t - LOOKBACK + 1:t + 1]
            Sigma, shrink = ledoit_wolf(window)
            shrinks.append(shrink)
            mu = (window - rf[t - LOOKBACK + 1:t + 1, None]).mean(axis=0)
            for name, build in PORTFOLIOS.items():
                target = build(Sigma, mu)
                trade = np.abs(target - held[name]).sum()
                turnover[name] += trade
                out[name][t] -= trade * cost_bps / 1e4  # cost paid at the rebalance close
                held[name] = target
                weights_log[name][returns.index[t]] = target

    first_trade = next(t for t in range(start, len(R)) if is_month_end[t]) + 1
    idx = returns.index[first_trade:]
    results = pd.DataFrame({k: v[first_trade:] for k, v in out.items()}, index=idx)
    results["SPY"] = returns[MARKET].loc[idx]
    results["60/40 SPY/IEF"] = 0.6 * returns["SPY"].loc[idx] + 0.4 * returns["IEF"].loc[idx]  # daily rebalanced
    years = len(idx) / TRADING_DAYS
    turnover = {k: v / years for k, v in turnover.items()}
    weights = {k: pd.DataFrame(v, index=list(returns.columns)).T for k, v in weights_log.items()}
    return results, turnover, weights, float(np.mean(shrinks))


# ----------------------------------------------------------------- data

def load_data(start: str) -> tuple[pd.DataFrame, pd.Series]:
    tickers = list(UNIVERSE) + [CASH_RATE_TICKER]
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    missing = [t for t in tickers if t not in data or data[t].dropna().empty]
    if missing:
        raise SystemExit(f"No price data returned for: {', '.join(missing)}")
    prices = data[list(UNIVERSE)].dropna()  # common history only, so every asset is always present
    cash = data[CASH_RATE_TICKER].reindex(prices.index).ffill().fillna(0.0) / 100 / TRADING_DAYS
    returns = prices.pct_change().iloc[1:]
    return returns, cash.loc[returns.index].clip(lower=0.0)


# ----------------------------------------------------------------- report

def fmt_table(df: pd.DataFrame, formats: dict) -> str:
    cols = list(df.columns)
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        cells = ["-" if pd.isna(row[c]) else formats.get(c, "{:.2f}").format(row[c]) for c in cols]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


PCT = "{:+.1%}"
RISK_FORMATS = {"CAGR": PCT, "Vol": "{:.1%}", "MaxDD": "{:.1%}", "VaR95": "{:.2%}", "CVaR95": "{:.2%}",
                "CF-VaR95": "{:.2%}", "Alpha": PCT, "PSR": "{:.0%}", "Turnover/yr": "{:.1f}x", "DSR": "{:.0%}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default="2007-06-01")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit of turnover, basis points")
    parser.add_argument("--chart", default=str(FOLDER / "quant_research.png"))
    parser.add_argument("--report", default=str(FOLDER / "quant_research.md"))
    args = parser.parse_args()

    print("Downloading prices...")
    returns, cash = load_data(args.start)
    first, last = returns.index[0], returns.index[-1]
    print(f"{len(returns)} trading days, {first:%Y-%m-%d} to {last:%Y-%m-%d}\n")

    # 1. Risk metrics
    assets = risk_table(returns, cash)
    assets.index = [f"{UNIVERSE[t]} ({t})" for t in assets.index]
    assets.loc[f"{UNIVERSE[MARKET]} ({MARKET})", ["Alpha", "Alpha t"]] = np.nan
    assets = assets.sort_values("Sharpe", ascending=False)
    print("1. Asset risk metrics (daily VaR/CVaR; alpha and beta vs SPY)")
    print(assets.round(3).to_string(), "\n")

    # 2. GARCH
    print("2. Fitting GARCH(1,1) to SPY...")
    g = garch_study(returns[MARKET])
    f = g["fit"]
    print(f"   train to {g['split_date']:%Y-%m-%d}: alpha={f['alpha']:.3f} beta={f['beta']:.3f} "
          f"persistence={f['persistence']:.4f} half-life={f['half_life']:.0f} days")
    for k, v in g["losses"].items():
        print(f"   QLIKE out-of-sample {k:<15} {v:.4f}")
    for k, (x, n, p) in g["var_tests"].items():
        print(f"   99% VaR {k:<13} breaches {x}/{n} ({x / n:.2%}, expected 1%) Kupiec p={p:.3f}")
    print("   Volatility forecast: " + ", ".join(f"{k}d {v:.1%}" for k, v in g["term"].items()), "\n")

    # 3. Walk-forward portfolios
    print("3. Walk-forward portfolios...")
    results, turnover, weights, avg_shrink = walk_forward(returns, cash, args.cost_bps)
    rf = cash.loc[results.index]
    port = risk_table(results.assign(**{MARKET: results[MARKET]}), rf)
    port.loc[MARKET, ["Alpha", "Alpha t"]] = np.nan  # SPY regressed on itself is meaningless
    port["Turnover/yr"] = pd.Series(turnover)
    trial_srs = [((results[k] - rf).mean() / (results[k] - rf).std()) for k in PORTFOLIOS]
    port["DSR"] = {k: deflated_sharpe((results[k] - rf).to_numpy(), trial_srs) for k in PORTFOLIOS}
    port_cols = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "CVaR95", "Beta", "Alpha", "Alpha t",
                 "PSR", "DSR", "Turnover/yr"]
    port = port[port_cols]
    print(port.round(3).to_string(), "\n")
    print(f"   Average Ledoit-Wolf shrinkage intensity: {avg_shrink:.2f}")

    # Current allocations (from the latest window)
    window = returns.to_numpy()[-LOOKBACK:]
    Sigma, _ = ledoit_wolf(window)
    mu = (window - cash.to_numpy()[-LOOKBACK:, None]).mean(axis=0)
    current = pd.DataFrame({k: build(Sigma, mu) for k, build in PORTFOLIOS.items()}, index=list(UNIVERSE))
    rp_rc = risk_contributions(current["Risk parity"].to_numpy(), Sigma)
    ew_rc = risk_contributions(current["Equal weight"].to_numpy(), Sigma)

    # 4. Efficient frontier, full sample (in-sample)
    Sig_full, _ = ledoit_wolf(returns.to_numpy())
    mu_full = returns.mean().to_numpy()
    frontier = []
    for lam in np.geomspace(0.3, 3000, 60):
        w = mean_variance(Sig_full, mu_full, lam, iters=600)
        frontier.append((math.sqrt(w @ Sig_full @ w * TRADING_DAYS), w @ mu_full * TRADING_DAYS))
    frontier = np.array(sorted(frontier))

    # ------------------------------------------------------------- chart
    fig, ax = plt.subplots(3, 2, figsize=(16, 16))
    colors = dict(zip(list(PORTFOLIOS) + ["SPY", "60/40 SPY/IEF"],
                      ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "black", "gray"]))

    a = ax[0, 0]
    for k in results:
        a.plot((1 + results[k]).cumprod(), label=f"{k} (SR {port.loc[k, 'Sharpe']:.2f})", color=colors[k],
               linewidth=2 if k in PORTFOLIOS else 1.2, linestyle="-" if k in PORTFOLIOS else "--")
    a.set_yscale("log")
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_yticks([0.5, 1, 2, 4, 8])
    a.set_title(f"Walk-forward portfolios: growth of \\$1 (monthly rebalance, {args.cost_bps:g} bp cost)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for k in results:
        a.plot(drawdown(results[k]), color=colors[k], linewidth=1.2 if k in PORTFOLIOS else 0.8,
               linestyle="-" if k in PORTFOLIOS else "--", label=k)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title("Drawdowns: V_t / max(V_s, s<=t) - 1")
    a.legend(fontsize=8, loc="lower left")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    vols = np.sqrt(np.diag(Sig_full) * TRADING_DAYS)
    a.plot(frontier[:, 0], frontier[:, 1], color="black", linewidth=2, label="Efficient frontier (long-only)")
    a.scatter(vols, mu_full * TRADING_DAYS, color="tab:blue")
    for t, x, y in zip(UNIVERSE, vols, mu_full * TRADING_DAYS):
        a.annotate(t, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    rf_full = cash.mean() * TRADING_DAYS
    sr_front = (frontier[:, 1] - rf_full) / frontier[:, 0]
    tx, ty = frontier[np.argmax(sr_front)]
    a.plot([0, tx * 1.6], [rf_full, rf_full + (ty - rf_full) * 1.6], color="tab:red", linestyle=":",
           label=f"Capital market line (max SR {sr_front.max():.2f})")
    a.set_xlabel("Annual volatility")
    a.set_ylabel("Annual mean return (arithmetic)")
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_xlim(0, vols.max() * 1.1)
    a.set_title("Markowitz efficient frontier, full sample (in-sample: hindsight!)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    recent = g["cond_vol"].loc[g["split_date"]:]
    a.plot(recent, color="tab:red", linewidth=1, label="GARCH(1,1) conditional vol")
    a.plot(g["realised"].loc[g["split_date"]:], color="gray", linewidth=0.8, alpha=0.8,
           label="Realised vol, next 21 days")
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title(f"SPY volatility: GARCH forecast vs what happened (test period from {g['split_date']:%Y})")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[2, 0]
    x = np.arange(len(UNIVERSE))
    bw = 0.16
    for i, k in enumerate(PORTFOLIOS):
        a.bar(x + (i - 2) * bw, current[k], bw, label=k, color=colors[k])
    a.set_xticks(x, list(UNIVERSE))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title(f"Current weights (past {LOOKBACK} days to {last:%d %b %Y})")
    a.legend(fontsize=8)
    a.grid(alpha=0.3, axis="y")

    a = ax[2, 1]
    a.bar(x - 0.2, ew_rc, 0.4, label="Equal weight", color=colors["Equal weight"])
    a.bar(x + 0.2, rp_rc, 0.4, label="Risk parity", color=colors["Risk parity"])
    a.axhline(1 / len(UNIVERSE), color="black", linestyle=":", linewidth=1)
    a.set_xticks(x, list(UNIVERSE))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title("Share of portfolio risk, RC_i = w_i (Sigma w)_i / w'Sigma w")
    a.legend(fontsize=8)
    a.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Quant research: {len(UNIVERSE)} ETFs, {first:%b %Y} to {last:%b %Y}", fontsize=15)
    fig.text(0.5, 0.005, "Past performance is not a forecast. Backtests use only data available at each "
             "rebalance; the efficient frontier panel uses hindsight.", ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    fig.savefig(args.chart, dpi=110)
    plt.close(fig)

    # ------------------------------------------------------------- report
    best = port.loc[list(PORTFOLIOS), "Sharpe"].idxmax()
    lw, ll = g["live"], g["losses"]
    best_vol_model = min(ll, key=ll.get)
    lines = [
        "# Quant research report",
        "",
        f"- **Generated:** {date.today():%d %B %Y} by `quant_research.py`. Not financial advice.",
        f"- **Data:** {len(UNIVERSE)} ETFs, daily, dividends included, {first:%d %b %Y} to {last:%d %b %Y} "
        f"({len(returns)} days). Cash = 13-week T-bill rate.",
        "",
        "## 1. Asset risk metrics",
        "",
        "Sharpe and Sortino use returns over cash. VaR/CVaR are one-day, 95%. Alpha is annualised "
        "CAPM alpha vs SPY with a Newey-West t-statistic (|t| > 2 is roughly significant). "
        "PSR = probability the true Sharpe is above zero, allowing for skew and fat tails.",
        "",
        fmt_table(assets, RISK_FORMATS),
        "",
        f"Fat tails: excess kurtosis ranges from {assets['ExKurt'].min():.1f} to {assets['ExKurt'].max():.1f} "
        "(a normal distribution has 0). CVaR95 / VaR95 averages "
        f"{(assets['CVaR95'] / assets['VaR95']).mean():.2f}; under a normal distribution it would be 1.25, so "
        "losses beyond the VaR are worse than a normal model says. Cornish-Fisher (`CF-VaR95`) is only "
        "trustworthy for moderate kurtosis (roughly below 8): for "
        f"{', '.join(assets.index[assets['ExKurt'] > 8]) or 'no asset here'} it is outside its valid range "
        "and should be ignored in favour of the historical figures.",
        "",
        "## 2. GARCH(1,1) volatility (SPY)",
        "",
        f"Fitted on {first:%Y} to {g['split_date']:%b %Y} only: alpha = {f['alpha']:.3f}, beta = {f['beta']:.3f}, "
        f"persistence alpha+beta = {f['persistence']:.4f}, shock half-life = {f['half_life']:.0f} trading days.",
        "",
        "Out-of-sample one-day variance forecasts (QLIKE loss = mean(log h + r^2/h), lower is better):",
        "",
        "| Model | QLIKE |", "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in ll.items()],
        "",
        f"Best: **{best_vol_model}**.",
        "",
        "99% one-day VaR backtest on the test period (Kupiec test; p < 0.05 means the breach rate is wrong):",
        "",
        "| VaR model | Breaches | Rate | Kupiec p |", "|---|---|---|---|",
        *[f"| {k} | {x}/{n} | {x / n:.2%} | {p:.3f} |" for k, (x, n, p) in g["var_tests"].items()],
        "",
        f"FHS uses the empirical 1% quantile of standardised residuals ({g['z_fhs']:.2f}) instead of the normal "
        f"{norm_ppf(0.01):.2f}, which accounts for fat tails.",
        "",
        f"Current forecast (refitted on all data; alpha {lw['alpha']:.3f}, beta {lw['beta']:.3f}), "
        "annualised average volatility over the next:",
        "",
        "| Horizon (days) | " + " | ".join(str(k) for k in g["term"]) + " |",
        "|---" * (len(g["term"]) + 1) + "|",
        "| Vol | " + " | ".join(f"{v:.1%}" for v in g["term"].values()) + " |",
        "",
        f"Long-run volatility: {math.sqrt(lw['long_run_var'] * TRADING_DAYS) / 100:.1%}. Forecasts "
        "mean-revert towards it: E[h(t+k)] = V + (alpha+beta)^(k-1) (h(t+1) - V).",
        "",
        "## 3. Walk-forward portfolio construction",
        "",
        f"Rebalanced every month end using only the past {LOOKBACK} days, Ledoit-Wolf shrunk covariance "
        f"(average shrinkage {avg_shrink:.2f}), long-only, fully invested, {args.cost_bps:g} bp cost per unit "
        "of turnover. DSR = Deflated Sharpe Ratio: the probability the Sharpe beats what the best of "
        f"{len(PORTFOLIOS)} tried portfolios would show by luck.",
        "",
        fmt_table(port, RISK_FORMATS),
        "",
        f"Best Sharpe among the constructed portfolios: **{best}** ({port.loc[best, 'Sharpe']:.2f}), against "
        f"{port.loc['SPY', 'Sharpe']:.2f} for SPY and {port.loc['60/40 SPY/IEF', 'Sharpe']:.2f} for 60/40. "
        + ("None of the constructed portfolios beat the simple 60/40 benchmark on Sharpe."
           if port.loc[list(PORTFOLIOS), 'Sharpe'].max() < port.loc['60/40 SPY/IEF', 'Sharpe']
           else "At least one constructed portfolio beat 60/40 on Sharpe."),
        "",
        f"The Sharpe ratios of the five span only {port.loc[list(PORTFOLIOS), 'Sharpe'].min():.2f} to "
        f"{port.loc[list(PORTFOLIOS), 'Sharpe'].max():.2f}; differences this small are well inside sampling "
        "error. Largest CAPM alpha t-statistic among them: "
        f"{port.loc[list(PORTFOLIOS), 'Alpha t'].abs().max():.2f} (below 2 is not significant). The question "
        "this setup tests is from DeMiguel, Garlappi & Uppal (2009): whether optimised portfolios beat naive "
        "ones out of sample once estimation error and turnover are counted. Max Sharpe needs expected returns, "
        f"which are noisy, and traded {port.loc['Max Sharpe', 'Turnover/yr']:.1f}x a year against "
        f"{port.loc['Equal weight', 'Turnover/yr']:.1f}x for equal weight.",
        "",
        "### Current allocations",
        "",
        fmt_table(current.assign(**{"Risk share (RP)": rp_rc, "Risk share (EW)": ew_rc}),
                  {c: "{:.1%}" for c in list(current.columns) + ["Risk share (RP)", "Risk share (EW)"]}),
        "",
        "With equal weights the risk is dominated by the volatile assets; risk parity equalises it.",
        "",
        "## Limits",
        "",
        "- One history (2007 onwards) with one big bond bull market and the 2022 bond crash; results depend on it.",
        "- Sharpe ratios, alphas and PSRs assume returns are stationary; regimes change.",
        "- Costs are a flat rate; taxes, spreads in stress and fund fees beyond the ETF price are ignored.",
        "- The efficient frontier chart uses full-sample means: it shows what was optimal in hindsight, "
        "not what was achievable.",
        "",
        "Chart: `quant_research.png`.",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"\nChart saved to {args.chart}\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
