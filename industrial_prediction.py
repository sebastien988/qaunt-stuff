"""Predict the US industrial sector and its largest companies.

The sector is the Industrial Select Sector SPDR fund (XLI). The companies are
36 large US industrials with a long price history.

Three questions, each answered with a method fixed in advance (nothing here is
searched or tuned, unlike nvidia_learning.py) and checked walk-forward, so
every past forecast used only earlier data.

 1. Where is the sector heading over the next h trading days?
      Baseline     the 10/50/90% quantiles of all past h-day returns.
      Vol-scaled   z = ln(P[t+h]/P[t]) / (sigma[t] sqrt(h)), with sigma the EWMA
                   daily volatility (lambda = 0.94). Forecast = past quantiles
                   of z times today's sigma sqrt(h): wide when markets are
                   rough, narrow when calm.
      Ridge        z = a + b'x + e, beta = (Z'Z/n + lambda I)^-1 Z'z/n, on
                   momentum, trend, volatility and VIX features; forecast =
                   (a + b'x + quantiles of e) times sigma sqrt(h).
    Scored by pinball loss, L_q(y, f) = max(q (y - f), (q - 1)(y - f)), as
    skill = 1 - loss / loss of the baseline. The headline forecast comes from
    whichever method has the best walk-forward skill.

 2. How wide is each company's likely range?
      Past quantiles of z pooled over all companies, times that company's
      sigma sqrt(h). Checked by how often the price landed inside.

 3. Which companies should beat the sector?
      Pooled cross-sectional ridge regression. Each day every feature is
      z-scored across companies; the target is the company's h-day return
      relative to the average company. Checked with the rank information
      coefficient, IC = Spearman correlation(forecast score, realised relative
      return), and with the return of the top third minus the bottom third,
      both on non-overlapping periods.

Saves industrial_prediction.png and industrial_prediction.md.

Usage:
    python industrial_prediction.py
    python industrial_prediction.py --days 60
"""

import argparse
import math
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-missing days before a company listed
FOLDER = Path(__file__).resolve().parent
PRICE_CACHE = FOLDER / "industrial_price_cache.csv"

SECTOR, MARKET, VIX = "XLI", "SPY", "^VIX"
COMPANIES = {
    "GE": "GE Aerospace", "CAT": "Caterpillar", "RTX": "RTX", "UNP": "Union Pacific", "HON": "Honeywell",
    "BA": "Boeing", "DE": "Deere", "LMT": "Lockheed Martin", "ETN": "Eaton", "UPS": "UPS", "ADP": "ADP",
    "GD": "General Dynamics", "NOC": "Northrop Grumman", "MMM": "3M", "ITW": "Illinois Tool Works",
    "EMR": "Emerson", "CSX": "CSX", "WM": "Waste Management", "FDX": "FedEx", "NSC": "Norfolk Southern",
    "PH": "Parker-Hannifin", "TT": "Trane", "CTAS": "Cintas", "JCI": "Johnson Controls", "PCAR": "Paccar",
    "CMI": "Cummins", "URI": "United Rentals", "GWW": "Grainger", "ROK": "Rockwell Automation",
    "FAST": "Fastenal", "LHX": "L3Harris", "TDG": "TransDigm", "PWR": "Quanta Services", "AME": "Ametek",
    "RSG": "Republic Services", "CPRT": "Copart",
}
START = "2005-01-01"
QUANTILES = np.array([0.1, 0.5, 0.9])
EWMA_LAMBDA = 0.94
MIN_HISTORY = 1000  # trading days before the first walk-forward forecast
REFIT_EVERY = 21
SECTOR_RIDGE = 1.0  # features are standardised, so 1 is substantial shrinkage
STOCK_RIDGE = 0.01
POOL_STEP = 5  # use every 5th day when pooling company data (neighbouring days are near-duplicates)


# ----------------------------------------------------------------- data

def load_prices() -> pd.DataFrame:
    """Daily closes with dividends, downloaded at most once a day."""
    tickers = [SECTOR, MARKET, VIX] + list(COMPANIES)
    cached = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True) if PRICE_CACHE.exists() else None
    if (cached is not None and date.fromtimestamp(PRICE_CACHE.stat().st_mtime) == date.today()
            and set(tickers) <= set(cached.columns)):
        return cached[tickers]
    try:
        data = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
        data = data[tickers].dropna(subset=[SECTOR]).ffill()
        bad = [t for t in tickers if data[t].count() < MIN_HISTORY + 500 or pd.isna(data[t].iloc[-1])]
        if bad:
            raise ValueError(f"no usable prices for {', '.join(bad)}")
        data.to_csv(PRICE_CACHE)
        return data
    except Exception as error:
        if cached is None or not set(tickers) <= set(cached.columns):
            raise SystemExit(f"Could not download prices ({error}). Try again in a few minutes.")
        print(f"  WARNING: price download failed; using prices saved to {cached.index[-1]:%Y-%m-%d}.")
        return cached[tickers]


def ewma_vol(prices):
    """EWMA daily volatility: sigma2[t] = lambda sigma2[t-1] + (1 - lambda) r[t]^2."""
    return np.sqrt((np.log(prices).diff() ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False, min_periods=60).mean())


def pinball(y: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    diff = y[:, None] - forecasts
    return np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean(axis=1)


# ----------------------------------------------------------------- 1. sector

def sector_features(prices: pd.DataFrame) -> pd.DataFrame:
    p, log_ret = prices[SECTOR], np.log(prices[SECTOR]).diff()
    mom = lambda s, d: np.log(s / s.shift(d))
    return pd.DataFrame({
        "return 21d": mom(p, 21), "return 63d": mom(p, 63), "return 252d": mom(p, 252),
        "volatility 21d/126d": log_ret.rolling(21).std() / log_ret.rolling(126).std(),
        "vs 200d average": p / p.rolling(200).mean() - 1,
        "vs S&P 500 63d": mom(p, 63) - mom(prices[MARKET], 63),
        "VIX level": np.log(prices[VIX]), "VIX change 21d": mom(prices[VIX], 21),
    }).dropna()


METHODS = ["Baseline", "Vol-scaled", "Ridge"]


def sector_fit(X, y, scale, end, rows):
    """Quantile forecasts (log returns) of each method for `rows`, fitted on rows before `end`."""
    out = {}
    out["Baseline"] = np.tile(np.quantile(y[:end], QUANTILES), (len(rows), 1))
    z = y[:end] / scale[:end]
    s = scale[rows][:, None]
    out["Vol-scaled"] = np.quantile(z, QUANTILES) * s
    mean_x, std_x = X[:end].mean(axis=0), X[:end].std(axis=0) + 1e-12
    Z = (X[:end] - mean_x) / std_x
    beta = np.linalg.solve(Z.T @ Z / end + SECTOR_RIDGE * np.eye(X.shape[1]), Z.T @ (z - z.mean()) / end)
    errors = z - z.mean() - Z @ beta
    mu = z.mean() + ((X[rows] - mean_x) / std_x) @ beta
    out["Ridge"] = (mu[:, None] + np.quantile(errors, QUANTILES)) * s
    chance_up = {"Baseline": np.full(len(rows), (y[:end] > 0).mean()),
                 "Vol-scaled": np.full(len(rows), (z > 0).mean()),
                 "Ridge": (errors[None, :] > -mu[:, None]).mean(axis=1)}
    return out, chance_up, beta


def sector_study(prices: pd.DataFrame, days: int) -> dict:
    feats = sector_features(prices)
    idx, n = feats.index, len(feats)
    X = feats.to_numpy()
    p = prices[SECTOR].reindex(idx)
    y = np.log(p.shift(-days) / p).to_numpy()
    scale = (ewma_vol(prices[SECTOR]).reindex(idx) * math.sqrt(days)).to_numpy()

    forecasts = {m: np.full((n, 3), np.nan) for m in METHODS}
    for as_of in range(MIN_HISTORY, n, REFIT_EVERY):
        rows = np.arange(as_of, min(as_of + REFIT_EVERY, n))
        result, _, _ = sector_fit(X, y, scale, as_of - days + 1, rows)
        for m in METHODS:
            forecasts[m][rows] = result[m]

    known = (np.arange(n) >= MIN_HISTORY) & ~np.isnan(y)
    yk = np.nan_to_num(y)
    loss = {m: pinball(yk, forecasts[m]) for m in METHODS}
    half = np.flatnonzero(known)[known.sum() // 2]
    periods = {"all": known, "first half": known & (np.arange(n) < half), "second half": known & (np.arange(n) >= half)}
    skill = {m: {k: 1 - loss[m][mask].mean() / loss["Baseline"][mask].mean() for k, mask in periods.items()}
             for m in METHODS}
    coverage = {m: ((y >= forecasts[m][:, 0]) & (y <= forecasts[m][:, 2]))[known].mean() for m in METHODS}

    last = np.array([n - 1])
    live, chance_up, beta = sector_fit(X, y, scale, n - days, last)
    best = max(METHODS, key=lambda m: skill[m]["all"])
    return {
        "index": idx, "price": p.to_numpy(), "known": known, "loss": loss, "skill": skill, "coverage": coverage,
        "forecasts": forecasts, "live": {m: live[m][0] for m in METHODS},
        "chance_up": {m: float(chance_up[m][0]) for m in METHODS}, "best": best,
        "split": idx[half], "start": idx[MIN_HISTORY], "end": idx[known][-1],
        "weights": pd.Series(beta, index=feats.columns), "vol": scale[-1] / math.sqrt(days) * math.sqrt(252),
        "vol_median": np.nanmedian(scale) / math.sqrt(days) * math.sqrt(252),
    }


# ----------------------------------------------------------------- 2 and 3. companies

def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise each day across companies, clipped at +/-3."""
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0).clip(-3, 3)


def company_features(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    log_ret = np.log(p).diff()
    return {
        "momentum 12m (skip last month)": np.log(p.shift(21) / p.shift(252)),
        "momentum 6m": np.log(p / p.shift(126)),
        "last month's return": np.log(p / p.shift(21)),
        "last week's return": np.log(p / p.shift(5)),
        "volatility 63d": log_ret.rolling(63).std(),
        "below 252d high": p / p.rolling(252).max() - 1,
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def company_study(prices: pd.DataFrame, days: int) -> dict:
    p = prices[list(COMPANIES)]
    idx, n = p.index, len(p)
    names = list(company_features(p))
    F = np.stack([zscore_rows(f).to_numpy() for f in company_features(p).values()], axis=2)  # day x company x feature
    y = np.log(p.shift(-days) / p).to_numpy()
    simple = (p.shift(-days) / p - 1).to_numpy()
    relative = y - np.nanmean(y, axis=1, keepdims=True)
    target = relative / np.nanstd(y, axis=1, keepdims=True)
    scale = (ewma_vol(p) * math.sqrt(days)).to_numpy()
    z = y / scale
    usable = ~np.isnan(F).any(axis=2) & ~np.isnan(scale)
    first = int(np.argmax(usable.sum(axis=1) >= 20)) + MIN_HISTORY  # first walk-forward day

    def fit(end: int):
        """Ridge weights and pooled z quantiles from days before `end` (outcomes known)."""
        days_used = np.arange(0, end, POOL_STEP)
        ok = usable[days_used] & ~np.isnan(target[days_used])
        X, t = F[days_used][ok], target[days_used][ok]
        beta = np.linalg.solve(X.T @ X / len(t) + STOCK_RIDGE * np.eye(len(names)), X.T @ t / len(t))
        return beta, z[days_used][ok]

    score = np.full((n, len(COMPANIES)), np.nan)
    ranges = np.full((n, len(COMPANIES), 3), np.nan)
    for as_of in range(first, n, REFIT_EVERY):
        rows = slice(as_of, min(as_of + REFIT_EVERY, n))
        beta, pooled = fit(as_of - days + 1)
        score[rows] = np.where(usable[rows], F[rows] @ beta, np.nan)
        ranges[rows] = scale[rows][:, :, None] * np.quantile(pooled, QUANTILES)

    # Non-overlapping checks: one date every `days` trading days.
    ic, spread, dates = [], [], []
    for t in range(first, n - days, days):
        ok = ~np.isnan(score[t]) & ~np.isnan(relative[t])
        if ok.sum() < 15:
            continue
        s, third = score[t][ok], ok.sum() // 3
        order = np.argsort(s)
        ic.append(spearman(s, relative[t][ok]))
        spread.append(simple[t][ok][order[-third:]].mean() - simple[t][ok][order[:third]].mean())
        dates.append(idx[t + days])
    ic, spread = np.array(ic), np.array(spread)
    tstat = lambda v: v.mean() / v.std(ddof=1) * math.sqrt(len(v))
    have = ~np.isnan(ranges[:, :, 0]) & ~np.isnan(y)
    inside = ((y >= ranges[:, :, 0]) & (y <= ranges[:, :, 2]))[have].mean()

    # Today's forecasts.
    beta, pooled = fit(n - days)
    live_score = F[-1] @ beta
    dispersion = float(np.nanmedian(np.nanstd(y[n - days - 252:n - days], axis=1)))  # typical spread between companies
    tilt = live_score * dispersion  # expected log return relative to the average company
    table = pd.DataFrame(index=list(COMPANIES))
    table["Company"] = pd.Series(COMPANIES)
    table["Price"] = p.iloc[-1]
    table["Volatility"] = scale[-1] / math.sqrt(days) * math.sqrt(252)
    for q, column in zip(np.quantile(pooled, QUANTILES), ["Low", "Median", "High"]):
        table[column] = np.expm1(q * scale[-1] + tilt)
    table["Chance of rising"] = [(pooled * s + t > 0).mean() for s, t in zip(scale[-1], tilt)]
    table["Vs peers"] = np.expm1(tilt - np.nanmean(tilt))
    return {
        "table": table.sort_values("Vs peers", ascending=False), "weights": pd.Series(beta, index=names),
        "ic": ic, "ic_t": tstat(ic), "spread": spread, "spread_t": tstat(spread), "dates": dates,
        "coverage": inside, "start": idx[first], "periods": len(ic),
        "score": score, "ranges": ranges,  # the walk-forward forecasts themselves, for the tests
    }


# ----------------------------------------------------------------- chart

def save_chart(path: Path, sec: dict, com: dict, days: int) -> None:
    fig = plt.figure(figsize=(18, 13))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    best, live = sec["best"], sec["live"][sec["best"]]

    a = fig.add_subplot(grid[0, 0])
    n = len(sec["index"])
    shown = np.arange(n - 250, n)
    dates, price = sec["index"][shown], sec["price"][shown]
    made = shown - days
    earlier = sec["forecasts"][best]
    a.fill_between(dates, sec["price"][made] * np.exp(earlier[made, 0]), sec["price"][made] * np.exp(earlier[made, 2]),
                   color="gray", alpha=0.25, label=f"What this method forecast {days} days earlier (8-in-10 range)")
    a.plot(dates, price, color="black", label=f"Industrial sector ({SECTOR})")
    ahead = pd.bdate_range(dates[-1], periods=days + 1)
    grow = lambda move: price[-1] * np.exp(move * np.arange(days + 1) / days)
    a.fill_between(ahead, grow(live[0]), grow(live[2]), color="tab:blue", alpha=0.25,
                   label=f"Forecast, 8-in-10 range: {math.expm1(live[0]):+.1%} to {math.expm1(live[2]):+.1%}")
    a.plot(ahead, grow(live[1]), color="tab:blue", linestyle="--", label=f"Median: {math.expm1(live[1]):+.1%}")
    a.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    a.set_title(f"Industrial sector: next {days} trading days ({best} method, "
                f"chance of rising {sec['chance_up'][best]:.0%})")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(grid[1, 0])
    known = sec["known"]
    for method, color in [("Vol-scaled", "tab:blue"), ("Ridge", "tab:orange")]:
        gain = np.cumsum((sec["loss"]["Baseline"] - sec["loss"][method])[known])
        a.plot(sec["index"][known], gain, color=color,
               label=f"{method}: skill {sec['skill'][method]['all']:+.1%} "
                     f"(first half {sec['skill'][method]['first half']:+.1%}, "
                     f"second half {sec['skill'][method]['second half']:+.1%})")
    a.axhline(0, color="black", linewidth=0.8)
    a.set_title("Sector forecasts: cumulative advantage over the baseline (rising = forecasting better)")
    a.set_ylabel("Cumulative pinball loss saved")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.3)

    a = fig.add_subplot(grid[0, 1])
    table = com["table"].sort_values("Median")
    rows = np.arange(len(table))
    a.hlines(rows, table["Low"], table["High"], color="tab:blue", alpha=0.45, linewidth=5)
    a.scatter(table["Median"], rows, color="black", s=14, zorder=5)
    a.axvline(0, color="black", linewidth=0.8)
    a.set_yticks(rows, [f"{c} ({t})" for t, c in zip(table.index, table["Company"])], fontsize=7)
    a.set_ylim(-1, len(table))
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title(f"Each company: 8-in-10 range and median, next {days} trading days", fontsize=11)
    a.grid(alpha=0.3, axis="x")

    a = fig.add_subplot(grid[1, 1])
    a.plot(com["dates"], np.cumsum(com["spread"]), color="tab:green")
    a.axhline(0, color="black", linewidth=0.8)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_title(f"Stock picking check: top third minus bottom third, summed over {com['periods']} periods\n"
                f"mean IC {com['ic'].mean():+.3f} (t = {com['ic_t']:.1f}), mean spread "
                f"{com['spread'].mean():+.2%} per period (t = {com['spread_t']:.1f}); |t| < 2 is not significant",
                fontsize=10)
    a.grid(alpha=0.3)

    fig.suptitle(f"US industrial sector forecast, prices to {sec['index'][-1]:%d %B %Y}", fontsize=15)
    fig.text(0.5, 0.006, "Ranges come from past behaviour scaled by current volatility. Not financial advice.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the US industrial sector and its largest companies.")
    parser.add_argument("--days", type=int, default=20, help="trading days ahead to predict")
    parser.add_argument("--chart", default=str(FOLDER / "industrial_prediction.png"))
    parser.add_argument("--report", default=str(FOLDER / "industrial_prediction.md"))
    args = parser.parse_args()
    if not 1 <= args.days <= 126:
        raise SystemExit("--days must be between 1 and 126.")
    days = args.days

    print("Loading prices...")
    prices = load_prices()
    sec, com = sector_study(prices, days), company_study(prices, days)
    best, day, price = sec["best"], prices.index[-1], prices[SECTOR].iloc[-1]
    pct = lambda log_move: math.expm1(log_move)

    sector_rows = [
        f"| {m}{' (headline)' if m == best else ''} | {pct(sec['live'][m][1]):+.1%} | "
        f"{pct(sec['live'][m][0]):+.1%} to {pct(sec['live'][m][2]):+.1%} | {sec['chance_up'][m]:.0%} | "
        + ("-" if m == "Baseline" else f"{sec['skill'][m]['all']:+.1%}") + " | "
        + ("-" if m == "Baseline" else f"{sec['skill'][m]['first half']:+.1%} / {sec['skill'][m]['second half']:+.1%}")
        + f" | {sec['coverage'][m]:.0%} |" for m in METHODS]
    live = sec["live"][best]
    picking = ("statistically significant" if abs(com["ic_t"]) >= 2 and abs(com["spread_t"]) >= 2
               else "not statistically significant")
    ridge_helps = sec["skill"]["Ridge"]["all"] > sec["skill"]["Vol-scaled"]["all"]
    table = com["table"]
    company_rows = [
        f"| {r.Company} ({t}) | ${r.Price:,.2f} | {r.Median:+.1%} | {r.Low:+.1%} to {r.High:+.1%} | "
        f"{r._7:.0%} | {r._8:+.2%} | {r.Volatility:.0%} |" for t, r in zip(table.index, table.itertuples())]

    report = [
        "# US industrial sector forecast",
        "",
        f"- **Generated:** {date.today():%d %B %Y} by `industrial_prediction.py`. Not financial advice.",
        f"- **Prices to:** {day:%d %B %Y}. Sector = {SECTOR} at ${price:,.2f}. Horizon = {days} trading days.",
        "",
        "## Summary",
        "",
        f"- **Sector:** median {pct(live[1]):+.1%} (to about ${price * math.exp(live[1]):,.2f}), 8-in-10 range "
        f"{pct(live[0]):+.1%} to {pct(live[2]):+.1%} (${price * math.exp(live[0]):,.2f} to "
        f"${price * math.exp(live[2]):,.2f}), chance of rising {sec['chance_up'][best]:.0%}. From the {best} method.",
        f"- **Volatility now:** {sec['vol']:.0%} a year, against a typical {sec['vol_median']:.0%}. "
        "This sets the width of the range and is the part of the forecast that has a track record.",
        f"- **Direction:** the ridge model, which tries to predict direction from momentum, trend and VIX, "
        f"{'added to' if ridge_helps else 'did not improve on'} the volatility-only forecast in the walk-forward check.",
        f"- **Stock picking:** the ranking of companies against each other is **{picking}** "
        f"(mean IC {com['ic'].mean():+.3f}, t = {com['ic_t']:.1f}).",
        "",
        "## 1. Sector forecast",
        "",
        f"Walk-forward check from {sec['start']:%b %Y} to {sec['end']:%b %Y}; halves split at {sec['split']:%b %Y}. "
        "Skill is against the baseline; above 0 is better. \"Inside range\" should be 80%.",
        "",
        "| Method | Median | 8-in-10 range | Chance of rising | Skill | Skill, first / second half | Inside range |",
        "|---|---|---|---|---|---|---|",
        *sector_rows,
        "",
        "Ridge weights (per standard deviation of each feature, in units of volatility): "
        + ", ".join(f"{k} {v:+.3f}" for k, v in sec["weights"].items()) + ".",
        "",
        "## 2. Company forecasts",
        "",
        f"Sorted by forecast return against the average company (\"Vs peers\"). Ranges use each company's current volatility; in the "
        f"walk-forward check from {com['start']:%b %Y} the price landed inside the range {com['coverage']:.0%} "
        "of the time (target 80%).",
        "",
        "| Company | Price | Median | 8-in-10 range | Chance of rising | Vs peers | Volatility (a year) |",
        "|---|---|---|---|---|---|---|",
        *company_rows,
        "",
        "## 3. Does the company ranking work?",
        "",
        f"Checked on {com['periods']} non-overlapping {days}-day periods from {com['start']:%b %Y}.",
        "",
        "| Measure | Value | t-statistic |",
        "|---|---|---|",
        f"| Mean rank IC | {com['ic'].mean():+.3f} | {com['ic_t']:.2f} |",
        f"| Share of periods with IC above 0 | {(com['ic'] > 0).mean():.0%} | |",
        f"| Top third minus bottom third, per period | {com['spread'].mean():+.2%} | {com['spread_t']:.2f} |",
        "",
        f"The ranking is {picking}. "
        + ("" if picking == "statistically significant" else
           "Treat the \"Vs peers\" column as noise: it is small because the model found little to go on. ")
        + "No trading costs are included in the spread.",
        "",
        "Ridge weights (positive = companies high on this did better next period): "
        + ", ".join(f"{k} {v:+.3f}" for k, v in com["weights"].items()) + ".",
        "",
        "## Limits",
        "",
        "- The companies are today's large industrials. Ones that shrank or failed are missing, which flatters "
        "the past averages and so the medians above.",
        "- Medians mostly reflect that these shares rose on average in the past, not a view about the next "
        f"{days} days.",
        "- Each company's median is its volatility times the typical past risk-adjusted return of the group, so "
        "volatile shares get bigger medians by construction. Read the ranges, not the medians.",
        "- The model knows only prices and the VIX: no earnings, orders, tariffs, interest rates or news.",
        "- 8-in-10 means 2 in 10 outcomes fall outside the range.",
        "",
        "Chart: `industrial_prediction.png`.",
    ]
    Path(args.report).write_text("\n".join(report) + "\n")
    save_chart(Path(args.chart), sec, com, days)

    print("\n" + "\n".join(report[6:12]))
    print("\n" + "\n".join(report[16:16 + 3 + len(METHODS)]))
    print(f"\nTop and bottom 5 companies against their peers (ranking {picking}):")
    for t, r in list(zip(table.index, table.itertuples()))[:5] + list(zip(table.index, table.itertuples()))[-5:]:
        print(f"  {r.Company + ' (' + t + ')':<30} median {r.Median:+6.1%}   range {r.Low:+6.1%} to {r.High:+6.1%}"
              f"   vs peers {r._8:+.2%}")
    print(f"\nChart saved to {args.chart}\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
