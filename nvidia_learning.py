"""Nvidia prediction that keeps learning: every run searches for a better model.

The script remembers its work in nvidia_learning.json. On every run it:

  1. Downloads the latest prices and news tone.
  2. Re-scores the best model found so far (the "champion").
  3. Tries new candidate models (half are small changes to the champion, half
     are random) and keeps a candidate only if it beats the champion.
  4. Saves today's prediction, and scores the predictions saved by earlier
     runs against what the price actually did.
  5. Redraws nvidia_learning.png and rewrites nvidia_learning.md.

The model
    Target      z = ln(P[t+h] / P[t]) / (sigma[t] * sqrt(h)), the h-day log return
                in units of expected volatility (sigma = EWMA daily volatility,
                lambda = 0.94). Without volatility scaling, z is the log return.
    Mean        ridge regression: beta = (Z'WZ + lambda I)^-1 Z'W z, on
                standardised features, with optional exponential time weights W.
    Range       mean + the 10%, 50% and 90% quantiles of the model's past errors.
    A candidate is a choice of features, lambda, training window, weighting
    half-life and whether to scale by volatility.

The score
    Pinball (quantile) loss of the 10/50/90% forecasts:
        L_q(y, f) = max(q (y - f), (q - 1)(y - f))
    Skill = 1 - loss(model) / loss(baseline), where the baseline forecasts the
    unconditional past quantiles. Skill above 0 beats ignoring everything.

Honesty about "better"
    Candidates are chosen on a VALIDATION period, so the validation score can
    only go up from run to run. That alone proves little: try enough models
    and one will fit the past by luck. Two defences:
      - The validation period is cut into 4 blocks and a model is scored by
        its WORST block, so it has to work in every period, and a candidate
        must beat the champion by a clear margin to replace it.
      - The most recent 500 trading days at the first run, and every day
        after, are HELD OUT: never used to choose a model. The held-out skill
        is the real test, and it is not guaranteed to rise.
    All forecasts are walk-forward: each uses only earlier data.

Usage:
    python nvidia_learning.py               # 20 trading days ahead
    python nvidia_learning.py --days 5
    python nvidia_learning.py --seconds 120 # search harder this run (default 30 s, all CPU cores)
    python nvidia_learning.py --tries 200   # try a fixed number of models instead of a time budget
    python nvidia_learning.py --reset       # forget everything for this horizon
"""

import argparse
import json
import math
import multiprocessing
import os
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import yfinance as yf

from nvidia_prediction import TICKER, load_tone

FOLDER = Path(__file__).resolve().parent
STATE = FOLDER / "nvidia_learning.json"
PRICE_CACHE = FOLDER / "nvidia_price_cache.csv"
BLOCKS = 4  # validation blocks; a model is scored by its worst one
MARGIN = 0.001  # a candidate must beat the champion's score by this much

QUANTILES = np.array([0.1, 0.5, 0.9])
MIN_HISTORY = 750  # trading days of history before the first forecast
REFIT_EVERY = 21  # trading days between refits in the walk-forward check
HELD_OUT_DAYS = 500  # most recent days set aside at the first run
MIN_TRAIN = 400
EWMA_LAMBDA = 0.94

WINDOWS = [None, 750, 1000, 1500]  # training rows; None = all history
HALF_LIVES = [None, 250, 500, 1000]  # time-weighting half-life in days; None = equal weights


# ----------------------------------------------------------------- data

def rsi(prices: pd.Series, days: int) -> pd.Series:
    change = prices.diff()
    gains = change.clip(lower=0).ewm(alpha=1 / days, adjust=False).mean()
    losses = (-change.clip(upper=0)).ewm(alpha=1 / days, adjust=False).mean()
    return 100 - 100 / (1 + gains / losses)


def load_market() -> pd.DataFrame:
    """Daily closes of Nvidia, the Nasdaq 100 fund and the VIX, downloaded at most once a day."""
    tickers = [TICKER, "QQQ", "^VIX"]
    cached = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True) if PRICE_CACHE.exists() else None
    if cached is not None and date.fromtimestamp(PRICE_CACHE.stat().st_mtime) == date.today():
        return cached
    try:
        data = yf.download(tickers, start="2016-06-01", auto_adjust=True, progress=False)["Close"]
        data = data[tickers].dropna(subset=[TICKER]).ffill()
        if len(data) < 1000 or data.iloc[-1].isna().any():
            raise ValueError("incomplete download")
        data.to_csv(PRICE_CACHE)
        return data
    except Exception as error:
        if cached is None:
            raise SystemExit(f"Could not download prices ({error}). Try again in a few minutes.")
        print(f"  WARNING: price download failed; using prices saved to {cached.index[-1]:%Y-%m-%d}.")
        return cached


def build_features(market: pd.DataFrame, tone: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Features known at each close, and the EWMA daily volatility."""
    prices, qqq, vix = market[TICKER], market["QQQ"], market["^VIX"]

    # Shift by a day so each trading day only sees news published before it.
    tone = tone.reindex(pd.date_range(tone.index[0], max(tone.index[-1], prices.index[-1]))).shift(1)
    tone_avg = lambda d: tone.rolling(d, min_periods=d // 2).mean().reindex(prices.index)

    log_ret = np.log(prices).diff()
    sigma = np.sqrt((log_ret ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean())
    mom = lambda p, d: np.log(p / p.shift(d))

    features = pd.DataFrame({
        "news tone 7d": tone_avg(7),
        "news tone 30d": tone_avg(30),
        "news tone 90d": tone_avg(90),
        "news tone change": tone_avg(7) - tone_avg(90),
        "return 5d": mom(prices, 5),
        "return 21d": mom(prices, 21),
        "return 63d": mom(prices, 63),
        "return 126d": mom(prices, 126),
        "return 252d": mom(prices, 252),
        "volatility 21d": np.log(log_ret.rolling(21).std()),
        "volatility 21d/126d": log_ret.rolling(21).std() / log_ret.rolling(126).std(),
        "RSI 14d": rsi(prices, 14),
        "vs 200d average": prices / prices.rolling(200).mean() - 1,
        "below 252d high": prices / prices.rolling(252).max() - 1,
        "Nasdaq return 21d": mom(qqq, 21),
        "vs Nasdaq 63d": mom(prices, 63) - mom(qqq, 63),
        "VIX level": np.log(vix),
        "VIX change 21d": mom(vix, 21),
    }).dropna()
    return features, sigma.reindex(features.index)


# ----------------------------------------------------------------- model

class Data:
    def __init__(self, prices: pd.Series, features: pd.DataFrame, sigma: pd.Series, days: int):
        self.index = features.index
        self.names = list(features.columns)
        self.X = features.to_numpy()
        self.price = prices.reindex(self.index).to_numpy()
        self.y = np.log(prices.shift(-days) / prices).reindex(self.index).to_numpy()  # nan for the last days
        self.vol_scale = sigma.to_numpy() * math.sqrt(days)
        self.days = days
        self.n = len(self.index)


BASELINE = {"features": [], "lam": 1.0, "window": None, "half_life": None, "vol_scale": False}


def forecast_rows(data: Data, cfg: dict, as_of: int, rows: np.ndarray) -> np.ndarray | None:
    """Fit on outcomes known at row `as_of`, forecast `rows`.

    Returns columns q10, q50, q90 (log returns), mean (simple return), chance of rising.
    """
    end = as_of - data.days + 1  # rows before `end` have an outcome known by `as_of`
    start = max(0, end - cfg["window"]) if cfg["window"] else 0
    if end - start < MIN_TRAIN:
        return None
    cols = [data.names.index(f) for f in cfg["features"]]
    scale = data.vol_scale if cfg["vol_scale"] else np.ones(data.n)
    X, z = data.X[start:end][:, cols], data.y[start:end] / scale[start:end]

    age = np.arange(end - start)[::-1]
    w = 0.5 ** (age / cfg["half_life"]) if cfg["half_life"] else np.ones(end - start)
    w = w / w.sum()
    mean_x = w @ X
    std_x = np.sqrt(w @ (X - mean_x) ** 2) + 1e-12
    Z = (X - mean_x) / std_x
    mean_z = w @ z
    beta = np.linalg.solve(Z.T @ (Z * w[:, None]) + cfg["lam"] * np.eye(len(cols)), Z.T @ (w * (z - mean_z)))

    errors = z - (mean_z + Z @ beta)
    mu = mean_z + ((data.X[rows][:, cols] - mean_x) / std_x) @ beta
    s = scale[rows][:, None]
    out = np.empty((len(rows), 5))
    out[:, :3] = (mu[:, None] + np.quantile(errors, QUANTILES)) * s
    out[:, 3] = np.expm1((mu[:, None] + errors[None, :]) * s).mean(axis=1)
    out[:, 4] = (errors[None, :] > -mu[:, None]).mean(axis=1)
    return out


def walk_forward(data: Data, cfg: dict) -> np.ndarray:
    """Forecast for every row from MIN_HISTORY on, refitting every REFIT_EVERY days on earlier data only."""
    out = np.full((data.n, 5), np.nan)
    for as_of in range(MIN_HISTORY, data.n, REFIT_EVERY):
        rows = np.arange(as_of, min(as_of + REFIT_EVERY, data.n))
        result = forecast_rows(data, cfg, as_of, rows)
        if result is not None:
            out[rows] = result
    return out


def pinball(y: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    """Pinball loss per row, averaged over the quantiles."""
    diff = y[:, None] - forecasts[:, :3]
    return np.maximum(QUANTILES * diff, (QUANTILES - 1) * diff).mean(axis=1)


def skill(loss: np.ndarray, base_loss: np.ndarray, mask: np.ndarray) -> float:
    return 1 - loss[mask].mean() / base_loss[mask].mean() if mask.any() else float("nan")


# ----------------------------------------------------------------- search

_JOB: dict = {}  # shared with worker processes (inherited through fork)


def evaluate(cfg: dict) -> tuple[float, np.ndarray]:
    """Worst skill across the validation blocks, and the forecasts."""
    job = _JOB
    forecasts = walk_forward(job["data"], cfg)
    if np.isnan(forecasts[job["known"]]).any():
        return -np.inf, forecasts
    model_loss = pinball(job["y"], forecasts)
    return min(skill(model_loss, job["base_loss"], block) for block in job["blocks"]), forecasts


def score_only(cfg: dict) -> float:
    return evaluate(cfg)[0]


def search(rng, champion: dict, best: float, names: list[str], seconds: float, tries: int | None):
    """Try candidates until the time budget (or fixed count) is used; keep any that beat the champion.

    Candidates are scored in parallel, one batch at a time; each batch is half small changes to the
    current champion and half random models.
    """
    workers = os.cpu_count() or 1
    batch_size = workers * 2
    tried = improved = 0
    stop = time.monotonic() + seconds
    try:
        pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    except (ValueError, OSError):
        pool = None
    try:
        while (tries is None and time.monotonic() < stop) or (tries is not None and tried < tries):
            size = batch_size if tries is None else min(batch_size, tries - tried)
            batch = [mutate(rng, champion, names) if i % 2 == 0 else random_config(rng, names)
                     for i in range(size)]
            scores = pool.map(score_only, batch) if pool else [score_only(c) for c in batch]
            tried += size
            top = int(np.argmax(scores))
            if scores[top] > best + MARGIN:
                champion, best, improved = batch[top], scores[top], improved + 1
    finally:
        if pool:
            pool.close()
            pool.join()
    return champion, best, tried, improved, workers


def random_config(rng, names: list[str]) -> dict:
    count = int(rng.integers(1, 6))
    return {
        "features": sorted(rng.choice(names, size=count, replace=False).tolist()),
        "lam": float(10 ** rng.uniform(-1, 2)),  # features are standardised, so 1 is real shrinkage
        "window": WINDOWS[rng.integers(len(WINDOWS))],
        "half_life": HALF_LIVES[rng.integers(len(HALF_LIVES))],
        "vol_scale": bool(rng.integers(2)),
    }


def mutate(rng, cfg: dict, names: list[str]) -> dict:
    new = {**cfg, "features": list(cfg["features"])}
    for _ in range(int(rng.integers(1, 3))):
        move = rng.integers(5)
        if move == 0:  # add or drop a feature
            name = names[rng.integers(len(names))]
            if name in new["features"]:
                if len(new["features"]) > 1:
                    new["features"].remove(name)
            else:
                new["features"].append(name)
        elif move == 1:
            new["lam"] = float(np.clip(new["lam"] * 10 ** rng.normal(0, 0.4), 0.1, 1e3))
        elif move == 2:
            new["window"] = WINDOWS[rng.integers(len(WINDOWS))]
        elif move == 3:
            new["half_life"] = HALF_LIVES[rng.integers(len(HALF_LIVES))]
        else:
            new["vol_scale"] = not new["vol_scale"]
    new["features"] = sorted(new["features"])
    return new


def describe(cfg: dict) -> str:
    return (f"{len(cfg['features'])} features, ridge lambda {cfg['lam']:.3g}, "
            f"{'all history' if not cfg['window'] else 'last ' + str(cfg['window']) + ' days'}, "
            f"{'equal weights' if not cfg['half_life'] else 'half-life ' + str(cfg['half_life']) + ' days'}, "
            f"{'volatility-scaled' if cfg['vol_scale'] else 'not volatility-scaled'}")


# ----------------------------------------------------------------- chart

def save_chart(path: Path, data: Data, memory: dict, champ: np.ndarray, base_loss: np.ndarray, loss: np.ndarray,
               masks: dict, live: np.ndarray, weights: pd.Series, scored: pd.DataFrame) -> None:
    days, runs = data.days, pd.DataFrame(memory["runs"])
    fig, ax = plt.subplots(2, 2, figsize=(17, 11), gridspec_kw={"height_ratios": [3, 2]})

    # Price, what the model said beforehand, and the new forecast.
    a = ax[0, 0]
    shown = slice(data.n - 250, data.n)
    dates, price = data.index[shown], data.price[shown]
    made = np.arange(data.n - 250, data.n) - days  # the forecast for each day was made `days` earlier
    a.fill_between(dates, data.price[made] * np.exp(champ[made, 0]), data.price[made] * np.exp(champ[made, 2]),
                   color="gray", alpha=0.25, label=f"What the model forecast {days} days earlier (8-in-10 range)")
    a.plot(dates, price, color="black", label=f"{TICKER} price")
    ahead = pd.bdate_range(dates[-1], periods=days + 1)
    grow = lambda log_move: price[-1] * np.exp(log_move * np.arange(days + 1) / days)
    a.fill_between(ahead, grow(live[0]), grow(live[2]), color="tab:blue", alpha=0.25,
                   label=f"New forecast, 8-in-10 range: {math.expm1(live[0]):+.0%} to {math.expm1(live[2]):+.0%}")
    a.plot(ahead, grow(live[1]), color="tab:blue", linestyle="--", label=f"Median: {math.expm1(live[1]):+.1%}")
    if len(scored):
        hit = scored["inside"]
        a.errorbar(scored["target"], scored["q50"], yerr=[scored["q50"] - scored["q10"], scored["q90"] - scored["q50"]],
                   fmt="none", ecolor="tab:green", alpha=0.6, capsize=2)
        a.scatter(scored["target"], scored["actual"], c=np.where(hit, "tab:green", "tab:red"), s=18, zorder=5,
                  label=f"Saved forecasts now scored: {hit.sum()} of {len(hit)} inside their range")
    a.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    a.set_title(f"{TICKER}: forecast for the next {days} trading days (chance of rising {live[4]:.0%})")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.3)

    # Learning curve.
    a = ax[0, 1]
    a.step(runs["run"], runs["validation"], where="post", color="tab:blue", marker="o", markersize=4,
           label=f"Validation score (worst of {BLOCKS} periods; chooses the model; can only rise)")
    a.plot(runs["run"], runs["held_out"], color="tab:red", marker="o", markersize=4,
           label="Held-out skill (never used to choose; the real test)")
    a.axhline(0, color="black", linewidth=0.8)
    a.text(0.99, 0.02, "above 0 = better than ignoring every signal", transform=a.transAxes, ha="right",
           fontsize=8, color="gray")
    a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    a.set_xlabel("Run")
    a.set_title(f"Learning curve: run {len(runs)}, {memory['tried']} models tried so far")
    a.legend(loc="center left", fontsize=8)
    a.grid(alpha=0.3)

    # Where the model gained or lost against the baseline.
    a = ax[1, 0]
    known = masks["validation"] | masks["embargo"] | masks["held_out"]
    gain = np.cumsum(np.where(known, base_loss - loss, 0.0))[known]
    a.plot(data.index[known], gain, color="tab:purple")
    a.axvspan(data.index[masks["held_out"]][0], data.index[known][-1], color="tab:red", alpha=0.08,
              label="Held-out period")
    a.axhline(0, color="black", linewidth=0.8)
    a.set_title("Cumulative advantage over the baseline (rising = model forecasting better)")
    a.set_ylabel("Cumulative pinball loss saved")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.3)

    # What the champion pays attention to.
    a = ax[1, 1]
    if len(weights):
        ordered = weights.reindex(weights.abs().sort_values().index)
        a.barh(ordered.index, ordered, color=np.where(ordered > 0, "seagreen", "firebrick"))
    a.axvline(0, color="black", linewidth=0.8)
    a.set_title("Current model: ridge weights (green = higher value, higher forecast)")
    a.tick_params(axis="y", labelsize=8)
    a.grid(alpha=0.3, axis="x")

    fig.suptitle(f"{TICKER} learning forecast, prices to {data.index[-1]:%d %B %Y}", fontsize=14)
    fig.text(0.5, 0.006, "Skill = 1 - pinball loss of the model / pinball loss of unconditional past quantiles. "
             "Not financial advice.", ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path, dpi=115)
    plt.close(fig)


# ----------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Nvidia prediction that searches for a better model every run.")
    parser.add_argument("--days", type=int, default=20, help="trading days ahead to predict")
    parser.add_argument("--seconds", type=float, default=30, help="seconds to spend searching for a better model")
    parser.add_argument("--tries", type=int, default=None,
                        help="try exactly this many models instead of searching for --seconds")
    parser.add_argument("--reset", action="store_true", help="forget what was learned for this horizon")
    parser.add_argument("--chart", default=str(FOLDER / "nvidia_learning.png"))
    parser.add_argument("--report", default=str(FOLDER / "nvidia_learning.md"))
    args = parser.parse_args()
    if args.days < 1 or (args.tries is not None and args.tries < 0) or args.seconds < 0:
        raise SystemExit("--days must be at least 1, --tries and --seconds at least 0.")

    print("Loading prices and news...")
    market, tone = load_market(), load_tone()
    prices = market[TICKER]
    features, sigma = build_features(market, tone)
    data = Data(prices, features, sigma, args.days)
    if data.n < MIN_HISTORY + HELD_OUT_DAYS + 250:
        raise SystemExit("Not enough history.")

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    key = str(args.days)
    if args.reset:
        state.pop(key, None)
    memory = state.setdefault(key, {
        "held_out_start": str(data.index[data.n - HELD_OUT_DAYS].date()),
        "champion": None, "tried": 0, "runs": [], "predictions": {},
    })

    # Periods. Validation chooses the model; an embargo of `days` rows keeps its outcomes out of the held-out period.
    split = int(data.index.searchsorted(pd.Timestamp(memory["held_out_start"])))
    row = np.arange(data.n)
    known = ~np.isnan(data.y) & (row >= MIN_HISTORY)
    masks = {"validation": known & (row < split - args.days), "held_out": known & (row >= split)}
    masks["embargo"] = known & ~masks["validation"] & ~masks["held_out"]
    y = np.nan_to_num(data.y)
    base_loss = pinball(y, walk_forward(data, BASELINE))

    blocks = []
    for part in np.array_split(np.flatnonzero(masks["validation"]), BLOCKS):
        blocks.append(np.isin(row, part))

    _JOB.update(data=data, known=known, y=y, base_loss=base_loss, blocks=blocks)

    # Champion from earlier runs, re-scored on today's data, then this run's candidates.
    rng = np.random.default_rng([args.days, memory["tried"]])
    first_run = memory["champion"] is None
    if first_run:
        champion = {**BASELINE, "features": ["news tone 30d"], "vol_scale": True}  # a simple starting point
        memory["tried"] += 1
    else:
        champion = memory["champion"]
    best, champ = evaluate(champion)
    before = best
    budget = f"{args.tries} models" if args.tries is not None else f"{args.seconds:g} seconds"
    print(f"Training: searching for a better model for {budget} on {os.cpu_count() or 1} CPU cores...")
    started = time.monotonic()
    champion, best, tried, improved, workers = search(rng, champion, best, data.names, args.seconds, args.tries)
    elapsed = time.monotonic() - started
    print(f"  tried {tried} models in {elapsed:.1f} s ({tried / max(elapsed, 1e-9):.1f} per second)")
    if improved:
        best, champ = evaluate(champion)
    memory["tried"] += tried
    memory["champion"] = champion

    loss = pinball(y, champ)
    held_out = skill(loss, base_loss, masks["held_out"])
    average = skill(loss, base_loss, masks["validation"])
    inside = (data.y >= champ[:, 0]) & (data.y <= champ[:, 2])
    coverage = inside[masks["held_out"]].mean()
    memory["runs"].append({
        "run": len(memory["runs"]) + 1, "date": str(date.today()), "validation": best, "held_out": held_out,
        "tried": memory["tried"], "improved": improved > 0, "seconds": round(elapsed, 1), "this_run": tried,
    })

    # Today's forecast from a fresh fit, saved for later runs to score.
    last = data.n - 1
    live = forecast_rows(data, champion, last, np.array([last]))[0]
    base_live = forecast_rows(data, BASELINE, last, np.array([last]))[0]
    price, day = data.price[last], data.index[last]
    memory["predictions"][str(day.date())] = {
        "price": price, "q10": price * math.exp(live[0]), "q50": price * math.exp(live[1]),
        "q90": price * math.exp(live[2]), "chance_up": live[4], "run": len(memory["runs"]),
    }
    STATE.write_text(json.dumps(state, indent=1))

    # Score saved forecasts whose target day has arrived.
    scored = []
    for made, p in memory["predictions"].items():
        at = data.index.searchsorted(pd.Timestamp(made))
        if at + args.days < data.n and data.index[at] == pd.Timestamp(made):
            actual = data.price[at + args.days]
            scored.append({"made": made, "target": data.index[at + args.days], "actual": actual, **p,
                           "inside": p["q10"] <= actual <= p["q90"]})
    scored = pd.DataFrame(scored)

    # Ridge weights of the live fit, for the chart.
    cols = [data.names.index(f) for f in champion["features"]]
    end = last - args.days + 1
    start = max(0, end - champion["window"]) if champion["window"] else 0
    scale = data.vol_scale if champion["vol_scale"] else np.ones(data.n)
    X, z = data.X[start:end][:, cols], data.y[start:end] / scale[start:end]
    age = np.arange(end - start)[::-1]
    w = 0.5 ** (age / champion["half_life"]) if champion["half_life"] else np.ones(end - start)
    w = w / w.sum()
    Z = (X - w @ X) / (np.sqrt(w @ (X - w @ X) ** 2) + 1e-12)
    beta = np.linalg.solve(Z.T @ (Z * w[:, None]) + champion["lam"] * np.eye(len(cols)), Z.T @ (w * (z - w @ z)))
    weights = pd.Series(beta, index=champion["features"])

    # ------------------------------------------------------------- output
    run = memory["runs"][-1]
    to_price = lambda log_move: price * math.exp(log_move)
    change = ("first run: starting model set" if first_run and not improved else
              f"found {improved} better model{'s' if improved != 1 else ''}: validation score "
              f"{before:+.2%} -> {best:+.2%}" if improved else "no better model found this run; champion kept")
    verdict = ("beats" if held_out > 0 else "does not beat")
    summary = [
        f"{TICKER} at ${price:,.2f} on {day:%Y-%m-%d}.  Run {run['run']}, {memory['tried']} models tried in total "
        f"({tried} this run in {elapsed:.0f} s).",
        "",
        f"FORECAST for the next {args.days} trading days",
        f"  Median            {math.expm1(live[1]):+.1%}  (to about ${to_price(live[1]):,.2f})",
        f"  Average           {live[3]:+.1%}",
        f"  8 in 10 range     {math.expm1(live[0]):+.1%} to {math.expm1(live[2]):+.1%}  "
        f"(${to_price(live[0]):,.2f} to ${to_price(live[2]):,.2f})",
        f"  Chance of rising  {live[4]:.0%}",
        f"  Baseline (ignores every signal): median {math.expm1(base_live[1]):+.1%}, range "
        f"{math.expm1(base_live[0]):+.1%} to {math.expm1(base_live[2]):+.1%}, rising {base_live[4]:.0%}",
        "",
        f"LEARNING  {change}",
        f"  Model             {describe(champion)}",
        f"  Features          {', '.join(champion['features'])}",
        f"  Validation score  {best:+.2%}   (worst of {BLOCKS} periods; average skill {average:+.2%}; "
        f"{masks['validation'].sum()} days to {data.index[masks['validation']][-1]:%Y-%m-%d}; chooses the model)",
        f"  Held-out skill    {held_out:+.2%}   ({masks['held_out'].sum()} days from {memory['held_out_start']}; "
        "never used to choose)",
        f"  Held-out range    actual price landed inside the 8-in-10 range {coverage:.0%} of the time (target 80%)",
        f"  Verdict           on days it was not tuned on, the model {verdict} the baseline."
        + ("" if held_out > 0 else " Prefer the baseline forecast above."),
    ]
    if len(scored):
        summary.append(f"  Saved forecasts   {scored['inside'].sum()} of {len(scored)} earlier forecasts "
                       "landed inside their range")
    else:
        summary.append(f"  Saved forecasts   {len(memory['predictions'])} saved; the first can be scored "
                       f"{args.days} trading days after it was made")
    print("\n" + "\n".join(summary))

    save_chart(Path(args.chart), data, memory, champ, base_loss, loss, masks, live, weights, scored)

    runs = pd.DataFrame(memory["runs"])
    report = [
        f"# Nvidia ({TICKER}) learning forecast",
        "",
        f"- **Generated:** {date.today():%d %B %Y} by `nvidia_learning.py`. Not financial advice.",
        "- Skill = 1 - pinball loss of the model / pinball loss of a baseline that forecasts the unconditional "
        "past quantiles. Above 0 means better than ignoring every signal.",
        f"- The validation score (skill in the worst of {BLOCKS} validation periods) is used to choose the model, "
        "so it can only rise. Held-out skill is measured on days never used to choose it and is the honest figure.",
        "",
        "```", *summary, "```",
        "",
        "## Runs so far",
        "",
        "| Run | Date | Models tried | Validation score | Held-out skill | Found a better model |",
        "|---|---|---|---|---|---|",
        *[f"| {r.run} | {r.date} | {r.tried} | {r.validation:+.2%} | {r.held_out:+.2%} | "
          f"{'yes' if r.improved else 'no'} |" for r in runs.itertuples()],
        "",
        "## Limits",
        "",
        "- The more models are tried, the more the validation skill flatters the model. Trust the held-out skill.",
        f"- {args.days}-day outcomes overlap from one day to the next, so there is far less independent evidence "
        "than the day counts suggest; small skill differences are noise.",
        "- The history is 2017 onwards, a period in which Nvidia rose strongly.",
        "",
        "Chart: `nvidia_learning.png`. Memory: `nvidia_learning.json` (delete it, or use `--reset`, to start again).",
    ]
    Path(args.report).write_text("\n".join(report) + "\n")
    print(f"\nChart saved to {args.chart}\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
