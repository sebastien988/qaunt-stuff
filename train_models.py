"""Train stock-ranking models on the local data library (./data), walk-forward.

Question: using only price and volume, can a model rank stocks by how much they
will beat the average stock over the next month (21 trading days)?

Data
    Every stock in data/daily/stocks_* (S&P 500, Nasdaq-100, S&P 400, S&P 600:
    whatever download_data.py has fetched so far). Sampled every 21 trading
    days, so neighbouring targets do not overlap.

Features (cross-sectionally ranked each date to [-0.5, 0.5]; missing = 0)
    rev_1w, rev_1m       last 5- and 21-day return (short-term reversal)
    mom_3m, mom_6m       63- and 126-day return
    mom_12_1             return from 12 months ago to 1 month ago (classic momentum)
    vol_1m, vol_3m       daily-return volatility, 21 and 63 days
    idio_vol             volatility not explained by the market (1 year)
    beta                 1-year beta to SPY
    dist_high            distance below the 52-week high
    trend_200            price / 200-day average - 1
    max_ret              biggest one-day gain in the last month (lottery effect)
    skew_3m              63-day return skewness
    size                 log 63-day average dollar volume (liquidity / size proxy)
    volume_trend         log(21-day / 252-day dollar volume)

Target
    rank of the next 21-day return across stocks, in [-0.5, 0.5].

Models
    momentum   no training: score = mom_12_1 (the textbook baseline)
    ridge      beta = (X'X/n + lambda I)^-1 X'y/n, lambda fixed in advance
    mlp        numpy neural net, 2 hidden layers (32, 16, ReLU), Adam, early
               stopping on the last 15% of the training dates, 3-seed average

Walk-forward
    For each test year, the models are trained only on samples whose targets
    were fully known before that year began (one-month embargo), then scored on
    that year. Nothing in the test year is used for training or for choosing
    settings; all settings are fixed in this file.

Scoring, per test date
    IC        Spearman correlation between score and realised return
    L/S       return of top 10% minus bottom 10% (equal weight), minus 10 bp per
              side of turnover; no borrow fees
    Long      top 10% minus the equal-weight average of all stocks

Big caveat: SURVIVORSHIP BIAS. The stock lists are today's index members, so
companies that failed or left the index are missing. Results look better than
a real investor could have achieved.

Usage:
    python train_models.py
    python train_models.py --first-test-year 2012 --cost-bps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "daily"
MODELS = HERE / "models"

HORIZON = 21
START = "1998-01-01"
MIN_PRICE = 5.0
RIDGE_LAMBDA = 0.05
TOP_FRAC = 0.10
FEATURES = ["rev_1w", "rev_1m", "mom_3m", "mom_6m", "mom_12_1", "vol_1m", "vol_3m",
            "idio_vol", "beta", "dist_high", "trend_200", "max_ret", "skew_3m",
            "size", "volume_trend"]


# ----------------------------------------------------------------------------- data

def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    spy = pd.read_csv(DATA / "etfs" / "SPY.csv", index_col=0, parse_dates=True)
    spy = spy.loc[START:, "Adj Close"]
    files = sorted(p for d in DATA.glob("stocks_*") for p in d.glob("*.csv"))
    px, dv = {}, {}
    for p in files:
        if p.stem in px:
            continue
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True,
                             usecols=["Date", "Close", "Adj Close", "Volume"])
        except (ValueError, KeyError):
            continue
        df = df.loc[START:]
        if len(df) < 300:
            continue
        px[p.stem] = df["Adj Close"]
        dv[p.stem] = (df["Close"] * df["Volume"]).where(lambda s: s > 0)
        dv[p.stem + "__raw"] = df["Close"]
    raw = pd.DataFrame({k[:-5]: v for k, v in dv.items() if k.endswith("__raw")}).reindex(spy.index)
    px = pd.DataFrame(px).reindex(spy.index)
    dv = pd.DataFrame({k: v for k, v in dv.items() if not k.endswith("__raw")}).reindex(spy.index)
    print(f"Loaded {px.shape[1]} stocks, {px.index[0].date()} to {px.index[-1].date()}")
    return px, dv, spy, raw


def build_features(px: pd.DataFrame, dv: pd.DataFrame, spy: pd.Series, raw: pd.DataFrame):
    """Return dates, tickers, X [T, N, F] (ranked), y [T, N] (ranked), fwd [T, N] raw, valid mask."""
    idx = np.arange(252 + 21, len(px.index), HORIZON)
    dates = px.index[idx]
    r = px.pct_change(fill_method=None)
    rm = spy.pct_change()

    def at(frame: pd.DataFrame) -> np.ndarray:
        return frame.iloc[idx].to_numpy(dtype=float)

    feats = {}
    feats["rev_1w"] = at(px / px.shift(5) - 1)
    feats["rev_1m"] = at(px / px.shift(21) - 1)
    feats["mom_3m"] = at(px / px.shift(63) - 1)
    feats["mom_6m"] = at(px / px.shift(126) - 1)
    feats["mom_12_1"] = at(px.shift(21) / px.shift(252) - 1)
    feats["vol_1m"] = at(r.rolling(21, min_periods=15).std())
    feats["vol_3m"] = at(r.rolling(63, min_periods=40).std())
    mr, mm = r.rolling(252, min_periods=150).mean(), rm.rolling(252, min_periods=150).mean()
    cov = r.mul(rm, axis=0).rolling(252, min_periods=150).mean() - mr.mul(mm, axis=0)
    var_m = rm.rolling(252, min_periods=150).var(ddof=0)
    beta = cov.div(var_m, axis=0)
    var_r = r.rolling(252, min_periods=150).var(ddof=0)
    feats["beta"] = at(beta)
    feats["idio_vol"] = at(np.sqrt((var_r - beta.pow(2).mul(var_m, axis=0)).clip(lower=0)))
    feats["dist_high"] = at(px / px.rolling(252, min_periods=150).max() - 1)
    feats["trend_200"] = at(px / px.rolling(200, min_periods=150).mean() - 1)
    feats["max_ret"] = at(r.rolling(21, min_periods=15).max())
    feats["skew_3m"] = at(r.rolling(63, min_periods=40).skew())
    feats["size"] = at(np.log(dv.rolling(63, min_periods=40).mean()))
    feats["volume_trend"] = at(np.log(dv.rolling(21, min_periods=15).mean()
                                      / dv.rolling(252, min_periods=150).mean()))

    fwd = at(px.shift(-HORIZON) / px - 1)
    price_ok = at(raw) >= MIN_PRICE
    valid = price_ok & np.isfinite(feats["mom_12_1"]) & np.isfinite(feats["vol_3m"])

    X = np.stack([cs_rank(feats[f], valid) for f in FEATURES], axis=-1)
    X = np.nan_to_num(X, nan=0.0)
    y = cs_rank(fwd, valid & np.isfinite(fwd))
    return dates, list(px.columns), X, y, fwd, valid, feats


def cs_rank(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per row, in [-0.5, 0.5]; NaN outside mask."""
    a = np.where(mask & np.isfinite(a), a, np.nan)
    ranks = pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()
    return ranks - 0.5


# ----------------------------------------------------------------------------- models

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb / len(X) + lam * np.diag([0.0] + [1.0] * X.shape[1])
    return np.linalg.solve(A, Xb.T @ y / len(X))


def predict_ridge(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


class MLP:
    def __init__(self, n_in: int, hidden=(32, 16), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.rng = rng

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def predict(self, X):
        return self.forward(X)[-1][:, 0]

    def fit(self, X, y, Xv, yv, epochs=30, batch=1024, lr=1e-3, l2=1e-4, patience=3):
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t, best, bad = 0, np.inf, 0
        best_state = ([W.copy() for W in self.W], [b.copy() for b in self.b])
        for _ in range(epochs):
            order = self.rng.permutation(len(X))
            for s in range(0, len(X), batch):
                bi = order[s:s + batch]
                acts = self.forward(X[bi])
                grad = 2 * (acts[-1][:, 0] - y[bi])[:, None] / len(bi)
                gW, gb = [], []
                for i in range(len(self.W) - 1, -1, -1):
                    gW.insert(0, acts[i].T @ grad + l2 * self.W[i])
                    gb.insert(0, grad.sum(0))
                    if i:
                        grad = (grad @ self.W[i].T) * (acts[i] > 0)
                t += 1
                for j, (p, g) in enumerate(zip(params, gW + gb)):
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    p -= lr * (m[j] / (1 - 0.9**t)) / (np.sqrt(v[j] / (1 - 0.999**t)) + 1e-8)
            loss = np.mean((self.predict(Xv) - yv) ** 2)
            if loss < best - 1e-6:
                best, bad = loss, 0
                best_state = ([W.copy() for W in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience:
                    break
        self.W, self.b = best_state
        return self


def fit_mlp_ensemble(X3, y2, mask, seeds=(0, 1, 2)):
    """X3 [T, N, F]; train on last 85% dates, validate on the final 15%."""
    rows_with_data = np.flatnonzero(mask.any(axis=1))
    cut = rows_with_data[int(len(rows_with_data) * 0.85)]
    tr, va = mask.copy(), mask.copy()
    tr[cut:], va[:cut] = False, False
    nets = [MLP(X3.shape[-1], seed=s).fit(X3[tr], y2[tr], X3[va], y2[va]) for s in seeds]
    return nets


def predict_mlp(nets, X):
    return np.mean([n.predict(X) for n in nets], axis=0)


# ----------------------------------------------------------------------------- evaluation

def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return np.corrcoef(ra, rb)[0, 1]


def evaluate(scores: np.ndarray, fwd: np.ndarray, mask: np.ndarray, cost_bps: float):
    """Per date: IC, long-short net return, long-only excess vs equal weight, decile returns."""
    rows, prev_long, prev_short = [], set(), set()
    deciles = []
    for t in range(scores.shape[0]):
        m = mask[t] & np.isfinite(fwd[t]) & np.isfinite(scores[t])
        if m.sum() < 30:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        s, f, ids = scores[t, m], fwd[t, m], np.flatnonzero(m)
        k = max(3, int(len(s) * TOP_FRAC))
        order = np.argsort(s)
        lo, hi = order[:k], order[-k:]
        longs, shorts = set(ids[hi]), set(ids[lo])
        turn_l = 1 - len(longs & prev_long) / k if prev_long else 1.0
        turn_s = 1 - len(shorts & prev_short) / k if prev_short else 1.0
        cost = cost_bps / 1e4 * 2 * (turn_l + turn_s)
        ls = f[hi].mean() - f[lo].mean() - cost
        lo_ex = f[hi].mean() - f.mean() - cost_bps / 1e4 * 2 * turn_l
        rows.append((spearman(s, f), ls, lo_ex, turn_l))
        prev_long, prev_short = longs, shorts
        q = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=False)
        deciles.append(pd.Series(f).groupby(q.to_numpy()).mean().to_numpy() - f.mean())
    df = pd.DataFrame(rows, columns=["ic", "ls", "long_excess", "turnover"])
    return df, np.nanmean(np.array(deciles), axis=0) if deciles else np.full(10, np.nan)


def summarise(df: pd.DataFrame) -> dict:
    d = df.dropna()
    per_year = 252 / HORIZON
    ic_t = d.ic.mean() / d.ic.std() * np.sqrt(len(d))
    return {
        "months": len(d),
        "mean_ic": d.ic.mean(),
        "ic_t_stat": ic_t,
        "hit_rate_ic_pos": (d.ic > 0).mean(),
        "ls_ann_return": d.ls.mean() * per_year,
        "ls_sharpe": d.ls.mean() / d.ls.std() * np.sqrt(per_year),
        "long_excess_ann": d.long_excess.mean() * per_year,
        "long_excess_ir": d.long_excess.mean() / d.long_excess.std() * np.sqrt(per_year),
        "avg_turnover": d.turnover.mean(),
    }


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test-year", type=int, default=2010)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    args = ap.parse_args()

    px, dv, spy, raw = load_panel()
    dates, tickers, X, y, fwd, valid, feats = build_features(px, dv, spy, raw)
    labelled = valid & np.isfinite(y)
    print(f"{len(dates)} monthly dates, {labelled.sum():,} labelled stock-months")

    T = len(dates)
    scores = {k: np.full((T, len(tickers)), np.nan) for k in ("momentum", "ridge", "mlp")}
    scores["momentum"] = np.where(valid, X[..., FEATURES.index("mom_12_1")], np.nan)
    years = sorted({d.year for d in dates if d.year >= args.first_test_year})
    ridge_coefs = {}
    for yr in years:
        test = np.array([d.year == yr for d in dates])
        first = np.argmax(test)
        train_end = first - 1  # target of date first-1 ends at date first: embargo of one step
        if train_end < 24:
            continue
        tr_mask = labelled.copy()
        tr_mask[train_end:] = False
        w = fit_ridge(X[tr_mask], y[tr_mask])
        ridge_coefs[yr] = w
        nets = fit_mlp_ensemble(X, y, tr_mask)
        for t in np.flatnonzero(test):
            scores["ridge"][t] = np.where(valid[t], predict_ridge(w, X[t]), np.nan)
            scores["mlp"][t] = np.where(valid[t], predict_mlp(nets, X[t]), np.nan)
        print(f"  {yr}: trained on {tr_mask.sum():,} rows")

    test_rows = np.array([d.year >= years[0] for d in dates])
    results, deciles, summaries = {}, {}, {}
    for name, sc in scores.items():
        sc = np.where(test_rows[:, None], sc, np.nan)
        df, dec = evaluate(sc, fwd, valid, args.cost_bps)
        df.index = dates
        results[name], deciles[name] = df, dec
        summaries[name] = summarise(df)
    summ = pd.DataFrame(summaries).T
    print("\nOut-of-sample results\n", summ.round(3).to_string())

    # final models on all labelled data, and today's ranking
    final_w = fit_ridge(X[labelled], y[labelled])
    final_nets = fit_mlp_ensemble(X, y, labelled)
    MODELS.mkdir(exist_ok=True)
    np.savez(MODELS / "ridge.npz", coef=final_w, features=np.array(FEATURES))
    np.savez(MODELS / "mlp.npz", **{f"net{i}_W{j}": W for i, n in enumerate(final_nets)
                                    for j, W in enumerate(n.W)},
             **{f"net{i}_b{j}": b for i, n in enumerate(final_nets) for j, b in enumerate(n.b)})

    # latest date with features (target unknown): live ranking
    last_idx = len(px.index) - 1
    live = latest_features(px, dv, spy, raw, last_idx)
    live_scores = pd.DataFrame({
        "ridge": predict_ridge(final_w, live.to_numpy()),
        "mlp": predict_mlp(final_nets, live.to_numpy()),
    }, index=live.index)
    live_scores["blend"] = live_scores.rank(pct=True).mean(axis=1)
    live_scores = live_scores.sort_values("blend", ascending=False)
    live_scores.to_csv(MODELS / "latest_ranking.csv")

    make_chart(results, deciles, summ, ridge_coefs, final_w, args)
    write_report(summ, final_w, live_scores, px.index[last_idx], len(tickers), dates, years, args)
    (MODELS / "summary.json").write_text(json.dumps(summ.round(4).to_dict(), indent=2))
    print("\nSaved train_models.png, train_models.md, models/")


def latest_features(px, dv, spy, raw, last_idx) -> pd.DataFrame:
    """Recompute features on the most recent day (same code path, single row)."""
    sub_px, sub_dv, sub_raw = px.iloc[: last_idx + 1], dv.iloc[: last_idx + 1], raw.iloc[: last_idx + 1]
    sub_spy = spy.iloc[: last_idx + 1]
    # pad so that build_features' sampling hits the last row
    offset = (len(sub_px) - 1 - (252 + 21)) % HORIZON
    sub_px, sub_dv, sub_raw, sub_spy = (sub_px.iloc[offset:], sub_dv.iloc[offset:],
                                        sub_raw.iloc[offset:], sub_spy.iloc[offset:])
    d, tick, X, _, _, valid, _ = build_features(sub_px, sub_dv, sub_spy, sub_raw)
    m = valid[-1]
    return pd.DataFrame(X[-1][m], index=np.array(tick)[m], columns=FEATURES)


def make_chart(results, deciles, summ, ridge_coefs, final_w, args) -> None:
    colors = {"momentum": "tab:gray", "ridge": "tab:blue", "mlp": "tab:red"}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Stock-ranking models trained on the local data library (walk-forward, out-of-sample)",
                 fontsize=14)

    a = ax[0, 0]
    for k, df in results.items():
        eq = (1 + df.ls.dropna()).cumprod()
        a.plot(eq.index, eq, color=colors[k],
               label=f"{k} (Sharpe {summ.loc[k, 'ls_sharpe']:.2f})")
    a.set_yscale("log")
    a.set_title(f"Long top 10% / short bottom 10%, growth of $1 ({args.cost_bps:.0f} bp/side)")
    a.legend()
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for k, df in results.items():
        a.plot(df.index, df.ic.rolling(12, min_periods=6).mean(), color=colors[k],
               label=f"{k} (mean IC {summ.loc[k, 'mean_ic']:.3f}, t={summ.loc[k, 'ic_t_stat']:.1f})")
    a.axhline(0, color="k", lw=0.8)
    a.set_title("Rank IC (Spearman), 12-month rolling mean")
    a.legend()
    a.grid(alpha=0.3)

    a = ax[1, 0]
    xs = np.arange(10)
    wdt = 0.27
    for i, (k, dec) in enumerate(deciles.items()):
        a.bar(xs + (i - 1) * wdt, dec * 100, wdt, color=colors[k], label=k)
    a.set_xticks(xs, [f"D{i + 1}" for i in xs])
    a.axhline(0, color="k", lw=0.8)
    a.set_title("Next-month return vs average stock, by predicted decile (D10 = best)")
    a.set_ylabel("% per month")
    a.legend()
    a.grid(alpha=0.3, axis="y")

    a = ax[1, 1]
    order = np.argsort(final_w[1:])
    a.barh(np.array(FEATURES)[order], final_w[1:][order], color="tab:blue")
    a.axvline(0, color="k", lw=0.8)
    a.set_title("Ridge weights (final model, all data): + means 'higher rank -> better'")
    a.grid(alpha=0.3, axis="x")

    fig.text(0.5, 0.005, "Survivorship bias: universe is today's index members. Not investment advice.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(HERE / "train_models.png", dpi=110)


def write_report(summ, final_w, live, last_date, n_stocks, dates, years, args) -> None:
    lines = [
        "# Stock-ranking models trained on the local data library",
        "",
        f"Universe: {n_stocks} US stocks from `data/daily/stocks_*`; monthly samples "
        f"{dates[0].date()} to {dates[-1].date()}; walk-forward test years {years[0]}-{years[-1]}; "
        f"costs {args.cost_bps:.0f} bp per side.",
        "",
        "## Out-of-sample results",
        "",
        "| model | mean IC | IC t-stat | IC>0 | L/S return/yr | L/S Sharpe | top-10% excess/yr | IR | turnover |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, r in summ.iterrows():
        lines.append(f"| {k} | {r.mean_ic:.3f} | {r.ic_t_stat:.1f} | {r.hit_rate_ic_pos:.0%} | "
                     f"{r.ls_ann_return:.1%} | {r.ls_sharpe:.2f} | {r.long_excess_ann:.1%} | "
                     f"{r.long_excess_ir:.2f} | {r.avg_turnover:.0%} |")
    best = summ.ls_sharpe.idxmax()
    lines += [
        "",
        f"Best long/short Sharpe out of sample: **{best}**. Compare the trained models with the "
        "untrained momentum baseline. If they don't clearly beat it, the training added nothing.",
        "",
        "## Ridge weights (final model)",
        "",
        "| feature | weight |",
        "|---|---|",
        *[f"| {f} | {w:+.4f} |" for f, w in sorted(zip(FEATURES, final_w[1:]), key=lambda x: -abs(x[1]))],
        "",
        f"## Latest ranking ({last_date.date()}), next ~21 trading days",
        "",
        "Top 15 (blend of ridge + MLP ranks):",
        "",
        "| ticker | ridge | mlp | blend |",
        "|---|---|---|---|",
        *[f"| {t} | {r.ridge:+.3f} | {r.mlp:+.3f} | {r.blend:.2f} |" for t, r in live.head(15).iterrows()],
        "",
        "Bottom 15:",
        "",
        "| ticker | ridge | mlp | blend |",
        "|---|---|---|---|",
        *[f"| {t} | {r.ridge:+.3f} | {r.mlp:+.3f} | {r.blend:.2f} |" for t, r in live.tail(15).iterrows()],
        "",
        "Full ranking: `models/latest_ranking.csv`.",
        "",
        "## Caveats",
        "",
        "- **Survivorship bias**: only today's index members are in the data, so failed companies are missing.",
        "- Predictions are *relative* (stock vs average stock), not market direction.",
        "- A typical good IC is 0.02-0.05. Most of each stock's monthly return is still unexplained noise.",
        "- Shorting costs (borrow fees) are ignored; the long-only column is closer to what you could actually do.",
        "",
    ]
    (HERE / "train_models.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
