"""Predict Nvidia's share price move from the tone of the news about it.

News data: the GDELT project reads worldwide online news and publishes the
average tone (positive or negative wording) of each day's English-language
articles mentioning Nvidia, back to 2017.

Method: take the average news tone over the last 30 days and sort it into low,
middle or high compared with the past. The prediction is what Nvidia's share
price did after past days in the same group: the average move, the usual
range, and how often it rose.

The script also checks the method on itself: for every day since 2020 it makes
the prediction using only earlier data, then compares it with what happened
and with a forecast that ignores the news.

Every run prints the prediction, saves a chart (nvidia_prediction.png) and
rewrites a report for other tools to read (nvidia_prediction.md).

Usage:
    python nvidia_prediction.py              # 20 and 5 trading days ahead
    python nvidia_prediction.py --days 10
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import requests
import yfinance as yf

FOLDER = Path(__file__).resolve().parent
CACHE = FOLDER / "news_tone_cache.json"
TICKER = "NVDA"
NEWS_QUERY = "nvidia sourcelang:english"
NEWS_START = "20170101000000"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

TONE_DAYS = 30  # days of news averaged into the signal
GROUPS = ["low", "middle", "high"]
MIN_HISTORY = 750  # trading days of past outcomes needed before predicting


def fetch_tone() -> dict[str, float]:
    """Daily average news tone from GDELT, which often asks callers to wait."""
    params = {
        "query": NEWS_QUERY, "mode": "timelinetone", "format": "json",
        "startdatetime": NEWS_START, "enddatetime": f"{date.today():%Y%m%d}235959",
    }
    for wait in [0, 15, 45, 90]:
        time.sleep(wait)
        try:
            response = requests.get(GDELT_URL, params=params, timeout=90)
            points = response.json()["timeline"][0]["data"]
            return {p["date"][:8]: p["value"] for p in points}
        except (requests.RequestException, ValueError, KeyError, IndexError):
            print("  news service busy, retrying...")
    raise RuntimeError("GDELT did not return the news data.")


def load_tone() -> pd.Series:
    """Daily news tone, downloaded at most once a day."""
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else None
    if cached and cached["fetched"] == str(date.today()):
        data = cached["data"]
    else:
        try:
            data = fetch_tone()
            CACHE.write_text(json.dumps({"fetched": str(date.today()), "data": data}))
        except RuntimeError as error:
            if not cached:
                raise SystemExit(f"{error} Try again in a few minutes.")
            print(f"  WARNING: using news saved on {cached['fetched']}; today's download failed.")
            data = cached["data"]
    tone = pd.Series({pd.Timestamp(day): value for day, value in data.items()}).asfreq("D")
    return tone[tone.index < pd.Timestamp(date.today())]  # today's news is still coming in


def load_prices() -> pd.Series:
    data = yf.download(TICKER, start="2016-06-01", auto_adjust=True, progress=False)
    if data.empty:
        raise SystemExit(f"No price data returned for {TICKER}.")
    close = data["Close"]
    return (close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close).dropna()


def build_table(prices: pd.Series, tone: pd.Series, days: int) -> pd.DataFrame:
    """One row per trading day: the news signal and what the price did next."""
    # Shift by a day so each trading day only sees news published before it.
    tone = tone.reindex(pd.date_range(tone.index[0], max(tone.index[-1], prices.index[-1])))
    signal = tone.shift(1).rolling(TONE_DAYS, min_periods=TONE_DAYS // 2).mean()
    table = pd.DataFrame({
        "price": prices,
        "signal": signal.reindex(prices.index),
        "outcome": prices.shift(-days) / prices - 1,  # unknown for the latest days
    })
    return table.dropna(subset=["signal"])


def predict(table: pd.DataFrame, row: int, days: int) -> dict | None:
    """Prediction for one day, from days whose outcome was already known then."""
    past = table.iloc[: max(row - days + 1, 0)]
    if len(past) < MIN_HISTORY:
        return None
    low, high = past["signal"].quantile([1 / 3, 2 / 3])
    group_of = lambda s: np.where(s < low, 0, np.where(s < high, 1, 2))
    group = int(group_of(table["signal"].iloc[row]))
    similar = past["outcome"][group_of(past["signal"]) == group]
    return {
        "group": group,
        "share of history": (past["signal"] < table["signal"].iloc[row]).mean(),
        "similar days": len(similar),
        "expected": similar.mean(),
        "median": similar.median(),
        "low": similar.quantile(0.1),
        "high": similar.quantile(0.9),
        "chance up": (similar > 0).mean(),
        "expected ignoring news": past["outcome"].mean(),
        "chance up ignoring news": (past["outcome"] > 0).mean(),
    }


def check_method(table: pd.DataFrame, days: int) -> pd.DataFrame:
    """Past predictions, each made without later data, next to what happened."""
    rows = []
    for row in range(len(table)):
        outcome = table["outcome"].iloc[row]
        if np.isnan(outcome):
            continue
        forecast = predict(table, row, days)
        if forecast:
            rows.append({"date": table.index[row], "outcome": outcome, **forecast})
    return pd.DataFrame(rows).set_index("date")


def score_check(checked: pd.DataFrame) -> dict:
    """How the past predictions compared with what happened."""
    rose = (checked["outcome"] > 0).astype(float)
    score = {
        "start": checked.index[0],
        "end": checked.index[-1],
        "groups": [],
        "miss with news": (checked["expected"] - checked["outcome"]).abs().mean(),
        "miss ignoring news": (checked["expected ignoring news"] - checked["outcome"]).abs().mean(),
        "brier with news": ((checked["chance up"] - rose) ** 2).mean(),
        "brier ignoring news": ((checked["chance up ignoring news"] - rose) ** 2).mean(),
    }
    for group, name in enumerate(GROUPS):
        rows = checked[checked["group"] == group]
        if len(rows):
            score["groups"].append({
                "name": name, "days": len(rows), "predicted": rows["expected"].mean(),
                "actual": rows["outcome"].mean(), "rose": (rows["outcome"] > 0).mean(),
            })
    helped = (score["brier with news"] < score["brier ignoring news"]
              and score["miss with news"] < score["miss ignoring news"])
    score["verdict"] = "helped" if helped else "did not help"
    return score


def print_prediction(forecast: dict, score: dict, price: float, days: int) -> None:
    print(f"\nPREDICTION for the next {days} trading days, "
          f"from {forecast['similar days']} past days with similar news tone:")
    print(f"  Average move      {forecast['expected']:+.1%}  (to about ${price * (1 + forecast['expected']):,.2f})")
    print(f"  Typical move      {forecast['median']:+.1%}")
    print(f"  8 in 10 cases     {forecast['low']:+.1%} to {forecast['high']:+.1%}  "
          f"(${price * (1 + forecast['low']):,.2f} to ${price * (1 + forecast['high']):,.2f})")
    print(f"  Rose in           {forecast['chance up']:.0%} of cases")
    print(f"  Ignoring the news {forecast['expected ignoring news']:+.1%} average, "
          f"rose in {forecast['chance up ignoring news']:.0%} of cases")

    print(f"\n  How the method did, {score['start']:%Y-%m-%d} to {score['end']:%Y-%m-%d} "
          f"(each prediction used only earlier data):")
    print(f"    {'News tone was':<16}{'Days':>6}{'Predicted':>11}{'Actual':>9}{'Rose':>8}")
    for g in score["groups"]:
        print(f"    {g['name']:<16}{g['days']:>6}{g['predicted']:>+11.1%}{g['actual']:>+9.1%}{g['rose']:>8.0%}")
    print(f"    Average miss: {score['miss with news']:.1%} with news, "
          f"{score['miss ignoring news']:.1%} ignoring news")
    print(f"    Chance-of-rising error (Brier score, lower is better): "
          f"{score['brier with news']:.3f} with news, {score['brier ignoring news']:.3f} ignoring news")
    print(f"    Verdict: over this period, using the news {score['verdict']}.")


def load_headlines() -> list[tuple[str, str]]:
    """Recent headlines as (date, title); empty if they cannot be fetched."""
    try:
        stories = yf.Ticker(TICKER).news[:8]
    except Exception:
        return []
    found = [(str(s.get("content", {}).get("pubDate", ""))[:10], s.get("content", {}).get("title", ""))
             for s in stories]
    return [(day, title) for day, title in found if title]


def write_report(path: Path, day, price: float, signal: float, tone: pd.Series,
                 results: list[tuple[int, dict, dict]], headlines: list[tuple[str, str]]) -> None:
    """Markdown report that can be read without any other context."""
    first_days, first, first_score = results[0]
    more_negative = 1 - first["share of history"]
    lines = [
        f"# Nvidia ({TICKER}) price prediction from news tone",
        "",
        f"- **Generated:** {date.today():%d %B %Y}",
        f"- **Starting price:** ${price:,.2f} ({TICKER}, {day:%d %B %Y})",
        "- **Produced by:** `nvidia_prediction.py`, a statistical script. This is not financial advice.",
        "",
        "## Summary",
        "",
        f"Nvidia is predicted to move about **{first['expected']:+.1%} over the next {first_days} trading days**, "
        f"to roughly **${price * (1 + first['expected']):,.2f}**. The range of likely outcomes is very wide, "
        f"and a check against past data found that using the news **{first_score['verdict']}** compared with "
        "ignoring it. Treat this as \"what usually happened in similar conditions\", not as a price target.",
        "",
    ]
    for days, forecast, _ in results:
        lines += [
            f"## Prediction: next {days} trading days",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Average move | {forecast['expected']:+.1%} (to about ${price * (1 + forecast['expected']):,.2f}) |",
            f"| Typical (median) move | {forecast['median']:+.1%} |",
            f"| Range in 8 of 10 similar past cases | {forecast['low']:+.1%} to {forecast['high']:+.1%} "
            f"(${price * (1 + forecast['low']):,.2f} to ${price * (1 + forecast['high']):,.2f}) |",
            f"| Share of similar past cases where the price rose | {forecast['chance up']:.0%} |",
            f"| Forecast ignoring the news | {forecast['expected ignoring news']:+.1%} average, "
            f"rose in {forecast['chance up ignoring news']:.0%} of cases |",
            "",
            f"Based on {forecast['similar days']} past trading days when the news tone was similarly "
            f"{GROUPS[forecast['group']]}.",
            "",
        ]
    lines += [
        "## The news signal",
        "",
        f"- **Current value:** {signal:+.2f}, the average tone of news about Nvidia over the "
        f"{TONE_DAYS} days before {day:%d %B %Y}.",
        f"- **How unusual:** more negative than {more_negative:.0%} of earlier days. "
        f"It falls in the \"{GROUPS[first['group']]}\" group.",
        "- **What tone means:** a score of how positive or negative the wording of articles is. "
        f"Above 0 is positive wording, below 0 is negative. The long-run average for Nvidia is {tone.mean():+.2f}.",
        "- **Source:** the GDELT project, which scores each day's English-language online news articles "
        f"mentioning Nvidia, from {tone.index[0]:%Y} onwards.",
        "",
    ]
    if headlines:
        lines += [
            "## Recent headlines (context only, not used in the calculation)",
            "",
            "From Yahoo Finance. These are headlines as published; the script did not verify them.",
            "",
            *[f"- {when}: {title}" for when, title in headlines],
            "",
        ]
    lines += [
        "## How the prediction is made",
        "",
        f"1. Average the daily news tone over the last {TONE_DAYS} days, using only news published before "
        "the day of the prediction.",
        "2. Sort that value into low, middle or high, compared with all earlier days.",
        "3. Look up what Nvidia's share price did over the following trading days after every past day "
        "in the same group.",
        "4. Report the average move, the median, the range covering 8 of 10 cases, and how often the price rose.",
        "",
        "## How reliable it is",
        "",
        "The method was checked on past trading days. Each past prediction used only data available "
        "before that day.",
        "",
    ]
    for days, _, score in results:
        lines += [
            f"### {days}-day predictions, {score['start']:%d %B %Y} to {score['end']:%d %B %Y}",
            "",
            "| News tone was | Days | Predicted move | Actual move | Price rose |",
            "|---|---|---|---|---|",
            *[f"| {g['name'].capitalize()} | {g['days']} | {g['predicted']:+.1%} | {g['actual']:+.1%} | {g['rose']:.0%} |"
              for g in score["groups"]],
            "",
            f"- The average miss of the forecast was {score['miss with news']:.1%} with the news and "
            f"{score['miss ignoring news']:.1%} ignoring it.",
            f"- The chance-of-rising forecast scored {score['brier with news']:.3f} with the news and "
            f"{score['brier ignoring news']:.3f} ignoring it (Brier score; lower is better).",
            f"- **Verdict: using the news {score['verdict']}.**",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- The forecast range is wide; see the average miss above.",
        "- The news signal measures wording, not facts. It does not know what the stories are about "
        "or whether they are true.",
        f"- The history starts in {tone.index[0]:%Y}, a period in which Nvidia's share price rose strongly, "
        "so the averages lean positive.",
        "- Multi-day outcomes overlap from one day to the next, so the day counts overstate how much "
        "independent evidence there is.",
        "- This file is rewritten each time the script runs. The numbers go out of date as prices "
        "and news change; check the generated date.",
        "",
    ]
    path.write_text("\n".join(lines))


def save_chart(table: pd.DataFrame, tone: pd.Series, forecast: dict, days: int, path: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 2]})

    recent = table["price"].iloc[-250:]
    last_day, last_price = recent.index[-1], recent.iloc[-1]
    ahead = pd.bdate_range(last_day, periods=days + 1)
    path_to = lambda move: last_price * (1 + move) ** (np.arange(days + 1) / days)
    top.plot(recent.index, recent, color="black", label=f"{TICKER} price")
    top.fill_between(ahead, path_to(forecast["low"]), path_to(forecast["high"]), color="tab:blue", alpha=0.2,
                     label=f"8 in 10 past cases: {forecast['low']:+.0%} to {forecast['high']:+.0%}")
    top.plot(ahead, path_to(forecast["expected"]), color="tab:blue", linestyle="--",
             label=f"Average past case: {forecast['expected']:+.1%}")
    top.set_title(f"{TICKER}: prediction for the next {days} trading days from news tone "
                  f"({GROUPS[forecast['group']]}), rose in {forecast['chance up']:.0%} of similar past cases")
    top.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    top.legend(loc="upper left")
    top.grid(alpha=0.3)

    smooth = tone.rolling(TONE_DAYS, min_periods=TONE_DAYS // 2).mean()
    bottom.plot(tone.index, tone, color="lightgray", linewidth=0.6, label="Daily news tone")
    bottom.plot(smooth.index, smooth, color="tab:red", label=f"{TONE_DAYS}-day average (the signal)")
    bottom.axhline(0, color="black", linewidth=0.5)
    bottom.set_ylabel("Tone (above 0 = positive wording)")
    bottom.legend(loc="lower left")
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Nvidia's share price move from news tone.")
    parser.add_argument("--days", type=int, nargs="+", default=[20, 5],
                        help="trading days ahead to predict; the first one is charted (20 is about a month)")
    parser.add_argument("--chart", default=str(FOLDER / "nvidia_prediction.png"))
    parser.add_argument("--report", default=str(FOLDER / "nvidia_prediction.md"))
    args = parser.parse_args()
    if min(args.days) < 1:
        raise SystemExit("--days must be at least 1.")

    print("Loading prices and news...")
    prices, tone = load_prices(), load_tone()

    results, tables = [], {}
    for days in args.days:
        table = build_table(prices, tone, days)
        forecast = predict(table, len(table) - 1, days)
        if forecast is None:
            raise SystemExit("Not enough history to make a prediction.")
        results.append((days, forecast, score_check(check_method(table, days))))
        tables[days] = table

    first_days, first, _ = results[0]
    table = tables[first_days]
    price, day, signal = table["price"].iloc[-1], table.index[-1], table["signal"].iloc[-1]
    print(f"\n{TICKER} at ${price:,.2f} on {day:%Y-%m-%d}")
    print(f"News tone, last {TONE_DAYS} days: {signal:+.2f} "
          f"({GROUPS[first['group']]}: more positive than {first['share of history']:.0%} of past days)")
    for days, forecast, score in results:
        print_prediction(forecast, score, price, days)

    headlines = load_headlines()
    if headlines:
        print("\nLatest headlines (shown for context; the prediction uses the tone data above):")
        for when, title in headlines:
            print(f"  {when}  {title}")

    save_chart(table, tone, first, first_days, Path(args.chart))
    write_report(Path(args.report), day, price, signal, tone, results, headlines)
    print(f"\nChart saved to {args.chart}")
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
