"""Pick crypto coins to hold for the next two weeks, and test how the rule did.

Rule, applied every 14 days:
    1. Score each coin by the average of its 14-day and 28-day price change.
    2. If Bitcoin is below its 50-day average, hold nothing (cash): when the
       whole market is falling, most coins fall with it.
    3. Otherwise hold the 5 highest-scoring coins in equal amounts for 14 days.

The script prints today's ranking and picks, then replays the rule from 2019
with a trading cost, against holding Bitcoin and holding every coin equally.

Caution when reading the test: a list of today's large coins leaves out the
ones that collapsed, which flatters any replay. To reduce that, the test also
includes 16 coins that used to be large and have since fallen 90% or more
(FALLEN below); they are never recommended. Some failures, such as the
original Terra LUNA, have no price history available and are still missing.

Usage:
    python crypto_scan.py
    python crypto_scan.py --top 3 --cost-bps 30
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import yfinance as yf

FOLDER = Path(__file__).resolve().parent

# Coin symbol -> Yahoo Finance ticker (some carry a numeric id).
COINS = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD", "BNB": "BNB-USD",
    "DOGE": "DOGE-USD", "ADA": "ADA-USD", "TRX": "TRX-USD", "AVAX": "AVAX-USD", "LINK": "LINK-USD",
    "DOT": "DOT-USD", "LTC": "LTC-USD", "BCH": "BCH-USD", "XLM": "XLM-USD", "ATOM": "ATOM-USD",
    "UNI": "UNI7083-USD", "NEAR": "NEAR-USD", "ETC": "ETC-USD", "HBAR": "HBAR-USD", "AAVE": "AAVE-USD",
    "ALGO": "ALGO-USD", "FIL": "FIL-USD", "ICP": "ICP-USD", "XMR": "XMR-USD", "SHIB": "SHIB-USD",
    "SUI": "SUI20947-USD", "APT": "APT21794-USD", "ARB": "ARB11841-USD", "OP": "OP-USD",
    "HYPE": "HYPE32196-USD", "INJ": "INJ-USD", "RENDER": "RENDER-USD", "TAO": "TAO22974-USD",
    "ZEC": "ZEC-USD", "XTZ": "XTZ-USD", "VET": "VET-USD", "PEPE": "PEPE24478-USD", "KAS": "KAS-USD",
}
# Formerly large coins, used only in the test so that it includes failures.
FALLEN = {
    "FTT": "FTT-USD", "CEL": "CEL-USD", "WAVES": "WAVES-USD", "EOS": "EOS-USD", "NEO": "NEO-USD",
    "DASH": "DASH-USD", "XEM": "XEM-USD", "QTUM": "QTUM-USD", "OMG": "OMG-USD", "ZIL": "ZIL-USD",
    "BTG": "BTG-USD", "LSK": "LSK-USD", "BSV": "BSV-USD", "THETA": "THETA-USD", "IOTA": "IOTA-USD",
    "LUNA": "LUNA20314-USD",
}
MARKET = "BTC"
HOLD_DAYS = 14
SCORE_DAYS = [14, 28]
MARKET_TREND_DAYS = 50
VOL_DAYS = 28
SKIP_FIRST_DAYS = 30  # the first days of a new listing often contain bad prices
BROKEN_JUMP = 10  # a 10-fold move in a day means the ticker switched to another coin
HISTORY_NEEDED = 120  # days of clean prices before a coin can be picked
TEST_START = "2019-01-01"
RECENT_START = "2023-01-01"
PERIODS_PER_YEAR = 365 / HOLD_DAYS


def load_prices() -> pd.DataFrame:
    """Daily closing prices in US dollars, with unusable data removed."""
    tickers = COINS | FALLEN
    data = yf.download(list(tickers.values()), start="2017-01-01", auto_adjust=True, progress=False)["Close"]
    prices = pd.DataFrame({coin: data[ticker] for coin, ticker in tickers.items() if ticker in data})
    prices = prices.where(prices > 0)  # tiny prices are sometimes stored as zero
    for coin in prices:  # keep only the history after the last impossible jump
        change = prices[coin].dropna().pct_change() + 1
        broken = change[(change > BROKEN_JUMP) | (change < 1 / BROKEN_JUMP)]
        if len(broken):
            prices.loc[prices.index < broken.index[-1], coin] = np.nan
    listed = prices.notna().cumsum()
    prices = prices.where(listed > SKIP_FIRST_DAYS).ffill(limit=3)
    usable = [coin for coin in prices if prices[coin].count() > HISTORY_NEEDED and pd.notna(prices[coin].iloc[-1])]
    dropped = sorted(set(tickers) - set(usable))
    if dropped:
        print(f"Skipping coins without enough recent data: {', '.join(dropped)}")
    if MARKET not in usable:
        raise SystemExit(f"No price data for {MARKET}.")
    return prices[usable]


def scores(prices: pd.DataFrame) -> pd.DataFrame:
    """Average of the recent price changes; missing until a coin has enough history."""
    score = sum(prices.pct_change(days, fill_method=None) for days in SCORE_DAYS) / len(SCORE_DAYS)
    return score.where(prices.notna().cumsum() > HISTORY_NEEDED)


def market_rising(prices: pd.DataFrame) -> pd.Series:
    return prices[MARKET] > prices[MARKET].rolling(MARKET_TREND_DAYS).mean()


def backtest(prices: pd.DataFrame, top: int, cost_bps: float, offset: int = 0) -> pd.DataFrame:
    """Returns of each 14-day holding period for the rule and the comparisons.

    Picks use the closing price of the first day of the period; the return is
    measured to the close 14 days later.
    """
    score, rising = scores(prices), market_rising(prices)
    ahead = prices.shift(-HOLD_DAYS) / prices - 1
    days = prices.index[prices.index >= TEST_START][offset::HOLD_DAYS]
    days = [d for d in days if d + pd.Timedelta(days=HOLD_DAYS) <= prices.index[-1]]

    rows, held = [], set()
    for day in days:
        ranked = score.loc[day].dropna()
        picks = set(ranked.nlargest(top).index) if rising.loc[day] else set()
        traded = len(picks ^ held) / top  # share of the money bought plus share sold
        rows.append({
            "date": day,
            "Strategy": (ahead.loc[day, list(picks)].mean() if picks else 0.0) - traded * cost_bps / 10_000,
            "Bitcoin": ahead.loc[day, MARKET],
            "All coins equally": ahead.loc[day, ranked.index].mean(),
            "in market": bool(picks),
        })
        held = picks
    return pd.DataFrame(rows).set_index("date")


def summarize(returns: pd.Series) -> dict[str, float]:
    equity = (1 + returns).cumprod()
    return {
        "Annual return": equity.iloc[-1] ** (PERIODS_PER_YEAR / len(returns)) - 1,
        "Sharpe ratio": returns.mean() / returns.std() * np.sqrt(PERIODS_PER_YEAR),
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Worst 2 weeks": returns.min(),
        "Winning periods": (returns > 0).mean(),
    }


def print_test(results: pd.DataFrame, title: str) -> None:
    names = ["Strategy", "Bitcoin", "All coins equally"]
    stats = {name: summarize(results[name]) for name in names}
    invested = results["Strategy"][results["in market"]]
    stats["Strategy"]["Winning periods"] = (invested > 0).mean()  # cash periods are neither wins nor losses
    print(f"\n{title}: {results.index[0]:%Y-%m-%d} to {results.index[-1]:%Y-%m-%d}, "
          f"{len(results)} two-week periods, strategy invested in {results['in market'].mean():.0%} of them\n")
    print(f"  {'':<17}" + "".join(f"{name:>19}" for name in names))
    for metric in stats["Strategy"]:
        form = ".2f" if metric == "Sharpe ratio" else ".0%"
        print(f"  {metric:<17}" + "".join(f"{stats[name][metric]:>19{form}}" for name in names))


def save_chart(ranking: pd.DataFrame, picks: list[str], rising: bool, results: pd.DataFrame,
               day: pd.Timestamp, path: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.1, 1]})

    colors = ["seagreen" if coin in picks else ("lightgray" if not rising else "steelblue") for coin in ranking.index]
    top.bar(ranking.index, ranking["score"], color=colors)
    top.axhline(0, color="black", linewidth=0.6)
    top.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    top.tick_params(axis="x", rotation=60)
    top.set_ylabel("Score (avg of 14- and 28-day change)")
    verdict = (f"picks for the next {HOLD_DAYS} days: {', '.join(picks)}" if rising
               else f"Bitcoin is below its {MARKET_TREND_DAYS}-day average, so the rule holds cash")
    top.set_title(f"Crypto ranking on {day:%d %B %Y}: {verdict}")
    top.grid(axis="y", alpha=0.3)

    for name, color in [("Strategy", "seagreen"), ("Bitcoin", "darkorange"), ("All coins equally", "gray")]:
        equity = (1 + results[name]).cumprod()
        bottom.plot(equity.index, equity, label=f"{name}: \\${equity.iloc[-1]:,.0f} from \\$1", color=color,
                    linewidth=2 if name == "Strategy" else 1.2)
    bottom.set_yscale("log")
    bottom.set_ylabel("Value of \\$1 (log scale)")
    bottom.set_title("Replay of the rule since 2019, including 16 coins that have since collapsed", fontsize=10)
    bottom.legend(loc="upper left")
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank crypto coins for the next two weeks and test the rule.")
    parser.add_argument("--top", type=int, default=5, help="number of coins to hold")
    parser.add_argument("--cost-bps", type=float, default=20.0, help="cost per trade in basis points (20 = 0.2%%)")
    parser.add_argument("--chart", default=str(FOLDER / "crypto_scan.png"))
    args = parser.parse_args()
    if args.top < 1:
        raise SystemExit("--top must be at least 1.")

    prices = load_prices()
    day = prices.index[-1]
    score = scores(prices).iloc[-1].dropna().sort_values(ascending=False)
    score = score[[coin for coin in score.index if coin in COINS]]  # fallen coins are for the test only
    rising = bool(market_rising(prices).iloc[-1])
    swing = prices.pct_change(fill_method=None).iloc[-VOL_DAYS:].std() * np.sqrt(HOLD_DAYS)
    ranking = pd.DataFrame({
        "price": prices.iloc[-1][score.index],
        "14 days": prices.pct_change(14, fill_method=None).iloc[-1][score.index],
        "28 days": prices.pct_change(28, fill_method=None).iloc[-1][score.index],
        "score": score,
        "swing": swing[score.index],
    })
    picks = list(ranking.index[: args.top]) if rising else []

    average = prices[MARKET].rolling(MARKET_TREND_DAYS).mean().iloc[-1]
    print(f"\nCrypto ranking, prices to {day:%Y-%m-%d} (today's price may still be moving)")
    print(f"Bitcoin ${prices[MARKET].iloc[-1]:,.0f} is {'ABOVE' if rising else 'BELOW'} "
          f"its {MARKET_TREND_DAYS}-day average of ${average:,.0f}\n")
    print(f"  {'':<4}{'Coin':<8}{'Price':>14}{'14 days':>10}{'28 days':>10}{'Score':>9}{'Usual 2-week swing':>21}")
    for rank, (coin, row) in enumerate(ranking.iterrows(), start=1):
        mark = "  <-- pick" if coin in picks else ""
        print(f"  {rank:<4}{coin:<8}{row['price']:>14,.6g}{row['14 days']:>+10.1%}{row['28 days']:>+10.1%}"
              f"{row['score']:>+9.1%}{row['swing']:>20.0%} {mark}")

    if rising:
        print(f"\nRULE SAYS: hold {', '.join(picks)} in equal amounts for the next {HOLD_DAYS} days, then re-run.")
    else:
        print(f"\nRULE SAYS: hold cash for the next {HOLD_DAYS} days. Bitcoin is below its "
              f"{MARKET_TREND_DAYS}-day average, so no coins are picked.")
        print(f"If the market were rising, the picks would be: {', '.join(ranking.index[: args.top])}")

    results = backtest(prices, args.top, args.cost_bps)
    print_test(results, "TEST, whole period")
    print_test(results[results.index >= RECENT_START], "TEST, recent years")

    # The result should not hinge on which day the two-week cycle happens to start.
    annual = [summarize(backtest(prices, args.top, args.cost_bps, offset)["Strategy"])["Annual return"]
              for offset in range(HOLD_DAYS)]
    print(f"\n  Starting the two-week cycle on other days, the strategy's annual return ranges "
          f"from {min(annual):.0%} to {max(annual):.0%} (middle: {np.median(annual):.0%}).")

    save_chart(ranking, picks, rising, results, day, Path(args.chart))
    print(f"\nChart saved to {args.chart}")


if __name__ == "__main__":
    main()
