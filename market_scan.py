"""Rank markets by recent strength and by price/earnings ratio, as PNGs.

Each market is an ETF. The score is the average of its 3-, 6- and 12-month
returns (the measure the "Leaders" strategy in combined_algo.py uses).
Strength is a description of the recent past, not a forecast.

The P/E ratio is the price of the fund's shares divided by their last 12
months of earnings, as reported by Yahoo Finance. It only exists for stock
markets; bonds, commodities and bitcoin have no earnings.

Saves market_scan.png (every market) and market_top10.png (the 10 strongest,
and the 10 with the lowest P/E).

Usage:
    python market_scan.py
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

MARKETS = {
    "SPY": "US large companies", "QQQ": "US technology (Nasdaq 100)", "IWM": "US small companies",
    "EFA": "Developed markets ex-US", "EEM": "Emerging markets", "VNQ": "US property",
    "TLT": "Long US Treasuries", "IEF": "Medium US Treasuries", "LQD": "US corporate bonds",
    "HYG": "US high-yield bonds", "GLD": "Gold", "SLV": "Silver", "DBC": "Commodities (broad)",
    "USO": "Oil", "IBIT": "Bitcoin", "EWJ": "Japan", "VGK": "Europe", "FXI": "China",
    "INDA": "India", "EWZ": "Brazil", "EWT": "Taiwan", "EWY": "South Korea",
    "XLK": "US tech sector", "XLF": "US financials", "XLE": "US energy", "XLV": "US healthcare",
    "XLI": "US industrials", "XLU": "US utilities", "XLP": "US consumer staples",
    "XLY": "US consumer discretionary", "SMH": "Semiconductors",
}
PERIODS = {"1 month": 21, "3 months": 63, "6 months": 126, "12 months": 252}  # trading days
SCORE_PERIODS = ["3 months", "6 months", "12 months"]
TREND_DAYS = 200
COLOR_LIMIT = 0.30  # returns beyond +/-30% get the strongest colour
TOP = 10


def load_pe(tickers) -> pd.Series:
    """Trailing P/E ratio per market; missing where there is none."""
    ratios = {}
    for ticker in tickers:
        try:
            ratios[ticker] = yf.Ticker(ticker).info.get("trailingPE")
        except Exception:
            ratios[ticker] = None  # treated as not available
    return pd.to_numeric(pd.Series(ratios), errors="coerce")


def scan() -> tuple[pd.DataFrame, pd.Timestamp]:
    """One row per market, strongest first."""
    prices = yf.download(list(MARKETS), period="2y", auto_adjust=True, progress=False)["Close"].ffill()
    missing = [t for t in MARKETS if t not in prices or prices[t].count() <= max(PERIODS.values())]
    if missing:
        print(f"Skipping markets without a year of prices: {', '.join(missing)}")
    prices = prices[[t for t in MARKETS if t not in missing]]
    if prices.empty:
        raise SystemExit("No price data returned.")

    latest = prices.iloc[-1]
    table = pd.DataFrame({name: latest / prices.iloc[-1 - days] - 1 for name, days in PERIODS.items()})
    table["Score"] = table[SCORE_PERIODS].mean(axis=1)
    table["Below 12-month high"] = latest / prices.iloc[-PERIODS["12 months"]:].max() - 1
    table["uptrend"] = latest > prices.rolling(TREND_DAYS).mean().iloc[-1]
    table["label"] = [f"{MARKETS[t]} ({t})" for t in table.index]
    table["P/E"] = load_pe(table.index)
    return table.sort_values("Score", ascending=False), prices.index[-1]


def save_chart(table: pd.DataFrame, day: pd.Timestamp, path: Path) -> None:
    columns = list(PERIODS) + ["Score", "Below 12-month high"]
    values = table[columns].to_numpy()
    rows = len(table)

    fig, ax = plt.subplots(figsize=(12, 0.36 * rows + 2.2))
    ax.imshow(values, cmap="RdYlGn", vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, aspect="auto")
    for row in range(rows):
        for col in range(len(columns)):
            weight = "bold" if columns[col] == "Score" else "normal"
            ink = "white" if abs(values[row, col]) > 0.75 * COLOR_LIMIT else "black"  # readable on dark cells
            ax.text(col, row, f"{values[row, col]:+.1%}", ha="center", va="center", fontsize=9,
                    fontweight=weight, color=ink)
        trend = "up" if table["uptrend"].iloc[row] else "down"
        ax.text(len(columns) - 0.35, row, trend, va="center", fontsize=9,
                color="darkgreen" if trend == "up" else "firebrick")
    ax.text(len(columns) - 0.35, -0.75, f"{TREND_DAYS}-day\ntrend", va="center", fontsize=9)

    headers = {"Score": "Score\n(avg of 3, 6, 12)", "Below 12-month high": "Below\n12-month high"}
    ax.set_xticks(range(len(columns)), [headers.get(c, c) for c in columns])
    ax.xaxis.tick_top()
    ax.set_yticks(range(rows), [f"{rank}. {label}" for rank, label in enumerate(table["label"], start=1)])
    ax.tick_params(length=0)
    ax.axvline(3.5, color="white", linewidth=3)
    ax.axvline(4.5, color="white", linewidth=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(f"Markets ranked by recent strength, prices to {day:%d %B %Y}", fontsize=13, y=0.995)
    fig.text(0.5, 0.012, "Returns include dividends. Recent strength describes the past; it is not a forecast.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.025, 0.97, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_top_chart(table: pd.DataFrame, day: pd.Timestamp, path: Path) -> None:
    """The strongest markets with their P/E, and the lowest-P/E markets with their score."""
    strongest = table.head(TOP)
    cheapest = table.dropna(subset=["P/E"]).sort_values("P/E").head(TOP)
    pe_text = lambda pe: "no P/E" if np.isnan(pe) else f"P/E {pe:.1f}"

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6.5))

    left.barh(range(len(strongest)), strongest["Score"], color="seagreen")
    for row, (score, pe) in enumerate(zip(strongest["Score"], strongest["P/E"])):
        left.text(score, row, f"  {score:+.1%}   {pe_text(pe)}", va="center", fontsize=10)
    left.set_yticks(range(len(strongest)), [f"{i}. {label}" for i, label in enumerate(strongest["label"], 1)])
    left.set_xlim(0, strongest["Score"].max() * 1.45)
    left.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    left.set_title(f"Top {TOP} by recent strength\n(score = average of 3-, 6- and 12-month returns)", fontsize=11)

    right.barh(range(len(cheapest)), cheapest["P/E"], color="steelblue")
    for row, (pe, score) in enumerate(zip(cheapest["P/E"], cheapest["Score"])):
        right.text(pe, row, f"  {pe:.1f}", va="center", fontsize=10)
        right.text(cheapest["P/E"].max() * 1.17, row, f"score {score:+.1%}", va="center", fontsize=10,
                   color="darkgreen" if score > 0 else "firebrick")
    right.set_yticks(range(len(cheapest)), [f"{i}. {label}" for i, label in enumerate(cheapest["label"], 1)])
    right.set_xlim(0, cheapest["P/E"].max() * 1.5)
    right.set_title(f"Top {TOP} by lowest P/E ratio\n(price divided by the last 12 months of earnings)", fontsize=11)

    for axis in (left, right):
        axis.invert_yaxis()
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", alpha=0.3)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.suptitle(f"Markets: strongest and cheapest, prices to {day:%d %B %Y}", fontsize=13)
    fig.text(0.5, 0.012, "P/E from Yahoo Finance; stock markets only. A low P/E means cheap relative to earnings, "
             "not necessarily a good investment.", ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank markets by recent strength and save a PNG.")
    parser.add_argument("--chart", default=str(FOLDER / "market_scan.png"))
    parser.add_argument("--top-chart", default=str(FOLDER / "market_top10.png"))
    args = parser.parse_args()

    table, day = scan()
    print(f"\nMarkets ranked by recent strength, prices to {day:%Y-%m-%d}\n")
    shown = table[["label"] + list(PERIODS) + ["Score", "P/E"]].set_index("label")
    formats = {c: "{:+.1%}".format for c in shown.columns}
    formats["P/E"] = "{:.1f}".format
    print(shown.to_string(formatters=formats, index_names=False, na_rep="-"))
    save_chart(table, day, Path(args.chart))
    save_top_chart(table, day, Path(args.top_chart))
    print(f"\nCharts saved to {args.chart} and {args.top_chart}")


if __name__ == "__main__":
    main()
