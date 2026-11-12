"""Scan bond markets: yield, recent returns and risk, saved as a PNG.

Each bond market is an ETF. For every fund the scan shows:
    - yield: the income it pays per year, as reported by Yahoo Finance
    - returns over 1, 3, 6 and 12 months, interest included
    - score: the average of the 3-, 6- and 12-month returns (recent strength)
    - volatility: how much the price moves, a measure of risk
    - whether the price is above its 200-day average

Funds are ranked by score. The chart also shows US Treasury yields by
maturity (the yield curve) now, 3 months ago and 12 months ago.
Recent strength describes the past; it is not a forecast.

Usage:
    python bond_scan.py
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

BONDS = {
    "BIL": "US Treasury bills (1-3 months)",
    "SHY": "US Treasuries, 1-3 years",
    "IEI": "US Treasuries, 3-7 years",
    "IEF": "US Treasuries, 7-10 years",
    "TLT": "US Treasuries, 20+ years",
    "GOVT": "US Treasuries, all maturities",
    "TIP": "US inflation-linked Treasuries",
    "AGG": "US bond market (total)",
    "MBB": "US mortgage bonds",
    "MUB": "US municipal bonds",
    "VCSH": "US corporate, short-term",
    "LQD": "US corporate, investment grade",
    "HYG": "US corporate, high yield",
    "FLOT": "US floating-rate bonds",
    "BKLN": "US senior loans",
    "CWB": "US convertible bonds",
    "BNDX": "International bonds (USD-hedged)",
    "EMB": "Emerging-market bonds (USD)",
    "EMLC": "Emerging-market bonds (local currency)",
}
# Yahoo symbols for US Treasury yields, in percent a year.
TREASURY_YIELDS = {"3 months": "^IRX", "5 years": "^FVX", "10 years": "^TNX", "30 years": "^TYX"}
PERIODS = {"1 month": 21, "3 months": 63, "6 months": 126, "12 months": 252}  # trading days
SCORE_PERIODS = ["3 months", "6 months", "12 months"]
TREND_DAYS = 200
VOL_DAYS = 60
RETURN_COLOR_LIMIT = 0.10  # returns beyond +/-10% get the strongest colour


def load_yields(tickers) -> pd.Series:
    """Yearly income as a share of price; missing where Yahoo has none."""
    found = {}
    for ticker in tickers:
        try:
            found[ticker] = yf.Ticker(ticker).info.get("yield")
        except Exception:
            found[ticker] = None  # treated as not available
    return pd.to_numeric(pd.Series(found), errors="coerce")


def scan() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Bond funds ranked by score, and the Treasury yield history."""
    symbols = list(BONDS) + list(TREASURY_YIELDS.values())
    data = yf.download(symbols, period="2y", auto_adjust=True, progress=False)["Close"].ffill()
    usable = [t for t in BONDS if t in data and data[t].count() > max(PERIODS.values())]
    skipped = sorted(set(BONDS) - set(usable))
    if skipped:
        print(f"Skipping funds without a year of prices: {', '.join(skipped)}")
    if not usable:
        raise SystemExit("No price data returned.")
    prices = data[usable]

    latest = prices.iloc[-1]
    table = pd.DataFrame({"Yield": load_yields(usable)})
    for name, days in PERIODS.items():
        table[name] = latest / prices.iloc[-1 - days] - 1
    table["Score"] = table[SCORE_PERIODS].mean(axis=1)
    table["Volatility"] = prices.pct_change().iloc[-VOL_DAYS:].std() * np.sqrt(252)
    table["uptrend"] = latest > prices.rolling(TREND_DAYS).mean().iloc[-1]
    table["label"] = [f"{BONDS[t]} ({t})" for t in table.index]

    curve = pd.DataFrame({name: data[symbol] for name, symbol in TREASURY_YIELDS.items() if symbol in data})
    return table.sort_values("Score", ascending=False), curve.dropna(how="all"), prices.index[-1]


def save_chart(table: pd.DataFrame, curve: pd.DataFrame, day: pd.Timestamp, path: Path) -> None:
    columns = ["Yield"] + list(PERIODS) + ["Score", "Volatility"]
    rows = len(table)
    fig, (grid, lines) = plt.subplots(
        2, 1, figsize=(12.5, 0.36 * rows + 6.5), gridspec_kw={"height_ratios": [0.36 * rows + 1, 4]}
    )

    # Each kind of column gets its own colour scale.
    def cell_color(column: str, value: float):
        if np.isnan(value):
            return (0.93, 0.93, 0.93, 1.0)
        if column == "Yield":
            return plt.cm.Blues(0.1 + 0.6 * min(value / 0.08, 1))
        if column == "Volatility":
            return plt.cm.Oranges(0.05 + 0.6 * min(value / 0.15, 1))
        return plt.cm.RdYlGn(0.5 + 0.5 * np.clip(value / RETURN_COLOR_LIMIT, -1, 1))

    colors = np.array([[cell_color(c, table[c].iloc[r]) for c in columns] for r in range(rows)])
    grid.imshow(colors, aspect="auto")
    for row in range(rows):
        for col, column in enumerate(columns):
            value = table[column].iloc[row]
            form = ".1%" if column in ("Yield", "Volatility") else "+.1%"
            text = "n/a" if np.isnan(value) else f"{value:{form}}"
            strong = column not in ("Yield", "Volatility") and abs(value) > 0.75 * RETURN_COLOR_LIMIT
            grid.text(col, row, text, ha="center", va="center", fontsize=9,
                      fontweight="bold" if column == "Score" else "normal",
                      color="white" if strong else "black")
        trend = "up" if table["uptrend"].iloc[row] else "down"
        grid.text(len(columns) - 0.35, row, trend, va="center", fontsize=9,
                  color="darkgreen" if trend == "up" else "firebrick")
    grid.text(len(columns) - 0.35, -0.75, f"{TREND_DAYS}-day\ntrend", va="center", fontsize=9)
    headers = {"Yield": "Yield\n(income a year)", "Score": "Score\n(avg of 3, 6, 12)",
               "Volatility": "Volatility\n(risk, a year)"}
    grid.set_xticks(range(len(columns)), [headers.get(c, c) for c in columns])
    grid.xaxis.tick_top()
    grid.set_yticks(range(rows), [f"{rank}. {label}" for rank, label in enumerate(table["label"], start=1)])
    grid.tick_params(length=0)
    for x in (0.5, 4.5, 5.5):
        grid.axvline(x, color="white", linewidth=3)
    for spine in grid.spines.values():
        spine.set_visible(False)

    if not curve.empty:
        snapshots = {"Now": -1, "3 months ago": -1 - PERIODS["3 months"], "12 months ago": -1 - PERIODS["12 months"]}
        styles = {"Now": ("black", "-", 2.2), "3 months ago": ("tab:blue", "--", 1.4),
                  "12 months ago": ("gray", ":", 1.4)}
        for name, position in snapshots.items():
            if len(curve) >= -position:
                color, style, width = styles[name]
                values = curve.iloc[position]
                lines.plot(curve.columns, values, marker="o", color=color, linestyle=style, linewidth=width,
                           label=f"{name}: " + ", ".join(f"{v:.2f}%" for v in values))
        lines.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f%%"))
        lines.set_title("US Treasury yields by maturity (the yield curve)", fontsize=11)
        lines.set_xlabel("Maturity")
        lines.legend(loc="best", fontsize=9)
        lines.grid(alpha=0.3)
    else:
        lines.axis("off")

    fig.suptitle(f"Bond markets ranked by recent strength, prices to {day:%d %B %Y}", fontsize=13, y=0.995)
    fig.text(0.5, 0.008, "Returns include interest. Yield as reported by Yahoo Finance. "
             "Recent strength describes the past; it is not a forecast.", ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.02, 0.97, 0.985))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan bond markets and save a PNG.")
    parser.add_argument("--chart", default=str(FOLDER / "bond_scan.png"))
    args = parser.parse_args()

    table, curve, day = scan()
    print(f"\nBond markets ranked by recent strength, prices to {day:%Y-%m-%d}\n")
    shown = table[["label", "Yield"] + list(PERIODS) + ["Score", "Volatility"]].set_index("label")
    formats = {c: "{:+.1%}".format for c in shown.columns}
    formats["Yield"] = formats["Volatility"] = "{:.1%}".format
    print(shown.to_string(formatters=formats, index_names=False, na_rep="-"))
    if not curve.empty:
        print("\nUS Treasury yields now: " + ", ".join(f"{name} {value:.2f}%" for name, value in curve.iloc[-1].items()))
    save_chart(table, curve, day, Path(args.chart))
    print(f"\nChart saved to {args.chart}")


if __name__ == "__main__":
    main()
