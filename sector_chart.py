"""Chart each US stock-market sector over the past ten years.

One panel per sector, showing how $1 invested grew (dividends included)
against the S&P 500, plus a panel ranking the sectors by yearly return.
Each sector is its SPDR sector ETF. A sector whose fund is younger than the
period is drawn from the fund's first day, with the S&P 500 restarted there.

Usage:
    python sector_chart.py
    python sector_chart.py --years 5
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import yfinance as yf

FOLDER = Path(__file__).resolve().parent

SECTORS = {
    "XLK": "Technology", "XLC": "Communication services", "XLY": "Consumer discretionary",
    "XLF": "Financials", "XLI": "Industrials", "XLV": "Healthcare", "XLP": "Consumer staples",
    "XLE": "Energy", "XLU": "Utilities", "XLB": "Materials", "XLRE": "Property",
}
MARKET = "SPY"
DAYS_PER_YEAR = 365.25


def yearly_return(prices: pd.Series) -> float:
    years = (prices.index[-1] - prices.index[0]).days / DAYS_PER_YEAR
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart each US sector over the past years.")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--chart", default=str(FOLDER / "sector_chart.png"))
    args = parser.parse_args()
    if args.years < 1:
        raise SystemExit("--years must be at least 1.")

    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)
    data = yf.download(list(SECTORS) + [MARKET], start=start, auto_adjust=True, progress=False)["Close"]
    missing = [t for t in list(SECTORS) + [MARKET] if t not in data or data[t].dropna().empty]
    if missing:
        raise SystemExit(f"No price data returned for: {', '.join(missing)}")

    # Each sector over its own available history, with the market over the same days.
    series = {}
    for ticker in SECTORS:
        prices = data[ticker].dropna()
        series[ticker] = (prices, data[MARKET].dropna().loc[prices.index[0]:])
    order = sorted(SECTORS, key=lambda t: yearly_return(series[t][0]), reverse=True)
    full_start = data[MARKET].dropna().index[0]
    market_yearly = yearly_return(data[MARKET].dropna())

    print(f"\nUS sectors, {full_start:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d}\n")
    print(f"  {'Sector':<32}{'A year':>8}{'$1 became':>11}{'S&P 500 a year':>17}")
    for ticker in order:
        prices, market = series[ticker]
        note = "" if prices.index[0] <= full_start + pd.Timedelta(days=7) else f"  (since {prices.index[0]:%b %Y})"
        print(f"  {SECTORS[ticker] + ' (' + ticker + ')':<32}{yearly_return(prices):>+8.1%}"
              f"{prices.iloc[-1] / prices.iloc[0]:>11.2f}{yearly_return(market):>+17.1%}{note}")

    fig, axes = plt.subplots(3, 4, figsize=(19, 12), sharey=True)
    panels = axes.flatten()
    for panel, ticker in zip(panels, order):
        prices, market = series[ticker]
        ahead = yearly_return(prices) >= yearly_return(market)
        color = "seagreen" if ahead else "firebrick"
        panel.plot(market.index, market / market.iloc[0], color="gray", linewidth=1, label="S&P 500")
        panel.plot(prices.index, prices / prices.iloc[0], color=color, linewidth=1.8, label=SECTORS[ticker])
        late = prices.index[0] > full_start + pd.Timedelta(days=7)
        panel.set_title(f"{SECTORS[ticker]} ({ticker})\n{yearly_return(prices):+.1%} a year, "
                        f"\\$1 became \\${prices.iloc[-1] / prices.iloc[0]:.2f}"
                        + (f" (since {prices.index[0]:%b %Y})" if late else ""), fontsize=10)
        panel.set_xlim(full_start, data.index[-1])
        panel.xaxis.set_major_locator(matplotlib.dates.YearLocator(max(1, args.years // 5)))
        panel.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        panel.set_yscale("log")
        panel.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
        panel.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        panel.set_yticks([0.5, 1, 2, 4, 8])
        panel.grid(alpha=0.3)
        panel.legend(loc="upper left", fontsize=8)
        panel.tick_params(labelleft=True)

    # Last panel: every sector ranked by yearly return.
    ranking = panels[len(order)]
    ranking.remove()
    ranking = fig.add_subplot(3, 4, len(order) + 1)
    yearly = [yearly_return(series[t][0]) for t in order]
    bars = ranking.barh([SECTORS[t] for t in order], yearly,
                        color=["seagreen" if y >= market_yearly else "firebrick" for y in yearly])
    ranking.bar_label(bars, labels=[f"{y:+.1%}" for y in yearly], padding=3, fontsize=8)
    ranking.axvline(market_yearly, color="gray", linestyle="--", linewidth=1)
    ranking.text(market_yearly, len(order) - 0.4, f" S&P 500 {market_yearly:+.1%}", color="gray", fontsize=8)
    ranking.invert_yaxis()
    ranking.set_xlim(0, max(yearly) * 1.25)
    ranking.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ranking.tick_params(axis="y", labelsize=8, length=0)
    ranking.set_title("Return a year, all sectors", fontsize=10)
    for side in ("top", "right"):
        ranking.spines[side].set_visible(False)

    fig.suptitle(f"US stock-market sectors over the past {args.years} years, "
                 f"{full_start:%B %Y} to {data.index[-1]:%B %Y} (green = beat the S&P 500, red = trailed it)",
                 fontsize=14)
    fig.text(0.5, 0.008, "Growth of $1 with dividends reinvested, log scale. Past returns are not a forecast.".replace("$", "\\$"),
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(args.chart, dpi=120)
    print(f"\nChart saved to {args.chart}")


if __name__ == "__main__":
    main()
