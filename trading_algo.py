"""Moving-average crossover trading strategy, trained by walk-forward optimisation.

Rule: hold the asset while the fast moving average is above the slow one,
otherwise stay in cash.

Training: for every calendar year, the fast/slow lengths are chosen as the pair
with the best Sharpe ratio over the preceding --train-years years, then traded
for that year. Every year in the report is therefore traded with settings that
were picked without seeing that year's prices.

The trained strategy is compared with a fixed crossover (--fast/--slow) and
with buying and holding, all over the same period.

Usage:
    python trading_algo.py                      # SPY, all available history
    python trading_algo.py AAPL --train-years 10
    python trading_algo.py ^GSPC                # S&P 500 index, back to 1927
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # write the chart to a file; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252
FAST_CHOICES = [5, 10, 20, 30, 50, 75, 100]
SLOW_CHOICES = [50, 100, 150, 200, 250, 300]


def load_prices(ticker: str, start: str, end: str | None) -> pd.Series:
    """Daily closing prices, adjusted for splits and dividends."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise SystemExit(f"No price data returned for {ticker!r}.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def crossover_position(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """1 when holding, 0 when in cash.

    The signal is known at today's close, so it can only be traded tomorrow.
    The shift is what keeps the backtest from using future information.
    """
    signal = prices.rolling(fast).mean() > prices.rolling(slow).mean()
    return signal.astype(float).shift(1).fillna(0.0)


def strategy_returns(position: pd.Series, market: pd.Series, cost_bps: float) -> pd.Series:
    """Daily returns of a position series, net of trading costs."""
    trades = position.diff().abs()
    trades.iloc[0] = position.iloc[0]  # entering on the first day is a trade too
    return position * market - trades * cost_bps / 10_000


def sharpe(returns: pd.Series) -> float:
    volatility = returns.std()
    if not volatility > 0:
        return float("-inf")  # never traded in this window
    return returns.mean() / volatility * np.sqrt(TRADING_DAYS)


def best_pair(pair_returns: dict, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    """The fast/slow pair with the highest Sharpe ratio in [start, end)."""
    scores = {
        pair: sharpe(returns[(returns.index >= start) & (returns.index < end)])
        for pair, returns in pair_returns.items()
    }
    return max(scores, key=scores.get)


def walk_forward(prices: pd.Series, train_years: int, cost_bps: float):
    """Pick parameters on past data year by year.

    Returns the stitched out-of-sample position, the pair used in each year,
    and the pair trained on the most recent window (for trading from here on).
    """
    market = prices.pct_change().fillna(0.0)
    pairs = [(f, s) for f in FAST_CHOICES for s in SLOW_CHOICES if f < s]
    positions = {pair: crossover_position(prices, *pair) for pair in pairs}
    pair_returns = {pair: strategy_returns(positions[pair], market, cost_bps) for pair in pairs}

    # Training may only use days on which every pair has a slow average.
    first_usable = prices.index[max(SLOW_CHOICES)]
    first_year = first_usable.year + 1 + train_years
    last_year = prices.index[-1].year
    if first_year > last_year:
        raise SystemExit(
            f"Not enough history to train on {train_years} years; "
            "use a smaller --train-years or an earlier --start."
        )

    position = pd.Series(np.nan, index=prices.index)
    chosen = {}
    for year in range(first_year, last_year + 1):
        train_start = pd.Timestamp(year - train_years, 1, 1)
        train_end = pd.Timestamp(year, 1, 1)
        pair = best_pair(pair_returns, train_start, train_end)
        chosen[year] = pair
        in_year = prices.index.year == year
        position[in_year] = positions[pair][in_year]

    latest = prices.index[-1] + pd.Timedelta(days=1)
    next_pair = best_pair(pair_returns, latest - pd.DateOffset(years=train_years), latest)
    return position.dropna(), chosen, next_pair


def summarize(returns: pd.Series, position: pd.Series) -> dict[str, float]:
    """Performance figures for a series of daily returns."""
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    return {
        "Total return": equity.iloc[-1] - 1,
        "Annual return": equity.iloc[-1] ** (1 / years) - 1,
        "Annual volatility": returns.std() * np.sqrt(TRADING_DAYS),
        "Sharpe ratio": sharpe(returns),
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Trades": position.diff().abs().sum(),
        "Time in market": position.mean(),
    }


def print_report(results: dict[str, dict], chosen: dict, next_pair: tuple[int, int]) -> None:
    names = list(results)
    print(f"{'':<20}" + "".join(f"{name:>14}" for name in names))
    for metric in results[names[0]]:
        row = f"{metric:<20}"
        for name in names:
            value = results[name][metric]
            if metric == "Sharpe ratio":
                row += f"{value:>14.2f}"
            elif metric == "Trades":
                row += f"{int(value):>14}"
            else:
                row += f"{value:>14.1%}"
        print(row)

    print("\nSettings chosen by training (fast/slow days):")
    years = sorted(chosen)
    run_start = years[0]
    for year, following in zip(years, years[1:] + [None]):
        if following is None or chosen[following] != chosen[year]:
            span = str(year) if run_start == year else f"{run_start}-{year}"
            print(f"  {span:<11}{chosen[year][0]}/{chosen[year][1]}")
            run_start = following
    print(f"\nSettings trained on the latest data, for trading from here: {next_pair[0]}/{next_pair[1]}")


def save_chart(prices: pd.Series, position: pd.Series, curves: dict[str, pd.Series], title: str, path: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    top.plot(prices.index, prices, color="black", linewidth=1, label="Price")
    top.fill_between(
        prices.index, 0, 1, where=position > 0, transform=top.get_xaxis_transform(),
        color="green", alpha=0.12, label="Trained strategy holding",
    )
    top.set_yscale("log")
    top.set_title(title)
    top.set_ylabel("Price (log scale)")
    top.legend(loc="upper left")

    for name, equity in curves.items():
        bottom.plot(equity.index, equity, label=name)
    bottom.set_yscale("log")
    bottom.set_ylabel("Growth of 1 (log scale)")
    bottom.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and backtest a moving-average crossover strategy.")
    parser.add_argument("ticker", nargs="?", default="SPY")
    parser.add_argument("--start", default="1900-01-01", help="default: all available history")
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-years", type=int, default=5, help="years of history used to pick settings")
    parser.add_argument("--fast", type=int, default=50, help="fast length of the fixed comparison strategy")
    parser.add_argument("--slow", type=int, default=200, help="slow length of the fixed comparison strategy")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per trade in basis points")
    parser.add_argument("--chart", default="backtest.png", help="where to save the chart")
    args = parser.parse_args()

    if args.fast >= args.slow:
        raise SystemExit("--fast must be smaller than --slow.")
    if args.train_years < 1:
        raise SystemExit("--train-years must be at least 1.")

    prices = load_prices(args.ticker, args.start, args.end)
    if len(prices) <= max(SLOW_CHOICES):
        raise SystemExit(f"Only {len(prices)} days of data; need more than {max(SLOW_CHOICES)}.")

    trained_position, chosen, next_pair = walk_forward(prices, args.train_years, args.cost_bps)

    # Compare everything over the period the trained strategy was actually traded.
    period = trained_position.index
    market = prices.pct_change().fillna(0.0)[period]
    fixed_position = crossover_position(prices, args.fast, args.slow)[period]
    always_in = pd.Series(1.0, index=period)
    fixed_name = f"Fixed {args.fast}/{args.slow}"

    positions = {"Trained": trained_position, fixed_name: fixed_position, "Buy & hold": always_in}
    returns = {
        "Trained": strategy_returns(trained_position, market, args.cost_bps),
        fixed_name: strategy_returns(fixed_position, market, args.cost_bps),
        "Buy & hold": market,
    }

    print(f"\n{args.ticker}: {period[0]:%Y-%m-%d} to {period[-1]:%Y-%m-%d} "
          f"({len(period) / TRADING_DAYS:.0f} years), trained on the prior {args.train_years} years\n")
    print_report({name: summarize(returns[name], positions[name]) for name in returns}, chosen, next_pair)

    curves = {name: (1 + r).cumprod() for name, r in returns.items()}
    title = f"{args.ticker}: crossover retrained yearly on the prior {args.train_years} years"
    save_chart(prices[period], trained_position, curves, title, args.chart)
    print(f"Chart saved to {args.chart}")


if __name__ == "__main__":
    main()
