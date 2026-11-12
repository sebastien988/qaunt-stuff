"""Multi-strategy, multi-asset trading algorithm with a backtest.

Four strategies each decide how much of ten ETFs (stocks, bonds, gold,
commodities, property) to hold. Their allocations are averaged, and whatever
is not invested sits in cash.

    trend       several moving-average crossovers per asset, averaged
    momentum    is the asset up over the last 3, 6 and 12 months?
    leaders     hold the 3 strongest assets, if they are rising
    dip         buy sharp short-term drops in assets that are in an uptrend

The first three are slow and are only updated at each month end, which keeps
trading costs down; the dip strategy trades daily.

The signal settings are textbook values, fixed in advance; none were fitted to
the results. Signals from today's close are traded at tomorrow's close, every
trade pays a cost, and cash earns the Treasury-bill rate.

Usage:
    python combined_algo.py
    python combined_algo.py --cost-bps 10
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # write the chart to a file; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252

UNIVERSE = {
    "SPY": "US large companies",
    "QQQ": "US technology",
    "IWM": "US small companies",
    "EFA": "Developed markets ex-US",
    "EEM": "Emerging markets",
    "VNQ": "US property",
    "TLT": "Long US Treasuries",
    "IEF": "Medium US Treasuries",
    "GLD": "Gold",
    "DBC": "Commodities",
}
CASH_RATE_TICKER = "^IRX"  # 13-week Treasury bill yield, in percent a year

TREND_PAIRS = [(20, 100), (50, 200), (100, 300)]
MOMENTUM_LOOKBACKS = [63, 126, 252]  # about 3, 6 and 12 months
LEADERS_HELD = 3
DIP_RSI_DAYS = 2
DIP_ENTRY, DIP_EXIT = 10, 70
DIP_TREND_DAYS = 200
DIP_POSITION = 0.2  # share of the dip strategy given to each open position
VOL_WINDOW = 60
HISTORY_NEEDED = 300  # days of prices before an asset can be traded
MIN_ASSETS = 5  # the simulation starts once this many assets are tradable


def load_data(start: str, end: str | None) -> tuple[pd.DataFrame, pd.Series]:
    """Adjusted daily closes for the universe, and the daily return on cash."""
    tickers = list(UNIVERSE) + [CASH_RATE_TICKER]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    missing = [t for t in tickers if t not in data or data[t].dropna().empty]
    if missing:
        raise SystemExit(f"No price data returned for: {', '.join(missing)}")

    prices = data[list(UNIVERSE)].dropna(how="all")
    prices = prices.ffill()  # a missing day after an asset starts trading keeps the last price
    cash = (data[CASH_RATE_TICKER].reindex(prices.index).ffill().fillna(0.0) / 100 / TRADING_DAYS)
    return prices, cash.clip(lower=0.0)


def inverse_volatility(returns: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """Weights that give calmer assets more money, summing to 1 across tradable assets."""
    inverse = (1 / returns.rolling(VOL_WINDOW).std()).where(tradable)
    return inverse.div(inverse.sum(axis=1), axis=0).fillna(0.0)


def trend_weights(prices: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Share of crossovers that are bullish, per asset."""
    bullish = sum(
        (prices.rolling(fast).mean() > prices.rolling(slow).mean()).astype(float)
        for fast, slow in TREND_PAIRS
    )
    return base * bullish / len(TREND_PAIRS)


def momentum_weights(prices: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Share of lookbacks over which the asset has risen."""
    rising = sum((prices.pct_change(days) > 0).astype(float) for days in MOMENTUM_LOOKBACKS)
    return base * rising / len(MOMENTUM_LOOKBACKS)


def leaders_weights(prices: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """Equal weight in the strongest assets, skipping any that are falling."""
    strength = sum(prices.pct_change(days) for days in MOMENTUM_LOOKBACKS) / len(MOMENTUM_LOOKBACKS)
    strength = strength.where(tradable)
    rank = strength.rank(axis=1, ascending=False)
    held = (rank <= LEADERS_HELD) & (strength > 0)
    return held.astype(float) / LEADERS_HELD


def rsi(prices: pd.DataFrame, days: int) -> pd.DataFrame:
    change = prices.diff()
    gains = change.clip(lower=0).ewm(alpha=1 / days, adjust=False).mean()
    losses = (-change.clip(upper=0)).ewm(alpha=1 / days, adjust=False).mean()
    return 100 - 100 / (1 + gains / losses)


def dip_weights(prices: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """Buy a sharp drop in an uptrend; sell once the price has bounced."""
    strength = rsi(prices, DIP_RSI_DAYS)
    uptrend = prices > prices.rolling(DIP_TREND_DAYS).mean()

    holding = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    holding[(strength < DIP_ENTRY) & uptrend & tradable] = 1.0
    holding[strength > DIP_EXIT] = 0.0
    holding = holding.ffill().fillna(0.0)  # keep the position between entry and exit

    weights = holding * DIP_POSITION
    total = weights.sum(axis=1)
    return weights.div(total.clip(lower=1.0), axis=0)  # never more than fully invested


def hold_through_month(weights: pd.DataFrame) -> pd.DataFrame:
    """Keep each month-end allocation until the next month end."""
    months = weights.index.to_period("M")
    month_end = pd.Series(months[:-1] != months[1:], index=weights.index[:-1])
    month_end = month_end.reindex(weights.index, fill_value=False)
    return weights.where(month_end, np.nan).ffill().fillna(0.0)


def build_strategies(prices: pd.DataFrame, returns: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Target weights of each strategy, and the days the simulation covers."""
    tradable = prices.notna().cumsum() > HISTORY_NEEDED
    base = inverse_volatility(returns, tradable)
    strategies = {
        "Trend": hold_through_month(trend_weights(prices, base)),
        "Momentum": hold_through_month(momentum_weights(prices, base)),
        "Leaders": hold_through_month(leaders_weights(prices, tradable)),
        "Dip": dip_weights(prices, tradable),
    }
    enough = tradable.sum(axis=1) >= MIN_ASSETS
    if not enough.any():
        raise SystemExit(f"Fewer than {MIN_ASSETS} assets have {HISTORY_NEEDED} days of history.")
    return strategies, prices.index[prices.index >= enough.idxmax()]


def simulate(target: pd.DataFrame, returns: pd.DataFrame, cash: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Daily results of rebalancing to the target weights at each close.

    The weights decided at one close are held over the following day. Turnover
    is measured against where yesterday's holdings have drifted to.
    """
    held = target.shift(1).fillna(0.0)
    in_cash = 1 - held.sum(axis=1)
    gross = (held * returns).sum(axis=1) + in_cash * cash
    drifted = (held * (1 + returns)).div(1 + gross, axis=0)
    turnover = (target - drifted).abs().sum(axis=1)
    return pd.DataFrame({
        "return": gross - turnover * cost_bps / 10_000,
        "turnover": turnover,
        "invested": held.sum(axis=1),
    })


def summarize(result: pd.DataFrame, cash: pd.Series) -> dict[str, float]:
    returns = result["return"]
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    excess = returns - cash
    by_year = (1 + returns).groupby(returns.index.year).prod() - 1
    return {
        "Total return": equity.iloc[-1] - 1,
        "Annual return": equity.iloc[-1] ** (1 / years) - 1,
        "Annual volatility": returns.std() * np.sqrt(TRADING_DAYS),
        "Sharpe ratio": excess.mean() / excess.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Worst year": by_year.min(),
        "Losing years": (by_year < 0).sum() / len(by_year),
        "Average invested": result["invested"].mean(),
        "Turnover per year": result["turnover"].sum() / years,
    }


def print_table(summaries: dict[str, dict], width: int = 13) -> None:
    names = list(summaries)
    print(f"{'':<19}" + "".join(f"{name:>{width}}" for name in names))
    for metric in summaries[names[0]]:
        row = f"{metric:<19}"
        for name in names:
            value = summaries[name][metric]
            if metric == "Sharpe ratio":
                row += f"{value:>{width}.2f}"
            elif metric == "Turnover per year":
                row += f"{value:>{width - 1}.1f}x"
            else:
                row += f"{value:>{width}.1%}"
        print(row)


def save_chart(results: dict[str, pd.DataFrame], weights: pd.DataFrame, path: str) -> None:
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(11, 11), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 2]}
    )
    for name, result in results.items():
        equity = (1 + result["return"]).cumprod()
        width = 2 if name == "Combined" else 1
        top.plot(equity.index, equity, label=name, linewidth=width)
        middle.plot(equity.index, equity / equity.cummax() - 1, label=name, linewidth=width)
    top.set_yscale("log")
    top.set_title("Combined algorithm against simple alternatives")
    top.set_ylabel("Growth of 1 (log scale)")
    top.legend(loc="upper left")
    middle.set_ylabel("Drawdown")
    middle.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    monthly = weights.resample("ME").mean()
    bottom.stackplot(monthly.index, monthly.T.values, labels=monthly.columns, linewidth=0)
    bottom.set_ylim(0, 1)
    bottom.set_ylabel("Combined allocation")
    bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bottom.legend(loc="upper left", ncol=5, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a combined multi-strategy ETF algorithm.")
    parser.add_argument("--start", default="1990-01-01", help="default: all available history")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit traded, in basis points")
    parser.add_argument("--chart", default="combined.png", help="where to save the chart")
    args = parser.parse_args()

    prices, cash = load_data(args.start, args.end)
    returns = prices.pct_change().fillna(0.0)
    strategies, period = build_strategies(prices, returns)
    combined = sum(strategies.values()) / len(strategies)
    tradable = prices.notna().cumsum() > HISTORY_NEEDED

    def run(weights: pd.DataFrame) -> pd.DataFrame:
        return simulate(weights.loc[period], returns.loc[period], cash[period], args.cost_bps)

    def fixed(allocation: dict[str, float]) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for ticker, weight in allocation.items():
            weights[ticker] = weight
        return weights

    equal = tradable.astype(float).div(tradable.sum(axis=1).clip(lower=1), axis=0)
    results = {
        "Combined": run(combined),
        "SPY": run(fixed({"SPY": 1.0})),
        "60/40": run(fixed({"SPY": 0.6, "IEF": 0.4})),
        "Equal weight": run(equal),
    }
    cash_period = cash[period]

    print(f"\n{period[0]:%Y-%m-%d} to {period[-1]:%Y-%m-%d} ({len(period) / TRADING_DAYS:.0f} years), "
          f"{args.cost_bps:g} bps per trade\n")
    print_table({name: summarize(result, cash_period) for name, result in results.items()})

    print("\nEach strategy on its own:\n")
    print_table({name: summarize(run(weights), cash_period) for name, weights in strategies.items()})

    today = combined.iloc[-1]
    print(f"\nAllocation to hold after the close of {prices.index[-1]:%Y-%m-%d}:")
    for ticker, weight in today.sort_values(ascending=False).items():
        if weight >= 0.005:
            print(f"  {ticker:<5}{weight:>6.1%}  {UNIVERSE[ticker]}")
    print(f"  {'Cash':<5}{1 - today.sum():>6.1%}")

    save_chart(results, combined.loc[period], args.chart)
    print(f"\nChart saved to {args.chart}")


if __name__ == "__main__":
    main()
