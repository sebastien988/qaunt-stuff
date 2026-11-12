"""Chart every algorithm in this folder against the S&P 500.

Draws the trained algorithm (trained_algo.py), the combined multi-strategy
algorithm (combined_algo.py) and the two moving-average crossovers
(trading_algo.py) over the same period, alongside buying and holding the
S&P 500 (SPY, dividends included). Everything pays the same trading cost and
earns the Treasury-bill rate on cash.

The trained algorithm's settings were fitted to the data left of the dashed
line, so only the part to the right of it is a fair test.

Usage:
    python compare_chart.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

import combined_algo as combined
import trading_algo as crossover
import trained_algo as trained

TRADING_DAYS = 252


def crossover_returns(position: pd.Series, market: pd.Series, cash: pd.Series, cost_bps: float) -> pd.Series:
    """Daily returns of a 0/1 position in the market, with interest on cash."""
    trades = position.diff().abs().fillna(0.0)
    return position * market + (1 - position) * cash - trades * cost_bps / 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart all algorithms against the S&P 500.")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--train-years", type=int, default=5, help="for the trained crossover")
    parser.add_argument("--train-until", default="2015-01-01", help="for the trained algorithm")
    parser.add_argument("--chart", default="comparison.png")
    args = parser.parse_args()

    prices, cash = combined.load_data("1990-01-01", None)
    returns = prices.pct_change().fillna(0.0)

    # Combined algorithm, built exactly as in combined_algo.py.
    strategies, period = combined.build_strategies(prices, returns)
    weights = sum(strategies.values()) / len(strategies)
    combined_returns = combined.simulate(
        weights.loc[period], returns.loc[period], cash[period], args.cost_bps
    )["return"]

    # Crossovers on SPY, built as in trading_algo.py.
    spy = prices["SPY"].dropna()
    trained_position, _, _ = crossover.walk_forward(spy, args.train_years, args.cost_bps)
    fixed_position = crossover.crossover_position(spy, 50, 200)
    market = returns["SPY"]

    # Trained algorithm: strategy mix and leverage fitted before the cutoff.
    _, trained_returns, _, _, _ = trained.train(prices, cash, args.cost_bps, args.train_until)
    cutoff = pd.Timestamp(args.train_until)
    test = period >= cutoff

    series = {
        "Trained algorithm": trained_returns,
        "Combined algorithm": combined_returns,
        "Trained crossover": crossover_returns(trained_position.reindex(period), market[period], cash[period], args.cost_bps),
        "Fixed 50/200 crossover": crossover_returns(fixed_position.reindex(period), market[period], cash[period], args.cost_bps),
        "S&P 500 (buy & hold)": market[period],
    }
    if any(r.isna().any() for r in series.values()):
        raise SystemExit("The crossover strategies do not cover the whole period.")

    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "black"]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
    )
    print(f"\n{period[0]:%Y-%m-%d} to {period[-1]:%Y-%m-%d}\n")
    print(f"{'':<24}{'Growth of $1':>13}{'Annual':>9}{'Sharpe':>8}{'Max drawdown':>14}{'Annual since ' + str(cutoff.year):>20}")
    since = {}
    for (name, daily), color in zip(series.items(), colors):
        equity = (1 + daily).cumprod()
        drawdown = equity / equity.cummax() - 1
        annual = equity.iloc[-1] ** (TRADING_DAYS / len(daily)) - 1
        excess = daily - cash[period]
        sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
        since[name] = (1 + daily[test]).prod() ** (TRADING_DAYS / test.sum()) - 1
        print(f"{name:<24}{equity.iloc[-1]:>13.2f}{annual:>9.1%}{sharpe:>8.2f}{drawdown.min():>14.1%}{since[name]:>20.1%}")

        width = 2.2 if name == "S&P 500 (buy & hold)" else 1.6
        label = f"{name}: ${equity.iloc[-1]:.2f}, {annual:.1%} a year, worst fall {drawdown.min():.0%}"
        top.plot(equity.index, equity, color=color, linewidth=width, label=label)
        bottom.plot(drawdown.index, drawdown, color=color, linewidth=1)

    top.set_yscale("log")
    top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    top.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    top.set_yticks([0.5, 1, 2, 4, 8, 16])
    top.set_title(
        f"All algorithms against the S&P 500, {period[0]:%Y} to {period[-1]:%Y}\n"
        f"Since {cutoff.year}, on data the trained algorithm had not seen: "
        f"trained {since['Trained algorithm']:.1%} a year, S&P 500 {since['S&P 500 (buy & hold)']:.1%} a year",
        fontsize=11,
    )
    for axis in (top, bottom):
        axis.axvline(cutoff, color="gray", linestyle="--", linewidth=1)
    top.text(cutoff, 0.02, " trained algorithm fitted to the data left of this line",
             transform=top.get_xaxis_transform(), fontsize=8, color="gray", ha="right")
    top.set_ylabel("Value of $1 invested (log scale)")
    top.grid(alpha=0.3)
    top.legend(loc="upper left", title="Final value, annual return, largest drawdown")

    bottom.set_ylabel("Fall from previous high")
    bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.chart, dpi=130)
    print(f"\nChart saved to {args.chart}")


if __name__ == "__main__":
    main()
