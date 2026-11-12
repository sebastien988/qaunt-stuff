"""Train the combined algorithm to beat the S&P 500, then test it on unseen data.

Training picks, from the data before --train-until only:
    - how to split the money between the four strategies of combined_algo.py
    - how much leverage to use (1x, 1.5x or 2x; borrowing costs the
      Treasury-bill rate plus a spread)
The pick is the combination with the highest return whose volatility was no
higher than the S&P 500's. It is then run, unchanged, on the later data.

Only the test period says anything about the future: the training period is
the data the settings were chosen to look good on.

Usage:
    python trained_algo.py
    python trained_algo.py --train-until 2012-01-01
"""

import argparse
import itertools

import numpy as np
import pandas as pd

import combined_algo as combined

TRADING_DAYS = 252
MIX_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]
LEVERAGE_CHOICES = [1.0, 1.5, 2.0]
BORROW_SPREAD = 0.01  # yearly cost of borrowed money above the Treasury-bill rate
MARKET = "SPY"


def leveraged_returns(weights, returns, cash, cost_bps: float) -> pd.Series:
    """Daily returns of the weights, net of trading and borrowing costs."""
    result = combined.simulate(weights, returns, cash, cost_bps)
    borrowed = (result["invested"] - 1).clip(lower=0)
    return result["return"] - borrowed * BORROW_SPREAD / TRADING_DAYS


def train(prices: pd.DataFrame, cash: pd.Series, cost_bps: float, train_until: str):
    """Choose the mix and leverage on the training data.

    Returns the chosen weights, their daily returns over the whole period,
    the strategy mix, the leverage, and the period simulated.
    """
    returns = prices.pct_change().fillna(0.0)
    strategies, period = combined.build_strategies(prices, returns)
    in_training = period < pd.Timestamp(train_until)
    if in_training.sum() < 2 * TRADING_DAYS or (~in_training).sum() < TRADING_DAYS:
        raise SystemExit("--train-until must leave at least 2 years to train on and 1 year to test on.")
    market_volatility = returns[MARKET][period][in_training].std()

    best = None
    mixes = [m for m in itertools.product(MIX_STEPS, repeat=len(strategies)) if abs(sum(m) - 1) < 1e-9]
    for mix, leverage in itertools.product(mixes, LEVERAGE_CHOICES):
        weights = sum(share * w for share, w in zip(mix, strategies.values())) * leverage
        daily = leveraged_returns(weights.loc[period], returns.loc[period], cash[period], cost_bps)
        trained_on = daily[in_training]
        if trained_on.std() > market_volatility:
            continue
        growth = (1 + trained_on).prod()
        if best is None or growth > best[0]:
            best = (growth, weights, daily, dict(zip(strategies, mix)), leverage)
    if best is None:
        raise SystemExit("No combination was as calm as the market in the training period.")
    return best[1], best[2], best[3], best[4], period


def describe(daily: pd.Series) -> dict[str, float]:
    equity = (1 + daily).cumprod()
    return {
        "Growth of $1": equity.iloc[-1],
        "Annual return": equity.iloc[-1] ** (TRADING_DAYS / len(daily)) - 1,
        "Annual volatility": daily.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": (equity / equity.cummax() - 1).min(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the combined algorithm, then test it on later data.")
    parser.add_argument("--train-until", default="2015-01-01", help="train on data before this date")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    prices, cash = combined.load_data("1990-01-01", None)
    weights, daily, mix, leverage, period = train(prices, cash, args.cost_bps, args.train_until)
    market = prices[MARKET].pct_change().fillna(0.0)[period]
    in_training = period < pd.Timestamp(args.train_until)

    chosen = ", ".join(f"{share:.0%} {name}" for name, share in mix.items() if share)
    print(f"\nChosen in training: {chosen}, at {leverage:g}x leverage")

    for title, mask in [("TRAINING (settings were fitted to this)", in_training),
                        ("TEST (never seen in training)", ~in_training)]:
        days = period[mask]
        print(f"\n{title}: {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}\n")
        print(f"{'':<20}{'Trained':>12}{'S&P 500':>12}")
        ours, theirs = describe(daily[mask]), describe(market[mask])
        for metric in ours:
            form = ".2f" if metric == "Growth of $1" else ".1%"
            print(f"{metric:<20}{ours[metric]:>12{form}}{theirs[metric]:>12{form}}")

    today = weights.iloc[-1]
    print(f"\nAllocation to hold after the close of {prices.index[-1]:%Y-%m-%d}:")
    for ticker, weight in today.sort_values(ascending=False).items():
        if weight >= 0.005:
            print(f"  {ticker:<9}{weight:>6.1%}  {combined.UNIVERSE[ticker]}")
    label = "Borrowed" if today.sum() > 1 else "Cash"
    print(f"  {label:<9}{abs(1 - today.sum()):>6.1%}")


if __name__ == "__main__":
    main()
