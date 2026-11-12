# Quant research report

- **Generated:** 12 November 2026 by `quant_research.py`. Not financial advice.
- **Data:** 12 ETFs, daily, dividends included, 04 Jun 2007 to 12 Nov 2026 (4894 days). Cash = 13-week T-bill rate.

## 1. Asset risk metrics

Sharpe and Sortino use returns over cash. VaR/CVaR are one-day, 95%. Alpha is annualised CAPM alpha vs SPY with a Newey-West t-statistic (|t| > 2 is roughly significant). PSR = probability the true Sharpe is above zero, allowing for skew and fat tails.

| | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | Skew | ExKurt | VaR95 | CVaR95 | CF-VaR95 | Beta | Alpha | Alpha t | PSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US technology (QQQ) | +15.7% | 22.4% | 0.70 | 0.99 | -53.4% | 0.29 | -0.07 | 7.32 | 2.30% | 3.36% | 2.07% | 1.05 | +4.6% | 2.51 | 100% |
| US large companies (SPY) | +10.6% | 19.7% | 0.53 | 0.75 | -55.2% | 0.19 | -0.00 | 14.42 | 1.81% | 3.02% | 1.63% | 1.00 | - | - | 99% |
| Gold (GLD) | +9.0% | 18.3% | 0.48 | 0.68 | -45.6% | 0.20 | -0.27 | 6.78 | 1.80% | 2.72% | 1.78% | 0.06 | +8.2% | 2.01 | 98% |
| US small companies (IWM) | +7.8% | 24.7% | 0.37 | 0.51 | -58.6% | 0.13 | -0.32 | 6.10 | 2.38% | 3.65% | 2.47% | 1.12 | -2.7% | -1.17 | 95% |
| High-yield bonds (HYG) | +4.7% | 10.8% | 0.34 | 0.49 | -34.2% | 0.14 | 0.76 | 43.50 | 0.82% | 1.60% | 0.35% | 0.37 | -0.2% | -0.12 | 94% |
| Corporate bonds (LQD) | +3.9% | 8.8% | 0.31 | 0.44 | -25.0% | 0.16 | 0.04 | 56.41 | 0.73% | 1.20% | 0.26% | 0.09 | +1.8% | 0.94 | 91% |
| US property (VNQ) | +5.3% | 29.5% | 0.27 | 0.39 | -70.6% | 0.08 | 0.02 | 16.77 | 2.43% | 4.43% | 2.38% | 1.12 | -3.6% | -0.96 | 89% |
| Medium Treasuries (IEF) | +3.1% | 7.0% | 0.26 | 0.38 | -23.9% | 0.13 | 0.14 | 2.59 | 0.69% | 0.96% | 0.67% | -0.10 | +2.9% | 1.95 | 88% |
| Developed ex-US (EFA) | +4.3% | 21.7% | 0.24 | 0.33 | -61.0% | 0.07 | -0.03 | 13.73 | 2.01% | 3.29% | 1.85% | 0.97 | -5.1% | -2.37 | 85% |
| Emerging markets (EEM) | +3.9% | 28.1% | 0.22 | 0.32 | -66.4% | 0.06 | 0.55 | 18.49 | 2.44% | 4.12% | 1.93% | 1.17 | -6.0% | -1.79 | 84% |
| Long Treasuries (TLT) | +2.8% | 15.1% | 0.16 | 0.23 | -48.4% | 0.06 | 0.08 | 3.19 | 1.51% | 2.05% | 1.47% | -0.23 | +4.9% | 1.54 | 76% |
| Commodities (DBC) | +2.0% | 19.3% | 0.12 | 0.17 | -76.4% | 0.03 | -0.41 | 3.17 | 1.91% | 2.94% | 2.05% | 0.38 | -1.6% | -0.40 | 71% |

Fat tails: excess kurtosis ranges from 2.6 to 56.4 (a normal distribution has 0). CVaR95 / VaR95 averages 1.60; under a normal distribution it would be 1.25, so losses beyond the VaR are worse than a normal model says. Cornish-Fisher (`CF-VaR95`) is only trustworthy for moderate kurtosis (roughly below 8): for US large companies (SPY), High-yield bonds (HYG), Corporate bonds (LQD), US property (VNQ), Developed ex-US (EFA), Emerging markets (EEM) it is outside its valid range and should be ignored in favour of the historical figures.

## 2. GARCH(1,1) volatility (SPY)

Fitted on 2007 to Feb 2017 only: alpha = 0.135, beta = 0.850, persistence alpha+beta = 0.9849, shock half-life = 46 trading days.

Out-of-sample one-day variance forecasts (QLIKE loss = mean(log h + r^2/h), lower is better):

| Model | QLIKE |
|---|---|
| GARCH(1,1) | 0.7872 |
| EWMA 0.94 | 0.8581 |
| 21-day rolling | 0.9158 |

Best: **GARCH(1,1)**.

99% one-day VaR backtest on the test period (Kupiec test; p < 0.05 means the breach rate is wrong):

| VaR model | Breaches | Rate | Kupiec p |
|---|---|---|---|
| GARCH-normal | 51/2447 | 2.08% | 0.000 |
| GARCH-FHS | 26/2447 | 1.06% | 0.758 |

FHS uses the empirical 1% quantile of standardised residuals (-2.85) instead of the normal -2.33, which accounts for fat tails.

Current forecast (refitted on all data; alpha 0.147, beta 0.833), annualised average volatility over the next:

| Horizon (days) | 1 | 5 | 21 | 63 | 252 |
|---|---|---|---|---|---|
| Vol | 24.3% | 24.1% | 23.5% | 22.4% | 20.7% |

Long-run volatility: 19.7%. Forecasts mean-revert towards it: E[h(t+k)] = V + (alpha+beta)^(k-1) (h(t+1) - V).

## 3. Walk-forward portfolio construction

Rebalanced every month end using only the past 252 days, Ledoit-Wolf shrunk covariance (average shrinkage 0.04), long-only, fully invested, 5 bp cost per unit of turnover. DSR = Deflated Sharpe Ratio: the probability the Sharpe beats what the best of 5 tried portfolios would show by luck.

| | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | CVaR95 | Beta | Alpha | Alpha t | PSR | DSR | Turnover/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Equal weight | +7.0% | 12.3% | 0.50 | 0.70 | -34.6% | 0.20 | 1.88% | 0.57 | -0.7% | -0.60 | 98% | 96% | 0.4x |
| Inverse vol | +6.0% | 9.3% | 0.52 | 0.73 | -24.9% | 0.24 | 1.39% | 0.40 | +0.0% | 0.04 | 99% | 97% | 0.5x |
| Min variance | +3.3% | 5.6% | 0.36 | 0.51 | -17.5% | 0.19 | 0.84% | 0.13 | +0.4% | 0.36 | 94% | 88% | 1.2x |
| Risk parity | +5.4% | 7.7% | 0.54 | 0.75 | -20.7% | 0.26 | 1.14% | 0.27 | +0.9% | 0.66 | 99% | 97% | 0.5x |
| Max Sharpe | +7.4% | 11.7% | 0.55 | 0.75 | -26.9% | 0.28 | 1.86% | 0.21 | +3.9% | 1.55 | 99% | 97% | 5.9x |
| SPY | +12.2% | 19.7% | 0.61 | 0.86 | -47.2% | 0.26 | 3.03% | 1.00 | - | - | 100% | - | - |
| 60/40 SPY/IEF | +8.9% | 11.4% | 0.69 | 0.98 | -27.7% | 0.32 | 1.72% | 0.56 | +1.0% | 1.67 | 100% | - | - |

Best Sharpe among the constructed portfolios: **Max Sharpe** (0.55), against 0.61 for SPY and 0.69 for 60/40. None of the constructed portfolios beat the simple 60/40 benchmark on Sharpe.

The Sharpe ratios of the five span only 0.36 to 0.55; differences this small are well inside sampling error. Largest CAPM alpha t-statistic among them: 1.55 (below 2 is not significant). The question this setup tests is from DeMiguel, Garlappi & Uppal (2009): whether optimised portfolios beat naive ones out of sample once estimation error and turnover are counted. Max Sharpe needs expected returns, which are noisy, and traded 5.9x a year against 0.4x for equal weight.

### Current allocations

| | Equal weight | Inverse vol | Min variance | Risk parity | Max Sharpe | Risk share (RP) | Risk share (EW) |
|---|---|---|---|---|---|---|---|
| SPY | 8.3% | 6.9% | 0.0% | 5.2% | 18.1% | 8.3% | 9.9% |
| QQQ | 8.3% | 4.5% | 0.0% | 3.7% | 0.0% | 8.3% | 14.6% |
| IWM | 8.3% | 5.1% | 0.0% | 3.7% | 6.2% | 8.3% | 13.5% |
| EFA | 8.3% | 5.8% | 0.0% | 4.2% | 9.9% | 8.3% | 12.1% |
| EEM | 8.3% | 3.6% | 0.0% | 2.9% | 0.0% | 8.3% | 18.8% |
| VNQ | 8.3% | 6.7% | 0.0% | 7.7% | 21.7% | 8.3% | 4.9% |
| TLT | 8.3% | 9.5% | 0.0% | 9.8% | 0.0% | 8.3% | 3.6% |
| IEF | 8.3% | 16.1% | 30.5% | 16.2% | 0.0% | 8.3% | 2.1% |
| LQD | 8.3% | 14.9% | 7.8% | 13.1% | 0.0% | 8.3% | 2.9% |
| HYG | 8.3% | 19.1% | 51.4% | 15.9% | 0.0% | 8.3% | 2.7% |
| GLD | 8.3% | 3.3% | 0.0% | 2.8% | 0.0% | 8.3% | 16.3% |
| DBC | 8.3% | 4.5% | 10.3% | 14.9% | 44.1% | 8.3% | -1.3% |

With equal weights the risk is dominated by the volatile assets; risk parity equalises it.

## Limits

- One history (2007 onwards) with one big bond bull market and the 2022 bond crash; results depend on it.
- Sharpe ratios, alphas and PSRs assume returns are stationary; regimes change.
- Costs are a flat rate; taxes, spreads in stress and fund fees beyond the ETF price are ignored.
- The efficient frontier chart uses full-sample means: it shows what was optimal in hindsight, not what was achievable.

Chart: `quant_research.png`.
