# US industrial sector forecast

- **Generated:** 12 November 2026 by `industrial_prediction.py`. Not financial advice.
- **Prices to:** 12 November 2026. Sector = XLI at $169.44. Horizon = 20 trading days.

## Summary

- **Sector:** median +1.8% (to about $172.49), 8-in-10 range -7.3% to +9.4% ($157.10 to $185.36), chance of rising 61%. From the Vol-scaled method.
- **Volatility now:** 24% a year, against a typical 16%. This sets the width of the range and is the part of the forecast that has a track record.
- **Direction:** the ridge model, which tries to predict direction from momentum, trend and VIX, did not improve on the volatility-only forecast in the walk-forward check.
- **Stock picking:** the ranking of companies against each other is **not statistically significant** (mean IC -0.009, t = -0.5).

## 1. Sector forecast

Walk-forward check from Dec 2009 to Oct 2026; halves split at May 2018. Skill is against the baseline; above 0 is better. "Inside range" should be 80%.

| Method | Median | 8-in-10 range | Chance of rising | Skill | Skill, first / second half | Inside range |
|---|---|---|---|---|---|---|
| Baseline | +1.2% | -5.0% to +6.7% | 61% | - | - | 83% |
| Vol-scaled (headline) | +1.8% | -7.3% to +9.4% | 61% | +3.3% | +2.8% / +3.8% | 80% |
| Ridge | +1.8% | -7.3% to +9.4% | 62% | +2.7% | +1.6% / +3.5% | 80% |

Ridge weights (per standard deviation of each feature, in units of volatility): return 21d -0.016, return 63d -0.010, return 252d +0.000, volatility 21d/126d -0.006, vs 200d average +0.009, vs S&P 500 63d -0.004, VIX level +0.001, VIX change 21d +0.003.

## 2. Company forecasts

Sorted by forecast return against the average company ("Vs peers"). Ranges use each company's current volatility; in the walk-forward check from Dec 2009 the price landed inside the range 79% of the time (target 80%).

| Company | Price | Median | 8-in-10 range | Chance of rising | Vs peers | Volatility (a year) |
|---|---|---|---|---|---|---|
| Quanta Services (PWR) | $606.16 | +6.4% | -22.9% to +40.3% | 61% | +1.00% | 84% |
| Caterpillar (CAT) | $605.28 | +4.3% | -16.0% to +25.7% | 61% | +0.74% | 57% |
| Eaton (ETN) | $359.04 | +4.7% | -18.0% to +29.3% | 61% | +0.66% | 64% |
| Johnson Controls (JCI) | $142.55 | +3.6% | -14.7% to +22.3% | 61% | +0.39% | 51% |
| Cummins (CMI) | $455.40 | +2.8% | -11.4% to +16.9% | 61% | +0.39% | 39% |
| Rockwell Automation (ROK) | $404.20 | +4.0% | -16.7% to +25.7% | 60% | +0.33% | 58% |
| Boeing (BA) | $171.55 | +2.8% | -11.7% to +17.2% | 61% | +0.32% | 40% |
| Deere (DE) | $678.79 | +1.9% | -8.3% to +11.5% | 60% | +0.16% | 27% |
| United Rentals (URI) | $1,139.14 | +3.2% | -14.3% to +21.0% | 60% | +0.14% | 48% |
| Trane (TT) | $424.05 | +2.5% | -11.3% to +16.0% | 60% | +0.13% | 38% |
| Ametek (AME) | $241.04 | +2.5% | -11.5% to +16.2% | 60% | +0.07% | 38% |
| FedEx (FDX) | $319.30 | +1.9% | -8.9% to +12.1% | 60% | +0.05% | 29% |
| Lockheed Martin (LMT) | $604.09 | +2.6% | -12.6% to +17.9% | 60% | +0.02% | 42% |
| CSX (CSX) | $52.40 | +1.8% | -9.1% to +12.2% | 59% | -0.04% | 30% |
| Parker-Hannifin (PH) | $957.98 | +2.2% | -11.0% to +15.2% | 59% | -0.05% | 36% |
| GE Aerospace (GE) | $330.86 | +2.4% | -12.2% to +16.9% | 59% | -0.07% | 40% |
| Emerson (EMR) | $166.92 | +1.8% | -9.4% to +12.6% | 59% | -0.07% | 31% |
| Copart (CPRT) | $32.00 | +2.2% | -11.4% to +15.7% | 59% | -0.08% | 38% |
| Paccar (PCAR) | $106.33 | +1.7% | -8.9% to +11.7% | 59% | -0.10% | 29% |
| Grainger (GWW) | $1,502.61 | +1.6% | -8.8% to +11.6% | 59% | -0.11% | 28% |
| UPS (UPS) | $110.38 | +1.9% | -9.9% to +13.2% | 59% | -0.12% | 32% |
| Union Pacific (UNP) | $304.20 | +1.4% | -7.7% to +9.8% | 59% | -0.14% | 24% |
| 3M (MMM) | $186.24 | +2.3% | -12.0% to +16.4% | 59% | -0.17% | 39% |
| Norfolk Southern (NSC) | $333.24 | +1.3% | -7.8% to +9.9% | 58% | -0.18% | 25% |
| Honeywell (HON) | $223.45 | +1.7% | -9.6% to +12.6% | 59% | -0.19% | 31% |
| Waste Management (WM) | $244.11 | +1.5% | -8.6% to +11.0% | 58% | -0.22% | 27% |
| L3Harris (LHX) | $271.84 | +2.1% | -11.7% to +15.7% | 59% | -0.22% | 38% |
| Northrop Grumman (NOC) | $549.13 | +1.6% | -9.4% to +12.2% | 58% | -0.24% | 30% |
| Illinois Tool Works (ITW) | $286.75 | +1.2% | -7.7% to +9.6% | 58% | -0.25% | 24% |
| RTX (RTX) | $198.36 | +1.5% | -9.1% to +11.6% | 58% | -0.25% | 29% |
| TransDigm (TDG) | $1,203.00 | +1.7% | -9.9% to +12.9% | 58% | -0.26% | 32% |
| Fastenal (FAST) | $55.55 | +1.0% | -6.8% to +8.2% | 57% | -0.31% | 21% |
| ADP (ADP) | $307.92 | +1.3% | -8.1% to +10.1% | 58% | -0.32% | 25% |
| Cintas (CTAS) | $227.52 | +1.2% | -8.0% to +9.9% | 58% | -0.32% | 25% |
| Republic Services (RSG) | $253.00 | +1.1% | -7.5% to +9.2% | 57% | -0.34% | 23% |
| General Dynamics (GD) | $363.07 | +1.1% | -7.7% to +9.5% | 57% | -0.34% | 24% |

## 3. Does the company ranking work?

Checked on 212 non-overlapping 20-day periods from Dec 2009.

| Measure | Value | t-statistic |
|---|---|---|
| Mean rank IC | -0.009 | -0.52 |
| Share of periods with IC above 0 | 47% | |
| Top third minus bottom third, per period | +0.05% | 0.22 |

The ranking is not statistically significant. Treat the "Vs peers" column as noise: it is small because the model found little to go on. No trading costs are included in the spread.

Ridge weights (positive = companies high on this did better next period): momentum 12m (skip last month) +0.017, momentum 6m -0.022, last month's return -0.007, last week's return -0.008, volatility 63d +0.040, below 252d high +0.007.

## Limits

- The companies are today's large industrials. Ones that shrank or failed are missing, which flatters the past averages and so the medians above.
- Medians mostly reflect that these shares rose on average in the past, not a view about the next 20 days.
- Each company's median is its volatility times the typical past risk-adjusted return of the group, so volatile shares get bigger medians by construction. Read the ranges, not the medians.
- The model knows only prices and the VIX: no earnings, orders, tariffs, interest rates or news.
- 8-in-10 means 2 in 10 outcomes fall outside the range.

Chart: `industrial_prediction.png`.
