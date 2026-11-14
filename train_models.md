# Stock-ranking models trained on the local data library

Universe: 211 US stocks from `data/daily/stocks_*`; monthly samples 1999-02-03 to 2026-10-22; walk-forward test years 2010-2026; costs 10 bp per side.

## Out-of-sample results

| model | mean IC | IC t-stat | IC>0 | L/S return/yr | L/S Sharpe | top-10% excess/yr | IR | turnover |
|---|---|---|---|---|---|---|---|---|
| momentum | 0.007 | 0.5 | 52% | 4.8% | 0.20 | 8.4% | 0.56 | 28% |
| ridge | 0.031 | 2.8 | 58% | 15.1% | 0.88 | 11.5% | 1.10 | 47% |
| mlp | 0.024 | 2.9 | 56% | 6.8% | 0.50 | 3.7% | 0.41 | 70% |

Best long/short Sharpe out of sample: **ridge**. Compare the trained models with the untrained momentum baseline. If they don't clearly beat it, the training added nothing.

## Ridge weights (final model)

| feature | weight |
|---|---|
| size | -0.0206 |
| rev_1m | -0.0092 |
| beta | +0.0088 |
| vol_3m | +0.0081 |
| mom_12_1 | +0.0077 |
| idio_vol | +0.0063 |
| mom_3m | -0.0057 |
| mom_6m | -0.0046 |
| dist_high | -0.0023 |
| skew_3m | -0.0022 |
| max_ret | +0.0022 |
| rev_1w | -0.0020 |
| volume_trend | +0.0018 |
| trend_200 | +0.0013 |
| vol_1m | +0.0012 |

## Latest ranking (2026-11-13), next ~21 trading days

Top 15 (blend of ridge + MLP ranks):

| ticker | ridge | mlp | blend |
|---|---|---|---|
| EME | +0.022 | +0.053 | 0.99 |
| GNRC | +0.025 | +0.043 | 0.99 |
| FIX | +0.023 | +0.041 | 0.98 |
| JBL | +0.021 | +0.045 | 0.98 |
| DVA | +0.023 | +0.039 | 0.98 |
| P | +0.018 | +0.046 | 0.96 |
| MCHP | +0.018 | +0.038 | 0.95 |
| APTV | +0.020 | +0.033 | 0.94 |
| ALB | +0.019 | +0.026 | 0.91 |
| LII | +0.017 | +0.034 | 0.91 |
| LITE | +0.020 | +0.023 | 0.91 |
| LRCX | +0.018 | +0.027 | 0.91 |
| CIEN | +0.021 | +0.022 | 0.90 |
| HPE | +0.015 | +0.035 | 0.90 |
| KEYS | +0.020 | +0.021 | 0.89 |

Bottom 15:

| ticker | ridge | mlp | blend |
|---|---|---|---|
| COST | -0.023 | -0.016 | 0.11 |
| CTAS | -0.008 | -0.026 | 0.10 |
| MSFT | -0.008 | -0.030 | 0.09 |
| APP | -0.005 | -0.061 | 0.09 |
| AWK | -0.010 | -0.029 | 0.08 |
| ABBV | -0.016 | -0.022 | 0.07 |
| MCD | -0.016 | -0.022 | 0.07 |
| ICE | -0.014 | -0.026 | 0.07 |
| ABT | -0.011 | -0.029 | 0.07 |
| ADP | -0.015 | -0.026 | 0.07 |
| AJG | -0.010 | -0.040 | 0.06 |
| BRK-B | -0.025 | -0.027 | 0.03 |
| HD | -0.015 | -0.045 | 0.03 |
| KO | -0.019 | -0.032 | 0.02 |
| MA | -0.018 | -0.040 | 0.02 |

Full ranking: `models/latest_ranking.csv`.

## Caveats

- **Survivorship bias**: only today's index members are in the data, so failed companies are missing.
- Predictions are *relative* (stock vs average stock), not market direction.
- A typical good IC is 0.02-0.05. Most of each stock's monthly return is still unexplained noise.
- Shorting costs (borrow fees) are ignored; the long-only column is closer to what you could actually do.
