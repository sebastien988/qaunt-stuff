# Nvidia (NVDA) learning forecast

- **Generated:** 13 November 2026 by `nvidia_learning.py`. Not financial advice.
- Skill = 1 - pinball loss of the model / pinball loss of a baseline that forecasts the unconditional past quantiles. Above 0 means better than ignoring every signal.
- The validation score (skill in the worst of 4 validation periods) is used to choose the model, so it can only rise. Held-out skill is measured on days never used to choose it and is the honest figure.

```
NVDA at $162.85 on 2026-11-13.  Run 13, 1953 models tried in total (992 this run in 30 s).

FORECAST for the next 20 trading days
  Median            +1.0%  (to about $164.55)
  Average           +3.3%
  8 in 10 range     -11.6% to +22.3%  ($143.99 to $199.25)
  Chance of rising  54%
  Baseline (ignores every signal): median +3.7%, range -11.5% to +20.6%, rising 64%

LEARNING  no better model found this run; champion kept
  Model             3 features, ridge lambda 0.1, last 1000 days, equal weights, not volatility-scaled
  Features          Nasdaq return 21d, news tone 7d, volatility 21d/126d
  Validation score  +4.60%   (worst of 4 periods; average skill +5.92%; 1080 days to 2024-09-09; chooses the model)
  Held-out skill    -9.79%   (485 days from 2024-10-08; never used to choose)
  Held-out range    actual price landed inside the 8-in-10 range 91% of the time (target 80%)
  Verdict           on days it was not tuned on, the model does not beat the baseline. Prefer the baseline forecast above.
  Saved forecasts   2 saved; the first can be scored 20 trading days after it was made
```

## Runs so far

| Run | Date | Models tried | Validation score | Held-out skill | Found a better model |
|---|---|---|---|---|---|
| 1 | 2026-11-12 | 81 | +1.62% | -7.52% | yes |
| 2 | 2026-11-12 | 161 | +2.67% | -8.69% | yes |
| 3 | 2026-11-12 | 241 | +2.94% | -9.36% | yes |
| 4 | 2026-11-12 | 321 | +3.05% | -9.40% | yes |
| 5 | 2026-11-12 | 401 | +4.60% | -9.79% | yes |
| 6 | 2026-11-12 | 481 | +4.60% | -9.79% | no |
| 7 | 2026-11-12 | 561 | +4.60% | -9.79% | no |
| 8 | 2026-11-13 | 641 | +4.60% | -9.79% | no |
| 9 | 2026-11-13 | 721 | +4.60% | -9.79% | no |
| 10 | 2026-11-13 | 801 | +4.60% | -9.79% | no |
| 11 | 2026-11-13 | 881 | +4.60% | -9.79% | no |
| 12 | 2026-11-13 | 961 | +4.60% | -9.79% | no |
| 13 | 2026-11-13 | 1953 | +4.60% | -9.79% | no |

## Limits

- The more models are tried, the more the validation skill flatters the model. Trust the held-out skill.
- 20-day outcomes overlap from one day to the next, so there is far less independent evidence than the day counts suggest; small skill differences are noise.
- The history is 2017 onwards, a period in which Nvidia rose strongly.

Chart: `nvidia_learning.png`. Memory: `nvidia_learning.json` (delete it, or use `--reset`, to start again).
