# Market data library (~1 GB)

Built by `../download_data.py` (resumable; re-run to fill gaps or refresh).
Not committed to git (`data/` is in `.gitignore`).

| Folder | What | Source |
|---|---|---|
| `daily/etfs` | ~110 ETFs: equity, sector, international, bonds, commodities, vol, leveraged | Yahoo |
| `daily/indices` | US + world indices, VIX family, Treasury yields (^IRX/^FVX/^TNX/^TYX) | Yahoo |
| `daily/futures` | Equity, rates, energy, metals, ags, FX futures, DXY | Yahoo |
| `daily/fx` | 23 currency pairs | Yahoo |
| `daily/crypto` | 28 coins (USD) | Yahoo |
| `daily/stocks_large` | S&P 500 + Nasdaq-100 | Yahoo |
| `daily/stocks_mid` | S&P 400 | Yahoo |
| `daily/stocks_small` | S&P 600 | Yahoo |
| `hourly/` | 1h bars, last ~730 days (ETFs, indices, futures, FX, crypto, large caps) | Yahoo |
| `minute/` | 1m bars, last ~29 days (key ETFs/futures/crypto + 80 S&P names) | Yahoo |
| `macro/fred` | ~53 rates, credit spreads, inflation, labour, growth, liquidity series | FRED |
| `macro/fama_french` | FF3, FF5, momentum, 49 industry portfolios (daily, in %) | Ken French library |
| `meta/` | Index constituents with GICS sectors; `manifest.csv` (ticker, rows, date range, file) | Wikipedia |

## Columns
Daily / intraday: `Date|Datetime, Open, High, Low, Close, Adj Close, Volume[, Dividends, Stock Splits]`
- `Close` is raw; `Adj Close` is adjusted for splits and dividends, so use it for returns.
- Ticker file names: `^` becomes `IDX_` and `=` becomes `_` (e.g. `IDX_VIX.csv`, `ES_F.csv`, `EURUSD_X.csv`).

## Loading
```python
import pandas as pd
spy = pd.read_csv("data/daily/etfs/SPY.csv", index_col=0, parse_dates=True)
ret = spy["Adj Close"].pct_change()

# panel of all large caps
from pathlib import Path
px = pd.concat({p.stem: pd.read_csv(p, index_col=0, parse_dates=True)["Adj Close"]
                for p in Path("data/daily/stocks_large").glob("*.csv")}, axis=1)
```

## Caveats
- **Survivorship bias**: stock lists are *today's* index members. Delisted or removed companies are missing, so backtests on them will look too good.
- Fama-French CSVs have text header/footer lines; skip them when parsing (`skiprows`).
- Yahoo data occasionally has bad prints or gaps; sanity-check before trusting a result.
