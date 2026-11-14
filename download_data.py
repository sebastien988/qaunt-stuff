"""
Build a local ~1 GB market-data library in ./data

Contents
  data/daily/<category>/<TICKER>.csv   full-history daily OHLCV + dividends/splits
  data/hourly/<TICKER>.csv             last ~730 days of 1h bars
  data/minute/<TICKER>.csv             last ~29 days of 1m bars
  data/macro/fred/<SERIES>.csv         FRED macro / rates series
  data/macro/fama_french/*.csv         Fama-French factors (daily)
  data/meta/*.csv                      index constituents (with sectors), manifest

Resumable: files that already exist are skipped. Stops when the size budget is hit.
Usage:  .venv/bin/python download_data.py [--budget-gb 1.0]
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) quant-research-data/1.0"}
FLOAT_FMT = "%.7g"  # 7 significant digits: plenty for prices, keeps files small

ETFS = [
    # US equity broad / style / size
    "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "IWB", "IWD", "IWF", "IWN", "IWO",
    "MDY", "IJH", "IJR", "RSP", "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPLV", "SCHD", "VIG",
    # sectors
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "XBI", "IBB", "KRE", "KBE", "ITB", "XHB", "XME", "XOP", "OIH", "IYR", "VNQ",
    # international
    "EFA", "EEM", "VEA", "VWO", "EWJ", "EWZ", "FXI", "MCHI", "EWG", "EWU", "EWC", "EWA",
    "EWY", "EWT", "INDA", "EWW", "EZU", "ACWI", "VXUS",
    # bonds
    "AGG", "BND", "TLT", "IEF", "IEI", "SHY", "SHV", "BIL", "TIP", "LQD", "HYG", "JNK",
    "EMB", "MUB", "BNDX", "VCIT", "VCSH", "GOVT", "MBB",
    # commodities / currencies / vol / alternatives
    "GLD", "IAU", "SLV", "GDX", "USO", "UNG", "DBC", "DBA", "PDBC", "UUP", "FXE", "FXY",
    "VXX", "UVXY", "SVXY", "BITO", "IBIT",
    # leveraged / inverse (useful for studying decay)
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SSO", "SDS", "SH", "TMF", "TBT",
]

INDICES = [
    "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^SP400", "^SP600", "^NYA", "^W5000",
    "^VIX", "^VVIX", "^VIX3M", "^SKEW", "^MOVE",
    "^IRX", "^FVX", "^TNX", "^TYX",
    "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^HSI", "000001.SS", "^AXJO",
    "^GSPTSE", "^BSESN", "^KS11", "^TWII", "^BVSP", "^MXX",
]

FUTURES = [
    "ES=F", "NQ=F", "YM=F", "RTY=F", "ZB=F", "ZN=F", "ZF=F", "ZT=F",
    "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZS=F", "ZW=F", "KC=F", "SB=F", "CC=F", "CT=F", "LE=F", "HE=F",
    "6E=F", "6J=F", "6B=F", "6A=F", "6C=F", "DX-Y.NYB", "BTC=F",
]

FX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X",
    "USDCNY=X", "USDHKD=X", "USDSGD=X", "USDINR=X", "USDMXN=X", "USDBRL=X", "USDZAR=X",
    "USDTRY=X", "USDKRW=X", "USDSEK=X", "USDNOK=X", "EURJPY=X", "EURGBP=X", "EURCHF=X",
    "GBPJPY=X", "AUDJPY=X",
]

CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD",
    "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD", "XLM-USD", "ATOM-USD",
    "ETC-USD", "XMR-USD", "FIL-USD", "HBAR-USD", "ALGO-USD", "VET-USD", "NEAR-USD",
    "AAVE-USD", "UNI7083-USD", "MKR-USD", "SHIB-USD", "USDT-USD", "USDC-USD",
]

FRED_SERIES = [
    # rates & curve
    "DFF", "FEDFUNDS", "SOFR", "DTB3", "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5",
    "DGS7", "DGS10", "DGS20", "DGS30", "T10Y2Y", "T10Y3M", "DFII10", "T10YIE", "T5YIFR",
    "MORTGAGE30US",
    # credit & stress
    "BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLC0A4CBBB", "TEDRATE", "STLFSI4", "NFCI",
    # inflation
    "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIACO",
    # growth & labour
    "GDP", "GDPC1", "INDPRO", "PAYEMS", "UNRATE", "ICSA", "CCSA", "JTSJOL", "RSAFS",
    "HOUST", "PERMIT", "UMCSENT", "USREC", "CFNAI",
    # money, fx, commodities, vol
    "M2SL", "WALCL", "RRPONTSYD", "DTWEXBGS", "DCOILWTICO", "DCOILBRENTEU", "DHHNGSP",
    "VIXCLS",
]

FAMA_FRENCH = {
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum_daily": "F-F_Momentum_Factor_daily_CSV.zip",
    "industry49_daily": "49_Industry_Portfolios_daily_CSV.zip",
}
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


# ----------------------------------------------------------------------------- helpers

def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_")


class Budget:
    def __init__(self, gb: float):
        self.limit = int(gb * 1024**3)
        self.used = dir_size(ROOT) if ROOT.exists() else 0

    def add(self, n: int) -> None:
        self.used += n

    @property
    def full(self) -> bool:
        return self.used >= self.limit

    def __str__(self) -> str:
        return f"{self.used / 1024**2:,.0f} MB / {self.limit / 1024**2:,.0f} MB"


def write_frame(df: pd.DataFrame, path: Path, budget: Budget) -> int:
    df = df.dropna(how="all")
    price_cols = [c for c in ("Open", "High", "Low", "Close", "Adj Close") if c in df]
    if price_cols:
        df = df.dropna(subset=price_cols, how="all")
    if df.empty:
        return 0
    if "Volume" in df:
        df["Volume"] = df["Volume"].fillna(0).round().astype("int64")
    for c in ("Dividends", "Stock Splits", "Capital Gains"):
        if c in df and (df[c].fillna(0) == 0).all():
            df = df.drop(columns=c)  # drop all-zero action columns to save space
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, float_format=FLOAT_FMT)
    budget.add(path.stat().st_size)
    return len(df)


def yf_batch(tickers: list[str], tries: int = 4, **kw) -> dict[str, pd.DataFrame]:
    """Download a batch; returns {ticker: frame}. Retries with backoff on rate limits."""
    for attempt in range(tries):
        try:
            data = yf.download(
                tickers, group_by="ticker", auto_adjust=False, actions=True,
                threads=True, progress=False, **kw,
            )
            break
        except Exception as e:  # noqa: BLE001
            wait = 10 * (attempt + 1)
            print(f"    batch error ({e}); retry in {wait}s")
            time.sleep(wait)
    else:
        return {}
    out = {}
    if data is None or data.empty:
        return out
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                out[t] = data[t].copy()
    else:
        out[tickers[0]] = data.copy()
    return out


def fetch_set(label: str, tickers: list[str], folder: Path, budget: Budget,
              batch: int = 40, pause: float = 1.5, **kw) -> list[dict]:
    manifest: list[dict] = []
    for attempt, size in enumerate((batch, 8, 2)):  # retry pass for random Yahoo misses
        todo = [t for t in dict.fromkeys(tickers) if not (folder / f"{safe_name(t)}.csv").exists()]
        if not todo or budget.full:
            break
        if attempt:
            print(f"  retry pass {attempt}: {len(todo)} missing")
            time.sleep(5)
        manifest += _fetch_pass(label, todo, folder, budget, size, pause, attempt == 0, **kw)
    return manifest


def _fetch_pass(label, todo, folder, budget, batch, pause, verbose, **kw) -> list[dict]:
    manifest = []
    if verbose:
        print(f"[{label}] {len(todo)} to download -> {folder.relative_to(ROOT.parent)}")
    for i in range(0, len(todo), batch):
        if budget.full:
            print(f"  budget reached ({budget}); stopping {label}")
            break
        chunk = todo[i:i + batch]
        frames = yf_batch(chunk, **kw)
        for t in chunk:
            df = frames.get(t)
            if df is None:
                continue
            path = folder / f"{safe_name(t)}.csv"
            rows = write_frame(df, path, budget)
            if rows:
                manifest.append({
                    "ticker": t, "dataset": label, "rows": rows,
                    "start": df.dropna(how="all").index.min(), "end": df.index.max(),
                    "file": str(path.relative_to(ROOT)),
                })
        if verbose:
            print(f"  {min(i + batch, len(todo))}/{len(todo)}  [{budget}]")
        time.sleep(pause)
    return manifest


# ----------------------------------------------------------------------------- universes

def wiki_table(url: str, match: str) -> pd.DataFrame:
    html = requests.get(url, headers=UA, timeout=30).text
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    cands = [t for t in tables if ("Symbol" in t.columns or "Ticker" in t.columns) and len(t) >= 90]
    return max(cands, key=len)


def get_constituents() -> dict[str, pd.DataFrame]:
    pages = {
        "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol"),
        "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol"),
        "nasdaq100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    }
    out = {}
    for name, (url, col) in pages.items():
        try:
            df = wiki_table(url, col)
            sym = "Symbol" if "Symbol" in df else "Ticker"
            df = df.rename(columns={sym: "Symbol"})
            df["Symbol"] = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
            out[name] = df
            print(f"  {name}: {len(df)} constituents")
        except Exception as e:  # noqa: BLE001
            print(f"  could not load {name}: {e}")
    return out


# ----------------------------------------------------------------------------- other sources

def fetch_fred(budget: Budget) -> None:
    folder = ROOT / "macro" / "fred"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[fred] {len(FRED_SERIES)} series")
    for s in FRED_SERIES:
        path = folder / f"{s}.csv"
        if path.exists():
            continue
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}",
                             headers=UA, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
            budget.add(path.stat().st_size)
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: failed ({e})")
        time.sleep(0.3)


def fetch_fama_french(budget: Budget) -> None:
    folder = ROOT / "macro" / "fama_french"
    folder.mkdir(parents=True, exist_ok=True)
    print("[fama-french]")
    for name, fname in FAMA_FRENCH.items():
        path = folder / f"{name}.csv"
        if path.exists():
            continue
        try:
            r = requests.get(FF_BASE + fname, headers=UA, timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                path.write_bytes(z.read(z.namelist()[0]))
            budget.add(path.stat().st_size)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: failed ({e})")


def fetch_minute(tickers: list[str], budget: Budget) -> list[dict]:
    """Yahoo serves 1m bars for the last ~30 days, max 8 days per request."""
    folder = ROOT / "minute"
    manifest = []
    todo = [t for t in tickers if not (folder / f"{safe_name(t)}.csv").exists()]
    print(f"[minute] {len(todo)} tickers")
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    windows = [(now - timedelta(days=7 * (k + 1)), now - timedelta(days=7 * k)) for k in range(4)]
    for i in range(0, len(todo), 20):
        if budget.full:
            print(f"  budget reached ({budget}); stopping minute")
            break
        chunk = todo[i:i + 20]
        parts: dict[str, list[pd.DataFrame]] = {t: [] for t in chunk}
        for start, end in windows:
            frames = yf_batch(chunk, interval="1m", start=start.date(), end=end.date(), prepost=False)
            for t, df in frames.items():
                parts[t].append(df)
            time.sleep(1)
        for t, dfs in parts.items():
            if not dfs:
                continue
            df = pd.concat(dfs).sort_index()
            df = df[~df.index.duplicated()]
            path = folder / f"{safe_name(t)}.csv"
            rows = write_frame(df, path, budget)
            if rows:
                manifest.append({"ticker": t, "dataset": "minute", "rows": rows,
                                 "start": df.index.min(), "end": df.index.max(),
                                 "file": str(path.relative_to(ROOT))})
        print(f"  {min(i + 20, len(todo))}/{len(todo)}  [{budget}]")
    return manifest


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="tiny test run")
    args = ap.parse_args()

    ROOT.mkdir(exist_ok=True)
    budget = Budget(args.budget_gb)
    print(f"Output: {ROOT}  budget {budget}")

    # 1. universes
    meta = ROOT / "meta"
    meta.mkdir(exist_ok=True)
    print("[constituents]")
    cons = get_constituents()
    for name, df in cons.items():
        df.to_csv(meta / f"{name}_constituents.csv", index=False)
    sp500 = cons.get("sp500", pd.DataFrame({"Symbol": []}))["Symbol"].tolist()
    ndx = cons.get("nasdaq100", pd.DataFrame({"Symbol": []}))["Symbol"].tolist()
    mid = cons.get("sp400", pd.DataFrame({"Symbol": []}))["Symbol"].tolist()
    small = cons.get("sp600", pd.DataFrame({"Symbol": []}))["Symbol"].tolist()
    large = list(dict.fromkeys(sp500 + ndx))

    if args.quick:
        large, mid, small = large[:5], [], []

    manifest: list[dict] = []
    d = ROOT / "daily"

    # 2. small, high-value sets first, then the big stock universes
    fetch_fred(budget)
    fetch_fama_french(budget)
    manifest += fetch_set("daily_etfs", ETFS[: 5 if args.quick else None], d / "etfs", budget, period="max")
    manifest += fetch_set("daily_indices", INDICES, d / "indices", budget, period="max")
    manifest += fetch_set("daily_futures", FUTURES, d / "futures", budget, period="max")
    manifest += fetch_set("daily_fx", FX, d / "fx", budget, period="max")
    manifest += fetch_set("daily_crypto", CRYPTO, d / "crypto", budget, period="max")
    manifest += fetch_set("daily_large_cap", large, d / "stocks_large", budget, period="max")
    manifest += fetch_set("daily_mid_cap", mid, d / "stocks_mid", budget, period="max")
    manifest += fetch_set("daily_small_cap", small, d / "stocks_small", budget, period="max")

    # 3. intraday
    hourly = (ETFS[:40] + INDICES[:14] + FUTURES[:18] + FX[:10] + CRYPTO[:15] + large)
    if args.quick:
        hourly = hourly[:5]
    manifest += fetch_set("hourly", hourly, ROOT / "hourly", budget, batch=30,
                          interval="1h", period="730d")
    minute = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "^VIX", "ES=F", "NQ=F", "CL=F", "GC=F",
              "BTC-USD", "ETH-USD", "EURUSD=X"] + sp500[:80]
    manifest += fetch_minute(minute[:3] if args.quick else minute, budget)

    # 4. manifest (merge with previous runs)
    mpath = meta / "manifest.csv"
    new = pd.DataFrame(manifest)
    if mpath.exists():
        old = pd.read_csv(mpath)
        new = pd.concat([old, new]).drop_duplicates(subset=["file"], keep="last")
    new.to_csv(mpath, index=False)
    print(f"\nDone. {len(new)} files in manifest; total size {dir_size(ROOT) / 1024**2:,.0f} MB")


if __name__ == "__main__":
    main()
