"""
Download every US-listed ETF (incl. international ones like VXUS) and their
daily total-return (dividend-adjusted) prices. Runs in GitHub Actions.

Outputs (in data/):
  etf_universe.csv   - symbol, name, exchange, flags (leveraged/inverse)
  prices.parquet     - daily adjusted close, one column per ETF (float32)
  fetch_log.txt      - what failed
"""
import io, re, time, sys
import pandas as pd
import requests
import yfinance as yf

START = "2005-01-01"
BATCH = 100
OUT = "data"

# 1) Universe: Nasdaq Trader's file lists every US-traded security (all exchanges),
#    with an ETF=Y flag. This includes NYSE Arca / Cboe listings like VXUS.
url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
txt = requests.get(url, timeout=60).text
df = pd.read_csv(io.StringIO(txt), sep="|")
df = df[df["Symbol"].notna() & ~df["Nasdaq Traded"].astype(str).str.startswith("File Creation")]
etf = df[(df["ETF"] == "Y") & (df["Test Issue"] == "N")].copy()

lev = re.compile(r"\b(2X|3X|-1X|-2X|-3X|ULTRA\w*|INVERSE|BEAR|LEVERAGED?|DAILY)\b|PROSHARES ULTRA|\dX ?LEVERAGED|LEVERAGE SHARES"
                 r"|\bSHORT\b(?![- ]?(TERM|DURATION|MATURITY|TREASURY|BOND|INCOME|CORP|HIGH|MUNI|TIPS|INVEST|GOVERN|DATED))", re.I)
# ultra-short-duration bond funds are ordinary, unleveraged funds
plain_bond = re.compile(r"ULTRA.?SHORT (INCOME|BOND|FIXED|TREASURY|DURATION|TERM|MUNI|GOVERN)", re.I)
etf["leveraged_or_inverse"] = etf["Security Name"].map(lambda n: bool(lev.search(n)) and not (plain_bond.search(n) and "PROSHARES" not in n.upper()))
etf["yf_symbol"] = etf["Symbol"].str.replace(".", "-", regex=False)
etf = etf[["Symbol", "yf_symbol", "Security Name", "Listing Exchange", "leveraged_or_inverse"]]
etf.columns = ["symbol", "yf_symbol", "name", "exchange", "leveraged_or_inverse"]

import os
os.makedirs(OUT, exist_ok=True)
etf.to_csv(f"{OUT}/etf_universe.csv", index=False)
print(f"{len(etf)} ETFs in universe")

# 2) Prices: dividend-adjusted closes in batches, with retries.
syms = etf["yf_symbol"].tolist() + ["^GSPC", "^SP500TR"]
frames, failed = [], []
for i in range(0, len(syms), BATCH):
    chunk = syms[i:i + BATCH]
    for attempt in range(4):
        try:
            d = yf.download(chunk, start=START, auto_adjust=True, progress=False,
                            threads=True, group_by="column")["Close"]
            if isinstance(d, pd.Series):
                d = d.to_frame(chunk[0])
            frames.append(d.astype("float32"))
            break
        except Exception as e:
            print(f"batch {i} attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(20 * (attempt + 1))
    else:
        failed += chunk
    time.sleep(3)
    print(f"{min(i + BATCH, len(syms))}/{len(syms)}")

prices = pd.concat(frames, axis=1)
prices = prices.loc[:, prices.notna().sum() > 0]
missing = sorted(set(syms) - set(prices.columns))
prices.to_parquet(f"{OUT}/prices.parquet", compression="zstd")

with open(f"{OUT}/fetch_log.txt", "w") as f:
    f.write(f"universe={len(etf)} downloaded={prices.shape[1]} rows={prices.shape[0]}\n")
    f.write("missing:\n" + "\n".join(missing))
print(f"saved {prices.shape}")
