"""
Long-term 2- and 3-ETF mixes vs SPY, 2011-03 .. latest.
A mix "wins" if, over the full period AND separately in 2011-2019 and 2020-2026,
it has higher return and lower volatility than SPY. Rebalanced monthly.
Output: out/combos_long.json, out/combos_long.csv
"""
import itertools, json, sys
import numpy as np, pandas as pd

DATA, OUT = "data", "out"
START, SPLIT = "2011-03-01", "2020-01-01"
uni = pd.read_csv(f"{DATA}/etf_universe.csv")
names = dict(zip(uni.yf_symbol, uni.name))
lev = set(uni.loc[uni.leveraged_or_inverse, "yf_symbol"])
px = pd.read_parquet(f"{DATA}/prices.parquet").astype(float).sort_index()
px = px.loc[:px["SPY"].last_valid_index()]

cols = [c for c in px.columns if not c.startswith("^") and c not in lev]
p = px.loc[START:, cols]
p = p.loc[:, p.iloc[:10].notna().all() & (p.notna().mean() > 0.98)].ffill(limit=5)
d = p.pct_change(fill_method=None).iloc[1:]
d = d.loc[:, ((d == 0).mean() < 0.15) & (d.abs().max() < 0.40)]
m = (1 + d).resample("ME").prod() - 1
m = m.loc[:"2026-10-31"]
d = d.loc[:"2026-10-31"]
# drop S&P 500 near-copies (VOO, IVV...) except SPY itself
cl = d.corrwith(d["SPY"]); d = d.drop(columns=[c for c in cl[cl > 0.995].index if c != "SPY"])
m = m[d.columns]
print("funds", m.shape[1], "months", len(m))

h1 = m.index < SPLIT
def mstats(R):
    """R: months x k. returns cagr, vol (monthly-based, annualised)."""
    n = R.shape[0]
    return np.prod(1 + R, 0) ** (12 / n) - 1, R.std(0, ddof=1) * np.sqrt(12)
M = m.values; b = m["SPY"].values[:, None]
spy = {k: [float(x[0]) for x in mstats(b[s])] for k, s in (("full", slice(None)), ("h1", h1), ("h2", ~h1))}

# pool: de-duplicate near-identical funds (corr > .95), keep the higher-Sharpe one
sh = (m.mean() / m.std()).sort_values(ascending=False)
corr = m.corr().values; idx = {c: i for i, c in enumerate(m.columns)}
pool = []
for t in sh.index:
    if all(corr[idx[t], idx[s]] < 0.95 for s in pool): pool.append(t)
print("distinct funds", len(pool))

def grid(k, step):
    return np.array([w for w in itertools.product(range(step, 101, step), repeat=k) if sum(w) == 100]) / 100

def search(k, funds, step):
    G = grid(k, step); P = [idx[f] for f in funds]; found = []
    sets = (("full", slice(None)), ("h1", h1), ("h2", ~h1))
    for combo in itertools.combinations(P, k):
        R = M[:, combo] @ G.T
        ok = np.ones(len(G), bool); st = {}
        for key, s in sets:
            c, v = mstats(R[s]); st[key] = (c, v)
            ok &= (c > spy[key][0]) & (v < spy[key][1])
            if not ok.any(): break
        for j in np.where(ok)[0]:
            found.append(dict(funds=[m.columns[i] for i in combo], w=G[j].tolist(),
                              **{f"{key}_{n}": float(st[key][q][j]) for key in st for q, n in ((0, "cagr"), (1, "vol"))}))
    return found

res2 = search(2, pool, 10)
print("2-fund winners", len(res2))
pool3 = pool[:140]
res3 = search(3, pool3, 10)
print("3-fund winners", len(res3))

# daily-vol check + rolling 3y win rate for reported mixes
def daily(funds, w):
    w = np.asarray(w); dd = d[funds]; parts = []
    for _, g in dd.groupby(dd.index.to_period("M")):
        v = (1 + g).cumprod() * w; parts.append(v.sum(1).pct_change().fillna(float((1 + g.iloc[0]) @ w - 1)))
    r = pd.concat(parts); mm = (1 + r).resample("ME").prod() - 1
    out = {}
    for key, a, z in (("full", None, None), ("h1", None, "2019-12-31"), ("h2", SPLIT, None)):
        rr, mx = r.loc[a:z], mm.loc[a:z]
        wl = (1 + rr).cumprod()
        out[key] = dict(cagr=float((1 + mx).prod() ** (12 / len(mx)) - 1), vol=float(rr.std() * np.sqrt(252)),
                        mdd=float((wl / wl.cummax() - 1).min()))
    rb = (1 + m["SPY"]).rolling(36).apply(np.prod, raw=True); rp = (1 + mm).rolling(36).apply(np.prod, raw=True)
    out["win36"] = float((rp > rb)[rb.notna()].mean())
    return out

spy_d = daily(["SPY"], [1.0])
def report(res, n=12):
    df = pd.DataFrame(res)
    if df.empty: return []
    df["edge"] = (df.full_cagr - spy["full"][0]) - 0.5 * (df.full_vol - spy["full"][1])  # more return, less vol
    df["key"] = df.funds.map(lambda f: tuple(sorted(f)))
    df = df.sort_values("edge", ascending=False).drop_duplicates("key")
    out = []
    for _, r in df.head(n * 3).iterrows():
        s = daily(r.funds, r.w)
        if not all(s[k]["cagr"] > spy_d[k]["cagr"] and s[k]["vol"] < spy_d[k]["vol"] for k in ("full", "h1", "h2")):
            continue
        out.append(dict(mix=" + ".join(f"{int(round(w*100))}% {f}" for f, w in zip(r.funds, r.w)),
                        funds={f: names.get(f, f) for f in r.funds}, **s))
        if len(out) >= n: break
    return out

top2, top3 = report(res2), report(res3)
# how common are the fund ingredients among all winners
ing = pd.Series([f for r in res2 + res3 for f in r["funds"]]).value_counts().head(15)
json.dump(dict(spy=spy_d, n_funds=int(m.shape[1]), n_pool=len(pool), n_pool3=len(pool3),
               n_win2=len(res2), n_win3=len(res3), n_pairs=len(pool) * (len(pool) - 1) // 2,
               top2=top2, top3=top3, ingredients=ing.to_dict()),
          open(f"{OUT}/combos_long.json", "w"), indent=1)
print(json.dumps(dict(spy=spy_d, ing=ing.to_dict()), indent=1))
for r in top2 + top3:
    print(r["mix"], {k: (round(r[k]["cagr"], 3), round(r[k]["vol"], 3)) for k in ("full", "h1", "h2")}, round(r["full"]["mdd"], 3), round(r["win36"], 2))
