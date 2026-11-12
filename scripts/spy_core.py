"""
SPY-core mixes: SPY + 1 or 2 other ETFs vs SPY alone, 2011-03 .. 2026-10, rebalanced monthly.
Hindsight test: higher return AND lower volatility than SPY over full period and in both halves.
Fair test: pick on 2011-2019 only, check 2020-2026; compare with random SPY-core mixes.
Output: out/spy_core.json
"""
import itertools, json
import numpy as np, pandas as pd

DATA, OUT = "data", "out"
START, SPLIT, END = "2011-03-01", "2020-01-01", "2026-10-31"
uni = pd.read_csv(f"{DATA}/etf_universe.csv")
names = dict(zip(uni.yf_symbol, uni.name))
lev = set(uni.loc[uni.leveraged_or_inverse, "yf_symbol"])
px = pd.read_parquet(f"{DATA}/prices.parquet").astype(float).sort_index()

cols = [c for c in px.columns if not c.startswith("^") and c not in lev]
p = px.loc[START:END, cols]
p = p.loc[:, p.iloc[:10].notna().all() & (p.notna().mean() > 0.98)].ffill(limit=5)
d = p.pct_change(fill_method=None).iloc[1:]
d = d.loc[:, ((d == 0).mean() < 0.15) & (d.abs().max() < 0.40)]
cl = d.corrwith(d["SPY"]); d = d.drop(columns=[c for c in cl[cl > 0.995].index if c != "SPY"])
m = (1 + d).resample("ME").prod() - 1

# distinct partners (drop near-duplicates, corr > .95), SPY excluded
sh = (m.mean() / m.std()).sort_values(ascending=False)
C = m.corr()
partners = []
for t in sh.index:
    if t == "SPY": continue
    if all(C.loc[t, s] < 0.95 for s in partners): partners.append(t)
print("partners", len(partners))

M = m.values; col = {c: i for i, c in enumerate(m.columns)}; S = col["SPY"]
h1 = (m.index < SPLIT); h2 = ~h1
def st(R):
    return np.prod(1 + R, 0) ** (12 / R.shape[0]) - 1, R.std(0, ddof=1) * np.sqrt(12)
spy = {k: [float(x[0]) for x in st(M[s][:, [S]])] for k, s in (("full", slice(None)), ("h1", h1), ("h2", h2))}

def grid3(step=10):
    return np.array([w for w in itertools.product(range(step, 101, step), repeat=3) if sum(w) == 100]) / 100
W2 = np.array([[a / 100, 1 - a / 100] for a in range(10, 100, 10)])   # SPY weight 10..90%
W3 = grid3()

rows = []   # every SPY-core mix with its stats per window
def run(idx_sets, W, k):
    for idx in idx_sets:
        R = {key: M[s][:, idx] @ W.T for key, s in (("full", slice(None)), ("h1", h1), ("h2", h2))}
        S_ = {key: st(R[key]) for key in R}
        for j in range(len(W)):
            rows.append((k, idx, j, *(float(S_[key][q][j]) for key in ("full", "h1", "h2") for q in (0, 1))))

run([(S, col[a]) for a in partners], W2, 2)
run([(S, col[a], col[b]) for a, b in itertools.combinations(partners, 2)], W3, 3)
df = pd.DataFrame(rows, columns=["k", "idx", "j", "fc", "fv", "c1", "v1", "c2", "v2"])
beats = lambda c, v, key: (df[c] > spy[key][0]) & (df[v] < spy[key][1])
df["b_full"], df["b1"], df["b2"] = beats("fc", "fv", "full"), beats("c1", "v1", "h1"), beats("c2", "v2", "h2")
df["all3"] = df.b_full & df.b1 & df.b2

def label(r):
    W = W2 if r.k == 2 else W3
    return " + ".join(f"{int(round(w*100))}% {m.columns[i]}" for i, w in zip(r.idx, W[r.j]))

out = dict(spy=spy, partners=len(partners))
for k in (2, 3):
    sub = df[df.k == k]
    win = sub[sub.all3].copy()
    win["key"] = win.idx.map(lambda t: tuple(sorted(t)))
    win["edge"] = (win.fc - spy["full"][0]) - 0.5 * (win.fv - spy["full"][1])
    top = win.sort_values("edge", ascending=False).drop_duplicates("key").head(8)
    # fair test
    picked = sub[sub.b1]
    out[f"k{k}"] = dict(
        mixes_tested=int(len(sub)), sets_tested=int(sub.idx.nunique()),
        hindsight_mixes=int(len(win)), hindsight_sets=int(win.key.nunique()),
        fair_picked=int(len(picked)), fair_held=int(picked.b2.sum()),
        fair_rate=float(picked.b2.mean()) if len(picked) else None,
        base_rate=float(sub.b2.mean()),
        top=[dict(mix=label(r), names={m.columns[i]: names.get(m.columns[i], m.columns[i]) for i in r.idx},
                  full=[r.fc, r.fv], h1=[r.c1, r.v1], h2=[r.c2, r.v2]) for _, r in top.iterrows()])
    # top fair picks by 2011-2019 edge, and how they did after
    fp = picked.assign(e1=(picked.c1 - spy["h1"][0]) - 0.5 * (picked.v1 - spy["h1"][1]))
    fp = fp.assign(key=fp.idx.map(lambda t: tuple(sorted(t)))).sort_values("e1", ascending=False).drop_duplicates("key").head(5)
    out[f"k{k}"]["fair_top"] = [dict(mix=label(r), h1=[r.c1, r.v1], h2=[r.c2, r.v2], held=bool(r.b2)) for _, r in fp.iterrows()]

# partner frequency among hindsight winners
w = df[df.all3]
out["ingredients"] = pd.Series([m.columns[i] for t in w.idx for i in t if i != S]).value_counts().head(12).to_dict()
json.dump(out, open(f"{OUT}/spy_core.json", "w"), indent=1, default=float)
print(json.dumps({k: (v if k not in ("k2", "k3") else {a: b for a, b in v.items() if a not in ("top", "fair_top")}) for k, v in out.items()}, indent=1))
for k in ("k2", "k3"):
    print(k, "hindsight top")
    for r in out[k]["top"]: print(" ", r["mix"], [round(x, 3) for x in r["h1"] + r["h2"] + r["full"]])
    print(k, "fair top (picked on 2011-19)")
    for r in out[k]["fair_top"]: print(" ", r["mix"], [round(x, 3) for x in r["h1"] + r["h2"]], r["held"])

# ---- strict re-check: every hindsight winner with daily-return volatility ----
def daily_stats(f, w):
    w = np.asarray(w); dd = d[f]
    W_ = (1 + dd).groupby(dd.index.to_period("M")).cumprod()
    port = (W_ * w).sum(1)
    first = dd.index.to_period("M").to_series(index=dd.index).ne(dd.index.to_period("M").to_series(index=dd.index).shift())
    prev = port.shift(1).where(~first, 1.0)
    r = port / prev - 1
    res = {}
    for key, a, z in (("h1", None, "2019-12-31"), ("h2", SPLIT, None), ("full", None, None)):
        rr = r.loc[a:z]; mm = (1 + rr).resample("ME").prod() - 1; wl = (1 + rr).cumprod()
        res[key] = [float((1 + mm).prod() ** (12 / len(mm)) - 1), float(rr.std() * np.sqrt(252)), float((wl / wl.cummax() - 1).min())]
    return res
sd = daily_stats(["SPY"], [1.0])
strict = []
for _, r in df[df.all3].iterrows():
    W = W2 if r.k == 2 else W3
    f = [m.columns[i] for i in r.idx]; s_ = daily_stats(f, W[r.j])
    if all(s_[k][0] > sd[k][0] and s_[k][1] < sd[k][1] for k in ("h1", "h2", "full")):
        strict.append(dict(mix=label(r), k=int(r.k), spy_w=float(W[r.j][0]), **s_))
sdf = pd.DataFrame(strict)
out["spy_daily"] = sd
out["strict_count"] = int(len(sdf)); out["strict_sets"] = int(sdf.mix.map(lambda x: tuple(sorted(t.split("% ")[1] for t in x.split(" + ")))).nunique()) if len(sdf) else 0
if len(sdf):
    sdf["key"] = sdf.mix.map(lambda x: tuple(sorted(t.split("% ")[1] for t in x.split(" + "))))
    sdf["edge"] = sdf.full.map(lambda v: v[0]) - sd["full"][0] - 0.5 * (sdf.full.map(lambda v: v[1]) - sd["full"][1])
    best = sdf.sort_values("edge", ascending=False).drop_duplicates("key")
    out["strict_top"] = best.head(10).drop(columns=["key", "edge"]).to_dict("records")
    out["strict_max_spy_w"] = float(sdf.spy_w.max())
    out["strict_spy_w_counts"] = sdf.spy_w.round(1).value_counts().sort_index().to_dict()
json.dump(out, open(f"{OUT}/spy_core.json", "w"), indent=1, default=float)
print("strict winners", len(sdf), "sets", out["strict_sets"], "SPY weight counts", out.get("strict_spy_w_counts"))
for r in out.get("strict_top", []):
    print(" ", r["mix"], [round(x, 3) for k in ("h1", "h2", "full") for x in r[k][:2]], round(r["full"][2], 3))
