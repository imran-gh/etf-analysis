"""
ETF vs S&P 500 analysis.

Selection window (train): 2011-03 .. 2019-12  (starts when VXUS has data)
Validation window (test): 2020-01 .. latest   (never used for selection)

Outputs (out/): metrics.csv, results.json
"""
import json, itertools, sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
TRAIN = ("2011-03-01", "2019-12-31")
TEST = ("2020-01-01", None)
BENCH, RF = "SPY", "BIL"
import os; os.makedirs(OUT, exist_ok=True)

uni = pd.read_csv(f"{DATA}/etf_universe.csv")
px = pd.read_parquet(f"{DATA}/prices.parquet").astype("float64")
px.index = pd.to_datetime(px.index)
px = px.sort_index()
last = px[BENCH].last_valid_index()
px = px.loc[:last]

# ---------- universe filters ----------
log = {"universe": int(len(uni)), "with_prices": int(px.shape[1])}
names = dict(zip(uni.yf_symbol, uni.name))
lev = set(uni.loc[uni.leveraged_or_inverse, "yf_symbol"])
cols = [c for c in px.columns if not c.startswith("^") and c not in lev]
log["after_lev_filter"] = len(cols)

full = px.loc[TRAIN[0]:, cols]
first_ok = full.apply(lambda s: s.first_valid_index())
cols = [c for c in cols if first_ok[c] is not None and first_ok[c] <= pd.Timestamp(TRAIN[0]) + pd.Timedelta(days=10)]
full = full[cols].ffill(limit=5)
ok = full.notna().mean() > 0.98
full = full.loc[:, ok]
dret = full.pct_change(fill_method=None).iloc[1:]
stale = (dret == 0).mean() > 0.15          # illiquid: price often unchanged
spike = dret.abs().max() > 0.40             # bad data / splits not adjusted
dret = dret.loc[:, ~stale & ~spike]
log["long_history_clean"] = int(dret.shape[1])
cols = list(dret.columns)

mret = (1 + dret).resample("ME").prod() - 1
mret = mret.iloc[:-1] if mret.index[-1].month == last.month and last.day < 25 else mret

def metrics(m, d, rf_m):
    """m: monthly returns frame, d: daily returns frame, rf_m: monthly rf series"""
    n = len(m)
    growth = (1 + m).prod()
    cagr = growth ** (12 / n) - 1
    vol = d.std() * np.sqrt(252)
    wealth = (1 + d).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    ex = m.sub(rf_m, axis=0)
    sharpe = ex.mean() / m.std() * np.sqrt(12)
    dn = m.where(m < 0, 0)
    sortino = ex.mean() / np.sqrt((dn ** 2).mean()) * np.sqrt(12)
    b = m[BENCH]
    beta = m.apply(lambda s: s.cov(b)) / b.var()
    up, down = b > 0, b < 0
    upcap = m[up].mean() / b[up].mean()
    dncap = m[down].mean() / b[down].mean()
    act = m.sub(b, axis=0)
    ir = act.mean() / act.std() * np.sqrt(12)
    # rolling 36m: share of windows where fund's cumulative return > SPY's
    roll = (1 + m).rolling(36).apply(np.prod, raw=True)
    win36 = roll.gt(roll[BENCH], axis=0)[roll[BENCH].notna()].mean()
    return pd.DataFrame(dict(cagr=cagr, vol=vol, mdd=mdd, sharpe=sharpe, sortino=sortino,
                             beta=beta, upcap=upcap, dncap=dncap, ir=ir, win36=win36))

def window(a, b):
    m = mret.loc[a:b]; d = dret.loc[a:b]
    return metrics(m, d, m[RF] if RF in m else 0 * m[BENCH])

tr, te, fu = window(*TRAIN), window(*TEST), window(TRAIN[0], None)
spy = {k: v.loc[BENCH].to_dict() for k, v in (("train", tr), ("test", te), ("full", fu))}

# ---------- significance of beating SPY (full period), BH-FDR ----------
from scipy import stats
act = mret.sub(mret[BENCH], axis=0)
def nw_t(x, lags=6):
    x = x.dropna().values; n = len(x); mu = x.mean(); e = x - mu
    g0 = (e @ e) / n
    s = g0 + 2 * sum((1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n for l in range(1, lags + 1))
    return mu / np.sqrt(s / n)
tstat = act.drop(columns=BENCH).apply(nw_t)
p = pd.Series(stats.t.sf(tstat, len(act) - 1), index=tstat.index)
order = p.sort_values()
bh = (order * len(order) / np.arange(1, len(order) + 1))[::-1].cummin()[::-1].clip(upper=1)
fdr_q = bh.reindex(p.index)

# ---------- assemble table ----------
tab = pd.concat({"tr": tr, "te": te, "fu": fu}, axis=1)
tab.columns = [f"{a}_{b}" for a, b in tab.columns]
tab["name"] = [names.get(c, c) for c in tab.index]
tab["q_beat_spy"] = fdr_q
tab.to_csv(f"{OUT}/metrics.csv", float_format="%.5f")

s_tr = tr.loc[BENCH]
crit = (tr.cagr > s_tr.cagr) & (tr.vol < s_tr.vol) & (tr.win36 >= 0.6)
picked = tab[crit.reindex(tab.index, fill_value=False)].drop(index=BENCH, errors="ignore")
picked = picked.sort_values("tr_sharpe", ascending=False)
s_te = te.loc[BENCH]
picked["held_up"] = (picked.te_cagr > s_te.cagr) & (picked.te_vol < s_te.vol)

# full-period (in-sample) view, de-duplicated near-clones
s_fu = fu.loc[BENCH]
fp = tab[(tab.fu_cagr > s_fu.cagr) & (tab.fu_vol < s_fu.vol)].drop(index=BENCH, errors="ignore")
riskadj = tab[(tab.fu_sharpe > s_fu.sharpe) & (tab.fu_vol < s_fu.vol)].drop(index=BENCH, errors="ignore")

def dedupe(df, key, k=25, thr=0.97):
    keep = []
    corr = mret[df.index].corr()
    for t in df.sort_values(key, ascending=False).index:
        if all(corr.loc[t, s] < thr for s in keep):
            keep.append(t)
        if len(keep) >= k: break
    return df.loc[keep]

# ---------- combinations ----------
# Pool: de-duplicated top funds by train Sharpe + core diversifiers
divers = [t for t in ["VXUS", "VEA", "VWO", "BND", "AGG", "IEF", "TLT", "GLD", "IAU", "VNQ",
                      "QQQ", "VTI", "VIG", "USMV", "SPLV", "MTUM", "QUAL", "BIL", "SHY", "TIP", "DBC"] if t in cols]
cand = tab.drop(index=BENCH).sort_values("tr_sharpe", ascending=False)
pool = list(dedupe(cand.head(300), "tr_sharpe", k=24).index)
pool = list(dict.fromkeys(pool + divers))[:40]
log["combo_pool"] = len(pool)

M_tr = mret.loc[TRAIN[0]:TRAIN[1], pool].values
M_te = mret.loc[TEST[0]:, pool].values
b_tr = mret.loc[TRAIN[0]:TRAIN[1], BENCH].values
b_te = mret.loc[TEST[0]:, BENCH].values

def grid(k, step=10):
    return np.array([w for w in itertools.product(range(step, 101, step), repeat=k) if sum(w) == 100]) / 100

def stats_m(R, b):
    cagr = np.prod(1 + R, axis=0) ** (12 / R.shape[0]) - 1
    vol = R.std(axis=0, ddof=1) * np.sqrt(12)
    W = np.cumprod(1 + R, axis=0); mdd = (W / np.maximum.accumulate(W, axis=0) - 1).min(axis=0)
    cw = np.cumprod(1 + R, axis=0); cb = np.cumprod(1 + b)
    roll = cw[36:] / cw[:-36]; rb = cb[36:] / cb[:-36]
    win = (roll > rb[:, None]).mean(axis=0)
    return cagr, vol, mdd, win

bc_tr, bv_tr, bm_tr, _ = stats_m(b_tr[:, None], b_tr)
bc_te, bv_te, bm_te, _ = stats_m(b_te[:, None], b_te)
res = []
for k in (2, 3, 4):
    G = grid(k, 10 if k < 4 else 20 if len(pool) > 30 else 10)
    for idx in itertools.combinations(range(len(pool)), k):
        R = M_tr[:, idx] @ G.T
        c, v, m, w = stats_m(R, b_tr)
        ok = (c > bc_tr[0]) & (v < 0.9 * bv_tr[0]) & (w >= 0.6)
        for j in np.where(ok)[0]:
            res.append((idx, G[j], c[j], v[j], m[j], w[j]))
log["combos_passing_train"] = len(res)

combos = []
if res:
    df = pd.DataFrame(res, columns=["idx", "w", "c", "v", "m", "win"])
    df["score"] = (df.c - bc_tr[0]) / df.v      # excess return per unit vol
    df = df.sort_values("score", ascending=False)
    seen = set()
    for _, r in df.iterrows():
        key = tuple(sorted(r.idx))
        if key in seen: continue
        seen.add(key)
        Rte = M_te[:, list(r.idx)] @ r.w
        c2, v2, m2, w2 = (x[0] for x in stats_m(Rte[:, None], b_te))
        combos.append(dict(funds=[pool[i] for i in r.idx], weights=[round(float(x), 2) for x in r.w],
                           train=dict(cagr=r.c, vol=r.v, mdd=r.m, win36=r.win),
                           test=dict(cagr=float(c2), vol=float(v2), mdd=float(m2), win36=float(w2)),
                           held_up=bool(c2 > bc_te[0] and v2 < bv_te[0])))
        if len(combos) >= 30: break

# ---------- growth curves for dashboard ----------
def curve(tickers, weights=None):
    m = mret.loc[TRAIN[0]:, tickers]
    r = m @ np.array(weights) if weights is not None else m.iloc[:, 0]
    return [round(float(x), 4) for x in (1 + r).cumprod()]

def rows(df, n=25):
    keep = ["name", "tr_cagr", "tr_vol", "tr_win36", "tr_sharpe", "te_cagr", "te_vol", "te_mdd", "te_win36",
            "fu_cagr", "fu_vol", "fu_mdd", "fu_sharpe", "fu_beta", "fu_upcap", "fu_dncap", "q_beat_spy"]
    out = df.head(n)[[c for c in keep if c in df]].copy()
    if "held_up" in df: out["held_up"] = df.held_up.head(n)
    return [dict(ticker=t, **{k: (None if pd.isna(v) else (round(float(v), 4) if not isinstance(v, (str, bool, np.bool_)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)))
                               for k, v in r.items()}) for t, r in out.iterrows()]

result = dict(
    as_of=str(mret.index[-1].date()), train=TRAIN, test=[TEST[0], str(mret.index[-1].date())],
    log=log, spy=spy,
    picked=rows(picked, 40), picked_count=int(len(picked)), picked_held=int(picked.held_up.sum()) if len(picked) else 0,
    full_beat=rows(dedupe(fp, "fu_sharpe") if len(fp) else fp), full_beat_count=int(len(fp)),
    riskadj=rows(dedupe(riskadj, "fu_sharpe") if len(riskadj) else riskadj), riskadj_count=int(len(riskadj)),
    significant=int((fdr_q < 0.10).sum()),
    combos=combos,
    curves=dict(dates=[str(d.date()) for d in mret.loc[TRAIN[0]:].index],
                SPY=curve([BENCH]),
                picks={t: curve([t]) for t in list(picked.index[:5])},
                combos={" + ".join(f"{int(w*100)}% {f}" for f, w in zip(c["funds"], c["weights"])): curve(c["funds"], c["weights"]) for c in combos[:5]}),
)
json.dump(result, open(f"{OUT}/results.json", "w"), default=float)
print(json.dumps(dict(log=log, picked=len(picked), held=result["picked_held"], full_beat=len(fp),
                      riskadj=len(riskadj), significant=result["significant"], combos=len(combos),
                      held_combos=sum(c["held_up"] for c in combos)), indent=1))
