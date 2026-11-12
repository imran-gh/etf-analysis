"""Turn out/results.json + out/combos_long.json into the dashboard's CSV files (out/dash/)."""
import json, pandas as pd, numpy as np
r = json.load(open('out/results.json')); D = 'out/dash/'
L = r['log']; f5 = r['five_year']; s = r['spy']
rows = [dict(window=lab, cagr=s[w]['cagr'], vol=s[w]['vol'], max_drawdown=s[w]['mdd'], sharpe=s[w]['sharpe'])
        for w, lab in [('train', '2011–2019'), ('test', '2020–2026'), ('full', '2011–2026')]]
v = f5['spy']; rows.append(dict(window='Last 5 years', cagr=v['cagr'], vol=v['vol'], max_drawdown=v['mdd'], sharpe=v['sharpe']))
pd.DataFrame(rows).round(4).to_csv(D + 'spy.csv', index=False)
fun = [('US-listed ETFs downloaded', L['universe']), ('Not leveraged or inverse', L['after_lev_filter']),
       ('Full history since 2011, clean data', L['long_history_clean']),
       ('Beat SPY with lower volatility, 2011–2019', r['picked_count']),
       ('…and still did in 2020–2026', r['picked_held']),
       ('Statistically significant after correcting for luck', r['significant'])]
pd.DataFrame(fun, columns=['stage', 'count']).to_csv(D + 'funnel.csv', index=False)
pd.DataFrame([dict(ticker=x['ticker'], name=x['name'], cagr_2011_19=x['tr_cagr'], vol_2011_19=x['tr_vol'], cagr_2020_26=x['te_cagr'],
                   vol_2020_26=x['te_vol'], held_up='Yes' if x['held_up'] else 'No') for x in r['picked']]).to_csv(D + 'picks.csv', index=False)
pd.DataFrame([dict(ticker=x['ticker'], name=x['name'], cagr=x['fu_cagr'], vol=x['fu_vol'], max_drawdown=x['fu_mdd'], sharpe=x['fu_sharpe'])
              for x in r['full_beat']]).to_csv(D + 'fullperiod.csv', index=False)
pd.DataFrame([dict(ticker=x['ticker'], name=x['name'], cagr=x['cagr'], vol=x['vol'], max_drawdown=x['mdd'], sharpe=x['sharpe'], beta=x['beta'])
              for x in f5['rows'][:20]]).to_csv(D + 'fiveyear.csv', index=False)
pd.DataFrame([dict(funds_checked=f5['n'], passed=f5['count'], start=f5['start'])]).to_csv(D + 'fiveyear_summary.csv', index=False)
cs = r['combos']; sel = cs[:8] + [c for c in cs[8:] if c['held_up']]
lab = lambda c: ' + '.join(f"{int(round(w*100))}% {f}" for f, w in zip(c['funds'], c['weights']))
pd.DataFrame([dict(rank=cs.index(c) + 1, portfolio=lab(c), cagr_2011_19=c['train']['cagr'], vol_2011_19=c['train']['vol'],
                   cagr_2020_26=c['test']['cagr'], vol_2020_26=c['test']['vol'], mdd_2020_26=c['test']['mdd'],
                   held_up='Yes' if c['held_up'] else 'No') for c in sel]).round(4).to_csv(D + 'combos.csv', index=False)
pd.DataFrame([dict(passed_train=L['combos_passing_train'], held_test=L['combos_held_test'],
                   random_held_rate=round(L['random_combo_held_rate'], 4))]).to_csv(D + 'combo_summary.csv', index=False)
# long-term (hindsight) mixes
c = json.load(open('out/combos_long.json'))
lm = [dict(size=n, mix=x['mix'], cagr=round(x['full']['cagr'], 4), vol=round(x['full']['vol'], 4), max_drawdown=round(x['full']['mdd'], 4),
           cagr_2011_19=round(x['h1']['cagr'], 4), cagr_2020_26=round(x['h2']['cagr'], 4), vol_2020_26=round(x['h2']['vol'], 4),
           win36=round(x['win36'], 4)) for n, k in ((2, 'top2'), (3, 'top3')) for x in c[k][:6]]
pd.DataFrame(lm).to_csv(D + 'longmix.csv', index=False)
pd.DataFrame([dict(funds=c['n_funds'], distinct=c['n_pool'], pool3=c['n_pool3'], win2=c['n_win2'], win3=c['n_win3'])]).to_csv(D + 'longmix_summary.csv', index=False)
