"""rq1_fb.py <gran> [B]: RQ1 primary design, sensitivity variants, pooled model, RMST bootstrap, per-agent results,
14-day rates and Table 2 statistics for one unit (line, hunk or file). Writes results/rq1/rq1_<gran>.json.
B = bootstrap replicates (paper: 1000)."""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
gran = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
path = fbc.out('rq1', f'rq1_{gran}.json')
res = json.load(open(path)) if os.path.exists(path) else {}
def save(): json.dump(res, open(path, 'w'), indent=1, default=float)
T = fbc.Timer(); d0 = fbc.load(gran); T(f'{gran} loaded {len(d0)}')

# A12 table-3 stats + A1 variants
for v in ['primary', 'all_prw', 'nv_unw', 'all_unw', 'sampled_nv_prw']:
    key = f'A1_{v}'
    if key in res: continue
    d = fbc.variant(d0, v, gran)
    r = fbc.counts(d); r['logrank'] = fbc.logrank(d)
    r['within'] = fbc.within(d); T(f'{v} within {fbc.fmt(r["within"]["agent"])}')
    res[key] = r; save()

dp = fbc.variant(d0, 'primary', gran)
# A2 pooled with PR clustering
for v in ['primary', 'all_unw']:
    if f'A2_{v}' in res: continue
    res[f'A2_{v}'] = fbc.pooled(fbc.variant(d0, v, gran)); T(f'pooled {v} {fbc.fmt(res[f"A2_{v}"])}'); save()

# A3 bootstrap
if 'A3' not in res:
    res['A3'] = fbc.Boot(dp).run(B, seed=1); res['A3']['B'] = B; T('boot done'); save()

# A5 per-agent
if 'A5' not in res:
    x = fbc.design(dp)
    _, hr = fbc._cox(x, ['ag_' + a for a in fbc.AGENTS], extra=['n_added_lines', 'n_source_files'])
    a5 = {}
    rng = np.random.default_rng(5)
    for a in fbc.AGENTS + ['Human']:
        s = dp[dp.agent_name == a]
        pr = s.groupby(['slug', 'pr_number']).agg(m=('event', 'mean'), n=('event', 'size'))
        m = pr.m.to_numpy(); bs = [m[rng.integers(0, len(m), len(m))].mean() for _ in range(B)]
        a5[a] = dict(units=int(len(s)), prs=int(len(pr)), repos=int(s.slug.nunique()), mod_prw=float(m.mean()),
                     mod_prw_lo=float(np.percentile(bs, 2.5)), mod_prw_hi=float(np.percentile(bs, 97.5)), mod_unw=float(s.event.mean()),
                     largest_pr_share=float(pr.n.max() / pr.n.sum()), largest_pr=str(pr.n.idxmax()),
                     hr=hr.get('ag_' + a))
    res['A5'] = a5; T('A5'); save()

# A6 14-day rate
if 'A6' not in res:
    dp2 = dp.copy(); dp2['m14'] = ((dp2.event == 1) & (dp2.duration_days <= 14)).astype(float)
    res['A6'] = {g: dict(prw=float(np.average(x.m14, weights=x.w)), unw=float(x.m14.mean()))
                 for g, x in list(dp2.groupby('group')) + [('Agent (all)', dp2[dp2.is_agent == 1])]}; save()

# A12
if 'A12' not in res:
    a12 = {}
    for g, x in list(dp.groupby('group')) + [('Agent (all)', dp[dp.is_agent == 1]), ('All', dp)]:
        mod = x[x.event == 1]; cen = x[x.event == 0]
        a12[g] = dict(units=int(len(x)), prs=int(x[['slug', 'pr_number']].drop_duplicates().shape[0]),
                      mod_unw=float(x.event.mean()), mod_prw=float(np.average(x.event, weights=x.w)),
                      median_days_to_mod=float(mod.duration_days.median()) if len(mod) else None,
                      median_days_to_mod_prw=float(np.interp(0.5, np.cumsum(mod.sort_values('duration_days').w) / mod.w.sum(), mod.sort_values('duration_days').duration_days)) if len(mod) else None,
                      median_followup_censored=float(cen.duration_days.median()) if len(cen) else None)
    res['A12'] = a12; save()
T('done')
