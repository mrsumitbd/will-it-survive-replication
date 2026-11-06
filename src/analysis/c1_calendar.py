"""c1_calendar.py [gran]: does calendar time explain the supervised-assistant HR?
(a) only PRs merged on or after 2025-05-16 (Codex launch; Copilot coding agent 2025-05-19), all groups, weights recomputed
(a2) as (a), but only human PRs restricted (agent PRs unchanged)
(b) strata = repository x merge month (full primary data)
(c) time split (0-14, 14-90, >90 days) per tier within (a)
Writes results/rq1/c1_calendar_<gran>.json"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
gran = sys.argv[1] if len(sys.argv) > 1 else 'line'
path = fbc.out('rq1', f'c1_calendar_{gran}.json')
res = {}
def save(): json.dump(res, open(path, 'w'), indent=1, default=float)
T = fbc.Timer(); d0 = fbc.load(gran)
p = pd.read_parquet(os.path.join(fbc.T, 'prs.parquet'))[['slug', 'pr_number', 'merged_at']]
p['m'] = pd.to_datetime(p.merged_at, utc=True)
d0 = d0.merge(p[['slug', 'pr_number', 'm']].drop_duplicates(['slug', 'pr_number']), on=['slug', 'pr_number'], how='left')
res['missing_merge_date'] = int(d0.m.isna().sum())
CUT = pd.Timestamp('2025-05-16', tz='UTC')
nv = d0[~d0.vendored]
def describe(d):
    c = fbc.counts(d); return dict(n=c['n'], prs=c['prs'], repos=c['repos'], groups={k: dict(prs=v['prs'], n=v['n'], mod_prw=v['mod_prw']) for k, v in c['groups'].items()})
# (a) all groups merged on/after the cutoff
da = fbc.prw(nv[nv.m >= CUT]); res['a_all_after_cut'] = describe(da)
res['a_all_after_cut']['within'] = fbc.within(da); T(f"(a) {res['a_all_after_cut']['within']}"); save()
# (a2) only human PRs restricted
da2 = fbc.prw(nv[(nv.is_agent == 1) | (nv.m >= CUT)]); res['a2_human_after_cut'] = describe(da2)
res['a2_human_after_cut']['within'] = fbc.within(da2); T(f"(a2) {res['a2_human_after_cut']['within']}"); save()
# (b) strata = repo x merge month on the primary data
dp = fbc.prw(nv); x = fbc.design(dp); x['slug'] = x.slug + '|' + x.m.dt.strftime('%Y-%m')
res['b_repo_x_month'] = {}
for name, ks in [('agent', ['is_agent']), ('tier', ['supervised', 'devin'])]:
    _, hr = fbc._cox(x, ks, extra=['n_added_lines', 'n_source_files']); res['b_repo_x_month'].update(hr)
res['b_repo_x_month']['strata'] = int(x.slug.nunique())
res['b_repo_x_month']['strata_with_agent_and_human'] = int((x.groupby('slug').is_agent.nunique() == 2).sum()); T(f"(b) {res['b_repo_x_month']}"); save()
# (c) time split per tier within (a)
xa = fbc.design(da); out = {}
for lo, hi in [(0, 14), (14, 90), (90, 1e9)]:
    s = xa[xa.duration_days > lo].copy(); s['event'] = ((s.event == 1) & (s.duration_days <= hi)).astype(int)
    s['duration_days'] = np.minimum(s.duration_days, hi) - lo
    for name, ks in [('agent', ['is_agent']), ('tier', ['supervised', 'devin'])]:
        try:
            _, hr = fbc._cox(s, ks, extra=['n_added_lines', 'n_source_files']); out[f'{name}_{lo}_{int(min(hi, 9999))}'] = hr
        except Exception as e:
            out[f'{name}_{lo}_{int(min(hi, 9999))}'] = repr(e)[:200]
    T(f'(c) {lo}-{hi}')
res['c_time_split_after_cut'] = out; save(); T('done')
