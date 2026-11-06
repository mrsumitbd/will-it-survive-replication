"""a4_ph.py [gran]: Schoenfeld-residual PH test and time-split HRs (0-14, 14-90, >90 days) of the primary model.
Writes results/rq1/a4_ph_<gran>.json."""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
gran = sys.argv[1] if len(sys.argv) > 1 else 'line'
T = fbc.Timer(); d = fbc.variant(fbc.load(gran), 'primary', gran); x = fbc.design(d)
res = {}
for name, dummies in [('agent', ['is_agent']), ('tier', ['supervised', 'devin'])]:
    cols = ['duration_days', 'event', 'n_added_lines', 'n_source_files', 'slug', 'w', 'cluster'] + dummies
    for mode in ['weighted_robust', 'unweighted']:
        try:
            if mode == 'weighted_robust':
                c = CoxPHFitter().fit(x[cols], 'duration_days', 'event', strata=['slug'], weights_col='w', cluster_col='cluster', robust=True)
                data = x[cols]
            else:
                cc = [k for k in cols if k not in ('w', 'cluster')]
                c = CoxPHFitter().fit(x[cc], 'duration_days', 'event', strata=['slug']); data = x[cc]
            r = proportional_hazard_test(c, data, time_transform='rank')
            s = r.summary
            res[f'{name}_{mode}'] = {k: dict(stat=float(s.loc[k, 'test_statistic']), p=float(s.loc[k, 'p'])) for k in s.index}
            T(f'{name} {mode} ok {res[f"{name}_{mode}"]}')
        except Exception as e:
            res[f'{name}_{mode}'] = repr(e)[:300]; T(f'{name} {mode} failed {e!r}')
        json.dump(res, open(fbc.out('rq1', f'a4_ph_{gran}.json'), 'w'), indent=1)
# time-split HR for agent and tiers (0-14, 14-90, >90 days), via episode splitting
from lifelines.utils import to_episodic_format
cuts = [14, 90]
ep = []
import pandas as pd
for lo, hi in [(0, 14), (14, 90), (90, 1e9)]:
    s = x[x.duration_days > lo].copy()
    s['start'] = lo; s['stop'] = np.minimum(s.duration_days, hi); s['ev'] = ((s.event == 1) & (s.duration_days <= hi)).astype(int)
    for k in ['is_agent', 'supervised', 'devin']:
        s[f'{k}_p{lo}'] = s[k]
    ep.append(s.assign(period=lo))
ep = pd.concat(ep)
for k in ['is_agent', 'supervised', 'devin']:
    for lo in [0, 14, 90]:
        ep[f'{k}_p{lo}'] = np.where(ep.period == lo, ep[k], 0)
from lifelines import CoxTimeVaryingFitter
out = {}
for name, ks in [('agent', ['is_agent']), ('tier', ['supervised', 'devin'])]:
    # per-period stratified Cox on the left-truncated risk sets (fit each period separately = equivalent to period interactions)
    for lo, hi in [(0, 14), (14, 90), (90, 1e9)]:
        s = x[x.duration_days > lo].copy(); s['event'] = ((s.event == 1) & (s.duration_days <= hi)).astype(int)
        s['duration_days'] = np.minimum(s.duration_days, hi) - lo
        s = fbc.prw(s) if False else s  # keep primary weights from the full data
        _, hr = fbc._cox(s, ks, extra=['n_added_lines', 'n_source_files'])
        out[f'{name}_{lo}_{int(min(hi, 9999))}'] = hr; T(f'split {name} {lo}-{hi} {hr}')
res['time_split'] = out
json.dump(res, open(fbc.out('rq1', f'a4_ph_{gran}.json'), 'w'), indent=1)
T('done')
