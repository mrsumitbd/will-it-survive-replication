"""a7_sub.py: substantial-modification event (theta in 25/50/75/100%) for hunk and file units, primary design
(non-vendored units, PR weights computed after the filter). Same construction as scripts/build_substantial.py."""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
CUTOFF = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
T = fbc.Timer(); l = fbc.load('line'); l = l[~l.vendored]
cov = ['is_agent', 'author_type', 'agent_name', 'group', 'n_added_lines', 'n_source_files', 'stars', 'contributors', 'language']
res = {}
for unit, keys in [('hunk', ['slug', 'pr_number', 'file', 'hunk']), ('file', ['slug', 'pr_number', 'file'])]:
    d = l.sort_values(keys + ['death_date'])
    d['n'] = d.groupby(keys).line.transform('size'); d['k'] = d.groupby(keys).event.cumsum()
    base = d.groupby(keys).agg(n_lines=('line', 'size'), n_dead=('event', 'sum'), birth_date=('birth_date', 'first'),
                               **{c: (c, 'first') for c in cov})
    for th in [0.25, 0.5, 0.75, 1.0]:
        need = np.ceil(th * d.n)
        hit = d[(d.event == 1) & (d.k >= need)].groupby(keys).death_date.min().rename('sub_death')
        u = base.join(hit).reset_index()
        u['event'] = u.sub_death.notna().astype(int)
        end = u.sub_death.where(u.event == 1, CUTOFF)
        u['duration_days'] = ((end - u.birth_date).dt.total_seconds() / 86400).clip(lower=0.001)
        u = fbc.prw(u)
        r = fbc.counts(u); r['within'] = fbc.within(u)
        res[f'{unit}|{int(th*100)}'] = r; T(f'{unit} {th} n={len(u)} {fbc.fmt(r["within"]["agent"])}')
        json.dump(res, open(fbc.out('rq1', 'a7_sub.json'), 'w'), indent=1, default=float)
T('done')
