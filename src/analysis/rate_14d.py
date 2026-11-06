"""rate_14d.py: share of lines and hunks modified within 14 days of the merge, per group: all units, without vendored
paths, and PR-weighted over all units (the churn window of GitClear). The primary-design values in the paper
(non-vendored and PR-weighted) are in results/rq1/rq1_<unit>.json (A6). Writes results/rq1/rate_14d.txt."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import TABLES, out
txt = []
for name in ['lines', 'hunks']:
    d = pd.read_parquet(os.path.join(TABLES, f'{name}.parquet')); d = d[d.event.notna()]
    d['m14'] = ((d.event == 1) & (d.duration_days <= 14)).astype(float)
    d['w'] = 1 / d.groupby(['slug', 'pr_number']).m14.transform('size')
    nv = d[~d.vendored]
    r = pd.DataFrame({'all': d.groupby('group').m14.mean(), 'no_vendored': nv.groupby('group').m14.mean(),
                      'pr_weighted': d.groupby('group').apply(lambda x: np.average(x.m14, weights=x.w))})
    txt += [name, (100 * r).round(1).to_string(), '']
open(out('rq1', 'rate_14d.txt'), 'w').write('\n'.join(txt)); print('\n'.join(txt))
