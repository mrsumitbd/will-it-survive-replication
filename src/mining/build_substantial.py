"""
build_substantial.py: 'substantial modification' event: a hunk or file unit is substantially modified at
the first time at least THRESH (default 50%) of its tracked added lines have been modified or deleted (per-line
git blame --reverse events from lines.parquet). Units below the threshold by the cutoff are censored.
Writes <out>_hunk/hunks.parquet and <out>_file/hunks.parquet (file units in the hunk schema, so rq1.py --gran hunk runs
all four weighting variants on them). Usage: python build_substantial.py <tables_dir> <out_prefix> [thresh]
"""
import sys, os, numpy as np, pandas as pd
CUTOFF = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
tables, out = sys.argv[1:3]; th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
l = pd.read_parquet(os.path.join(tables, 'lines.parquet'))
l = l[l.event.notna()]
cov = ['is_agent', 'author_type', 'agent_name', 'group', 'n_added_lines', 'n_source_files', 'stars', 'contributors', 'language']
for unit, keys in [('hunk', ['slug', 'pr_number', 'file', 'hunk']), ('file', ['slug', 'pr_number', 'file'])]:
    d = l.sort_values(keys + ['death_date'])
    d['n'] = d.groupby(keys).line.transform('size')
    d['k'] = d.groupby(keys).event.cumsum()  # deaths sorted by date, NaT (alive) last
    need = np.ceil(th * d.n)
    hit = d[(d.event == 1) & (d.k >= need)].groupby(keys).death_date.min().rename('sub_death')
    u = d.groupby(keys).agg(n_lines=('line', 'size'), n_dead=('event', 'sum'), birth_date=('birth_date', 'first'),
                            vendored=('vendored', 'first'), **{c: (c, 'first') for c in cov}).join(hit).reset_index()
    if unit == 'file':
        u['hunk'] = 0
    u['event'] = u.sub_death.notna().astype(int)
    end = u.sub_death.where(u.event == 1, CUTOFF)
    u['duration_days'] = ((end - u.birth_date).dt.total_seconds() / 86400).clip(lower=0.001)
    o = f'{out}_{unit}'; os.makedirs(o, exist_ok=True)
    u.to_parquet(os.path.join(o, 'hunks.parquet'))
    print(unit, len(u), 'substantial-mod rate', round(u.event.mean(), 3), 'any-mod rate', round((u.n_dead > 0).mean(), 3),
          u.groupby('group').event.mean().round(3).to_dict())
