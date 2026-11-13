"""a14_tasktype.py: RQ1 controlled for the task type of each PR (line level, Section 3, "Task type").

Task type = the conventional-commit type that AIDev assigns to each PR from its title and description
(data/aidev/pr_task_type_816.csv.gz). Rare types (build, chore, ci, perf, style, revert, other) and PRs without
a type are pooled as 'other'.
  1. task-type mix per group (PRs) and chi-square test
  2. primary design (PR-weighted) and unweighted, strata = repository x task type
  3. primary design, strata = repository, task-type dummies as covariates
  4. primary design within the two largest task types (feat, fix), strata = repository
Writes results/rq1/a14_tasktype.json
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
from paths import DATA
from lifelines import CoxPHFitter

KEEP = ['feat', 'fix', 'refactor', 'docs', 'test']
OUT = fbc.out('rq1', 'a14_tasktype.json')
tm = fbc.Timer(); res = {}

tt = pd.read_csv(os.path.join(DATA, 'aidev', 'pr_task_type_816.csv.gz')).rename(columns={'id': 'pr_id', 'type': 'task_raw'})
prs = pd.read_parquet(os.path.join(fbc.T, 'prs.parquet'))[['slug', 'pr_number', 'pr_id']]
d = fbc.load('line')
d = d.merge(prs, on=['slug', 'pr_number'], how='left').merge(tt, on='pr_id', how='left')
d['task'] = np.where(d.task_raw.isin(KEEP), d.task_raw, 'other')
tm(f'loaded {len(d)} lines')

p = d[~d.vendored][['slug', 'pr_number', 'group', 'task', 'task_raw']].drop_duplicates(['slug', 'pr_number'])
res['prs'] = int(len(p)); res['prs_without_type'] = int(p.task_raw.isna().sum())
res['prs_without_type_by_group'] = p[p.task_raw.isna()].group.value_counts().to_dict()
ct = pd.crosstab(p.group, p.task)
res['mix_counts'] = {g: {k: int(v) for k, v in r.items()} for g, r in ct.iterrows()}
res['mix_pct'] = {g: {k: float(v) for k, v in r.items()} for g, r in (ct.T / ct.sum(1) * 100).T.iterrows()}
chi2, pv, dof, _ = chi2_contingency(ct)
res['mix_test'] = dict(chi2=float(chi2), dof=int(dof), p=float(pv), cramers_v=float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))))


def cox(x, dummies, strata, extra):
    cols = ['duration_days', 'event', 'w', 'cluster'] + list(dummies) + list(extra) + list(strata)
    c = CoxPHFitter().fit(x[cols], 'duration_days', 'event', weights_col='w', cluster_col='cluster', robust=True, strata=list(strata))
    s = c.summary
    return {k: dict(hr=float(np.exp(s.loc[k].coef)), lo=float(s.loc[k]['exp(coef) lower 95%']),
                    hi=float(s.loc[k]['exp(coef) upper 95%']), p=float(s.loc[k].p)) for k in dummies}


def within(dd, strata, extra=()):
    x = fbc.design(dd); x['slug_task'] = x.slug + '|' + x.task
    for k in KEEP[1:] + ['other']:
        x['t_' + k] = (x.task == k).astype(int)
    base = ['n_added_lines', 'n_source_files'] + list(extra)
    r = dict(agent=cox(x, ['is_agent'], strata, base)['is_agent'])
    r.update(cox(x, ['supervised', 'devin'], strata, base))
    r['lines'] = int(len(x)); r['prs'] = int(x[['slug', 'pr_number']].drop_duplicates().shape[0])
    return r


prim = fbc.variant(d, 'primary', 'line'); unw = fbc.variant(d, 'nv_unw', 'line')
res['reference_primary_slug'] = within(prim, ['slug']); tm('reference ' + fbc.fmt(res['reference_primary_slug']['agent']))
res['primary_slug_x_task'] = within(prim, ['slug_task']); tm('slug x task ' + fbc.fmt(res['primary_slug_x_task']['agent']))
res['unweighted_slug_x_task'] = within(unw, ['slug_task']); tm('unweighted slug x task ' + fbc.fmt(res['unweighted_slug_x_task']['agent']))
res['primary_slug_task_covariates'] = within(prim, ['slug'], extra=['t_' + k for k in KEEP[1:] + ['other']])
tm('task covariates ' + fbc.fmt(res['primary_slug_task_covariates']['agent']))
s = prim.assign(st=prim.slug + '|' + prim.task).groupby('st').is_agent.agg(['min', 'max'])
res['slug_x_task_strata'] = int(len(s)); res['slug_x_task_strata_mixed'] = int(((s['min'] == 0) & (s['max'] == 1)).sum())
s = prim.groupby('slug').is_agent.agg(['min', 'max'])
res['slug_strata'] = int(len(s)); res['slug_strata_mixed'] = int(((s['min'] == 0) & (s['max'] == 1)).sum())
for k in ['feat', 'fix']:
    res['within_task_' + k] = within(fbc.prw(d[(~d.vendored) & (d.task == k)]), ['slug'])
    tm(f'{k}: ' + fbc.fmt(res['within_task_' + k]['agent']))

json.dump(res, open(OUT, 'w'), indent=1, default=float)
for k, v in res.items():
    if isinstance(v, dict) and 'agent' in v:
        print(f'{k:30s}', '  '.join(f"{t}={fbc.fmt(v[t])}" for t in ('agent', 'supervised', 'devin')), f"lines={v['lines']} prs={v['prs']}")
    else:
        print(k, v)
