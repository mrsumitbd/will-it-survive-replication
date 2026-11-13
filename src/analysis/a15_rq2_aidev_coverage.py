"""a15_rq2_aidev_coverage.py: how many of the commits that RQ2 classifies could inherit an AIDev task label?

RQ2 classifies the commits that first modify the primary lines (vendored and generated paths excluded).
AIDev labels PRs, not commits. A modifying commit belongs to an AIDev PR if
  (a) it is one of the commits of an AIDev PR (data/aidev/pr_commit_shas_816.csv.gz), or
  (b) it is the merge commit of a cohort PR (tables/prs.parquet, birth_sha).
Writes results/rq2/a15_rq2_aidev_coverage.json
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
from paths import DATA

d = fbc.load('line'); d = d[(~d.vendored) & (d.event == 1)]
dc = d[['slug', 'death_sha', 'death_date']].drop_duplicates(['slug', 'death_sha'])
res = dict(modifying_commits=int(len(dc)), death_date_min=str(dc.death_date.min()), death_date_max=str(dc.death_date.max()))
pc = set(pd.read_csv(os.path.join(DATA, 'aidev', 'pr_commit_shas_816.csv.gz')).sha.astype(str))
merges = set(pd.read_parquet(os.path.join(fbc.T, 'prs.parquet')).birth_sha.astype(str))
res['in_aidev_pr_commits'] = int(dc.death_sha.isin(pc).sum())
res['is_merge_commit_of_cohort_pr'] = int(dc.death_sha.isin(merges).sum())
cov = dc.death_sha.isin(pc | merges)
res['covered_any'] = int(cov.sum()); res['covered_any_pct'] = float(100 * cov.mean())
res['after_2025_08_01'] = int((pd.to_datetime(dc.death_date, utc=True) >= pd.Timestamp('2025-08-01', tz='UTC')).sum())
json.dump(res, open(fbc.out('rq2', 'a15_rq2_aidev_coverage.json'), 'w'), indent=1)
for k, v in res.items():
    print(k, v)
