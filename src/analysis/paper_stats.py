"""paper_stats.py: cohort attrition below the line of Table 1 and the descriptive numbers of Section 2.1.3 and 2.2.

Reads data/cohort/prs_cohort256.csv and data/tables/*.parquet. Writes results/cohort/cohort_stats.json and prints a text
version (results/cohort/paper_stats.txt via reproduce.sh)."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import TABLES, COHORT, out

prs = pd.read_parquet(os.path.join(TABLES, 'prs.parquet'))
c = pd.read_csv(os.path.join(COHORT, 'prs_cohort256.csv'))
lines = pd.read_parquet(os.path.join(TABLES, 'lines.parquet'), columns=['slug', 'pr_number', 'event', 'vendored', 'author_type', 'agent_name', 'kind'])
res = {}


def row(d, slug='slug', at='author_type'):
    return dict(repos=int(d[slug].nunique()), agent=int((d[at] == 'Agent').sum()), human=int((d[at] == 'Human').sum()))


# Table 1, rows below the line (merged PRs)
res['merged_in_256'] = row(c)
amb = c[c.duplicated(['slug', 'number'], keep=False)]
mined_repos = set(prs.slug)
c_hist = c[c.slug.isin(mined_repos)]
res['history_available'] = row(c_hist)
res['history_unavailable_repos'] = sorted(set(c.slug) - mined_repos)
c_unamb = c_hist[~c_hist.set_index(['slug', 'number']).index.isin(amb.set_index(['slug', 'number']).index)]
res['not_in_both_tables'] = row(c_unamb)
res['ambiguous_prs'] = int(amb[['slug', 'number']].drop_duplicates().shape[0])
res['ambiguous_rows'] = int(len(amb))
res['prs_parquet'] = row(prs)
ok = prs[prs.status == 'ok']
res['merge_located'] = row(ok)
res['merge_not_found'] = int((prs.status != 'ok').sum())
res['merge_not_found_by_status'] = {k: int(v) for k, v in prs.status.value_counts().items() if k != 'ok'}
res['merge_located_agent_prs'] = int((ok.author_type == 'Agent').sum())
ev = lines[lines.event.notna()]
mined = ev[['slug', 'pr_number', 'author_type', 'agent_name']].drop_duplicates(['slug', 'pr_number'])
res['mined_cohort'] = row(mined)
res['mined_cohort_by_agent'] = {k: int(v) for k, v in mined.agent_name.value_counts().items()}
res['no_tracked_source_line'] = int(len(ok) - len(mined))
nv = ev[~ev.vendored][['slug', 'pr_number', 'author_type']].drop_duplicates(['slug', 'pr_number'])
res['primary_prs'] = row(nv)
res['vendored_only_prs'] = int(len(mined) - len(nv))
res['line_kinds'] = {str(k): int(v) for k, v in lines.kind.value_counts().items()}
# largest PRs (lines added, all tracked lines and non-vendored)
size = ev.groupby(['slug', 'pr_number', 'author_type']).size().rename('n').reset_index()
res['largest_pr_lines'] = {a: int(g.n.max()) for a, g in size.groupby('author_type')}
size_nv = ev[~ev.vendored].groupby(['slug', 'pr_number', 'author_type']).size().rename('n').reset_index()
res['largest_pr_lines_non_vendored'] = {a: int(g.n.max()) for a, g in size_nv.groupby('author_type')}
# observation window
m = pd.to_datetime(ok.merged_at, utc=True)
res['last_merge'] = str(m.max()); res['first_merge'] = str(m.min())
res['min_followup_days'] = float((pd.Timestamp('2025-12-31 23:59:59', tz='UTC') - m.max()).total_seconds() / 86400)
json.dump(res, open(out('cohort', 'cohort_stats.json'), 'w'), indent=1)
for k, v in res.items():
    print(f'{k}: {v}')
