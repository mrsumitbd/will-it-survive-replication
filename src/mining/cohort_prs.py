"""cohort_prs.py: merged PRs of the filtered repositories (input of the miner): data/cohort/prs_cohort256.csv.

Usage: python src/mining/cohort_prs.py <aidev_dir> <repos.csv with column slug> <out.csv>
Columns: slug, number, merged_at, author_type (Agent or Human), agent_name (agent or Human), pr_id.
PRs listed in both AIDev tables are kept twice here; build_tables.py excludes them.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_repos import normalize_slug

aidev, repos, out = sys.argv[1:4]
keep = set(pd.read_csv(repos).slug)
cols = ['id', 'number', 'repo_url', 'merged_at']
a = pd.read_parquet(os.path.join(aidev, 'all_pull_request.parquet'), columns=cols + ['agent']).assign(author_type='Agent')
h = pd.read_parquet(os.path.join(aidev, 'human_pull_request.parquet'), columns=cols).assign(author_type='Human', agent='Human')
x = pd.concat([a, h], ignore_index=True)
x['slug'] = x.repo_url.map(normalize_slug)
x = x[x.slug.isin(keep) & x.merged_at.notna()]
x = x.rename(columns={'agent': 'agent_name', 'id': 'pr_id'})[['slug', 'number', 'merged_at', 'author_type', 'agent_name', 'pr_id']]
x.to_csv(out, index=False)
print(len(x), 'merged PRs in', x.slug.nunique(), 'repositories ->', out)
