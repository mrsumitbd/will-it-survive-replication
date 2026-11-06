"""aidev_extract.py: compact extract of the AIDev PR tables for the funnel (Table 1).

Input: the released AIDev parquet files (rebuild/aidev/, downloaded by download_data.py).
Output:
  data/aidev/aidev_summary.json        totals of the full AIDev release (Table 1 row "AIDev, all repositories", Section 2.1.1)
  data/aidev/aidev_prs_816.csv.gz      every PR (agent and human table) of the repositories that have both agent and human PRs:
                                       table, id, slug, number, agent, user_hash, created_at, merged_at
  user_hash is a SHA-1 of the GitHub login. The funnel only needs the number of distinct authors per repository.
Usage: python src/mining/aidev_extract.py [<aidev_dir> [<out_dir>]]   (default out_dir: data/aidev)
"""
import os, sys, json, hashlib
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import AIDEV
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_repos import normalize_slug

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join('rebuild', 'aidev')
cols = ['id', 'number', 'agent', 'user', 'repo_id', 'repo_url', 'created_at', 'merged_at']
a = pd.read_parquet(os.path.join(src, 'all_pull_request.parquet'), columns=cols)
h = pd.read_parquet(os.path.join(src, 'human_pull_request.parquet'), columns=[c for c in cols if c not in ('agent', 'repo_id')])
r = pd.read_parquet(os.path.join(src, 'all_repository.parquet'), columns=['id'])
a['slug'] = a.repo_url.apply(normalize_slug); h['slug'] = h.repo_url.apply(normalize_slug)
both = set(a.slug) & set(h.slug)
summary = dict(agent_prs=int(len(a)), agent_prs_by_agent={k: int(v) for k, v in a.agent.value_counts().items()},
               agent_repos_by_repo_id=int(a.repo_id.nunique()), agent_repos_by_repo_url=int(a.repo_url.nunique()),
               agent_repos_by_slug=int(a.slug.nunique()), all_repository_rows=int(len(r)),
               human_prs=int(len(h)), human_repos_by_slug=int(h.slug.nunique()),
               human_prs_in_repos_without_agent_prs=int((~h.slug.isin(both)).sum()),
               human_repos_without_agent_prs=int(h[~h.slug.isin(both)].slug.nunique()),
               repos_with_agent_and_human_prs=len(both))
a['table'] = 'agent'; h['table'] = 'human'; h['agent'] = 'Human'
x = pd.concat([a, h], ignore_index=True)
x = x[x.slug.isin(both)].copy()
x['user_hash'] = x.user.fillna('').map(lambda u: hashlib.sha1(u.encode()).hexdigest()[:16] if u else '')
AIDEV = sys.argv[2] if len(sys.argv) > 2 else AIDEV
os.makedirs(AIDEV, exist_ok=True)
x[['table', 'id', 'slug', 'number', 'agent', 'user_hash', 'created_at', 'merged_at']].sort_values(['table', 'id']).to_csv(
    os.path.join(AIDEV, 'aidev_prs_816.csv.gz'), index=False, compression={'method': 'gzip', 'mtime': 0})
json.dump(summary, open(os.path.join(AIDEV, 'aidev_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1)); print('rows', len(x))
