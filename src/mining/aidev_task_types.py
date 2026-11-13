"""aidev_task_types.py: AIDev task types and PR commits of the 816 repositories with agent and human PRs.

Input: the released AIDev parquet files (rebuild/aidev/, downloaded by download_data.py):
  pr_task_type.parquet, human_pr_task_type.parquet   conventional-commit type of each PR (from its title and description)
  pr_commits.parquet                                 the commits of each agent PR
Output (only PRs listed in data/aidev/aidev_prs_816.csv.gz):
  data/aidev/pr_task_type_816.csv.gz     id, type                     (RQ1 task-type sensitivity analysis)
  data/aidev/pr_commit_shas_816.csv.gz   pr_id, sha                   (share of RQ2 commits that belong to an AIDev PR)
Commit messages and authors are not kept.
Usage: python src/mining/aidev_task_types.py [<aidev_dir> [<out_dir>]]   (default out_dir: data/aidev)
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import AIDEV

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join('rebuild', 'aidev')
out = sys.argv[2] if len(sys.argv) > 2 else AIDEV
ids = set(pd.read_csv(os.path.join(AIDEV, 'aidev_prs_816.csv.gz'), usecols=['id']).id.astype('int64'))
t = pd.concat([pd.read_parquet(os.path.join(src, f), columns=['id', 'type']) for f in ('pr_task_type.parquet', 'human_pr_task_type.parquet')])
t = t[t.id.astype('int64').isin(ids)].drop_duplicates('id').sort_values('id')
c = pd.read_parquet(os.path.join(src, 'pr_commits.parquet'), columns=['pr_id', 'sha'])
c = c[c.pr_id.astype('int64').isin(ids)].drop_duplicates().sort_values(['pr_id', 'sha'])
os.makedirs(out, exist_ok=True)
gz = {'method': 'gzip', 'mtime': 0}
t.to_csv(os.path.join(out, 'pr_task_type_816.csv.gz'), index=False, compression=gz)
c.to_csv(os.path.join(out, 'pr_commit_shas_816.csv.gz'), index=False, compression=gz)
print('task types', len(t), t.type.value_counts().to_dict()); print('PR commits', len(c), 'of', c.pr_id.nunique(), 'PRs')
