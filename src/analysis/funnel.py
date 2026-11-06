"""funnel.py: repository filtering funnel, steps (1) to (5) of Table 1, from the shipped AIDev extract.

Inputs: data/aidev/aidev_summary.json (totals of the full AIDev release), data/aidev/aidev_prs_816.csv.gz (all PRs of the
816 repositories with agent and human PRs), data/cohort/repo_metadata_v1.csv (GitHub metadata, collected January 2026),
data/cohort/prs_cohort256.csv (merged PRs of the final cohort).
The filtering rules are those of src/mining/filter_repos.py (Xiao et al.).
Step (5), the code-ratio filter, needs the per-repository code ratio (cloc) of the final run, which is not part of the
package. The step is therefore reproduced from the cohort itself: the 256 repositories are the 246 repositories with a
merged PR in prs_cohort256.csv plus the Q1 repositories without any merged PR. The repositories removed by the code-ratio
step are then the remaining Q1 repositories (listed in the output).
Writes results/cohort/filtering_funnel.csv and results/cohort/funnel.json."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mining'))
from paths import AIDEV, COHORT, out
from filter_repos import MIN_STARS, is_bad_license

S = json.load(open(os.path.join(AIDEV, 'aidev_summary.json')))
x = pd.read_csv(os.path.join(AIDEV, 'aidev_prs_816.csv.gz'), dtype={'user_hash': str})
a, h = x[x.table == 'agent'], x[x.table == 'human']
rows = [dict(step='AIDev, all repositories', repos=S['agent_repos_by_repo_id'], agent_prs=S['agent_prs'], human_prs=S['human_prs'])]


def log(step, slugs):
    rows.append(dict(step=step, repos=len(slugs), agent_prs=int(a.slug.isin(slugs).sum()), human_prs=int(h.slug.isin(slugs).sum())))


cohort = set(a.slug) & set(h.slug)
log('(1) Repositories with agent and human PRs', cohort)
m = pd.read_csv(os.path.join(COHORT, 'repo_metadata_v1.csv'))
m = m[m.slug.isin(cohort)].drop_duplicates('slug')
meta_missing = sorted(cohort - set(m.slug))
m = m[m.stars >= MIN_STARS]
m = m[~m.license.apply(is_bad_license)]
log('(2) Software license', set(m.slug))
m = m[m.is_archived == False]
m = m[m.has_release == True]
u = x[x.slug.isin(m.slug)]
cc = u.groupby('slug').user_hash.nunique()
m = m[m.slug.map(cc).fillna(0) >= 2]
log('(3) Not archived, has release or tag, >=2 contributors', set(m.slug))
m = m.copy()
m['total_pr_count'] = m.slug.map(a.groupby('slug').size()).fillna(0) + m.slug.map(h.groupby('slug').size()).fillna(0)
keep = []
for lang, d in m.groupby('language'):
    if pd.isna(lang) or lang == 'Unknown':
        continue
    if len(d) < 4:
        keep.append(d); continue
    keep.append(d[(d.total_pr_count > d.total_pr_count.quantile(.25)) & (d.open_issues > d.open_issues.quantile(.25)) & (d.size_kb > d.size_kb.quantile(.25))])
q1 = set(pd.concat(keep).slug)
log('(4) Bottom-quartile removal per language', q1)
c = pd.read_csv(os.path.join(COHORT, 'prs_cohort256.csv'))
s246 = set(c.slug)
merged = set(x[x.merged_at.notna()].slug)
no_merged = {s for s in q1 if s not in merged}
s256 = s246 | no_merged
log('(5) Code ratio within the 97% interval', s256)
res = dict(rows=rows, metadata_missing=meta_missing, cohort246_subset_of_q1=s246 <= q1,
           q1_without_merged_pr=sorted(no_merged), removed_by_code_ratio_inferred=sorted(q1 - s256),
           step5_note='reproduced from the cohort list; per-repository code ratios of the final run are not shipped')
df = pd.DataFrame(rows); df.to_csv(out('cohort', 'filtering_funnel.csv'), index=False)
json.dump(res, open(out('cohort', 'funnel.json'), 'w'), indent=1)
print(df.to_string(index=False))
print('246 cohort repositories inside Q1:', res['cohort246_subset_of_q1'])
print('Q1 repositories without a merged PR:', len(no_merged), '| removed by the code-ratio step (inferred):', len(res['removed_by_code_ratio_inferred']))
