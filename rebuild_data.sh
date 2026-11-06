#!/usr/bin/env bash
# rebuild_data.sh: rebuild the shipped data/ from the AIDev release and GitHub (full pipeline).
#
# This is NOT needed to reproduce the paper: reproduce.sh works on the shipped data/. The rebuild needs network
# access, git, cloc (code ratio), an authenticated GitHub CLI `gh` (search API, RQ3b PR history), about 100 GB of
# disk for the bare clones of the 215 repositories, and several days (mining and the rate-limited GitHub search).
# Repositories keep changing (rewritten histories, deleted repositories, renamed owners), so a rebuild today will not
# give exactly the shipped data. See README.md, section "Full data rebuild", for what each step does.
#
#   bash rebuild_data.sh --list            list the steps
#   bash rebuild_data.sh <step> [<step>]   run some steps (in the order given)
#   bash rebuild_data.sh all               run every step in order
# Everything is written to rebuild/ (REBUILD_DIR); data/ is never overwritten. The last step compares the rebuilt
# files with data/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
export REBUILD_DIR="${REBUILD_DIR:-rebuild}"; RB="$REBUILD_DIR"
WORKERS="${WORKERS:-4}"
mkdir -p "$RB"

STEPS=(download aidev_extract filter cohort_prs clone mine tables sanitize parents file_covariates rq3_git rq3_pr_features rq3_inputs rq3_prep compare)

step_download() {          # 1. AIDev release from Hugging Face -> rebuild/aidev/*.parquet (needs: pip install requests)
  python src/mining/download_data.py
}
step_aidev_extract() {     # 2. Table 1 inputs: AIDev totals and the PRs of the 816 repositories with agent and human PRs
  python src/mining/aidev_extract.py "$RB/aidev" "$RB/aidev_extract"
}
step_filter() {            # 3. Xiao et al. filters: GitHub metadata (GITHUB_TOKEN in .env or the environment), Q1 filter,
                           #    code ratio with cloc -> rebuild/{repo_metadata_v1,code_ratio_v1,aidev_filtered_v1}.csv
                           #    src/mining/code_ratio.py <url> <out.jsonl> recomputes one code ratio without the clone/cloc
                           #    timeouts of filter_repos.py (the final run used it for repositories that timed out).
  python src/mining/filter_repos.py
}
step_cohort_prs() {        # 4. merged PRs of the 256 filtered repositories -> rebuild/prs_cohort256.csv
  python - "$RB" <<'EOF'
import sys, pandas as pd
rb = sys.argv[1]; f = pd.read_csv(f'{rb}/aidev_filtered_v1.csv')
f[['slug']].to_csv(f'{rb}/repos_filtered.csv', index=False); print(len(f), 'filtered repositories')
EOF
  python src/mining/cohort_prs.py "$RB/aidev" "$RB/repos_filtered.csv" "$RB/prs_cohort256.csv"
}
step_clone() {             # 5. bare clones of the cohort repositories -> rebuild/clones/<owner>__<name>.git
  mkdir -p "$RB/clones"
  python -c "import pandas as pd; print('\n'.join(sorted(set(pd.read_csv('$RB/prs_cohort256.csv').slug))))" | while read -r s; do
    d="$RB/clones/${s/\//__}.git"; [ -d "$d" ] && continue
    git clone --quiet --bare "https://github.com/$s.git" "$d" || echo "clone failed: $s"
  done
}
step_mine() {              # 6. line, hunk and file survival per repository (git blame --reverse) -> rebuild/mine/
                           #    the miner makes its own single-branch clone; this takes days for the whole cohort
  mkdir -p "$RB/mine"
  python -c "import pandas as pd; print('\n'.join(sorted(set(pd.read_csv('$RB/prs_cohort256.csv').slug))))" | \
    xargs -P "$WORKERS" -I{} python src/mining/mine_survival.py {} "$RB/prs_cohort256.csv" "$RB/mine"
}
step_tables() {            # 7. assemble lines/hunks/files/prs/commits parquet tables -> rebuild/tables/
  python - "$RB" <<'EOF'
import sys, pandas as pd
rb = sys.argv[1]; m = pd.read_csv(f'{rb}/aidev_filtered_v1.csv')
m[['slug', 'contributors']].to_csv(f'{rb}/contributors.csv', index=False)
EOF
  python src/mining/build_tables.py "$RB/mine" - "$RB/prs_cohort256.csv" "$RB/aidev_filtered_v1.csv" "$RB/contributors.csv" "$RB/tables"
}
step_sanitize() {          # 8. drop author names and e-mail addresses from the commit table
  mv "$RB/tables/commits.parquet" "$RB/tables/commits_raw.parquet"
  python src/mining/sanitize_commits.py "$RB/tables/commits_raw.parquet" "$RB/tables/commits.parquet"
}
step_parents() {           # 9. parent count of every modifying commit (RQ2 merge-commit check) -> rebuild/commit_parents.csv
  TABLES_DIR="$RB/tables" python src/mining/commit_parents.py "$RB/clones" "$RB/commit_parents.csv"
}
step_file_covariates() {   # 10. file history at merge time (RQ1 sensitivity, RQ3a baseline) -> rebuild/file_covariates/
  python src/mining/a9_extract.py "$RB/clones" "$RB/tables" "$RB/file_covariates" "$WORKERS"
}
step_rq3_git() {           # 11. RQ3 inputs from git: file content and tokens at merge, git process features -> rebuild/rq3_inputs/
  python src/mining/extract_rq3_git.py "$RB/tables/files.parquet" src/rq3 "$RB/rq3_inputs" --workers "$WORKERS"
}
step_rq3_pr_features() {   # 12. PR-history features from the GitHub search API (30 requests per minute; resumable)
                           #     extract_rq3_pr_features.py: original definition (needed by build_rq3_inputs.py)
                           #     extract_rq3_pr_features_fixed.py: leakage fix used in the paper (excludes the PR itself,
                           #     counts PRs merged before the merge time)
  python src/mining/extract_rq3_pr_features.py "$RB/tables/files.parquet" "$RB/prs_cohort256.csv" "$RB/aidev" "$RB/rq3_inputs/pr_features.jsonl"
  python src/mining/extract_rq3_pr_features_fixed.py "$RB/tables/files.parquet" "$RB/prs_cohort256.csv" "$RB/aidev" "$RB/rq3_inputs/pr_features_fixed.jsonl"
}
step_rq3_inputs() {        # 13. RQ3a and RQ3b inputs in the classifier format -> rebuild/rq3_256/
  python src/mining/build_rq3_inputs.py "$RB/tables" "$RB/rq3_inputs" "$RB/rq3_256" ab
}
step_rq3_prep() {          # 14. drop vendored files (RQ3a and RQ3b) and apply the full PR-history fix (RQ3b) -> rebuild/rq3, rebuild/rq3full
  python src/mining/d_prep.py "$RB/tables" "$RB/rq3_256" "$RB/rq3"
  python src/mining/d_full_prep.py "$RB/rq3_inputs/pr_features_fixed.jsonl" "$RB/rq3" "$RB/rq3full"
}
step_compare() {           # 15. compare the rebuilt files with the shipped data/
  python - "$RB" <<'EOF'
import sys, os, glob, pandas as pd
rb = sys.argv[1]
def cmp(name, a, b, keys):
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f'{name:40s} not rebuilt'); return
    x = pd.read_parquet(a) if a.endswith('.parquet') else pd.read_csv(a)
    y = pd.read_parquet(b) if b.endswith('.parquet') else pd.read_csv(b)
    k = [c for c in keys if c in x.columns and c in y.columns]
    same = len(x) == len(y) and x.sort_values(k).reset_index(drop=True).equals(y.sort_values(k).reset_index(drop=True)[x.columns])
    print(f'{name:40s} rebuilt {len(x):>9,} rows, shipped {len(y):>9,} rows, identical: {same}')
for t, k in [('lines', ['slug', 'pr_number', 'file', 'line']), ('hunks', ['slug', 'pr_number', 'file', 'hunk']), ('files', ['slug', 'pr_number', 'unit_id']),
             ('prs', ['slug', 'pr_number']), ('commits', ['sha'])]:
    cmp(f'tables/{t}', f'{rb}/tables/{t}.parquet', f'data/tables/{t}.parquet', k)
cmp('cohort/prs_cohort256', f'{rb}/prs_cohort256.csv', 'data/cohort/prs_cohort256.csv', ['slug', 'number', 'pr_id'])
cmp('rq2/commit_parents', f'{rb}/commit_parents.csv', 'data/rq2/commit_parents.csv', ['slug', 'sha'])
for s in ['agent', 'all']:
    cmp(f'rq3/rq3a_{s} events', f'{rb}/rq3/rq3a_{s}/data/survival_events_file_v1.csv', f'data/rq3/rq3a_{s}/survival_events_file_v1.csv', ['slug', 'pr_number', 'unit_id'])
    cmp(f'rq3/rq3b_{s}', f'{rb}/rq3full/rq3b_{s}/data/process_features_rq32.csv', f'data/rq3/rq3b_{s}/process_features_rq32.csv', ['slug', 'pr_number', 'unit_id'])
print('To use the rebuilt data, copy it into data/ (gzip the RQ3a code_content_file_v1.csv files) and run reproduce.sh.')
EOF
}

case "${1:-}" in
  ""|-h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  --list) printf '%s\n' "${STEPS[@]}"; exit 0 ;;
  all) set -- "${STEPS[@]}" ;;
esac
for s in "$@"; do
  declare -F "step_$s" >/dev/null || { echo "unknown step: $s (see --list)"; exit 2; }
  echo "== $s ($(date '+%H:%M:%S'))"; t=$(date +%s)
  "step_$s"
  echo "== $s done in $(( $(date +%s) - t )) s"
done
