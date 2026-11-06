"""A9: file-level covariates at merge time for every file unit (local bare clones in data/temp_repos).
prior_mod_3m = commits reachable from the PR base that touch the path in the 90 days before the merge (PR itself excluded)
file_age_days = merge time - first commit (reachable from base) that touches the path; 0 for new files
new_file = path absent at the PR base commit
Usage: python src/mining/a9_extract.py <clones_dir> <tables_dir> <out_dir> [workers]
  <clones_dir> holds bare clones named <owner>__<name>.git. The shipped output is data/file_covariates/."""
import os, sys, subprocess, json, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
R, TAB, O = (os.path.abspath(x) for x in sys.argv[1:4]); os.makedirs(O, exist_ok=True)
def git(repo, *a):
    r = subprocess.run(['git', '-C', repo, *a], capture_output=True)
    return r.stdout.decode('utf-8', 'replace') if r.returncode == 0 else None
def do(slug, rows):
    out = os.path.join(O, slug.replace('/', '__') + '.parquet')
    if os.path.exists(out): return slug, 'cached'
    repo = os.path.join(R, slug.replace('/', '__') + '.git'); res = []
    for x in rows.itertuples():
        T = pd.Timestamp(x.birth_date).tz_convert('UTC'); base = x.base_sha
        exists = git(repo, 'cat-file', '-e', f'{base}:{x.unit_id}') is not None
        if exists:
            n = git(repo, 'rev-list', '--count', f'--since={(T - pd.Timedelta(days=90)).isoformat()}', base, '--', x.unit_id)
            first = git(repo, 'log', '--format=%cI', '--diff-filter=A', base, '--', x.unit_id)
            first = first.strip().split('\n')[-1] if first and first.strip() else None
            age = max(0.0, (T - pd.Timestamp(first).tz_convert('UTC')).total_seconds() / 86400) if first else None
        else:
            n, age = 0, 0.0
        res.append(dict(slug=slug, pr_number=x.pr_number, file=x.unit_id, new_file=int(not exists),
                        prior_mod_3m=int(n) if n not in (None, '') else None, file_age_days=age))
    pd.DataFrame(res).to_parquet(out); return slug, f'ok {len(res)}'
if __name__ == '__main__':
    f = pd.read_parquet(os.path.join(TAB, 'files.parquet'))
    p = pd.read_parquet(os.path.join(TAB, 'prs.parquet'))[['slug', 'pr_number', 'base_sha']]
    f = f.merge(p, on=['slug', 'pr_number'])
    gs = sorted(f.groupby('slug'), key=lambda g: -len(g[1]))
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 5) as ex:
        fs = [ex.submit(do, s, g) for s, g in gs]
        for i, fu in enumerate(as_completed(fs)):
            try: s, st = fu.result()
            except Exception as e: s, st = '?', repr(e)
            print(f'[{i+1}/{len(fs)}] {s}: {st}', flush=True)
    print('done', flush=True)
