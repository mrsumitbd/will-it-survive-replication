"""
extract_rq3_git.py: RQ3 inputs from git for every file unit in tables/files.parquet (256-repo cohort).

Per file unit (slug, pr_number, path, birth_sha, birth_date):
  * RQ3a: file content at the merge commit (birth_sha) + tokenizer of src/rq3/extract_code_contents.py -> code_content_file.csv
  * RQ3b git process features (definitions of the earlier version, exact timestamps instead of day strings,
    history read from birth_sha instead of the clone-time HEAD):
      project_commit_velocity_3m      commits in (T-90d, T]
      file_modification_frequency_3m  commits touching the path in (T-90d, T]
      file_age_days                   T - first commit that added the path
      birth_day_of_week, birth_hour   (UTC)
Usage: python extract_rq3_git.py <files.parquet> <ease_src_dir> <out_dir> [--workers 3] [--slugs a,b]
"""
import sys, os, json, argparse, subprocess, tempfile, shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def git(repo, *args):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True)
    return r.stdout.decode('utf-8', 'replace') if r.returncode == 0 else None


def do_repo(slug, rows, src, out_dir):
    sys.path.insert(0, src)
    from extract_code_contents import tokenize_code
    tag = slug.replace('/', '__')
    done = os.path.join(out_dir, tag + '.done')
    if os.path.exists(done):
        return slug, 'cached'
    tmp = tempfile.mkdtemp(dir=os.environ.get('CLONE_DIR'))
    try:
        r = subprocess.run(['git', 'clone', '--no-checkout', '--quiet', f'https://github.com/{slug}.git', tmp], capture_output=True)
        if r.returncode != 0:
            return slug, 'clone_failed: ' + r.stderr.decode()[-200:]
        code, feats = [], []
        for x in rows.itertuples():
            T = pd.Timestamp(x.birth_date).tz_convert('UTC')
            since = (T - pd.Timedelta(days=90)).isoformat(); until = T.isoformat()
            content = git(tmp, 'show', f'{x.birth_sha}:{x.unit_id}')
            if content is not None:
                toks = tokenize_code(content)
                code.append(dict(slug=slug, pr_number=x.pr_number, unit_id=x.unit_id, birth_sha=x.birth_sha,
                                 code_content=content, code_tokens=' '.join(toks), num_tokens=len(toks)))
            vel = git(tmp, 'rev-list', '--count', f'--since={since}', f'--until={until}', x.birth_sha)
            freq = git(tmp, 'rev-list', '--count', f'--since={since}', f'--until={until}', x.birth_sha, '--', x.unit_id)
            first = git(tmp, 'log', '--reverse', '--format=%cI', '--diff-filter=A', x.birth_sha, '--', x.unit_id)
            if not first or not first.strip():
                first = git(tmp, 'log', '--reverse', '--format=%cI', x.birth_sha, '--', x.unit_id)
            age = 0.0
            if first and first.strip():
                age = max(0.0, (T - pd.Timestamp(first.split('\n')[0].strip()).tz_convert('UTC')).total_seconds() / 86400)
            feats.append(dict(slug=slug, pr_number=x.pr_number, unit_id=x.unit_id,
                              project_commit_velocity_3m=int(vel) if vel else 0,
                              file_modification_frequency_3m=int(freq) if freq else 0,
                              file_age_days=int(age), birth_day_of_week=T.weekday(), birth_hour=T.hour))
        pd.DataFrame(code).to_parquet(os.path.join(out_dir, tag + '.code.parquet'))
        pd.DataFrame(feats).to_parquet(os.path.join(out_dir, tag + '.feat.parquet'))
        json.dump(dict(n=len(rows), code=len(code)), open(done, 'w'))
        return slug, f'ok {len(code)}/{len(rows)}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('files'); ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=3); ap.add_argument('--slugs', default=''); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = pd.read_parquet(a.files)
    if a.slugs:
        f = f[f.slug.isin(a.slugs.split(','))]
    groups = list(f.groupby('slug'))
    groups.sort(key=lambda g: -len(g[1]))
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(do_repo, s, g, a.src, a.out) for s, g in groups]
        for i, fu in enumerate(as_completed(futs)):
            try:
                s, st = fu.result()
            except Exception as e:
                s, st = '?', repr(e)
            print(f'[{i + 1}/{len(futs)}] {s}: {st}', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
