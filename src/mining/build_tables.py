"""
build_tables.py: assemble line / hunk / file survival tables from mine_survival.py output.

Usage: python build_tables.py <mine_dir> [<override_dir>] <prs.csv> <repo_meta.csv> <contrib.csv> <out_dir>
  override_dir: per-repo outputs that replace mine_dir ones (e.g. reruns with fixes)
Outputs (parquet): lines, hunks, files, prs, commits
"""
import sys, glob, os, re
import numpy as np, pandas as pd

CUTOFF = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
VENDOR_RE = re.compile(r'(?:^|/)(?:vendor|vendors|third_party|thirdparty|node_modules|dist|build|out|generated|__generated__)/'
                       r'|\.min\.js$|\.bundle\.js$|_pb2(?:_grpc)?\.py$|\.pb\.go$|\.g\.cs$|\.designer\.cs$|esm\.sh/', re.I)


def load(kind, dirs):
    frames = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, f'*.{kind}.parquet')):
            frames[os.path.basename(f)] = f  # later dirs override
    out = [pd.read_parquet(f) for f in frames.values()]
    out = [x for x in out if len(x)]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    mine, override, prs_csv, meta_csv, contrib_csv, out = sys.argv[1:7]
    dirs = [mine] + ([override] if override != '-' else [])
    os.makedirs(out, exist_ok=True)
    prs_in = pd.read_csv(prs_csv)
    # PRs listed in BOTH the AIDev agent table and the human table (same repo, number and merge time, different ids):
    # authorship is ambiguous, and keeping both rows duplicates every unit of the PR -> exclude them
    amb = prs_in[prs_in.duplicated(['slug', 'number'], keep=False)][['slug', 'number']].drop_duplicates()
    print('ambiguous PRs (agent and human table) excluded:', len(amb), amb.values.tolist())
    prs_in = prs_in[~prs_in.set_index(['slug', 'number']).index.isin(amb.set_index(['slug', 'number']).index)]
    prs = load('prs', dirs)
    prs = prs[~prs.set_index(['slug', 'pr_number']).index.isin(amb.set_index(['slug', 'number']).index)]
    prs = prs.drop_duplicates(['slug', 'pr_number'])
    prs = prs.merge(prs_in[['slug', 'number', 'pr_id']].rename(columns={'number': 'pr_number'}), on=['slug', 'pr_number'], how='left')
    meta = pd.read_csv(meta_csv)[['slug', 'stars', 'language']].drop_duplicates('slug')
    contrib = pd.read_csv(contrib_csv)
    prs = prs.merge(meta, on='slug', how='left').merge(contrib, on='slug', how='left')
    prs['is_agent'] = (prs.author_type == 'Agent').astype(int)
    prs['group'] = np.where(prs.author_type == 'Human', 'Human', np.where(prs.agent_name == 'Devin', 'Devin (autonomous)', 'Supervised assistants'))
    prs.to_parquet(os.path.join(out, 'prs.parquet'))
    ok = prs[prs.status == 'ok']
    cov = ok[['slug', 'pr_number', 'is_agent', 'author_type', 'agent_name', 'group', 'n_added_lines', 'n_source_files', 'stars', 'contributors', 'language']]

    lines = load('lines', dirs)
    lines = lines.merge(cov, on=['slug', 'pr_number'], how='inner')
    lines['birth_date'] = pd.to_datetime(lines.birth_date, utc=True)
    lines['death_date'] = pd.to_datetime(lines.death_date, utc=True)
    end = lines.death_date.where(lines.event == 1, CUTOFF)
    lines['duration_days'] = ((end - lines.birth_date).dt.total_seconds() / 86400).clip(lower=0.001)
    lines['vendored'] = lines.file.str.contains(VENDOR_RE)
    lines.to_parquet(os.path.join(out, 'lines.parquet'))

    ok_lines = lines[lines.event.notna()]
    h = ok_lines.groupby(['slug', 'pr_number', 'file', 'hunk']).agg(
        n_lines=('line', 'size'), n_dead=('event', 'sum'), birth_date=('birth_date', 'first'),
        first_death=('death_date', 'min'), vendored=('vendored', 'first')).reset_index()
    h['event'] = (h.n_dead > 0).astype(int)
    end = h.first_death.where(h.event == 1, CUTOFF)
    h['duration_days'] = ((end - h.birth_date).dt.total_seconds() / 86400).clip(lower=0.001)
    h = h.merge(cov, on=['slug', 'pr_number'], how='inner')
    h.to_parquet(os.path.join(out, 'hunks.parquet'))

    files = load('files', dirs).rename(columns={'n_added_lines': 'file_added_lines'}).merge(cov, on=['slug', 'pr_number'], how='inner')
    files['birth_date'] = pd.to_datetime(files.birth_date, utc=True)
    files['death_date'] = pd.to_datetime(files.death_date, utc=True)
    end = files.death_date.where(files.event == 1, CUTOFF)
    files['duration_days'] = ((end - files.birth_date).dt.total_seconds() / 86400).clip(lower=0.001)
    files['vendored'] = files.unit_id.str.contains(VENDOR_RE)
    files.to_parquet(os.path.join(out, 'files.parquet'))

    commits = load('commits', dirs).drop_duplicates('sha')
    commits.to_parquet(os.path.join(out, 'commits.parquet'))
    print('prs', prs.status.value_counts().to_dict())
    print('lines', len(lines), lines.kind.value_counts().to_dict())
    print('hunks', len(h), 'files', len(files), 'commits', len(commits))


if __name__ == '__main__':
    main()
