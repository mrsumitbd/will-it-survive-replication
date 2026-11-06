"""filter_prs.py: copy the RQ1 tables with a list of PRs removed (attribution-audit sensitivity).
Usage: python filter_prs.py <tables_dir> <exclude.csv with slug,pr_number> <out_tables_dir>"""
import sys, os, shutil
import pandas as pd
src, exc, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
x = pd.read_csv(exc, dtype=str)
keys = set(zip(x.slug.str.lower(), x.pr_number.str.replace(r'\.0$', '', regex=True)))
for f in sorted(os.listdir(src)):
    if not f.endswith('.parquet'):
        continue
    d = pd.read_parquet(os.path.join(src, f))
    if {'slug', 'pr_number'} <= set(d.columns):
        k = list(zip(d.slug.str.lower(), d.pr_number.astype(str).str.replace(r'\.0$', '', regex=True)))
        drop = pd.Series([t in keys for t in k], index=d.index)
        hit = {t for t, m in zip(k, drop) if m}
        print(f, 'rows', len(d), 'dropped', int(drop.sum()), 'PRs matched', len(hit), 'of', len(keys), flush=True)
        d = d[~drop]
        d.to_parquet(os.path.join(out, f), index=False)
    else:
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
