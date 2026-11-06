"""commit_parents.py: parent count of every commit that modifies a tracked line or file (RQ2 merge-commit check).

Reads the death commits from data/tables/{lines,files}.parquet and asks a local bare clone of each repository
(`git rev-list --no-walk --parents`) for their parents. Writes data/rq2/commit_parents.csv (slug, sha, n_parents).
Only rebuild_data.sh needs this script; the analyses read the shipped CSV. Set TABLES_DIR to read rebuilt tables.
Usage: python src/mining/commit_parents.py <clones_dir> [<out.csv>]
  <clones_dir> holds one bare clone per repository, named <owner>__<name>.git
"""
import os, sys, subprocess
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import TABLES, DATA


def parents_from_clones(shas, clones):
    """shas: DataFrame with slug, death_sha. Returns {sha: n_parents} (same logic as the paper's run)."""
    par = {}
    for s, g in shas.groupby('slug'):
        r = subprocess.run(['git', '-C', os.path.join(clones, s.replace('/', '__') + '.git'), 'rev-list', '--no-walk', '--parents', '--stdin'],
                           input='\n'.join(g.death_sha) + '\n', capture_output=True, text=True)
        for line in r.stdout.split('\n'):
            p = line.split()
            if p:
                par[p[0]] = len(p) - 1
    return par


def death_shas():
    l = pd.read_parquet(os.path.join(TABLES, 'lines.parquet'), columns=['slug', 'event', 'death_sha'])
    f = pd.read_parquet(os.path.join(TABLES, 'files.parquet'), columns=['slug', 'event', 'death_sha'])
    return pd.concat([l[l.event == 1][['slug', 'death_sha']], f[f.event == 1][['slug', 'death_sha']]]).drop_duplicates()


if __name__ == '__main__':
    clones = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, 'rq2', 'commit_parents.csv')
    shas = death_shas()
    par = parents_from_clones(shas, clones)
    d = shas.assign(n_parents=shas.death_sha.map(par)).rename(columns={'death_sha': 'sha'})
    d = d.dropna(subset=['n_parents']).astype({'n_parents': int}).sort_values(['slug', 'sha'])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d[['slug', 'sha', 'n_parents']].to_csv(out, index=False)
    print(f'{len(d)} of {len(shas)} commits resolved -> {out}')
