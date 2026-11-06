"""D: RQ3 inputs without vendored/generated files; RQ3b PR-history leakage fix from the cached counts.
Leakage fix (no new GitHub queries possible here: gh is not installed and there is no token):
  old  prior = #PRs by author created before T (T = merge time)  -> includes the PR itself
  old  rate  = #(created before T and merged at any time) / prior -> includes the PR itself (merged) and later merges
  new  prior' = prior - 1 ; merged' = round(rate*prior) - 1 ; rate' = merged'/prior' (0 if prior' = 0)
  This removes the PR itself exactly. PRs created before T but merged after T stay counted as merged (needs merged:<T queries).
The RQ3a inputs of the paper come from this script. The RQ3b inputs of the paper come from d_full_prep.py (full fix).
Usage: python src/mining/d_prep.py <tables_dir> <rq3_256_dir (build_rq3_inputs.py output)> <out_dir>"""
import os, sys, shutil, pandas as pd, numpy as np
TAB, IN, O = sys.argv[1:4]
f = pd.read_parquet(f'{TAB}/files.parquet')
vend = set(zip(f[f.vendored].slug, f[f.vendored].pr_number, f[f.vendored].unit_id))
k = lambda d: pd.Series([t in vend for t in zip(d.slug, d.pr_number, d.unit_id)], index=d.index)
log = []
for sub in ['all', 'agent']:
    src = f'{IN}/rq3a_{sub}/data'; dst = f'{O}/rq3a_{sub}/data'; os.makedirs(dst, exist_ok=True)
    ev = pd.read_csv(f'{src}/survival_events_file_v1.csv'); v = k(ev)
    ev[~v].to_csv(f'{dst}/survival_events_file_v1.csv', index=False)
    n0 = 0; n1 = 0; first = True
    for ch in pd.read_csv(f'{src}/code_content_file_v1.csv', chunksize=2000):
        vv = k(ch); n0 += len(ch); n1 += int((~vv).sum())
        ch[~vv].to_csv(f'{dst}/code_content_file_v1.csv', index=False, mode='w' if first else 'a', header=first); first = False
    log.append(f'rq3a_{sub}: events {len(ev)} -> {int((~v).sum())} (vendored dropped {int(v.sum())}); code rows {n0} -> {n1}')
for sub in ['all', 'agent']:
    src = f'{IN}/rq3b_{sub}/data'; dst = f'{O}/rq3b_{sub}/data'; os.makedirs(dst, exist_ok=True)
    p = pd.read_csv(f'{src}/process_features_rq32.csv'); v = k(p); p = p[~v].copy()
    prior = p.contributor_prior_prs.astype(float); merged = np.round(p.contributor_acceptance_rate * prior)
    p['contributor_prior_prs_old'] = prior; p['contributor_acceptance_rate_old'] = p.contributor_acceptance_rate
    newp = (prior - 1).clip(lower=0); newm = (merged - 1).clip(lower=0)
    p['contributor_prior_prs'] = newp
    p['contributor_acceptance_rate'] = np.where(newp > 0, np.minimum(newm / newp.where(newp > 0, 1), 1.0), 0.0)
    p.drop(columns=['contributor_prior_prs_old', 'contributor_acceptance_rate_old']).to_csv(f'{dst}/process_features_rq32.csv', index=False)
    log.append(f'rq3b_{sub}: rows {len(v)} -> {len(p)} (vendored dropped {int(v.sum())}); prior==0 old {int((prior == 0).sum())} '
               f'(PR itself not counted? check) new {int((newp == 0).sum())}; bins {p.duration_bin.value_counts(normalize=True).sort_index().round(4).to_dict()}')
open(f'{O}/prep_log.txt', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))
