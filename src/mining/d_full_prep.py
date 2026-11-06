"""D2 full fix: rebuild RQ3b inputs (non-vendored) with PR-history features from the leakage-fixed re-query (pr_fixed/*.jsonl).
Usage: python src/mining/d_full_prep.py <pr_features_fixed.jsonl> <d_prep out_dir> <out_dir>"""
import os, sys, pandas as pd
PRF, IN, O = sys.argv[1:4]
prf = pd.read_json(PRF, lines=True)
log = [f'fixed PR features: {len(prf)} PRs']
for sub in ['all', 'agent']:
    p = pd.read_csv(f'{IN}/rq3b_{sub}/data/process_features_rq32.csv')
    orig = p.columns.tolist()
    p = p.drop(columns=['contributor_prior_prs', 'contributor_acceptance_rate', 'project_open_prs_at_birth'])
    m = p.merge(prf[['slug', 'pr_number', 'contributor_prior_prs', 'contributor_acceptance_rate', 'project_open_prs_at_birth']], on=['slug', 'pr_number'], how='left')
    miss = m.contributor_prior_prs.isna() | m.project_open_prs_at_birth.isna()
    log.append(f'rq3b_{sub}: {len(m)} rows, dropped for missing PR features {int(miss.sum())} ({miss.mean():.4f}) of {len(m)} non-vendored modified files; '
               f'repos dropped rows {m[miss].slug.value_counts().to_dict()}')
    m = m[~miss]
    d = f'{O}/rq3b_{sub}/data'; os.makedirs(d, exist_ok=True)
    m[orig].to_csv(f'{d}/process_features_rq32.csv', index=False)
    log.append(f'  bins {m.duration_bin.value_counts().sort_index().to_dict()} share {m.duration_bin.value_counts(normalize=True).sort_index().round(4).to_dict()}')
open(f'{O}/prep_log.txt', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))
