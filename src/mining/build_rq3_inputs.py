"""
build_rq3_inputs.py: assemble the classifier-format RQ3 inputs for the 256-repo cohort.
  RQ3a  <out>/rq3a_{all,agent}/data/{code_content_file_v1.csv, survival_events_file_v1.csv}
  RQ3b  <out>/rq3b_{all,agent}/data/process_features_rq32.csv   (needs pr_features.jsonl)
Usage: python build_rq3_inputs.py <tables_dir> <rq3_inputs_dir> <out_dir> [a|b|ab]
"""
import sys, os, glob, json
import pandas as pd

tables, inp, out = sys.argv[1:4]
what = sys.argv[4] if len(sys.argv) > 4 else 'ab'
f = pd.read_parquet(os.path.join(tables, 'files.parquet'))
f['is_agent'] = f.is_agent.astype(bool)
keys = ['slug', 'pr_number', 'unit_id']


def dur_bin(d):
    return 0 if d <= 1 else 1 if d <= 7 else 2 if d <= 30 else 3


if 'a' in what:
    code = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(inp, '*.code.parquet'))], ignore_index=True)
    ev = f[keys + ['event', 'duration_days', 'is_agent']]
    for sub, e in [('all', ev), ('agent', ev[ev.is_agent])]:
        d = os.path.join(out, f'rq3a_{sub}', 'data'); os.makedirs(d, exist_ok=True)
        e.to_csv(os.path.join(d, 'survival_events_file_v1.csv'), index=False)
        code.merge(e[keys], on=keys).to_csv(os.path.join(d, 'code_content_file_v1.csv'), index=False)
        print('rq3a', sub, len(e), 'events,', int(e.event.sum()), 'modified')

if 'b' in what:
    feat = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(inp, '*.feat.parquet'))], ignore_index=True)
    prf = pd.read_json(os.path.join(inp, 'pr_features.jsonl'), lines=True)
    m = f[f.event == 1][keys + ['is_agent', 'agent_name', 'birth_date', 'duration_days']]
    m = m.merge(feat, on=keys, how='inner').merge(prf, on=['slug', 'pr_number'], how='inner')
    m['duration_bin'] = m.duration_days.apply(dur_bin)
    m['birth_date'] = pd.to_datetime(m.birth_date, utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S')
    cols = ['slug', 'pr_number', 'unit_id', 'is_agent', 'agent_name', 'project_commit_velocity_3m', 'file_modification_frequency_3m',
            'file_age_days', 'contributor_prior_prs', 'contributor_acceptance_rate', 'project_open_prs_at_birth', 'birth_day_of_week',
            'birth_hour', 'birth_date', 'contributor', 'duration_days', 'duration_bin']
    m = m[cols]
    miss = m.contributor_prior_prs.isna() | m.project_open_prs_at_birth.isna()
    print('rq3b dropped (PR features unavailable):', int(miss.sum()), m[miss].slug.value_counts().to_dict())
    m = m[~miss]  # the earlier version turned missing values into 0 (nan_to_num); drop instead
    for sub, e in [('all', m), ('agent', m[m.is_agent])]:
        d = os.path.join(out, f'rq3b_{sub}', 'data'); os.makedirs(d, exist_ok=True)
        e.to_csv(os.path.join(d, 'process_features_rq32.csv'), index=False)
        print('rq3b', sub, len(e), e.duration_bin.value_counts(normalize=True).sort_index().round(3).to_dict())
