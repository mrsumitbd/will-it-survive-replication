"""a9_model.py: primary model with file covariates at merge time (data/file_covariates/). Writes results/rq1/a9.json."""
import sys, os, json, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbc
T = fbc.Timer()
feat = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(fbc.FILE_COV, '*.parquet')))], ignore_index=True)
res = {'feature_rows': int(len(feat)), 'feature_missing_age': int(feat.file_age_days.isna().sum())}
out = {}
for gran in ['line', 'hunk', 'file']:
    d = fbc.variant(fbc.load(gran), 'primary', gran)
    d = d.merge(feat, on=['slug', 'pr_number', 'file'], how='left')
    d['test'] = (d.file.str.lower().str.contains(r'(?:^|/)(?:test|tests|spec|specs|__tests__)/') |
                 d.file.str.lower().str.rsplit('/', n=1).str[-1].str.contains('test|spec')).astype(int)
    ok = d.prior_mod_3m.notna() & d.file_age_days.notna()
    r = {'units': int(len(d)), 'units_with_features': int(ok.sum())}
    s = d[ok]
    r['desc'] = {g: dict(new_file=float(np.average(x.new_file, weights=x.w)), test=float(np.average(x.test, weights=x.w)),
                         prior_mod_3m_median=float(x.prior_mod_3m.median()), file_age_median=float(x.file_age_days.median()))
                 for g, x in list(s.groupby('group')) + [('Agent (all)', s[s.is_agent == 1])]}
    r['base_same_rows'] = fbc.within(s); T(f'{gran} base {fbc.fmt(r["base_same_rows"]["agent"])}')
    ex = ['prior_mod_3m', 'file_age_days', 'new_file', 'test']
    r['with_file_covariates'] = fbc.within(s, extra=ex, extra_log=['prior_mod_3m', 'file_age_days'])
    x = fbc.design(s, ['prior_mod_3m', 'file_age_days'])
    _, cov = fbc._cox(x, ['is_agent'] + ex, extra=['n_added_lines', 'n_source_files'])
    r['covariate_hrs'] = cov
    T(f'{gran} +cov {fbc.fmt(r["with_file_covariates"]["agent"])}')
    out[gran] = r; res['A9'] = out
    json.dump(res, open(fbc.out('rq1', 'a9.json'), 'w'), indent=1, default=float)
T('done')
