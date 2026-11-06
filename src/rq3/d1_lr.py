"""D1: logistic-regression baseline on log file size (lines) + log prior 3-month changes, same 30x10 grouped CV as rq3a_tournament.
Run from an RQ3a subset dir (data/ present). Usage: python d1_lr.py <scripts_dir> <file_covariates_dir> (data/file_covariates)"""
import sys, os, glob, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import rq3a_tournament as T
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
df, X, y, groups = T.load_and_prepare_data()
feat = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(sys.argv[2], '*.parquet')))]).rename(columns={'file': 'unit_id'})
df = df.merge(feat[['slug', 'pr_number', 'unit_id', 'prior_mod_3m', 'file_age_days', 'new_file']], on=['slug', 'pr_number', 'unit_id'], how='left')
df['n_lines'] = df.code_content.fillna('').str.count('\n') + 1
F = np.column_stack([np.log1p(df.n_lines), np.log1p(df.prior_mod_3m.fillna(0))])
res = []
for rep in range(T.N_REPETITIONS):
    gkf = GroupKFold(n_splits=T.N_FOLDS, shuffle=True, random_state=T.RANDOM_STATE + rep)
    np.random.seed(T.RANDOM_STATE + rep); ug = np.unique(groups); m = {g: i for i, g in enumerate(np.random.permutation(ug))}
    sg = np.array([m[g] for g in groups])
    for k, (tr, va) in enumerate(gkf.split(F, y, sg)):
        if len(np.unique(y[va])) < 2: continue
        mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000)).fit(F[tr], y[tr])
        p = mdl.predict_proba(F[va])[:, 1]
        res.append(dict(rep=rep, fold=k, auc_roc=roc_auc_score(y[va], p), auc_pr=average_precision_score(y[va], p)))
r = pd.DataFrame(res); os.makedirs('results', exist_ok=True); r.to_csv('results/d1_lr_size_history_folds.csv', index=False)
out = dict(n=int(len(y)), pos_rate=float(y.mean()), prior_missing=int(df.prior_mod_3m.isna().sum()), folds=int(len(r)),
           **{f'{c}_{s}': float(getattr(r[c], s)()) for c in ['auc_roc', 'auc_pr'] for s in ['mean', 'median', 'std']},
           auc_roc_p2_5=float(r.auc_roc.quantile(.025)), auc_roc_p97_5=float(r.auc_roc.quantile(.975)),
           auc_pr_p2_5=float(r.auc_pr.quantile(.025)), auc_pr_p97_5=float(r.auc_pr.quantile(.975)))
json.dump(out, open('results/d1_lr_size_history.json', 'w'), indent=1); print(out)
