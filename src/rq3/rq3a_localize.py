"""
rq3a_localize.py: quantitative evaluation of line-level localization for RQ3a.

For every file that the RQ3a model is asked about:
  * cross-fitting exactly as rq3_explain.py (5 repo folds, seed 42; best model = RandomForest, count + SMOTE settings),
    so the model never saw the file's repository
  * SHAP (TreeExplainer) value of every unigram feature for that file
  * each line ADDED by the PR (from lines.parquet) is scored by the sum of the SHAP values of the unigram features its
    tokens contain (same tokenizer + comment/syntax filters as the BOW pipeline); LIME-style "important token" lines
    are the special case of thresholding this score
  * ground truth: the line was modified or deleted by the cutoff (per-line git blame --reverse event)
Metrics (only files with at least one modified and one unmodified added line, so ranking is defined):
  pooled line AUC-ROC / AUC-PR, mean per-file AUC-ROC, precision@top-10% of lines per file vs the base rate,
  plus a baseline that ranks lines randomly (AUC 0.5) and a 'line length' baseline (more tokens = riskier).
Usage (from the RQ3a subset directory, e.g. rq3_256/rq3a_agent):
  python rq3a_localize.py <scripts_dir> <ease_src_dir> <lines.parquet>
"""
import sys, os, json
import numpy as np, pandas as pd, shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

scripts, ease_src, lines_pq = sys.argv[1:4]
sys.path.insert(0, scripts); sys.path.insert(0, ease_src)
import rq3a_tournament as T
from extract_code_contents import tokenize_code

SEED = 42
sf = 'results/rq31_bow_tournament_summary_combined_count.csv'
best = (next(m for m in pd.read_csv(sf).sort_values('auc_pr_mean', ascending=False).model if m in ['RandomForest', 'XGBoost', 'CatBoost'])
        if os.path.exists(sf) else 'RandomForest')  # tree model for exact SHAP (RF ranked first in both 30x10 tournaments)
MAXF = int(os.environ.get('MAXF', '0'))  # >0: cap files per fold (smoke test)
df, X, y, groups = T.load_and_prepare_data()
L = pd.read_parquet(lines_pq, columns=['slug', 'pr_number', 'file', 'line', 'event', 'blank'])
L = L[L.event.notna()]
L['pr_number'] = L.pr_number.astype(str)
df['pr_number'] = df.pr_number.astype(str)
L = L.merge(df[['slug', 'pr_number', 'unit_id']].rename(columns={'unit_id': 'file'}).reset_index(), on=['slug', 'pr_number', 'file'])


def line_tokens(text, ext):
    t = ' '.join(tokenize_code(text) or [])
    t = T.remove_comments_from_tokens(t, ext)
    t = T.filter_syntax_tokens(t)
    return set(t.split()) if t else set()


rows = []
folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), None, df.slug.values)
for k, (tr, te) in enumerate(folds):
    _, model, vec, _ = T.train_final_model(X[tr], y[tr], best, 'smote', 'count')
    names = vec.get_feature_names_out()
    uni = {n: j for j, n in enumerate(names) if ' ' not in n}
    expl = shap.TreeExplainer(model)
    te_set = set(te)
    Lk = L[L['index'].isin(te_set)]
    Lk = Lk[Lk.groupby('index').event.transform('nunique') == 2]
    ids = sorted(Lk['index'].unique())[:MAXF] if MAXF else sorted(Lk['index'].unique())
    SV = expl.shap_values(vec.transform(X[ids]).toarray())
    SV = SV[1] if isinstance(SV, list) else SV
    SV = SV[:, :, 1] if SV.ndim == 3 else SV
    pos = {i: p for p, i in enumerate(ids)}
    for idx, g in Lk[Lk['index'].isin(pos)].groupby('index'):
        r = df.iloc[idx]
        sv = SV[pos[idx]]
        code = str(r.code_content).split('\n')
        for ln, ev in zip(g.line, g.event):
            text = code[ln - 1] if 0 < ln <= len(code) else ''
            toks = line_tokens(text, r.extension)
            score = float(sum(sv[uni[t]] for t in toks if t in uni))
            rows.append(dict(fold=k, file_idx=int(idx), slug=r.slug, line=int(ln), event=int(ev), score=score, n_tok=len(toks)))
    print('fold', k, 'lines so far', len(rows), flush=True)

R = pd.DataFrame(rows)
R.to_csv('results/rq3a_line_localization.csv', index=False)
rng = np.random.default_rng(SEED)
per_file, p10, p10_len, base = [], [], [], []
for _, g in R.groupby('file_idx'):
    per_file.append(roc_auc_score(g.event, g.score))
    k = max(1, int(np.ceil(0.1 * len(g))))
    p10.append(g.nlargest(k, 'score').event.mean())
    p10_len.append(g.assign(r=rng.random(len(g))).sort_values(['n_tok', 'r'], ascending=False).head(k).event.mean())
    base.append(g.event.mean())
out = dict(model=best, files=int(R.file_idx.nunique()), repos=int(R.slug.nunique()), lines=int(len(R)), line_mod_rate=float(R.event.mean()),
           pooled_auc_roc=float(roc_auc_score(R.event, R.score)), pooled_auc_pr=float(average_precision_score(R.event, R.score)),
           length_baseline_pooled_auc_roc=float(roc_auc_score(R.event, R.n_tok)),
           mean_file_auc_roc=float(np.mean(per_file)), median_file_auc_roc=float(np.median(per_file)),
           precision_at_10pct=float(np.mean(p10)), precision_at_10pct_length_baseline=float(np.mean(p10_len)),
           base_rate_mean_per_file=float(np.mean(base)))
json.dump(out, open('results/rq3a_line_localization_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
