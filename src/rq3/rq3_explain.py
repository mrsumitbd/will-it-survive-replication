"""
rq3_explain.py: out-of-sample LIME and SHAP for RQ3a (BOW) and RQ3b (process features).

An earlier version fitted the final model on ALL data and explained the same data (in-sample, AUC ~0.98). Here:
  * cross-fitting: repositories are split into 5 folds (GroupKFold, shuffled, seed 42); for each fold the tournament's
    best model (RQ3a: AUC-PR, RQ3b: Macro F1) is fit on the other 4 folds with the final-model settings of the tournament script
  * every file gets an out-of-sample prediction and SHAP values from the model that never saw its repository;
    LIME (default settings of the tournament script) explains 1/5 of the 100 sampled files per fold, again with that fold's model
  * SHAP: TreeExplainer on the best model if it is tree-based, else on the best-ranked tree-based model (reported)
Run from the RQ directory (with data/ and results/ from the tournament):
  python rq3_explain.py rq3a <scripts_dir>     |   python rq3_explain.py rq3b <scripts_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

mode, scripts = sys.argv[1], sys.argv[2]
sys.path.insert(0, scripts)
os.makedirs('results', exist_ok=True)
TREE = ['RandomForest', 'XGBoost', 'CatBoost']
SEED = 42


def folds(groups):
    return list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(groups)), None, groups))


def shap_values(model, X):
    sv = shap.TreeExplainer(model).shap_values(X)
    return sv


if mode == 'rq3a':
    import rq3a_tournament as T
    summ = pd.read_csv('results/rq31_bow_tournament_summary_combined_count.csv').sort_values('auc_pr_mean', ascending=False)
    best = summ.model.iloc[0]
    best_tree = next(m for m in summ.model if m in TREE)
    df, X, y, groups = T.load_and_prepare_data()
    smodel_name = best if best in TREE else best_tree
    p = np.zeros(len(y)); ps = np.zeros(len(y)); lime_parts, shap_parts = [], []
    for k, (tr, te) in enumerate(folds(df.slug.values)):
        _, model, vec, _ = T.train_final_model(X[tr], y[tr], best, 'smote', 'count')
        Xte = vec.transform(X[te])
        p[te] = model.predict_proba(Xte.toarray() if best in ['KNN', 'SVM', 'MLP'] else Xte)[:, 1]
        dte = df.iloc[te].reset_index(drop=True)
        lime_parts.append(T.apply_lime_explanations(model, vec, dte, X[te], y[te], p[te], max_explanations=20).assign(fold=k))
        if smodel_name != best:
            _, model, vec, _ = T.train_final_model(X[tr], y[tr], smodel_name, 'smote', 'count')
            Xte = vec.transform(X[te])
        ps[te] = model.predict_proba(Xte)[:, 1]
        sv = shap_values(model, Xte.toarray())
        sv = sv[1] if isinstance(sv, list) else sv
        sv = sv[:, :, 1] if sv.ndim == 3 else sv
        shap_parts.append(pd.DataFrame(sv, columns=vec.get_feature_names_out()).abs().mean().rename(f'abs{k}').to_frame()
                          .join(pd.DataFrame(sv, columns=vec.get_feature_names_out()).mean().rename(f'sig{k}'))
                          .join(pd.Series(np.asarray((Xte > 0).mean(0)).ravel(), index=vec.get_feature_names_out(), name=f'df{k}'))
                          .assign(**{f'n{k}': len(te)}))
    out = dict(best_model=best, shap_model=smodel_name, n=len(y), repos=int(df.slug.nunique()), pos_rate=float(y.mean()),
               oos_auc_roc=float(roc_auc_score(y, p)), oos_auc_pr=float(average_precision_score(y, p)),
               oos_f1=float(f1_score(y, (p >= .5).astype(int))), shap_model_oos_auc_pr=float(average_precision_score(y, ps)))
    lime_df = pd.concat(lime_parts, ignore_index=True); lime_df.to_csv('results/rq3a_lime_oos.csv', index=False)
    J = pd.concat(shap_parts, axis=1).fillna(0)  # vocabulary differs per fold; a token absent in a fold counts 0
    w = np.array([J[f'n{k}'].max() for k in range(5)], float); w /= w.sum()
    imp = pd.DataFrame({'mean_abs_shap': sum(w[k] * J[f'abs{k}'] for k in range(5)), 'mean_shap': sum(w[k] * J[f'sig{k}'] for k in range(5)),
                        'doc_freq': sum(w[k] * J[f'df{k}'] for k in range(5)), 'n_folds_in_vocab': sum((J[f'n{k}'] > 0).astype(int) for k in range(5))})
    imp = imp.sort_values('mean_abs_shap', ascending=False).rename_axis('token').reset_index()
    imp.to_csv('results/rq3a_shap_oos.csv', index=False)
    top50 = set(imp.token.head(50))
    lt = [t for s_ in lime_df.all_token_weights for t, _ in eval(s_)]
    out['lime_tokens_in_shap_top50'] = float(np.mean([t in top50 for t in lt])) if lt else None
    out['lime_top_tokens'] = pd.Series(lt).value_counts().head(20).to_dict()
    out['shap_top_tokens'] = imp.head(20)[['token', 'mean_abs_shap', 'mean_shap']].round(5).to_dict('records')
else:
    import rq3b_tournament as T
    from lime_analysis_rq32_binned_process import run_lime_analysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    summ = pd.read_csv('results/rq32_process_tournament_summary.csv')
    summ = summ[summ.metric == 'macro_f1'].sort_values('mean', ascending=False)
    best = summ.model.iloc[0]
    best_tree = next(m for m in summ.model if m in TREE)
    df = pd.read_csv(T.INPUT_FILE)
    X = np.nan_to_num(df[T.FEATURE_COLS].values.astype(float), nan=0, posinf=0, neginf=0)
    y = df[T.TARGET_COL].values
    sel = json.load(open('results/rq32_process_selected_features.json'))['selected_features']
    idx = [T.FEATURE_COLS.index(f) for f in sel]
    X = X[:, idx]

    def make(name):
        from catboost import CatBoostClassifier
        from xgboost import XGBClassifier
        if name == 'CatBoost':
            return CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=SEED, verbose=False, auto_class_weights='Balanced')
        if name == 'XGBoost':
            return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED, eval_metric='mlogloss', n_jobs=-1)
        from sklearn.base import clone
        return clone(T.get_models()[name])

    smodel_name = best if best in TREE else best_tree
    yp = np.zeros(len(y), int); pr = np.zeros((len(y), 4)); sv_all = np.zeros((len(y), len(sel), 4))
    exps, imps = [], []
    for k, (tr, te) in enumerate(folds(df.slug.values)):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        model = make(best).fit(Xtr, y[tr])
        yp[te] = np.asarray(model.predict(Xte)).ravel().astype(int); pr[te] = model.predict_proba(Xte)
        e, i = run_lime_analysis(model, Xte, y[te], yp[te], sel, max_samples=20)
        exps.append(e.assign(fold=k)); imps.append(i.assign(fold=k))
        smodel = model if smodel_name == best else make(smodel_name).fit(Xtr, y[tr])
        sv = shap_values(smodel, Xte)
        sv_all[te] = np.stack(sv, -1) if isinstance(sv, list) else sv
    Xz = np.zeros_like(X, dtype=float)  # per-fold standardised values for the value-vs-SHAP direction
    for tr, te in folds(df.slug.values):
        Xz[te] = StandardScaler().fit(X[tr]).transform(X[te])
    m = T.evaluate_multiclass(y, yp, pr, 4)
    out = dict(best_model=best, shap_model=smodel_name, selected_features=sel, n=len(y), repos=int(df.slug.nunique()), oos=m,
               oos_report=classification_report(y, yp, output_dict=True, zero_division=0))
    exp_df = pd.concat(exps, ignore_index=True); exp_df.to_csv('results/rq3b_lime_oos_explanations.csv', index=False)
    imp_df = pd.concat(imps, ignore_index=True); imp_df.to_csv('results/rq3b_lime_oos_importance_by_fold.csv', index=False)
    rows = []
    for c in range(4):
        s_ = sv_all[:, :, c]
        for j, f_ in enumerate(sel):
            rho = pd.Series(Xz[:, j]).corr(pd.Series(s_[:, j]), method='spearman')
            rows.append(dict(cls=c, bin=T.BIN_LABELS[c], feature=f_, mean_abs_shap=float(np.abs(s_[:, j]).mean()),
                             spearman_value_vs_shap=float(rho) if pd.notna(rho) else None))
    shp = pd.DataFrame(rows); shp.to_csv('results/rq3b_shap_oos.csv', index=False)
    out['shap_overall'] = shp.groupby('feature').mean_abs_shap.mean().sort_values(ascending=False).round(4).to_dict()
    ov = imp_df[imp_df.scope == 'Overall'].groupby('feature').mean_abs_weight.mean().sort_values(ascending=False)
    out['lime_overall'] = ov.round(4).to_dict()

json.dump(out, open(f'results/{mode}_explain_summary.json', 'w'), indent=1, default=float)
print('done', flush=True)
