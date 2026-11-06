import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')
INPUT_FILE = Path('data/process_features_rq32.csv')
OUTPUT_DIR = Path('results')
MODEL_DIR = Path('models')
N_REPS = int(__import__('os').environ.get('RQ3_N_REPS', '30'))  # paper: 30 (reproduce.sh --quick lowers it)
N_FOLDS = 10
RANDOM_STATE = 42
SPEARMAN_THRESHOLD = 0.7
VIF_THRESHOLD = 5.0
FEATURE_COLS = ['project_commit_velocity_3m', 'file_modification_frequency_3m', 'file_age_days', 'contributor_prior_prs', 'contributor_acceptance_rate', 'project_open_prs_at_birth', 'birth_day_of_week', 'birth_hour']
TARGET_COL = 'duration_bin'
GROUP_COL = 'slug'
BIN_LABELS = {0: 'Within 1 day', 1: '1 day to 1 week', 2: '1 week to 1 month', 3: 'More than 1 month'}

def autospearman(X, feature_names, correlation_threshold=0.7, vif_threshold=5.0):
    df = pd.DataFrame(X, columns=feature_names)
    selected = list(feature_names)
    removal_log = []
    print('\n  Part 1: Spearman correlation analysis...')
    while True:
        if len(selected) <= 1:
            break
        corrmat = df[selected].corr(method='spearman')
        abs_corrmat = corrmat.abs()
        max_corr, max_pair = (0, None)
        for i, col1 in enumerate(selected):
            for j, col2 in enumerate(selected):
                if i < j and abs_corrmat.loc[col1, col2] > max_corr:
                    max_corr = abs_corrmat.loc[col1, col2]
                    max_pair = (col1, col2)
        if max_pair is None or max_corr <= correlation_threshold:
            break
        col1, col2 = max_pair
        other_cols = [c for c in selected if c not in [col1, col2]]
        if len(other_cols) == 0:
            removed = col2
        else:
            avg1 = abs_corrmat.loc[col1, other_cols].mean()
            avg2 = abs_corrmat.loc[col2, other_cols].mean()
            removed = col1 if avg1 > avg2 else col2
        selected.remove(removed)
        removal_log.append({'feature': removed, 'reason': 'Spearman correlation', 'detail': f'ρ={max_corr:.3f} with {(max_pair[0] if removed == max_pair[1] else max_pair[1])}'})
        print(f"    Removed '{removed}' (ρ={max_corr:.3f})")
    print('\n  Part 2: VIF analysis...')
    while len(selected) > 1:
        X_vif = df[selected].values
        X_vif = X_vif + np.random.normal(0, 1e-10, X_vif.shape)
        vif_scores = []
        for i in range(X_vif.shape[1]):
            try:
                vif = variance_inflation_factor(X_vif, i)
                vif_scores.append(vif if not np.isinf(vif) else 100)
            except:
                vif_scores.append(0)
        max_vif = max(vif_scores)
        if max_vif < vif_threshold:
            break
        max_idx = vif_scores.index(max_vif)
        removed = selected[max_idx]
        selected.remove(removed)
        removal_log.append({'feature': removed, 'reason': 'VIF', 'detail': f'VIF={max_vif:.2f}'})
        print(f"    Removed '{removed}' (VIF={max_vif:.2f})")
    selected_indices = [feature_names.index(f) for f in selected]
    return (selected_indices, selected, removal_log)

def get_models():
    return {'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE), 'NaiveBayes': GaussianNB(), 'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'), 'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE), 'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1), 'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1), 'CatBoost': CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=False, auto_class_weights='Balanced'), 'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE, early_stopping=True)}

def evaluate_multiclass(y_true, y_pred, y_prob, n_classes=4):
    metrics = {}
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    for cls in range(n_classes):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        metrics[f'precision_class{cls}'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f'recall_class{cls}'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f'f1_class{cls}'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    try:
        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        else:
            metrics['auc_roc'] = 0.5
    except:
        metrics['auc_roc'] = 0.5
    return metrics

def run_tournament(X, y, groups, feature_names, n_reps=30, n_folds=10):
    models = get_models()
    all_results = []
    n_classes = len(np.unique(y))
    print(f'\n  Classes: {n_classes}')
    print(f'  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}')
    total_iterations = n_reps * n_folds * len(models)
    iteration = 0
    import json as _json, os as _os  # checkpoint per fold (resume after an interruption)
    _ck = str(OUTPUT_DIR / 'ckpt.jsonl'); _done = {}
    if _os.path.exists(_ck):
        for _l in open(_ck):
            _r = _json.loads(_l); _done.setdefault((_r['rep'], _r['fold']), []).append(_r)
    _ckf = open(_ck, 'a')
    for rep in range(n_reps):
        np.random.seed(RANDOM_STATE + rep)
        shuffle_idx = np.random.permutation(len(y))
        X_shuffled = X[shuffle_idx]
        y_shuffled = y[shuffle_idx]
        groups_shuffled = groups[shuffle_idx]
        gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE + rep)  # fix: plain GroupKFold repeats the same split every rep
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_shuffled, y_shuffled, groups_shuffled)):
            if (rep, fold) in _done:
                all_results.extend(_done[(rep, fold)]); iteration += len(models); continue
            _n0 = len(all_results)
            X_train, X_test = (X_shuffled[train_idx], X_shuffled[test_idx])
            y_train, y_test = (y_shuffled[train_idx], y_shuffled[test_idx])
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            for model_name, model in models.items():
                iteration += 1
                if iteration % 100 == 0:
                    print(f'    Progress: {iteration}/{total_iterations} ({100 * iteration / total_iterations:.1f}%)')
                try:
                    if model_name == 'CatBoost':
                        clf = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=False, auto_class_weights='Balanced')
                    elif model_name == 'XGBoost':
                        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
                    else:
                        from sklearn.base import clone
                        clf = clone(model)
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                    if hasattr(clf, 'predict_proba'):
                        y_prob = clf.predict_proba(X_test_scaled)
                    else:
                        y_prob = None
                    metrics = evaluate_multiclass(y_test, y_pred, y_prob, n_classes)
                    result = {'rep': rep, 'fold': fold, 'model': model_name, **metrics}
                    all_results.append(result)
                except Exception as e:
                    print(f'      Error with {model_name} (rep={rep}, fold={fold}): {e}')
                    continue
            for _r in all_results[_n0:]:
                _ckf.write(_json.dumps(_r, default=float) + '\n')
            _ckf.flush()
    results_df = pd.DataFrame(all_results)
    summary_records = []
    for model_name in models.keys():
        model_results = results_df[results_df['model'] == model_name]
        if len(model_results) == 0:
            continue
        for metric in ['macro_f1', 'weighted_f1', 'mcc', 'auc_roc']:
            scores = model_results[metric].values
            n_bootstrap = 1000
            boot_means = [np.mean(np.random.choice(scores, len(scores), replace=True)) for _ in range(n_bootstrap)]
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
            summary_records.append({'model': model_name, 'metric': metric, 'mean': scores.mean(), 'std': scores.std(), 'median': np.median(scores), 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'n_scores': len(scores)})
    summary_df = pd.DataFrame(summary_records)
    return (results_df, summary_df)

def create_wide_format(results_df, metric='macro_f1'):
    wide_df = results_df.pivot_table(index=['rep', 'fold'], columns='model', values=metric).reset_index(drop=True)
    return wide_df

def compute_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f'True_{l}' for l in labels], columns=[f'Pred_{l}' for l in labels])
    return cm_df

def main():
    print('=' * 70)
    print('RQ3.2 MODEL TOURNAMENT: Process Features')
    print('Predicting Time-to-Modification (Khatoonabadi-grounded)')
    print('=' * 70)
    print(f'Timestamp: {datetime.now().isoformat()}')
    print()
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    print('Loading data...')
    if not INPUT_FILE.exists():
        print(f'ERROR: {INPUT_FILE} not found.')
        print('Run extract_process_features_rq32.py first.')
        return
    df = pd.read_csv(INPUT_FILE)
    print(f'  Total samples: {len(df):,}')
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL, GROUP_COL] if c not in df.columns]
    if missing_cols:
        print(f'ERROR: Missing columns: {missing_cols}')
        return
    print('\nPreparing data...')
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    print(f'  Features: {len(FEATURE_COLS)}')
    print(f'  Samples: {len(y):,}')
    print(f'  Groups (repos): {len(np.unique(groups))}')
    print(f'\n  Class distribution:')
    for bin_val, label in BIN_LABELS.items():
        count = (y == bin_val).sum()
        pct = 100 * count / len(y)
        print(f'    Class {bin_val} ({label}): {count:,} ({pct:.1f}%)')
    print('\nHandling missing values...')
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    print('\n' + '=' * 70)
    print('AUTOSPEARMAN PREPROCESSING')
    print('=' * 70)
    print(f'  Spearman threshold: {SPEARMAN_THRESHOLD}')
    print(f'  VIF threshold: {VIF_THRESHOLD}')
    selected_indices, selected_features, removal_log = autospearman(X, FEATURE_COLS, SPEARMAN_THRESHOLD, VIF_THRESHOLD)
    print(f'\n  Original features: {len(FEATURE_COLS)}')
    print(f'  Selected features: {len(selected_features)}')
    print(f'  Selected: {selected_features}')
    feature_selection_results = {'original_features': FEATURE_COLS, 'selected_features': selected_features, 'removal_log': removal_log, 'spearman_threshold': SPEARMAN_THRESHOLD, 'vif_threshold': VIF_THRESHOLD}
    with open(OUTPUT_DIR / 'rq32_process_selected_features.json', 'w') as f:
        json.dump(feature_selection_results, f, indent=2)
    X_selected = X[:, selected_indices]
    print('\n' + '=' * 70)
    print('MODEL TOURNAMENT (Full Data, GroupKFold CV)')
    print('=' * 70)
    print(f'  Samples: {len(X_selected):,}')
    print(f'  Repetitions: {N_REPS}')
    print(f'  Folds: {N_FOLDS}')
    print(f'  Total CV iterations per model: {N_REPS * N_FOLDS}')
    results_df, summary_df = run_tournament(X_selected, y, groups, selected_features, N_REPS, N_FOLDS)
    results_df.to_csv(OUTPUT_DIR / 'rq32_process_tournament_scores.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'rq32_process_tournament_summary.csv', index=False)
    for metric in ['macro_f1', 'weighted_f1', 'mcc', 'auc_roc']:
        wide_df = create_wide_format(results_df, metric)
        wide_df.to_csv(OUTPUT_DIR / f'rq32_process_wide_{metric}.csv', index=False)
    print('\n' + '=' * 70)
    print('TOURNAMENT RESULTS')
    print('=' * 70)
    print('\nPrimary Metric: Macro F1 (treats all classes equally)')
    print('-' * 60)
    macro_f1_summary = summary_df[summary_df['metric'] == 'macro_f1'].sort_values('mean', ascending=False)
    for _, row in macro_f1_summary.iterrows():
        print(f"  {row['model']:20s}: {row['mean']:.4f} [{row['ci_lower']:.4f} - {row['ci_upper']:.4f}]")
    print('\nSecondary Metric: AUC-ROC (One-vs-Rest, macro)')
    print('-' * 60)
    auc_summary = summary_df[summary_df['metric'] == 'auc_roc'].sort_values('mean', ascending=False)
    for _, row in auc_summary.iterrows():
        print(f"  {row['model']:20s}: {row['mean']:.4f} [{row['ci_lower']:.4f} - {row['ci_upper']:.4f}]")
    best_model_name = macro_f1_summary.iloc[0]['model']
    best_macro_f1 = macro_f1_summary.iloc[0]['mean']
    print(f'\n  Best model (Macro F1): {best_model_name} ({best_macro_f1:.4f})')
    print('\n' + '=' * 70)
    print('TRAINING FINAL MODEL (Full Data)')
    print('=' * 70)
    print(f'  Training on: Full dataset ({len(X_selected):,} samples)')
    print(f'  Note: CV metrics are primary; this model is for LIME explanations')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    if best_model_name == 'CatBoost':
        final_model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=False, auto_class_weights='Balanced')
    elif best_model_name == 'XGBoost':
        final_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    else:
        final_model = get_models()[best_model_name]
    final_model.fit(X_scaled, y)
    y_pred = final_model.predict(X_scaled)
    y_prob = final_model.predict_proba(X_scaled) if hasattr(final_model, 'predict_proba') else None
    train_metrics = evaluate_multiclass(y, y_pred, y_prob)
    print(f'\n  Full data metrics (reference only - CV metrics are primary):')
    print(f"    Macro F1:    {train_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {train_metrics['weighted_f1']:.4f}")
    print(f"    MCC:         {train_metrics['mcc']:.4f}")
    print(f"    AUC-ROC:     {train_metrics['auc_roc']:.4f}")
    cm_df = compute_confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
    cm_df.to_csv(OUTPUT_DIR / 'rq32_process_confusion_matrix.csv')
    print(f'\n  Confusion Matrix:')
    print(cm_df.to_string())
    print(f'\n  Per-class performance (full data):')
    for cls in range(4):
        p = train_metrics[f'precision_class{cls}']
        r = train_metrics[f'recall_class{cls}']
        f = train_metrics[f'f1_class{cls}']
        print(f'    Class {cls} ({BIN_LABELS[cls]:20s}): P={p:.3f}, R={r:.3f}, F1={f:.3f}')
    model_artifacts = {'model': final_model, 'scaler': scaler, 'selected_features': selected_features, 'feature_indices': selected_indices, 'train_metrics': train_metrics, 'best_model_name': best_model_name, 'X_scaled': X_scaled, 'y': y, 'y_pred': y_pred, 'y_prob': y_prob, 'groups': groups}
    joblib.dump(model_artifacts, MODEL_DIR / 'rq32_process_best_model.joblib')
    print('\n' + '=' * 70)
    print('FILES GENERATED')
    print('=' * 70)
    print(f"  {OUTPUT_DIR / 'rq32_process_tournament_scores.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_tournament_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_wide_macro_f1.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_wide_auc_roc.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_confusion_matrix.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_selected_features.json'}")
    print(f"  {MODEL_DIR / 'rq32_process_best_model.joblib'}")
    print('\n' + '=' * 70)
    print('NEXT STEPS')
    print('=' * 70)
    print('  1. Run Scott-Knott ESD in R:')
    print('     Rscript src/analyze_scott_knott.R results/rq32_process_wide_macro_f1.csv results/rq32_process_sk')
    print('  2. Run LIME analysis for interpretability (separate script)')
    print('  3. Compare with RQ3.1 (BOW textual features) results')
    print('\n' + '=' * 70)
    print('BASELINE COMPARISON')
    print('=' * 70)
    class_counts = np.bincount(y, minlength=4)
    majority_class = np.argmax(class_counts)
    majority_pct = class_counts[majority_class] / len(y)
    majority_precision = majority_pct
    majority_recall = 1.0
    majority_class_f1 = 2 * majority_precision * majority_recall / (majority_precision + majority_recall)
    majority_macro_f1 = majority_class_f1 / 4
    random_f1 = 0.25
    print(f'\n  Class distribution:')
    for cls in range(4):
        pct = class_counts[cls] / len(y) * 100
        marker = ' ← majority' if cls == majority_class else ''
        print(f'    Class {cls} ({BIN_LABELS[cls]:20s}): {class_counts[cls]:,} ({pct:.1f}%){marker}')
    print(f'\n  Baselines:')
    print(f'    Random baseline (4-class):   Macro F1 ≈ {random_f1:.4f}')
    print(f'    Majority baseline (class {majority_class}): Macro F1 ≈ {majority_macro_f1:.4f}')
    print(f'\n  Best model ({best_model_name}): Macro F1 = {best_macro_f1:.4f}')
    improvement_random = (best_macro_f1 - random_f1) / random_f1 * 100
    improvement_majority = (best_macro_f1 - majority_macro_f1) / majority_macro_f1 * 100
    print(f'\n  Improvement:')
    print(f'    Over random baseline:   {improvement_random:+.1f}%')
    print(f'    Over majority baseline: {improvement_majority:+.1f}%')
    print(f'\n  Khatoonabadi et al. TSE 2024 reference:')
    print(f'    Their task: 3-class first response latency prediction')
    print(f'    Their best model: CatBoost')
    print(f'    Their AUC-ROC improvement: 29-39% over majority baseline')
    print('\n' + '=' * 70)
    print('COMPLETE')
    print('=' * 70)
if __name__ == '__main__':
    main()