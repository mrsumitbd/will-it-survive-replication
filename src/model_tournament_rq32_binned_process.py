"""
model_tournament_rq32_process.py

RQ3.2: When will AI-generated code be modified?
Predicts time-to-modification using Khatoonabadi-grounded process features.

Methodology (consistent with RQ3.1):
    1. AutoSpearman preprocessing (Spearman ρ > 0.7, VIF > 5)
    2. 30 repetitions × 10-fold GroupKFold CV on FULL data (no train/test split)
    3. Model tournament: LR, NB, KNN, SVM, RF, XGBoost, CatBoost, MLP
    4. Scott-Knott ESD for statistical comparison
    5. Train final model on full data for LIME explanations

Note: We use CV on full data (no held-out test set) because project-based 
splitting results in severely imbalanced class distributions in small 
held-out sets, making held-out evaluation unreliable. CV provides unbiased 
performance estimates via grouped cross-validation. This is consistent 
with RQ3.1's approach.

Features (8 Khatoonabadi-grounded process features):
    Git-based:
        1. project_commit_velocity_3m
        2. file_modification_frequency_3m
        3. file_age_days
    AIDev-based:
        4. contributor_prior_prs
        5. contributor_acceptance_rate
        6. project_open_prs_at_birth
    Temporal:
        7. birth_day_of_week
        8. birth_hour

Target: duration_bin (4 classes)
    0: Within 1 day (≤1)
    1: 1 day to 1 week (1 < x ≤ 7)
    2: 1 week to 1 month (7 < x ≤ 30)
    3: More than 1 month (> 30)

Usage:
    python src/model_tournament_rq32_process.py

Input:
    data/process_features_rq32.csv

Output:
    results/rq32_process_tournament_scores.csv
    results/rq32_process_tournament_summary.csv
    results/rq32_process_wide_*.csv (for Scott-Knott)
    results/rq32_process_confusion_matrix.csv
    results/rq32_process_selected_features.json
    models/rq32_process_best_model.joblib

References:
    - Khatoonabadi et al. "Predicting the First Response Latency of 
      Maintainers and Contributors in Pull Requests" IEEE TSE 2024.
    - Jiarpakdee et al. "AutoSpearman" ICSME 2018.
    - Tantithamthavorn et al. "Scott-Knott ESD" IEEE TSE 2017.
"""

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

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, matthews_corrcoef
)

import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
INPUT_FILE = Path("data/process_features_rq32.csv")
OUTPUT_DIR = Path("results")
MODEL_DIR = Path("models")

# Cross-validation settings
N_REPS = 30
N_FOLDS = 10
RANDOM_STATE = 42

# AutoSpearman thresholds
SPEARMAN_THRESHOLD = 0.7
VIF_THRESHOLD = 5.0

# Feature columns (8 Khatoonabadi-grounded features)
FEATURE_COLS = [
    'project_commit_velocity_3m',
    'file_modification_frequency_3m',
    'file_age_days',
    'contributor_prior_prs',
    'contributor_acceptance_rate',
    'project_open_prs_at_birth',
    'birth_day_of_week',
    'birth_hour'
]

# Target column
TARGET_COL = 'duration_bin'

# Group column (for GroupKFold)
GROUP_COL = 'slug'

# Duration bin labels
BIN_LABELS = {
    0: "Within 1 day",
    1: "1 day to 1 week",
    2: "1 week to 1 month",
    3: "More than 1 month"
}


def autospearman(X, feature_names, correlation_threshold=0.7, vif_threshold=5.0):
    """
    AutoSpearman: Automated feature selection for collinearity and multicollinearity.
    
    Reference:
        Jiarpakdee et al. "AutoSpearman: Automatically Mitigating Correlated 
        Software Metrics for Interpreting Defect Models" ICSME 2018.
    
    Args:
        X: Feature matrix (numpy array)
        feature_names: List of feature names
        correlation_threshold: Spearman threshold (default 0.7)
        vif_threshold: VIF threshold (default 5.0)
    
    Returns:
        selected_indices: Indices of selected features
        selected_features: List of selected feature names
        removal_log: Log of removed features and reasons
    """
    df = pd.DataFrame(X, columns=feature_names)
    selected = list(feature_names)
    removal_log = []
    
    # Part 1: Spearman correlation analysis
    print("\n  Part 1: Spearman correlation analysis...")
    while True:
        if len(selected) <= 1:
            break
        
        corrmat = df[selected].corr(method='spearman')
        abs_corrmat = corrmat.abs()
        
        # Find highest correlated pair above threshold
        max_corr, max_pair = 0, None
        for i, col1 in enumerate(selected):
            for j, col2 in enumerate(selected):
                if i < j and abs_corrmat.loc[col1, col2] > max_corr:
                    max_corr = abs_corrmat.loc[col1, col2]
                    max_pair = (col1, col2)
        
        if max_pair is None or max_corr <= correlation_threshold:
            break
        
        # Remove feature with higher avg correlation to others
        col1, col2 = max_pair
        other_cols = [c for c in selected if c not in [col1, col2]]
        
        if len(other_cols) == 0:
            removed = col2
        else:
            avg1 = abs_corrmat.loc[col1, other_cols].mean()
            avg2 = abs_corrmat.loc[col2, other_cols].mean()
            removed = col1 if avg1 > avg2 else col2
        
        selected.remove(removed)
        removal_log.append({
            'feature': removed,
            'reason': 'Spearman correlation',
            'detail': f'ρ={max_corr:.3f} with {max_pair[0] if removed == max_pair[1] else max_pair[1]}'
        })
        print(f"    Removed '{removed}' (ρ={max_corr:.3f})")
    
    # Part 2: VIF analysis
    print("\n  Part 2: VIF analysis...")
    while len(selected) > 1:
        X_vif = df[selected].values
        
        # Add small constant to avoid singular matrix
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
        
        # Remove feature with highest VIF
        max_idx = vif_scores.index(max_vif)
        removed = selected[max_idx]
        selected.remove(removed)
        removal_log.append({
            'feature': removed,
            'reason': 'VIF',
            'detail': f'VIF={max_vif:.2f}'
        })
        print(f"    Removed '{removed}' (VIF={max_vif:.2f})")
    
    # Return results
    selected_indices = [feature_names.index(f) for f in selected]
    return selected_indices, selected, removal_log


def get_models():
    """
    Get dictionary of models for tournament.
    
    Models selected based on Khatoonabadi et al. TSE 2024:
    - CatBoost (their best performer)
    - Plus standard SE model families for comparison
    """
    return {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'NaiveBayes': GaussianNB(),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_STATE
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbose=False,
            auto_class_weights='Balanced'
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True
        )
    }


def evaluate_multiclass(y_true, y_pred, y_prob, n_classes=4):
    """
    Evaluate multiclass classification performance.
    
    Returns dict with multiple metrics.
    """
    metrics = {}
    
    # Macro F1 (primary metric for multiclass - treats all classes equally)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted F1 (accounts for class imbalance)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Matthews Correlation Coefficient (robust to imbalance)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    for cls in range(n_classes):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        
        metrics[f'precision_class{cls}'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f'recall_class{cls}'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        metrics[f'f1_class{cls}'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # AUC-ROC (One-vs-Rest, macro average)
    try:
        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(
                y_true, y_prob, 
                multi_class='ovr', 
                average='macro'
            )
        else:
            metrics['auc_roc'] = 0.5
    except:
        metrics['auc_roc'] = 0.5
    
    return metrics


def run_tournament(X, y, groups, feature_names, n_reps=30, n_folds=10):
    """
    Run model tournament with repeated GroupKFold cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels for GroupKFold
        feature_names: List of feature names
        n_reps: Number of repetitions
        n_folds: Number of folds
    
    Returns:
        results_df: DataFrame with all scores
        summary_df: DataFrame with summary statistics
    """
    models = get_models()
    all_results = []
    
    n_classes = len(np.unique(y))
    print(f"\n  Classes: {n_classes}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    total_iterations = n_reps * n_folds * len(models)
    iteration = 0
    
    for rep in range(n_reps):
        # Shuffle data for this repetition
        np.random.seed(RANDOM_STATE + rep)
        shuffle_idx = np.random.permutation(len(y))
        X_shuffled = X[shuffle_idx]
        y_shuffled = y[shuffle_idx]
        groups_shuffled = groups[shuffle_idx]
        
        # GroupKFold
        gkf = GroupKFold(n_splits=n_folds)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_shuffled, y_shuffled, groups_shuffled)):
            X_train, X_test = X_shuffled[train_idx], X_shuffled[test_idx]
            y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]
            
            # Scale features (fit on train only)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            for model_name, model in models.items():
                iteration += 1
                if iteration % 100 == 0:
                    print(f"    Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")
                
                try:
                    # Clone model for fresh training
                    if model_name == 'CatBoost':
                        clf = CatBoostClassifier(
                            iterations=100, depth=6, learning_rate=0.1,
                            random_state=RANDOM_STATE, verbose=False,
                            auto_class_weights='Balanced'
                        )
                    elif model_name == 'XGBoost':
                        clf = XGBClassifier(
                            n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=RANDOM_STATE, use_label_encoder=False,
                            eval_metric='mlogloss', n_jobs=-1
                        )
                    else:
                        from sklearn.base import clone
                        clf = clone(model)
                    
                    # Train
                    clf.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred = clf.predict(X_test_scaled)
                    
                    # Get probabilities
                    if hasattr(clf, 'predict_proba'):
                        y_prob = clf.predict_proba(X_test_scaled)
                    else:
                        y_prob = None
                    
                    # Evaluate
                    metrics = evaluate_multiclass(y_test, y_pred, y_prob, n_classes)
                    
                    # Store results
                    result = {
                        'rep': rep,
                        'fold': fold,
                        'model': model_name,
                        **metrics
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"      Error with {model_name} (rep={rep}, fold={fold}): {e}")
                    continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Compute summary statistics
    summary_records = []
    for model_name in models.keys():
        model_results = results_df[results_df['model'] == model_name]
        
        if len(model_results) == 0:
            continue
        
        for metric in ['macro_f1', 'weighted_f1', 'mcc', 'auc_roc']:
            scores = model_results[metric].values
            
            # Bootstrap 95% CI
            n_bootstrap = 1000
            boot_means = [
                np.mean(np.random.choice(scores, len(scores), replace=True))
                for _ in range(n_bootstrap)
            ]
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
            
            summary_records.append({
                'model': model_name,
                'metric': metric,
                'mean': scores.mean(),
                'std': scores.std(),
                'median': np.median(scores),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_scores': len(scores)
            })
    
    summary_df = pd.DataFrame(summary_records)
    
    return results_df, summary_df


def create_wide_format(results_df, metric='macro_f1'):
    """
    Convert results to wide format for Scott-Knott ESD test.
    
    Each column = model, each row = one CV score.
    """
    # Pivot: rows = (rep, fold), columns = model
    wide_df = results_df.pivot_table(
        index=['rep', 'fold'],
        columns='model',
        values=metric
    ).reset_index(drop=True)
    
    return wide_df


def compute_confusion_matrix(y_true, y_pred, labels):
    """Compute and format confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{l}' for l in labels],
        columns=[f'Pred_{l}' for l in labels]
    )
    return cm_df


def main():
    print("=" * 70)
    print("RQ3.2 MODEL TOURNAMENT: Process Features")
    print("Predicting Time-to-Modification (Khatoonabadi-grounded)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found.")
        print("Run extract_process_features_rq32.py first.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"  Total samples: {len(df):,}")
    
    # Check for required columns
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL, GROUP_COL] if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return
    
    # Prepare data
    print("\nPreparing data...")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Samples: {len(y):,}")
    print(f"  Groups (repos): {len(np.unique(groups))}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    for bin_val, label in BIN_LABELS.items():
        count = (y == bin_val).sum()
        pct = 100 * count / len(y)
        print(f"    Class {bin_val} ({label}): {count:,} ({pct:.1f}%)")
    
    # Handle missing values
    print("\nHandling missing values...")
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # =======================================================================
    # AUTOSPEARMAN PREPROCESSING
    # Note: Applied to full dataset. This is consistent with RQ3.1 approach.
    # =======================================================================
    print("\n" + "=" * 70)
    print("AUTOSPEARMAN PREPROCESSING")
    print("=" * 70)
    print(f"  Spearman threshold: {SPEARMAN_THRESHOLD}")
    print(f"  VIF threshold: {VIF_THRESHOLD}")
    
    selected_indices, selected_features, removal_log = autospearman(
        X, FEATURE_COLS, SPEARMAN_THRESHOLD, VIF_THRESHOLD
    )
    
    print(f"\n  Original features: {len(FEATURE_COLS)}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Selected: {selected_features}")
    
    # Save feature selection results
    feature_selection_results = {
        'original_features': FEATURE_COLS,
        'selected_features': selected_features,
        'removal_log': removal_log,
        'spearman_threshold': SPEARMAN_THRESHOLD,
        'vif_threshold': VIF_THRESHOLD
    }
    with open(OUTPUT_DIR / 'rq32_process_selected_features.json', 'w') as f:
        json.dump(feature_selection_results, f, indent=2)
    
    # Use selected features
    X_selected = X[:, selected_indices]
    
    # =======================================================================
    # MODEL TOURNAMENT ON FULL DATA
    # Note: We use 30x10 GroupKFold CV on full data (no train/test split).
    # This is consistent with RQ3.1. We do not use a separate held-out test 
    # set because project-based splitting results in severely imbalanced 
    # class distributions in small held-out sets, making evaluation unreliable.
    # CV provides unbiased performance estimates via grouped cross-validation.
    # =======================================================================
    print("\n" + "=" * 70)
    print("MODEL TOURNAMENT (Full Data, GroupKFold CV)")
    print("=" * 70)
    print(f"  Samples: {len(X_selected):,}")
    print(f"  Repetitions: {N_REPS}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Total CV iterations per model: {N_REPS * N_FOLDS}")
    
    results_df, summary_df = run_tournament(
        X_selected, y, groups, selected_features, N_REPS, N_FOLDS
    )
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / 'rq32_process_tournament_scores.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'rq32_process_tournament_summary.csv', index=False)
    
    # Create wide format for Scott-Knott
    for metric in ['macro_f1', 'weighted_f1', 'mcc', 'auc_roc']:
        wide_df = create_wide_format(results_df, metric)
        wide_df.to_csv(OUTPUT_DIR / f'rq32_process_wide_{metric}.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TOURNAMENT RESULTS")
    print("=" * 70)
    
    # Primary metric: Macro F1
    print("\nPrimary Metric: Macro F1 (treats all classes equally)")
    print("-" * 60)
    
    macro_f1_summary = summary_df[summary_df['metric'] == 'macro_f1'].sort_values('mean', ascending=False)
    for _, row in macro_f1_summary.iterrows():
        print(f"  {row['model']:20s}: {row['mean']:.4f} [{row['ci_lower']:.4f} - {row['ci_upper']:.4f}]")
    
    # Secondary metric: AUC-ROC
    print("\nSecondary Metric: AUC-ROC (One-vs-Rest, macro)")
    print("-" * 60)
    
    auc_summary = summary_df[summary_df['metric'] == 'auc_roc'].sort_values('mean', ascending=False)
    for _, row in auc_summary.iterrows():
        print(f"  {row['model']:20s}: {row['mean']:.4f} [{row['ci_lower']:.4f} - {row['ci_upper']:.4f}]")
    
    # Best model
    best_model_name = macro_f1_summary.iloc[0]['model']
    best_macro_f1 = macro_f1_summary.iloc[0]['mean']
    print(f"\n  Best model (Macro F1): {best_model_name} ({best_macro_f1:.4f})")
    
    # =======================================================================
    # TRAIN FINAL MODEL ON FULL DATA
    # Note: Consistent with RQ3.1 - train on all data for LIME explanations.
    # CV metrics are the primary performance measure, not held-out test.
    # =======================================================================
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL (Full Data)")
    print("=" * 70)
    print(f"  Training on: Full dataset ({len(X_selected):,} samples)")
    print(f"  Note: CV metrics are primary; this model is for LIME explanations")
    
    # Scale full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Train best model on full data
    if best_model_name == 'CatBoost':
        final_model = CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=False,
            auto_class_weights='Balanced'
        )
    elif best_model_name == 'XGBoost':
        final_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='mlogloss', n_jobs=-1
        )
    else:
        final_model = get_models()[best_model_name]
    
    final_model.fit(X_scaled, y)
    
    # Get predictions on full data (for confusion matrix and LIME)
    y_pred = final_model.predict(X_scaled)
    y_prob = final_model.predict_proba(X_scaled) if hasattr(final_model, 'predict_proba') else None
    
    # Note: These metrics are on training data - CV metrics are the primary measure
    train_metrics = evaluate_multiclass(y, y_pred, y_prob)
    
    print(f"\n  Full data metrics (reference only - CV metrics are primary):")
    print(f"    Macro F1:    {train_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {train_metrics['weighted_f1']:.4f}")
    print(f"    MCC:         {train_metrics['mcc']:.4f}")
    print(f"    AUC-ROC:     {train_metrics['auc_roc']:.4f}")
    
    # Confusion matrix on full data
    cm_df = compute_confusion_matrix(y, y_pred, labels=[0, 1, 2, 3])
    cm_df.to_csv(OUTPUT_DIR / 'rq32_process_confusion_matrix.csv')
    
    print(f"\n  Confusion Matrix:")
    print(cm_df.to_string())
    
    # Per-class performance
    print(f"\n  Per-class performance (full data):")
    for cls in range(4):
        p = train_metrics[f'precision_class{cls}']
        r = train_metrics[f'recall_class{cls}']
        f = train_metrics[f'f1_class{cls}']
        print(f"    Class {cls} ({BIN_LABELS[cls]:20s}): P={p:.3f}, R={r:.3f}, F1={f:.3f}")
    
    # Save model and scaler
    model_artifacts = {
        'model': final_model,
        'scaler': scaler,
        'selected_features': selected_features,
        'feature_indices': selected_indices,
        'train_metrics': train_metrics,  # Reference only - CV metrics are primary
        'best_model_name': best_model_name,
        # Store data for LIME analysis
        'X_scaled': X_scaled,
        'y': y,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'groups': groups
    }
    joblib.dump(model_artifacts, MODEL_DIR / 'rq32_process_best_model.joblib')
    
    # Summary
    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    print(f"  {OUTPUT_DIR / 'rq32_process_tournament_scores.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_tournament_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_wide_macro_f1.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_wide_auc_roc.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_confusion_matrix.csv'}")
    print(f"  {OUTPUT_DIR / 'rq32_process_selected_features.json'}")
    print(f"  {MODEL_DIR / 'rq32_process_best_model.joblib'}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("  1. Run Scott-Knott ESD in R:")
    print("     Rscript src/analyze_scott_knott.R results/rq32_process_wide_macro_f1.csv results/rq32_process_sk")
    print("  2. Run LIME analysis for interpretability (separate script)")
    print("  3. Compare with RQ3.1 (BOW textual features) results")
    
    # Baseline comparison
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    
    # Compute majority class baseline
    class_counts = np.bincount(y, minlength=4)
    majority_class = np.argmax(class_counts)
    majority_pct = class_counts[majority_class] / len(y)
    
    # Majority baseline: predicts majority class for all samples
    # Macro F1 = average of per-class F1
    # For majority baseline: F1 for majority class ≈ 2*precision*recall/(precision+recall)
    # where precision = majority_pct, recall = 1.0 for majority class
    # For other classes: F1 = 0 (no predictions)
    majority_precision = majority_pct
    majority_recall = 1.0
    majority_class_f1 = 2 * majority_precision * majority_recall / (majority_precision + majority_recall)
    majority_macro_f1 = majority_class_f1 / 4  # Only 1 of 4 classes has non-zero F1
    
    # Random baseline for 4-class
    random_f1 = 0.25  # Expected macro F1 for random 4-class (1/n_classes)
    
    print(f"\n  Class distribution:")
    for cls in range(4):
        pct = class_counts[cls] / len(y) * 100
        marker = " ← majority" if cls == majority_class else ""
        print(f"    Class {cls} ({BIN_LABELS[cls]:20s}): {class_counts[cls]:,} ({pct:.1f}%){marker}")
    
    print(f"\n  Baselines:")
    print(f"    Random baseline (4-class):   Macro F1 ≈ {random_f1:.4f}")
    print(f"    Majority baseline (class {majority_class}): Macro F1 ≈ {majority_macro_f1:.4f}")
    
    print(f"\n  Best model ({best_model_name}): Macro F1 = {best_macro_f1:.4f}")
    
    improvement_random = (best_macro_f1 - random_f1) / random_f1 * 100
    improvement_majority = (best_macro_f1 - majority_macro_f1) / majority_macro_f1 * 100
    
    print(f"\n  Improvement:")
    print(f"    Over random baseline:   {improvement_random:+.1f}%")
    print(f"    Over majority baseline: {improvement_majority:+.1f}%")
    
    # Khatoonabadi comparison
    print(f"\n  Khatoonabadi et al. TSE 2024 reference:")
    print(f"    Their task: 3-class first response latency prediction")
    print(f"    Their best model: CatBoost")
    print(f"    Their AUC-ROC improvement: 29-39% over majority baseline")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()