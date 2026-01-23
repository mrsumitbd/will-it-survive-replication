"""
lime_analysis_rq32_process.py

LIME explanations for RQ3.2: When will AI-generated code be modified?

Generates instance-level explanations using LIME (Local Interpretable 
Model-agnostic Explanations) for the process features model.

Methodology:
    1. Load trained model and data from model_tournament_rq32_process.py
    2. Stratified sampling across prediction categories (TP, FP, TN, FN per class)
    3. Generate LIME explanations for each sample
    4. Aggregate feature importance across samples

Note: Even with weak overall signal (Macro F1 = 0.285), LIME can reveal
which features the model relies on for correct vs incorrect predictions.

Usage:
    python src/lime_analysis_rq32_process.py
    python src/lime_analysis_rq32_process.py --max-samples 200

Input:
    models/rq32_process_best_model.joblib
    data/process_features_rq32.csv

Output:
    results/rq32_process_lime_explanations.csv
    results/rq32_process_lime_feature_importance.csv

References:
    - Ribeiro et al. "Why Should I Trust You?: Explaining the Predictions 
      of Any Classifier" KDD 2016.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import lime
import lime.lime_tabular

import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_FILE = Path("models/rq32_process_best_model.joblib")
DATA_FILE = Path("data/process_features_rq32.csv")
OUTPUT_DIR = Path("results")

# LIME Configuration
LIME_NUM_FEATURES = 7  # All features (after AutoSpearman)
LIME_NUM_SAMPLES = 500  # Perturbation samples for each explanation
MAX_EXPLANATIONS = 100  # Max explanations per class (total = 4 classes × 100)

# Class labels
BIN_LABELS = {
    0: "Within 1 day",
    1: "1 day to 1 week",
    2: "1 week to 1 month",
    3: "More than 1 month"
}

RANDOM_STATE = 42


def get_prediction_category(y_true, y_pred, class_of_interest):
    """
    Determine prediction category for multiclass.
    
    For a given class of interest:
    - TP: True class = class_of_interest, Predicted = class_of_interest
    - FP: True class ≠ class_of_interest, Predicted = class_of_interest
    - TN: True class ≠ class_of_interest, Predicted ≠ class_of_interest
    - FN: True class = class_of_interest, Predicted ≠ class_of_interest
    """
    if y_true == class_of_interest and y_pred == class_of_interest:
        return 'TP'
    elif y_true != class_of_interest and y_pred == class_of_interest:
        return 'FP'
    elif y_true != class_of_interest and y_pred != class_of_interest:
        return 'TN'
    else:  # y_true == class_of_interest and y_pred != class_of_interest
        return 'FN'


def stratified_sample_indices(y_true, y_pred, max_per_category=25):
    """
    Get stratified sample indices across classes and prediction categories.
    
    For each class (0-3), sample up to max_per_category from each of:
    - Correct predictions (TP for that class)
    - Incorrect predictions (FN for that class)
    
    Returns:
        List of (index, true_class, pred_class, category) tuples
    """
    np.random.seed(RANDOM_STATE)
    
    samples = []
    
    for cls in range(4):
        # Correct: predicted == true == cls
        correct_mask = (y_true == cls) & (y_pred == cls)
        correct_indices = np.where(correct_mask)[0]
        
        # Incorrect: true == cls but predicted != cls
        incorrect_mask = (y_true == cls) & (y_pred != cls)
        incorrect_indices = np.where(incorrect_mask)[0]
        
        # Sample from correct predictions
        n_correct = min(len(correct_indices), max_per_category)
        if n_correct > 0:
            sampled_correct = np.random.choice(correct_indices, n_correct, replace=False)
            for idx in sampled_correct:
                samples.append((idx, cls, y_pred[idx], 'Correct'))
        
        # Sample from incorrect predictions (FN for this class)
        n_incorrect = min(len(incorrect_indices), max_per_category)
        if n_incorrect > 0:
            sampled_incorrect = np.random.choice(incorrect_indices, n_incorrect, replace=False)
            for idx in sampled_incorrect:
                samples.append((idx, cls, y_pred[idx], 'Incorrect'))
        
        print(f"    Class {cls} ({BIN_LABELS[cls]:20s}): {n_correct} correct, {n_incorrect} incorrect")
    
    return samples


def run_lime_analysis(model, X, y, y_pred, feature_names, max_samples=100):
    """
    Run LIME analysis on sampled instances.
    
    Args:
        model: Trained model with predict_proba
        X: Feature matrix (scaled)
        y: True labels
        y_pred: Predicted labels
        feature_names: List of feature names
        max_samples: Max samples per class
    
    Returns:
        explanations_df: DataFrame with LIME explanations
        importance_df: DataFrame with aggregated feature importance
    """
    print("\nInitializing LIME explainer...")
    
    # Create LIME explainer for tabular data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        class_names=[BIN_LABELS[i] for i in range(4)],
        mode='classification',
        random_state=RANDOM_STATE
    )
    
    # Get stratified samples
    print("\nStratified sampling...")
    samples = stratified_sample_indices(y, y_pred, max_per_category=max_samples // 4)
    print(f"  Total samples to explain: {len(samples)}")
    
    # Generate explanations
    explanations = []
    feature_weights_by_class = defaultdict(list)
    feature_weights_by_correctness = {'Correct': [], 'Incorrect': []}
    
    print("\nGenerating LIME explanations...")
    for i, (idx, true_cls, pred_cls, correctness) in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(samples)}")
        
        try:
            # Get LIME explanation
            exp = explainer.explain_instance(
                X[idx],
                model.predict_proba,
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
                top_labels=4
            )
            
            # Get probabilities
            probs = model.predict_proba(X[idx].reshape(1, -1))[0]
            
            # Extract feature weights for predicted class
            # LIME returns tuples like ('feature_name <= 123.45', weight)
            # We need to extract just the feature name
            raw_weights = exp.as_list(label=pred_cls)
            feature_weights = {}
            for feat_condition, weight in raw_weights:
                # Extract feature name (before any comparison operator)
                feat_name = feat_condition.split(' ')[0].split('<')[0].split('>')[0].split('=')[0].strip()
                # Map back to original feature name
                for orig_feat in feature_names:
                    if orig_feat in feat_condition or feat_name == orig_feat:
                        feature_weights[orig_feat] = weight
                        break
            
            # Store weights for aggregation
            for feat, weight in feature_weights.items():
                feature_weights_by_class[pred_cls].append((feat, weight))
                feature_weights_by_correctness[correctness].append((feat, weight))
            
            # Format top features
            sorted_weights = sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True)
            top_positive = [(f, w) for f, w in sorted_weights if w > 0][:3]
            top_negative = [(f, w) for f, w in sorted_weights if w < 0][:3]
            
            explanations.append({
                'sample_idx': idx,
                'true_class': true_cls,
                'true_label': BIN_LABELS[true_cls],
                'pred_class': pred_cls,
                'pred_label': BIN_LABELS[pred_cls],
                'correctness': correctness,
                'prob_class0': probs[0],
                'prob_class1': probs[1],
                'prob_class2': probs[2],
                'prob_class3': probs[3],
                'confidence': probs[pred_cls],
                'top_positive_features': '; '.join([f"{f}: {w:.4f}" for f, w in top_positive]),
                'top_negative_features': '; '.join([f"{f}: {w:.4f}" for f, w in top_negative]),
                'all_weights': str(feature_weights)
            })
            
        except Exception as e:
            print(f"  Warning: LIME failed for sample {idx}: {str(e)[:50]}")
    
    print(f"\nGenerated {len(explanations)} explanations")
    
    # Check if any explanations were generated
    if len(explanations) == 0:
        print("  WARNING: No explanations generated. Check for errors above.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create explanations DataFrame
    explanations_df = pd.DataFrame(explanations)
    
    # Aggregate feature importance
    print("\nAggregating feature importance...")
    importance_records = []
    
    # Overall importance (by absolute weight)
    all_weights = defaultdict(list)
    for exp in explanations:
        weights = eval(exp['all_weights'])
        for feat, weight in weights.items():
            all_weights[feat].append(abs(weight))
    
    for feat in feature_names:
        if feat in all_weights:
            weights = all_weights[feat]
            importance_records.append({
                'feature': feat,
                'scope': 'Overall',
                'mean_abs_weight': np.mean(weights),
                'std_abs_weight': np.std(weights),
                'median_abs_weight': np.median(weights),
                'n_samples': len(weights)
            })
    
    # Importance by class
    for cls in range(4):
        class_weights = defaultdict(list)
        class_exps = explanations_df[explanations_df['pred_class'] == cls]
        
        for _, exp in class_exps.iterrows():
            weights = eval(exp['all_weights'])
            for feat, weight in weights.items():
                class_weights[feat].append(weight)  # Keep sign for class-specific
        
        for feat in feature_names:
            if feat in class_weights:
                weights = class_weights[feat]
                importance_records.append({
                    'feature': feat,
                    'scope': f'Class_{cls}_{BIN_LABELS[cls]}',
                    'mean_abs_weight': np.mean(np.abs(weights)),
                    'mean_weight': np.mean(weights),  # Signed for direction
                    'std_abs_weight': np.std(np.abs(weights)),
                    'n_samples': len(weights)
                })
    
    # Importance by correctness
    for correctness in ['Correct', 'Incorrect']:
        corr_weights = defaultdict(list)
        corr_exps = explanations_df[explanations_df['correctness'] == correctness]
        
        for _, exp in corr_exps.iterrows():
            weights = eval(exp['all_weights'])
            for feat, weight in weights.items():
                corr_weights[feat].append(abs(weight))
        
        for feat in feature_names:
            if feat in corr_weights:
                weights = corr_weights[feat]
                importance_records.append({
                    'feature': feat,
                    'scope': correctness,
                    'mean_abs_weight': np.mean(weights),
                    'std_abs_weight': np.std(weights),
                    'n_samples': len(weights)
                })
    
    importance_df = pd.DataFrame(importance_records)
    
    return explanations_df, importance_df


def print_summary(explanations_df, importance_df, feature_names):
    """Print summary of LIME analysis."""
    
    print("\n" + "=" * 70)
    print("LIME ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Check for empty DataFrames
    if len(explanations_df) == 0 or len(importance_df) == 0:
        print("\n  No explanations to summarize. Check for errors during LIME generation.")
        return
    
    # Explanations by class and correctness
    print("\nExplanations by class and correctness:")
    summary = explanations_df.groupby(['true_class', 'correctness']).size().unstack(fill_value=0)
    print(summary)
    
    # Overall feature importance
    print("\n" + "-" * 70)
    print("OVERALL FEATURE IMPORTANCE (by mean absolute LIME weight)")
    print("-" * 70)
    
    overall = importance_df[importance_df['scope'] == 'Overall'].sort_values(
        'mean_abs_weight', ascending=False
    )
    
    for _, row in overall.iterrows():
        print(f"  {row['feature']:35s}: {row['mean_abs_weight']:.4f} (±{row['std_abs_weight']:.4f})")
    
    # Feature importance by class
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE BY PREDICTED CLASS")
    print("-" * 70)
    
    for cls in range(4):
        scope = f'Class_{cls}_{BIN_LABELS[cls]}'
        class_imp = importance_df[importance_df['scope'] == scope].sort_values(
            'mean_abs_weight', ascending=False
        )
        
        if len(class_imp) > 0:
            print(f"\n  Class {cls} ({BIN_LABELS[cls]}):")
            for _, row in class_imp.head(5).iterrows():
                direction = "↑" if row.get('mean_weight', 0) > 0 else "↓"
                print(f"    {row['feature']:33s}: {row['mean_abs_weight']:.4f} {direction}")
    
    # Correct vs Incorrect
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE: CORRECT vs INCORRECT PREDICTIONS")
    print("-" * 70)
    
    for correctness in ['Correct', 'Incorrect']:
        corr_imp = importance_df[importance_df['scope'] == correctness].sort_values(
            'mean_abs_weight', ascending=False
        )
        
        if len(corr_imp) > 0:
            print(f"\n  {correctness} predictions:")
            for _, row in corr_imp.head(5).iterrows():
                print(f"    {row['feature']:33s}: {row['mean_abs_weight']:.4f}")
    
    # Average confidence by correctness
    print("\n" + "-" * 70)
    print("MODEL CONFIDENCE")
    print("-" * 70)
    
    for correctness in ['Correct', 'Incorrect']:
        subset = explanations_df[explanations_df['correctness'] == correctness]
        if len(subset) > 0:
            avg_conf = subset['confidence'].mean()
            std_conf = subset['confidence'].std()
            print(f"  {correctness:10s}: avg confidence = {avg_conf:.3f} (±{std_conf:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="LIME Analysis for RQ3.2 Process Features"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_EXPLANATIONS,
        help=f"Maximum samples per class to explain (default: {MAX_EXPLANATIONS})"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("RQ3.2 LIME ANALYSIS: Process Features")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load model artifacts
    print("Loading model artifacts...")
    if not MODEL_FILE.exists():
        print(f"ERROR: {MODEL_FILE} not found.")
        print("Run model_tournament_rq32_process.py first.")
        return
    
    artifacts = joblib.load(MODEL_FILE)
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['selected_features']
    X_scaled = artifacts['X_scaled']
    y = artifacts['y']
    y_pred = artifacts['y_pred']
    
    print(f"  Model: {artifacts['best_model_name']}")
    print(f"  Features: {feature_names}")
    print(f"  Samples: {len(y):,}")
    
    # Class distribution
    print(f"\n  Class distribution:")
    for cls in range(4):
        count = (y == cls).sum()
        correct = ((y == cls) & (y_pred == cls)).sum()
        accuracy = correct / count * 100 if count > 0 else 0
        print(f"    Class {cls} ({BIN_LABELS[cls]:20s}): {count:,} samples, {accuracy:.1f}% correct")
    
    # Run LIME analysis
    explanations_df, importance_df = run_lime_analysis(
        model=model,
        X=X_scaled,
        y=y,
        y_pred=y_pred,
        feature_names=feature_names,
        max_samples=args.max_samples
    )
    
    # Print summary
    print_summary(explanations_df, importance_df, feature_names)
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    explanations_file = OUTPUT_DIR / 'rq32_process_lime_explanations.csv'
    importance_file = OUTPUT_DIR / 'rq32_process_lime_feature_importance.csv'
    
    explanations_df.to_csv(explanations_file, index=False)
    importance_df.to_csv(importance_file, index=False)
    
    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    print(f"  {explanations_file}")
    print(f"  {importance_file}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()