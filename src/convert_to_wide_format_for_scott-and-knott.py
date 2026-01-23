"""
Convert tournament scores from long format to wide format for Scott-Knott ESD analysis.

Usage:
    python src/convert_to_wide_format.py results/rq31_bow_tournament_scores_combined_count.csv

Output:
    results/rq31_bow_wide_auc_pr_combined_count.csv   (primary metric for ranking)
    results/rq31_bow_wide_auc_roc_combined_count.csv  (secondary metric)
"""

import pandas as pd
import sys
from pathlib import Path


def convert_to_wide_format(input_file: str):
    """
    Convert long-format tournament scores to wide format for Scott-Knott ESD.
    
    Args:
        input_file: Path to the long-format CSV (e.g., rq31_bow_tournament_scores_combined_count.csv)
    """
    # Load long-format scores
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df):,} rows from {input_file}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Metrics available: {[c for c in df.columns if c not in ['fold', 'repetition', 'model', 'time_seconds']]}")
    
    # Clean model names (add spaces for readability)
    model_name_map = {
        'LogisticRegression': 'Logistic Regression',
        'NaiveBayes': 'Naive Bayes',
        'RandomForest': 'Random Forest',
        'XGBoost': 'XGBoost',
        'CatBoost': 'CatBoost',
        'KNN': 'KNN',
        'SVM': 'SVM',
        'MLP': 'MLP'
    }
    
    df['model'] = df['model'].map(model_name_map).fillna(df['model'])
    
    # Determine output file prefix from input file name
    input_path = Path(input_file)
    # Extract suffix like "combined_count" from "rq31_bow_tournament_scores_combined_count.csv"
    file_suffix = input_path.stem.replace('rq31_bow_tournament_scores_', '')
    output_dir = input_path.parent
    
    # Create unique index for pivoting (combine repetition and fold)
    if 'repetition' in df.columns:
        df['eval_id'] = df['repetition'].astype(str) + '_' + df['fold'].astype(str)
    else:
        df['eval_id'] = df['fold'].astype(str)
    
    # Primary metric: AUC-PR (for imbalanced data)
    wide_auc_pr = df.pivot(index='eval_id', columns='model', values='auc_pr')
    wide_auc_pr = wide_auc_pr.reset_index(drop=True)
    
    auc_pr_file = output_dir / f"rq31_bow_wide_auc_pr_{file_suffix}.csv"
    wide_auc_pr.to_csv(auc_pr_file, index=False)
    print(f"\nPrimary metric (AUC-PR) saved to: {auc_pr_file}")
    print(f"  Shape: {wide_auc_pr.shape} (rows=folds, cols=models)")
    
    # Secondary metric: AUC-ROC
    wide_auc_roc = df.pivot(index='eval_id', columns='model', values='auc_roc')
    wide_auc_roc = wide_auc_roc.reset_index(drop=True)
    
    auc_roc_file = output_dir / f"rq31_bow_wide_auc_roc_{file_suffix}.csv"
    wide_auc_roc.to_csv(auc_roc_file, index=False)
    print(f"\nSecondary metric (AUC-ROC) saved to: {auc_roc_file}")
    print(f"  Shape: {wide_auc_roc.shape} (rows=folds, cols=models)")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print("\nAUC-PR (Primary - use for Scott-Knott ranking):")
    print("-" * 60)
    summary_pr = wide_auc_pr.mean().sort_values(ascending=False)
    for model, score in summary_pr.items():
        print(f"  {model:20s}: {score:.4f}")
    
    print("\nAUC-ROC (Secondary):")
    print("-" * 60)
    summary_roc = wide_auc_roc.mean().sort_values(ascending=False)
    for model, score in summary_roc.items():
        print(f"  {model:20s}: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. Update R script path:
   model_performance <- read_csv("{auc_pr_file}")

2. Run Scott-Knott ESD:
   Rscript analyze_scott_knott.R

3. Output: visuals/rq3_scott_knott_auc.pdf
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/convert_to_wide_format.py <input_file>")
        print("Example: python src/convert_to_wide_format.py results/rq31_bow_tournament_scores_combined_count.csv")
        sys.exit(1)
    
    convert_to_wide_format(sys.argv[1])