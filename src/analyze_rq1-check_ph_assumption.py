import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')
DATA_FILE = Path('data/prediction_features_v1.csv')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def check_ph_for_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    print(f"\n{'=' * 60}")
    print(f'Checking PH Assumption: {granularity.upper()}-Level')
    print('=' * 60)
    df_gran = df[df['granularity'] == granularity].copy()
    print(f'Observations: {len(df_gran):,}')
    cox_features = ['duration_days', 'event', 'is_agent', 'pr_churn', 'pr_files_changed', 'repo_stars', 'repo_contributors']
    df_cox = df_gran[cox_features].dropna().copy()
    print(f'After dropna: {len(df_cox):,}')
    for col in ['pr_churn', 'pr_files_changed', 'repo_stars', 'repo_contributors']:
        df_cox[f'{col}_log'] = np.log1p(df_cox[col])
        df_cox = df_cox.drop(columns=[col])
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col='duration_days', event_col='event', show_progress=False)
    print(f'\nModel fitted. Concordance: {cph.concordance_index_:.4f}')
    print(f'\nHazard Ratios:')
    for covar in cph.summary.index:
        hr = cph.summary.loc[covar, 'exp(coef)']
        p = cph.summary.loc[covar, 'p']
        print(f'  {covar}: HR={hr:.4f}, p={p:.2e}')
    print(f'\n--- Schoenfeld Residuals Test ---')
    print('H0: PH assumption holds (covariate effect is constant over time)')
    print('If p < 0.05, the PH assumption is VIOLATED for that covariate\n')
    try:
        ph_results = cph.check_assumptions(df_cox, show_plots=False, p_value_threshold=1.0)
        results_list = []
        for covar in cph.summary.index:
            results_list.append({'granularity': granularity, 'covariate': covar, 'hazard_ratio': cph.summary.loc[covar, 'exp(coef)'], 'hr_p_value': cph.summary.loc[covar, 'p']})
        results_df = pd.DataFrame(results_list)
        print('\nInterpretation:')
        print('- If all p-values > 0.05: PH assumption holds ✓')
        print('- If any p-value < 0.05: PH assumption violated for that covariate')
        return results_df
    except Exception as e:
        print(f'Error checking PH assumption: {e}')
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    print('=' * 60)
    print('RQ1: Proportional Hazards Assumption Check')
    print('=' * 60)
    print(f'\nLoading data from {DATA_FILE}...')
    df = pd.read_csv(DATA_FILE)
    df['event'] = df['event'].astype(int)
    df['duration_days'] = df['duration_days'].astype(float).clip(lower=0.001)
    df['is_agent'] = df['is_agent'].astype(int)
    print(f'Total observations: {len(df):,}')
    all_results = []
    for gran in ['file', 'line']:
        results = check_ph_for_granularity(df, gran)
        if not results.empty:
            all_results.append(results)
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_file = RESULTS_DIR / 'rq1_ph_assumption_test.csv'
        combined_df.to_csv(output_file, index=False)
        print(f'\n\nResults saved to: {output_file}')
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print('\nIf the Schoenfeld test shows p > 0.05 for all covariates:\n  → PH assumption holds\n  → Cox model results are valid\n\nIf p < 0.05 for some covariates:\n  → PH assumption is violated for those covariates\n  → Consider: stratified Cox model, time-varying coefficients,\n    or report as a limitation\n\nFor the main variable of interest (is_agent):\n  → If p > 0.05: The effect of AI authorship is constant over time ✓\n  → If p < 0.05: The effect changes over time (early vs late survival differs)\n')
if __name__ == '__main__':
    main()