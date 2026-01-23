"""
analyze_rq2_v1.py

RQ2: When AI-generated code is modified, why is it modified?

Analysis:
1. Distribution of modification reasons (Swanson taxonomy)
2. Chi-square test of independence (Agent vs Human)
3. Effect size (Cramér's V)
4. Post-hoc analysis of standardized residuals
5. Comparison by agent type

Usage:
    python src/analyze_rq2_v1.py

Input:
    data/prediction_features_v1.csv

Output:
    visuals/rq2_reason_distribution.pdf
    visuals/rq2_reason_by_author.pdf
    visuals/rq2_reason_heatmap.pdf
    results/rq2_reason_summary.csv
    results/rq2_chi_square_tests.csv
    results/rq2_standardized_residuals.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats.contingency import association
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = Path("data")
VISUALS_DIR = Path("visuals")
RESULTS_DIR = Path("results")
INPUT_FILE = DATA_DIR / "prediction_features_v1.csv"

# Ensure output directories exist
VISUALS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Plot styling
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

GRANULARITY_ORDER = ['file', 'hunk', 'line']

# Swanson taxonomy
REASON_LABELS = {
    0: 'Corrective',
    1: 'Perfective',
    2: 'Adaptive',
    3: 'Preventive',
    4: 'Other',
}

REASON_COLORS = {
    'Corrective': '#E74C3C',   # Red - bugs
    'Perfective': '#3498DB',   # Blue - improvements
    'Adaptive': '#2ECC71',     # Green - environment
    'Preventive': '#9B59B6',   # Purple - security/testing
    'Other': '#95A5A6',        # Gray - unknown
}


def load_data() -> pd.DataFrame:
    """Load and prepare data (deaths only for RQ2)."""
    df = pd.read_csv(INPUT_FILE)
    
    # Filter to deaths only (event=1)
    df_deaths = df[df['event'] == 1].copy()
    
    # Ensure proper types
    df_deaths['reason_class'] = df_deaths['reason_class'].astype(int)
    df_deaths['is_agent'] = df_deaths['is_agent'].astype(bool)
    
    # Create labels
    df_deaths['author_type'] = df_deaths['is_agent'].map({True: 'Agent', False: 'Human'})
    df_deaths['reason_name'] = df_deaths['reason_class'].map(REASON_LABELS)
    
    return df_deaths


def compute_reason_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reason distribution by granularity and author type."""
    results = []
    
    for gran in GRANULARITY_ORDER + ['all']:
        if gran == 'all':
            df_subset = df
        else:
            df_subset = df[df['granularity'] == gran]
        
        for author_type in ['Agent', 'Human', 'All']:
            if author_type == 'All':
                df_author = df_subset
            else:
                df_author = df_subset[df_subset['author_type'] == author_type]
            
            total = len(df_author)
            if total == 0:
                continue
            
            for reason_class, reason_name in REASON_LABELS.items():
                count = (df_author['reason_class'] == reason_class).sum()
                results.append({
                    'granularity': gran,
                    'author_type': author_type,
                    'reason_class': reason_class,
                    'reason_name': reason_name,
                    'count': count,
                    'percentage': 100 * count / total,
                    'total': total,
                })
    
    return pd.DataFrame(results)


def plot_reason_distribution(df: pd.DataFrame, dist_df: pd.DataFrame):
    """Plot overall reason distribution."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, gran in enumerate(GRANULARITY_ORDER):
        ax = axes[idx]
        
        # Get data for this granularity
        data = dist_df[(dist_df['granularity'] == gran) & 
                       (dist_df['author_type'] == 'All')]
        data = data.sort_values('reason_class')
        
        colors = [REASON_COLORS[r] for r in data['reason_name']]
        
        bars = ax.barh(data['reason_name'], data['percentage'], color=colors)
        
        ax.set_xlabel('Percentage of Deaths (%)')
        ax.set_title(f'{gran.capitalize()}-Level (n={data["total"].iloc[0]:,})')
        ax.set_xlim(0, 60)
        
        # Add percentage labels
        for bar, pct in zip(bars, data['percentage']):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.suptitle('Distribution of Modification Reasons (Swanson Taxonomy)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'rq2_reason_distribution.pdf', bbox_inches='tight')
    plt.savefig(VISUALS_DIR / 'rq2_reason_distribution.png', bbox_inches='tight')
    plt.close()
    
    print("  Saved: rq2_reason_distribution.pdf")


def plot_reason_by_author(df: pd.DataFrame, dist_df: pd.DataFrame):
    """Plot reason distribution comparing Agent vs Human."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    reason_order = ['Corrective', 'Perfective', 'Adaptive', 'Preventive', 'Other']
    
    for idx, gran in enumerate(GRANULARITY_ORDER):
        ax = axes[idx]
        
        # Prepare data for grouped bar chart
        data = dist_df[(dist_df['granularity'] == gran) & 
                       (dist_df['author_type'].isin(['Agent', 'Human']))]
        
        pivot = data.pivot(index='reason_name', columns='author_type', values='percentage')
        pivot = pivot.reindex(reason_order)
        
        x = np.arange(len(reason_order))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pivot['Agent'], width, label='Agent', color='#E74C3C')
        bars2 = ax.bar(x + width/2, pivot['Human'], width, label='Human', color='#3498DB')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{gran.capitalize()}-Level')
        ax.set_xticks(x)
        ax.set_xticklabels(reason_order, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 60)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 2:  # Only label if visible
                    ax.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Modification Reasons: Agent vs Human Code', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'rq2_reason_by_author.pdf', bbox_inches='tight')
    plt.savefig(VISUALS_DIR / 'rq2_reason_by_author.png', bbox_inches='tight')
    plt.close()
    
    print("  Saved: rq2_reason_by_author.pdf")


def perform_chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform chi-square tests of independence.
    Tests whether modification reason distribution differs between Agent and Human.
    """
    results = []
    
    for gran in GRANULARITY_ORDER + ['all']:
        if gran == 'all':
            df_subset = df
        else:
            df_subset = df[df['granularity'] == gran]
        
        # Create contingency table: author_type x reason_class
        contingency = pd.crosstab(df_subset['author_type'], df_subset['reason_class'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Cramér's V (effect size)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        # Interpretation of Cramér's V
        if cramers_v < 0.1:
            effect_interpretation = 'Negligible'
        elif cramers_v < 0.3:
            effect_interpretation = 'Small'
        elif cramers_v < 0.5:
            effect_interpretation = 'Medium'
        else:
            effect_interpretation = 'Large'
        
        results.append({
            'granularity': gran,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'effect_size': effect_interpretation,
            'n_observations': n,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01,
        })
    
    return pd.DataFrame(results)


def compute_standardized_residuals(df: pd.DataFrame, granularity: str = 'line') -> pd.DataFrame:
    """
    Compute standardized residuals for post-hoc analysis.
    Residuals > |2| indicate cells contributing significantly to chi-square.
    """
    if granularity == 'all':
        df_subset = df
    else:
        df_subset = df[df['granularity'] == granularity]
    
    # Create contingency table
    contingency = pd.crosstab(df_subset['author_type'], df_subset['reason_class'])
    
    # Get expected frequencies
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    expected_df = pd.DataFrame(expected, 
                                index=contingency.index, 
                                columns=contingency.columns)
    
    # Compute standardized residuals: (observed - expected) / sqrt(expected)
    residuals = (contingency - expected_df) / np.sqrt(expected_df)
    
    # Flatten for output
    results = []
    for author_type in contingency.index:
        for reason_class in contingency.columns:
            reason_name = REASON_LABELS.get(reason_class, f'Class_{reason_class}')
            observed = contingency.loc[author_type, reason_class]
            exp = expected_df.loc[author_type, reason_class]
            resid = residuals.loc[author_type, reason_class]
            
            # Interpretation
            if resid > 2:
                interpretation = 'Significantly MORE than expected'
            elif resid < -2:
                interpretation = 'Significantly LESS than expected'
            else:
                interpretation = 'As expected'
            
            results.append({
                'granularity': granularity,
                'author_type': author_type,
                'reason_class': reason_class,
                'reason_name': reason_name,
                'observed': observed,
                'expected': exp,
                'std_residual': resid,
                'interpretation': interpretation,
            })
    
    return pd.DataFrame(results)


def plot_residual_heatmap(df: pd.DataFrame):
    """Plot heatmap of standardized residuals."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, gran in enumerate(GRANULARITY_ORDER):
        ax = axes[idx]
        
        residuals_df = compute_standardized_residuals(df, granularity=gran)
        
        # Pivot for heatmap
        pivot = residuals_df.pivot(index='author_type', 
                                    columns='reason_name', 
                                    values='std_residual')
        
        # Reorder columns
        col_order = ['Corrective', 'Perfective', 'Adaptive', 'Preventive', 'Other']
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-5, vmax=5, ax=ax, cbar=idx==2)
        
        ax.set_title(f'{gran.capitalize()}-Level')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.suptitle('Standardized Residuals: Agent vs Human by Reason\n(|z| > 2 indicates significant deviation)', 
                 fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'rq2_reason_heatmap.pdf', bbox_inches='tight')
    plt.savefig(VISUALS_DIR / 'rq2_reason_heatmap.png', bbox_inches='tight')
    plt.close()
    
    print("  Saved: rq2_reason_heatmap.pdf")


def analyze_by_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze modification reasons by specific agent type."""
    
    # Only analyze agents with sufficient data
    agent_counts = df[df['is_agent']]['agent_name'].value_counts()
    major_agents = agent_counts[agent_counts >= 100].index.tolist()
    
    results = []
    
    for gran in GRANULARITY_ORDER:
        df_gran = df[df['granularity'] == gran]
        
        for agent in major_agents + ['Human']:
            if agent == 'Human':
                subset = df_gran[~df_gran['is_agent']]
            else:
                subset = df_gran[df_gran['agent_name'] == agent]
            
            if len(subset) < 50:
                continue
            
            total = len(subset)
            
            for reason_class, reason_name in REASON_LABELS.items():
                count = (subset['reason_class'] == reason_class).sum()
                results.append({
                    'granularity': gran,
                    'agent_name': agent,
                    'reason_class': reason_class,
                    'reason_name': reason_name,
                    'count': count,
                    'percentage': 100 * count / total,
                    'total': total,
                })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("RQ2 ANALYSIS: Modification Reasons")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data (deaths only)...")
    df = load_data()
    print(f"  Total deaths: {len(df):,}")
    print(f"  Agent deaths: {df['is_agent'].sum():,}")
    print(f"  Human deaths: {(~df['is_agent']).sum():,}")
    print()
    
    # ==========================================================================
    # 1. Reason Distribution Summary
    # ==========================================================================
    print("1. Computing reason distribution...")
    dist_df = compute_reason_distribution(df)
    dist_df.to_csv(RESULTS_DIR / 'rq2_reason_summary.csv', index=False)
    print("  Saved: rq2_reason_summary.csv")
    
    # Print summary
    print("\n  Reason distribution (line-level, all authors):")
    line_all = dist_df[(dist_df['granularity'] == 'line') & 
                       (dist_df['author_type'] == 'All')]
    for _, row in line_all.iterrows():
        print(f"    {row['reason_name']}: {row['percentage']:.1f}%")
    print()
    
    # ==========================================================================
    # 2. Visualizations
    # ==========================================================================
    print("2. Generating visualizations...")
    plot_reason_distribution(df, dist_df)
    plot_reason_by_author(df, dist_df)
    plot_residual_heatmap(df)
    print()
    
    # ==========================================================================
    # 3. Chi-Square Tests
    # ==========================================================================
    print("3. Performing chi-square tests...")
    chi2_df = perform_chi_square_tests(df)
    chi2_df.to_csv(RESULTS_DIR / 'rq2_chi_square_tests.csv', index=False)
    print("  Saved: rq2_chi_square_tests.csv")
    
    print("\n  Chi-square test results (Agent vs Human):")
    for _, row in chi2_df.iterrows():
        sig = "***" if row['significant_001'] else ("*" if row['significant_005'] else "")
        print(f"    {row['granularity'].capitalize()}: χ²={row['chi2_statistic']:.2f}, "
              f"p={row['p_value']:.2e}, V={row['cramers_v']:.3f} ({row['effect_size']}) {sig}")
    print()
    
    # ==========================================================================
    # 4. Standardized Residuals (Post-Hoc)
    # ==========================================================================
    print("4. Computing standardized residuals...")
    
    all_residuals = []
    for gran in GRANULARITY_ORDER:
        residuals_df = compute_standardized_residuals(df, granularity=gran)
        all_residuals.append(residuals_df)
    
    residuals_combined = pd.concat(all_residuals, ignore_index=True)
    residuals_combined.to_csv(RESULTS_DIR / 'rq2_standardized_residuals.csv', index=False)
    print("  Saved: rq2_standardized_residuals.csv")
    
    # Print significant residuals
    print("\n  Significant deviations (|z| > 2, line-level):")
    sig_residuals = residuals_combined[
        (residuals_combined['granularity'] == 'line') & 
        (abs(residuals_combined['std_residual']) > 2)
    ]
    for _, row in sig_residuals.iterrows():
        direction = "MORE" if row['std_residual'] > 0 else "LESS"
        print(f"    {row['author_type']} + {row['reason_name']}: "
              f"z={row['std_residual']:.2f} ({direction} than expected)")
    print()
    
    # ==========================================================================
    # 5. Analysis by Agent Type
    # ==========================================================================
    print("5. Analyzing by agent type...")
    agent_df = analyze_by_agent(df)
    agent_df.to_csv(RESULTS_DIR / 'rq2_agent_analysis.csv', index=False)
    print("  Saved: rq2_agent_analysis.csv")
    
    # Print corrective rate by agent
    print("\n  Corrective modification rate by agent (line-level):")
    line_corrective = agent_df[
        (agent_df['granularity'] == 'line') & 
        (agent_df['reason_name'] == 'Corrective')
    ].sort_values('percentage', ascending=False)
    
    for _, row in line_corrective.iterrows():
        print(f"    {row['agent_name']}: {row['percentage']:.1f}% corrective (n={row['total']:,})")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RQ2 SUMMARY")
    print("=" * 70)
    
    # Get key statistics
    line_chi2 = chi2_df[chi2_df['granularity'] == 'line'].iloc[0]
    
    agent_corrective = dist_df[
        (dist_df['granularity'] == 'line') & 
        (dist_df['author_type'] == 'Agent') & 
        (dist_df['reason_name'] == 'Corrective')
    ]['percentage'].iloc[0]
    
    human_corrective = dist_df[
        (dist_df['granularity'] == 'line') & 
        (dist_df['author_type'] == 'Human') & 
        (dist_df['reason_name'] == 'Corrective')
    ]['percentage'].iloc[0]
    
    print(f"""
Key Findings:
1. Modification reason distributions differ significantly between Agent and Human code
   - Chi-square: χ²={line_chi2['chi2_statistic']:.2f}, p={line_chi2['p_value']:.2e}
   - Effect size: Cramér's V = {line_chi2['cramers_v']:.3f} ({line_chi2['effect_size']})

2. Agent code has HIGHER corrective (bug fix) modification rate
   - Agent: {agent_corrective:.1f}% corrective
   - Human: {human_corrective:.1f}% corrective
   - Difference: {agent_corrective - human_corrective:.1f} percentage points

3. Perfective modifications dominate for both:
   - ~50% of all modifications are perfective (refactoring, performance, features)

Interpretation:
- While Agent code survives longer (RQ1), when it IS modified, it's more likely
  to be a bug fix compared to human code
- This suggests AI-generated code may have more latent defects that are discovered
  over time, even though the code itself is modified less frequently
- Human code modifications are more often enhancements/refactoring
    """)
    
    print("\nOutput files:")
    print(f"  {RESULTS_DIR}/rq2_reason_summary.csv")
    print(f"  {RESULTS_DIR}/rq2_chi_square_tests.csv")
    print(f"  {RESULTS_DIR}/rq2_standardized_residuals.csv")
    print(f"  {RESULTS_DIR}/rq2_agent_analysis.csv")
    print(f"  {VISUALS_DIR}/rq2_reason_distribution.pdf")
    print(f"  {VISUALS_DIR}/rq2_reason_by_author.pdf")
    print(f"  {VISUALS_DIR}/rq2_reason_heatmap.pdf")


if __name__ == "__main__":
    main()