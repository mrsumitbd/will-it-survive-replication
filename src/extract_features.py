import pandas as pd
import numpy as np
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
DATA_DIR = Path('data')
OUTPUT_FILE = DATA_DIR / 'prediction_features_v1.csv'
MAX_WORKERS = 4
GRANULARITIES = ['file', 'hunk', 'line']
REASON_MAPPING = {'Corrective (Bug Fix)': 0, 'Perfective (Refactor)': 1, 'Performance': 1, 'Documentation': 1, 'Adaptive (New Feature)': 1, 'Chore/Maintenance': 2, 'Security': 3, 'Preventive (Testing)': 3, 'Other': 4}
REASON_LABELS = {0: 'Corrective', 1: 'Perfective', 2: 'Adaptive', 3: 'Preventive', 4: 'Other'}
AGENT_NAME_MAPPING = {'Copilot': 'GitHub Copilot', 'GitHub Copilot': 'GitHub Copilot', 'OpenAI_Codex': 'OpenAI Codex', 'OpenAI Codex': 'OpenAI Codex', 'Devin': 'Devin', 'Cursor': 'Cursor', 'Claude_Code': 'Claude Code', 'Claude Code': 'Claude Code', 'Human': 'Human', 'Unknown Agent': 'Unknown Agent'}

def normalize_slug(url: str) -> str:
    s = str(url).lower().strip().replace('.git', '').rstrip('/')
    if 'github.com/' in s:
        return s.split('github.com/')[-1]
    if 'api.github.com/repos/' in s:
        return s.split('api.github.com/repos/')[-1]
    parts = s.split('/')
    if len(parts) >= 2:
        return f'{parts[-2]}/{parts[-1]}'
    return s

def extract_file_extension(unit_id: str) -> str:
    filepath = unit_id.split('::')[0]
    if '.' in filepath:
        ext = '.' + filepath.rsplit('.', 1)[-1].lower()
        return ext
    return '.unknown'

def mine_git_features(slug: str, pr_numbers: list) -> dict:
    results = {}
    temp_dir = tempfile.mkdtemp()
    clone_url = f'https://github.com/{slug}.git'
    try:
        clone_result = subprocess.run(['git', 'clone', '--filter=blob:none', clone_url, temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if clone_result.returncode != 0:
            return {pr: {'pr_additions': 0, 'pr_deletions': 0, 'pr_files_changed': 0} for pr in pr_numbers}
        for pr_num in pr_numbers:
            cmd = ['git', 'log', '--all', '--grep', f'#{pr_num}', '--format=%H', '-n', '1']
            res = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)
            commit_sha = res.stdout.strip()
            if commit_sha:
                cmd_stat = ['git', 'show', '--shortstat', '--oneline', commit_sha]
                res_stat = subprocess.run(cmd_stat, cwd=temp_dir, capture_output=True, text=True)
                output = res_stat.stdout.strip()
                additions, deletions, files = (0, 0, 0)
                if 'changed' in output:
                    try:
                        parts = output.split('\n')[-1].split(',')
                        for p in parts:
                            if 'file' in p:
                                files = int(p.strip().split()[0])
                            if 'insertion' in p:
                                additions = int(p.strip().split()[0])
                            if 'deletion' in p:
                                deletions = int(p.strip().split()[0])
                    except:
                        pass
                results[pr_num] = {'pr_additions': additions, 'pr_deletions': deletions, 'pr_files_changed': files}
            else:
                results[pr_num] = {'pr_additions': 0, 'pr_deletions': 0, 'pr_files_changed': 0}
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    for pr in pr_numbers:
        if pr not in results:
            results[pr] = {'pr_additions': 0, 'pr_deletions': 0, 'pr_files_changed': 0}
    return results

def load_survival_events() -> pd.DataFrame:
    dfs = []
    for gran in GRANULARITIES:
        filepath = DATA_DIR / f'survival_events_{gran}_v1.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f'  Loaded {len(df):,} {gran}-level events')
            dfs.append(df)
        else:
            print(f'  Warning: {filepath} not found')
    if not dfs:
        raise FileNotFoundError('No survival event files found')
    combined = pd.concat(dfs, ignore_index=True)
    print(f'  Total: {len(combined):,} events')
    return combined

def load_death_details() -> pd.DataFrame:
    dfs = []
    for gran in GRANULARITIES:
        filepath = DATA_DIR / f'death_details_{gran}_v1.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f'  Loaded {len(df):,} {gran}-level death details')
            dfs.append(df)
        else:
            print(f'  Warning: {filepath} not found')
    if not dfs:
        print('  No death details found - RQ3.3 labels will be unavailable')
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    print(f'  Total: {len(combined):,} death details')
    return combined

def main():
    print('=' * 70)
    print('FEATURE EXTRACTION (v1)')
    print('=' * 70)
    print()
    print('STEP 1: Loading survival events')
    print('-' * 40)
    df = load_survival_events()
    print('\nSTEP 2: Loading death details')
    print('-' * 40)
    df_death = load_death_details()
    if not df_death.empty:
        df_death_clean = df_death.drop_duplicates(subset=['slug', 'pr_number', 'unit_id', 'is_agent', 'granularity'])
        df = pd.merge(df, df_death_clean[['slug', 'pr_number', 'unit_id', 'is_agent', 'granularity', 'reason']], on=['slug', 'pr_number', 'unit_id', 'is_agent', 'granularity'], how='left')
        df.loc[df['event'] == 0, 'reason'] = 'No Churn'
        df['reason'] = df['reason'].fillna('Other')
        df['reason_class'] = df['reason'].map(REASON_MAPPING)
        df['reason_class'] = df['reason_class'].fillna(4).astype(int)
        df['reason_label'] = df['reason_class'].map(REASON_LABELS)
        df.loc[df['event'] == 0, 'reason_class'] = -1
        df.loc[df['event'] == 0, 'reason_label'] = 'No Churn'
        print(f"  Merged death reasons: {(df['reason'] != 'No Churn').sum():,} deaths with reasons")
    else:
        df['reason'] = 'Unknown'
        df['reason_class'] = -1
        df['reason_label'] = 'Unknown'
    print('\nSTEP 3: Loading repository metadata')
    print('-' * 40)
    repo_file = DATA_DIR / 'aidev_filtered_v1.csv'
    if repo_file.exists():
        df_repo = pd.read_csv(repo_file)
        print(f'  Loaded {len(df_repo)} repository records')
        repo_cols = {'slug': 'slug', 'stars': 'repo_stars', 'forks': 'repo_forks', 'size_kb': 'repo_size_kb', 'open_issues': 'repo_open_issues', 'contributors': 'repo_contributors', 'code_ratio': 'repo_code_ratio', 'language': 'repo_language', 'loc': 'repo_loc'}
        available_cols = {k: v for k, v in repo_cols.items() if k in df_repo.columns}
        df_repo_subset = df_repo[list(available_cols.keys())].rename(columns=available_cols)
        df = pd.merge(df, df_repo_subset, on='slug', how='left')
        print(f'  Merged repository metadata')
    else:
        print(f'  Warning: {repo_file} not found')
    print('\nSTEP 4: Loading PR metadata')
    print('-' * 40)
    agent_file = DATA_DIR / 'all_pull_request.parquet'
    human_file = DATA_DIR / 'human_pull_request.parquet'
    pr_dfs = []
    if agent_file.exists():
        df_agent = pd.read_parquet(agent_file)
        url_col = next((c for c in ['repo_url', 'url', 'repository_url'] if c in df_agent.columns), None)
        if url_col:
            df_agent['slug'] = df_agent[url_col].apply(normalize_slug)
        pr_dfs.append(df_agent)
        print(f'  Loaded {len(df_agent):,} agent PRs')
    if human_file.exists():
        df_human = pd.read_parquet(human_file)
        url_col = next((c for c in ['repo_url', 'url', 'repository_url'] if c in df_human.columns), None)
        if url_col:
            df_human['slug'] = df_human[url_col].apply(normalize_slug)
        pr_dfs.append(df_human)
        print(f'  Loaded {len(df_human):,} human PRs')
    if pr_dfs:
        df_pr = pd.concat(pr_dfs, ignore_index=True)
        if 'title' in df_pr.columns:
            df_pr['pr_title_len'] = df_pr['title'].astype(str).apply(len)
        if 'body' in df_pr.columns:
            df_pr['pr_body_len'] = df_pr['body'].astype(str).apply(len)
        pr_cols = ['slug', 'number']
        if 'pr_title_len' in df_pr.columns:
            pr_cols.append('pr_title_len')
        if 'pr_body_len' in df_pr.columns:
            pr_cols.append('pr_body_len')
        df_pr_subset = df_pr[pr_cols].drop_duplicates(subset=['slug', 'number'])
        df = pd.merge(df, df_pr_subset, left_on=['slug', 'pr_number'], right_on=['slug', 'number'], how='left')
        for col in ['pr_title_len', 'pr_body_len']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        print(f'  Merged PR text features')
    print('\nSTEP 5: Mining git features')
    print('-' * 40)
    repo_groups = df.groupby('slug')['pr_number'].apply(lambda x: x.unique().tolist()).to_dict()
    print(f'  Mining {len(repo_groups)} repositories...')
    git_stats_map = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(mine_git_features, slug, prs): slug for slug, prs in repo_groups.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc='  Git mining'):
            slug = futures[future]
            try:
                result = future.result()
                for pr_num, stats in result.items():
                    git_stats_map[slug, pr_num] = stats
            except Exception:
                pass
    df['pr_additions'] = df.apply(lambda r: git_stats_map.get((r['slug'], r['pr_number']), {}).get('pr_additions', 0), axis=1)
    df['pr_deletions'] = df.apply(lambda r: git_stats_map.get((r['slug'], r['pr_number']), {}).get('pr_deletions', 0), axis=1)
    df['pr_files_changed'] = df.apply(lambda r: git_stats_map.get((r['slug'], r['pr_number']), {}).get('pr_files_changed', 0), axis=1)
    print(f'  Added git features')
    print('\nSTEP 6: Computing derived features')
    print('-' * 40)
    df['file_extension'] = df['unit_id'].apply(extract_file_extension)
    df['agent_name'] = df['agent_name'].map(AGENT_NAME_MAPPING).fillna(df['agent_name'])
    if 'birth_date' in df.columns:
        df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
        df['birth_day_of_week'] = df['birth_date'].dt.dayofweek
        df['birth_month'] = df['birth_date'].dt.month
        df['birth_hour'] = df['birth_date'].dt.hour
    df['pr_churn'] = df['pr_additions'] + df['pr_deletions']
    df['pr_churn_ratio'] = df['pr_additions'] / (df['pr_deletions'] + 1)
    if 'hunk_size' not in df.columns:
        df['hunk_size'] = 0
    df['hunk_size'] = df['hunk_size'].fillna(0).astype(int)
    print(f'  Added derived features')
    print('\nSTEP 7: Final cleanup')
    print('-' * 40)
    output_cols = ['slug', 'pr_number', 'unit_id', 'granularity', 'event', 'duration_days', 'reason', 'reason_class', 'reason_label', 'is_agent', 'agent_name', 'file_extension', 'hunk_size', 'pr_title_len', 'pr_body_len', 'pr_additions', 'pr_deletions', 'pr_files_changed', 'pr_churn', 'pr_churn_ratio', 'repo_stars', 'repo_forks', 'repo_size_kb', 'repo_open_issues', 'repo_contributors', 'repo_code_ratio', 'repo_language', 'repo_loc', 'birth_day_of_week', 'birth_month', 'birth_hour']
    output_cols = [c for c in output_cols if c in df.columns]
    df_final = df[output_cols].copy()
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_final[col] = df_final[col].fillna(0)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'\nTotal observations: {len(df_final):,}')
    print(f'\nBy granularity:')
    for gran in df_final['granularity'].unique():
        count = (df_final['granularity'] == gran).sum()
        print(f'  {gran}: {count:,}')
    print(f'\nBy author type:')
    agent_count = df_final['is_agent'].sum()
    human_count = len(df_final) - agent_count
    print(f'  Agent: {agent_count:,}')
    print(f'  Human: {human_count:,}')
    print(f'\nTarget variable (event):')
    deaths = df_final['event'].sum()
    survived = len(df_final) - deaths
    print(f'  Deaths (event=1): {deaths:,} ({100 * deaths / len(df_final):.1f}%)')
    print(f'  Survived (event=0): {survived:,} ({100 * survived / len(df_final):.1f}%)')
    print(f'\nReason classes (deaths only):')
    deaths_df = df_final[df_final['event'] == 1]
    for cls in sorted(deaths_df['reason_class'].unique()):
        if cls >= 0:
            count = (deaths_df['reason_class'] == cls).sum()
            label = REASON_LABELS.get(cls, 'Unknown')
            print(f'  Class {cls} ({label}): {count:,} ({100 * count / len(deaths_df):.1f}%)')
    print(f'\nFeatures extracted: {len(output_cols)}')
    print(f'Output saved to: {OUTPUT_FILE}')
if __name__ == '__main__':
    main()