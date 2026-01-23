import pandas as pd
import numpy as np
import subprocess
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
SURVIVAL_FILE = Path('data/survival_events_file_v1.csv')
PR_PARQUET = Path('data/all_pull_request.parquet')
OUTPUT_FILE = Path('data/process_features_rq32.csv')
MAX_WORKERS = 4

def duration_to_bin(days: float) -> int:
    if days <= 1:
        return 0
    elif days <= 7:
        return 1
    elif days <= 30:
        return 2
    else:
        return 3

def parse_date(date_str) -> pd.Timestamp:
    if pd.isna(date_str):
        return None
    try:
        ts = pd.to_datetime(date_str)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts
    except:
        return None

def extract_git_features(repo_dir: str, filepath: str, birth_date: pd.Timestamp) -> dict:
    features = {'project_commit_velocity_3m': 0, 'file_modification_frequency_3m': 0, 'file_age_days': 0}
    if birth_date is None or pd.isna(birth_date):
        return features
    three_months_ago = birth_date - pd.Timedelta(days=90)
    birth_str = birth_date.strftime('%Y-%m-%d')
    three_months_str = three_months_ago.strftime('%Y-%m-%d')
    try:
        cmd = ['git', 'rev-list', '--count', f'--since={three_months_str}', f'--until={birth_str}', 'HEAD']
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            features['project_commit_velocity_3m'] = int(result.stdout.strip())
    except:
        pass
    try:
        cmd = ['git', 'log', '--oneline', f'--since={three_months_str}', f'--until={birth_str}', '--', filepath]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            features['file_modification_frequency_3m'] = len(lines)
    except:
        pass
    try:
        cmd = ['git', 'log', '--reverse', '--format=%ci', '--diff-filter=A', '--', filepath]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            first_commit_date = pd.to_datetime(first_line[:19])
            features['file_age_days'] = max(0, (birth_date - first_commit_date).days)
        else:
            cmd = ['git', 'log', '--reverse', '--format=%ci', '--', filepath]
            result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                first_line = result.stdout.strip().splitlines()[0]
                first_commit_date = pd.to_datetime(first_line[:19])
                features['file_age_days'] = max(0, (birth_date - first_commit_date).days)
    except:
        pass
    return features

def process_repo(slug: str, repo_files: pd.DataFrame, pr_data: pd.DataFrame) -> list:
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = f'https://github.com/{slug}.git'
    try:
        clone_result = subprocess.run(['git', 'clone', '--filter=blob:none', clone_url, temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if clone_result.returncode != 0:
            print(f'  Failed to clone {slug}')
            return results
        for _, row in repo_files.iterrows():
            filepath = row['unit_id']
            birth_date = parse_date(row['birth_date'])
            pr_number = row['pr_number']
            duration_days = row['duration_days']
            if birth_date is None or pd.isna(birth_date):
                continue
            git_features = extract_git_features(temp_dir, filepath, birth_date)
            pr_row = pr_data[pr_data['number'] == pr_number]
            contributor = pr_row['user'].iloc[0] if len(pr_row) > 0 and 'user' in pr_row.columns else None
            contributor_prior_prs = 0
            if contributor and pd.notna(contributor):
                mask = (pr_data['user'] == contributor) & (pr_data['created_at'] < birth_date)
                contributor_prior_prs = mask.sum()
            contributor_acceptance_rate = 0.0
            if contributor and pd.notna(contributor):
                mask = (pr_data['user'] == contributor) & (pr_data['created_at'] < birth_date)
                prior_prs = pr_data[mask]
                if len(prior_prs) > 0:
                    merged_count = prior_prs['merged_at'].notna().sum()
                    contributor_acceptance_rate = merged_count / len(prior_prs)
            project_open_prs = 0
            mask_created = pr_data['created_at'] < birth_date
            mask_not_closed = pr_data['closed_at'].isna() | (pr_data['closed_at'] > birth_date)
            project_open_prs = (mask_created & mask_not_closed).sum()
            birth_day_of_week = birth_date.weekday()
            birth_hour = birth_date.hour
            duration_bin = duration_to_bin(duration_days)
            results.append({'slug': slug, 'pr_number': pr_number, 'unit_id': filepath, 'is_agent': row['is_agent'], 'agent_name': row['agent_name'], 'duration_days': duration_days, 'duration_bin': duration_bin, 'project_commit_velocity_3m': git_features['project_commit_velocity_3m'], 'file_modification_frequency_3m': git_features['file_modification_frequency_3m'], 'file_age_days': git_features['file_age_days'], 'contributor_prior_prs': contributor_prior_prs, 'contributor_acceptance_rate': contributor_acceptance_rate, 'project_open_prs_at_birth': project_open_prs, 'birth_day_of_week': birth_day_of_week, 'birth_hour': birth_hour, 'birth_date': row['birth_date'], 'contributor': contributor})
    except subprocess.TimeoutExpired:
        print(f'  Timeout cloning {slug}')
    except Exception as e:
        print(f'  Error processing {slug}: {e}')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results

def normalize_slug(url: str) -> str:
    if pd.isna(url):
        return None
    s = str(url).lower().strip().replace('.git', '').rstrip('/')
    if 'github.com/' in s:
        return s.split('github.com/')[-1]
    if 'api.github.com/repos/' in s:
        return s.split('api.github.com/repos/')[-1]
    parts = s.split('/')
    if len(parts) >= 2:
        return f'{parts[-2]}/{parts[-1]}'
    return s

def main():
    print('=' * 70)
    print('RQ3.2 PROCESS FEATURES EXTRACTION')
    print('Khatoonabadi-grounded features for time-to-modification prediction')
    print('=' * 70)
    print()
    print('Loading survival events...')
    df_survival = pd.read_csv(SURVIVAL_FILE)
    print(f'  Total events: {len(df_survival):,}')
    df_modified = df_survival[df_survival['event'] == 1].copy()
    print(f'  Modified files (event=1): {len(df_modified):,}')
    print('\nLoading PR data...')
    df_prs = pd.read_parquet(PR_PARQUET)
    print(f'  Total PRs: {len(df_prs):,}')
    df_prs['slug'] = df_prs['repo_url'].apply(normalize_slug)
    print('  Parsing PR dates...')
    for col in ['created_at', 'closed_at', 'merged_at']:
        df_prs[col] = pd.to_datetime(df_prs[col], errors='coerce')
        if df_prs[col].dt.tz is not None:
            df_prs[col] = df_prs[col].dt.tz_convert('UTC').dt.tz_localize(None)
    repos = df_modified['slug'].unique()
    print(f'\nRepositories to process: {len(repos)}')
    all_results = []
    print('\nExtracting process features (this will clone repos)...')
    for slug in tqdm(repos, desc='Processing repos'):
        repo_files = df_modified[df_modified['slug'] == slug]
        repo_prs = df_prs[df_prs['slug'] == slug]
        if len(repo_prs) == 0:
            repo_prs = df_prs[df_prs['slug'].str.lower() == slug.lower()]
        results = process_repo(slug, repo_files, repo_prs)
        all_results.extend(results)
    df_features = pd.DataFrame(all_results)
    df_features.to_csv(OUTPUT_FILE, index=False)
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Total samples: {len(df_features):,}')
    if len(df_features) > 0:
        print(f'\nDuration bin distribution:')
        bin_labels = {0: 'Within 1 day', 1: '1 day to 1 week', 2: '1 week to 1 month', 3: 'More than 1 month'}
        for bin_val in sorted(df_features['duration_bin'].unique()):
            count = (df_features['duration_bin'] == bin_val).sum()
            pct = 100 * count / len(df_features)
            print(f"  Class {bin_val} ({bin_labels.get(bin_val, 'Unknown')}): {count:,} ({pct:.1f}%)")
        print(f'\nFeature statistics:')
        feature_cols = ['project_commit_velocity_3m', 'file_modification_frequency_3m', 'file_age_days', 'contributor_prior_prs', 'contributor_acceptance_rate', 'project_open_prs_at_birth', 'birth_day_of_week', 'birth_hour']
        for col in feature_cols:
            if col in df_features.columns:
                print(f'  {col}: mean={df_features[col].mean():.2f}, median={df_features[col].median():.2f}, std={df_features[col].std():.2f}')
    print(f'\nOutput saved to: {OUTPUT_FILE}')
if __name__ == '__main__':
    main()