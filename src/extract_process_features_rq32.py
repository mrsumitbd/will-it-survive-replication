"""
extract_process_features_rq32.py

Extracts Khatoonabadi-grounded process features for RQ3.2 (When will AI-generated code be modified?).

Features (8 total):
    Git-based (require repo cloning):
        1. project_commit_velocity_3m - Commits in project in 3 months before birth
        2. file_modification_frequency_3m - Times file was modified in 3 months before birth
        3. file_age_days - Days since file was first created
    
    AIDev-based (from parquet):
        4. contributor_prior_prs - Contributor's prior PRs in this project
        5. contributor_acceptance_rate - Merged PRs / Total PRs by contributor
        6. project_open_prs_at_birth - Number of open PRs at birth time
    
    Already available:
        7. birth_day_of_week - Day of week (0=Monday, 6=Sunday)
        8. birth_hour - Hour of day (0-23)

Grounding: Khatoonabadi et al., "Predicting the First Response Latency of 
           Maintainers and Contributors in Pull Requests", IEEE TSE 2024.

Target variable:
    duration_bin: 0 = within 1 day, 1 = 1 day to 1 week, 2 = 1 week to 1 month, 3 = more than 1 month

Usage:
    python src/extract_process_features_rq32.py

Input:
    data/survival_events_file_v1.csv
    data/all_pull_request.parquet

Output:
    data/process_features_rq32.csv
"""

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

# --- CONFIGURATION ---
SURVIVAL_FILE = Path("data/survival_events_file_v1.csv")
PR_PARQUET = Path("data/all_pull_request.parquet")
OUTPUT_FILE = Path("data/process_features_rq32.csv")

# Parallelization
MAX_WORKERS = 4

# Duration bins (Khatoonabadi-style)
# 0: within 1 day (<= 1)
# 1: 1 day to 1 week (1 < x <= 7)
# 2: 1 week to 1 month (7 < x <= 30)
# 3: more than 1 month (> 30)
def duration_to_bin(days: float) -> int:
    """Convert duration in days to bin class."""
    if days <= 1:
        return 0
    elif days <= 7:
        return 1
    elif days <= 30:
        return 2
    else:
        return 3


def parse_date(date_str) -> pd.Timestamp:
    """Parse date string to pandas Timestamp (timezone-naive for comparison)."""
    if pd.isna(date_str):
        return None
    try:
        # Use pandas to parse - handles various formats
        ts = pd.to_datetime(date_str)
        # Convert to timezone-naive for consistent comparison
        if ts.tzinfo is not None:
            ts = ts.tz_convert('UTC').tz_localize(None)
        return ts
    except:
        return None


def extract_git_features(repo_dir: str, filepath: str, birth_date: pd.Timestamp) -> dict:
    """
    Extract Git-based process features for a single file.
    
    Args:
        repo_dir: Path to cloned repository
        filepath: Path to the file within the repo
        birth_date: When the file was born (PR merge date) - pandas Timestamp
    
    Returns:
        Dictionary with git-based features
    """
    features = {
        'project_commit_velocity_3m': 0,
        'file_modification_frequency_3m': 0,
        'file_age_days': 0
    }
    
    if birth_date is None or pd.isna(birth_date):
        return features
    
    three_months_ago = birth_date - pd.Timedelta(days=90)
    
    # Format dates for git commands
    birth_str = birth_date.strftime('%Y-%m-%d')
    three_months_str = three_months_ago.strftime('%Y-%m-%d')
    
    try:
        # 1. Project commit velocity (commits in last 3 months before birth)
        cmd = [
            "git", "rev-list", "--count",
            f"--since={three_months_str}",
            f"--until={birth_str}",
            "HEAD"
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            features['project_commit_velocity_3m'] = int(result.stdout.strip())
    except:
        pass
    
    try:
        # 2. File modification frequency (commits touching this file in last 3 months)
        cmd = [
            "git", "log", "--oneline",
            f"--since={three_months_str}",
            f"--until={birth_str}",
            "--", filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            features['file_modification_frequency_3m'] = len(lines)
    except:
        pass
    
    try:
        # 3. File age (days since file was first created)
        cmd = ["git", "log", "--reverse", "--format=%ci", "--diff-filter=A", "--", filepath]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            first_commit_date = pd.to_datetime(first_line[:19])
            features['file_age_days'] = max(0, (birth_date - first_commit_date).days)
        else:
            # Fallback: try without --diff-filter
            cmd = ["git", "log", "--reverse", "--format=%ci", "--", filepath]
            result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                first_line = result.stdout.strip().splitlines()[0]
                first_commit_date = pd.to_datetime(first_line[:19])
                features['file_age_days'] = max(0, (birth_date - first_commit_date).days)
    except:
        pass
    
    return features


def process_repo(slug: str, repo_files: pd.DataFrame, pr_data: pd.DataFrame) -> list:
    """
    Process all files for a single repository.
    
    Args:
        slug: Repository slug (owner/repo)
        repo_files: DataFrame of files in this repo (from survival_events)
        pr_data: DataFrame of all PRs for this repo (from all_pull_request)
    
    Returns:
        List of feature dictionaries
    """
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = f"https://github.com/{slug}.git"
    
    try:
        # Clone with blobless filter for speed
        clone_result = subprocess.run(
            ["git", "clone", "--filter=blob:none", clone_url, temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300
        )
        
        if clone_result.returncode != 0:
            print(f"  Failed to clone {slug}")
            return results
        
        for _, row in repo_files.iterrows():
            filepath = row['unit_id']
            birth_date = parse_date(row['birth_date'])
            pr_number = row['pr_number']
            duration_days = row['duration_days']
            
            if birth_date is None or pd.isna(birth_date):
                continue
            
            # Extract Git-based features
            git_features = extract_git_features(temp_dir, filepath, birth_date)
            
            # Extract AIDev-based features
            # Get the user who submitted this PR
            pr_row = pr_data[pr_data['number'] == pr_number]
            contributor = pr_row['user'].iloc[0] if len(pr_row) > 0 and 'user' in pr_row.columns else None
            
            # 4. Contributor prior PRs (PRs by same user before this PR)
            contributor_prior_prs = 0
            if contributor and pd.notna(contributor):
                mask = (pr_data['user'] == contributor) & (pr_data['created_at'] < birth_date)
                contributor_prior_prs = mask.sum()
            
            # 5. Contributor acceptance rate (merged / total by this contributor before this PR)
            contributor_acceptance_rate = 0.0
            if contributor and pd.notna(contributor):
                mask = (pr_data['user'] == contributor) & (pr_data['created_at'] < birth_date)
                prior_prs = pr_data[mask]
                if len(prior_prs) > 0:
                    merged_count = prior_prs['merged_at'].notna().sum()
                    contributor_acceptance_rate = merged_count / len(prior_prs)
            
            # 6. Project open PRs at birth (PRs created before birth and not closed yet)
            project_open_prs = 0
            mask_created = pr_data['created_at'] < birth_date
            mask_not_closed = (pr_data['closed_at'].isna()) | (pr_data['closed_at'] > birth_date)
            project_open_prs = (mask_created & mask_not_closed).sum()
            
            # 7 & 8. Temporal features (already available but extracted here for completeness)
            birth_day_of_week = birth_date.weekday()  # 0=Monday, 6=Sunday
            birth_hour = birth_date.hour
            
            # Target variable: duration bin
            duration_bin = duration_to_bin(duration_days)
            
            results.append({
                # Identifiers
                'slug': slug,
                'pr_number': pr_number,
                'unit_id': filepath,
                'is_agent': row['is_agent'],
                'agent_name': row['agent_name'],
                
                # Target
                'duration_days': duration_days,
                'duration_bin': duration_bin,
                
                # Git-based features
                'project_commit_velocity_3m': git_features['project_commit_velocity_3m'],
                'file_modification_frequency_3m': git_features['file_modification_frequency_3m'],
                'file_age_days': git_features['file_age_days'],
                
                # AIDev-based features
                'contributor_prior_prs': contributor_prior_prs,
                'contributor_acceptance_rate': contributor_acceptance_rate,
                'project_open_prs_at_birth': project_open_prs,
                
                # Temporal features
                'birth_day_of_week': birth_day_of_week,
                'birth_hour': birth_hour,
                
                # Metadata for debugging
                'birth_date': row['birth_date'],
                'contributor': contributor
            })
    
    except subprocess.TimeoutExpired:
        print(f"  Timeout cloning {slug}")
    except Exception as e:
        print(f"  Error processing {slug}: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def normalize_slug(url: str) -> str:
    """Normalize repository URL to slug format."""
    if pd.isna(url):
        return None
    s = str(url).lower().strip().replace('.git', '').rstrip('/')
    
    if 'github.com/' in s:
        return s.split('github.com/')[-1]
    if 'api.github.com/repos/' in s:
        return s.split('api.github.com/repos/')[-1]
    
    parts = s.split('/')
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    
    return s


def main():
    print("=" * 70)
    print("RQ3.2 PROCESS FEATURES EXTRACTION")
    print("Khatoonabadi-grounded features for time-to-modification prediction")
    print("=" * 70)
    print()
    
    # Load survival events (file-level, modified only for RQ3.2)
    print("Loading survival events...")
    df_survival = pd.read_csv(SURVIVAL_FILE)
    print(f"  Total events: {len(df_survival):,}")
    
    # Filter to modified files only (event=1) for RQ3.2
    df_modified = df_survival[df_survival['event'] == 1].copy()
    print(f"  Modified files (event=1): {len(df_modified):,}")
    
    # Load PR data
    print("\nLoading PR data...")
    df_prs = pd.read_parquet(PR_PARQUET)
    print(f"  Total PRs: {len(df_prs):,}")
    
    # Normalize slug in PR data
    df_prs['slug'] = df_prs['repo_url'].apply(normalize_slug)
    
    # Parse dates in PR data and make timezone-naive
    print("  Parsing PR dates...")
    for col in ['created_at', 'closed_at', 'merged_at']:
        df_prs[col] = pd.to_datetime(df_prs[col], errors='coerce')
        # Convert to timezone-naive if timezone-aware
        if df_prs[col].dt.tz is not None:
            df_prs[col] = df_prs[col].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Get unique repos
    repos = df_modified['slug'].unique()
    print(f"\nRepositories to process: {len(repos)}")
    
    # Process by repository
    all_results = []
    
    print("\nExtracting process features (this will clone repos)...")
    
    # Sequential processing for now (can parallelize later)
    for slug in tqdm(repos, desc="Processing repos"):
        repo_files = df_modified[df_modified['slug'] == slug]
        repo_prs = df_prs[df_prs['slug'] == slug]
        
        if len(repo_prs) == 0:
            # Try case-insensitive match
            repo_prs = df_prs[df_prs['slug'].str.lower() == slug.lower()]
        
        results = process_repo(slug, repo_files, repo_prs)
        all_results.extend(results)
    
    # Create output DataFrame
    df_features = pd.DataFrame(all_results)
    
    # Save
    df_features.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(df_features):,}")
    
    if len(df_features) > 0:
        print(f"\nDuration bin distribution:")
        bin_labels = {
            0: "Within 1 day",
            1: "1 day to 1 week",
            2: "1 week to 1 month",
            3: "More than 1 month"
        }
        for bin_val in sorted(df_features['duration_bin'].unique()):
            count = (df_features['duration_bin'] == bin_val).sum()
            pct = 100 * count / len(df_features)
            print(f"  Class {bin_val} ({bin_labels.get(bin_val, 'Unknown')}): {count:,} ({pct:.1f}%)")
        
        print(f"\nFeature statistics:")
        feature_cols = [
            'project_commit_velocity_3m',
            'file_modification_frequency_3m', 
            'file_age_days',
            'contributor_prior_prs',
            'contributor_acceptance_rate',
            'project_open_prs_at_birth',
            'birth_day_of_week',
            'birth_hour'
        ]
        for col in feature_cols:
            if col in df_features.columns:
                print(f"  {col}: mean={df_features[col].mean():.2f}, median={df_features[col].median():.2f}, std={df_features[col].std():.2f}")
    
    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()