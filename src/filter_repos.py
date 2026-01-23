"""
filter_repos_v1.py

Filters repositories to identify "engineered" software projects following
Xiao et al.'s methodology.

Filtering criteria (Xiao et al.):
1. License: Exclude non-software licenses (CC, OFL, Unlicense, None, Other)
2. Repository state: Exclude archived repos, require >= 2 contributors, require releases
3. Statistical: Exclude Q1 (bottom 25%) per language for PRs, issues, size
4. Code ratio: Filter outliers beyond 97% CI per language

Key corrections from v0:
- Explicit 500+ star filter (matching human PR baseline)
- Uses repository.parquet for star counts
- Fixed Q1 filtering logic (per-metric, not all-metrics-AND)
- Added LOC estimation via cloc (Xiao uses LOC, not disk size)
- Observation cutoff date: Dec 31, 2025

Usage:
    python src/filter_repos_v1.py

Input:
    data/all_pull_request.parquet
    data/human_pull_request.parquet
    data/all_repository.parquet

Output:
    data/aidev_filtered_v1.csv
    data/repo_metadata_v1.csv (cached metadata)
    data/code_ratio_v1.csv (cached code ratios)
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
import subprocess
import tempfile
import time
import requests
import multiprocessing
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm

# --- CONFIGURATION ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Input files
AGENT_FILE = Path("data/all_pull_request.parquet")
HUMAN_FILE = Path("data/human_pull_request.parquet")
REPO_FILE = Path("data/all_repository.parquet")

# Output files (v1)
OUTPUT_FILE = Path("data/aidev_filtered_v1.csv")
META_CACHE_FILE = Path("data/repo_metadata_v1.csv")
RATIO_CACHE_FILE = Path("data/code_ratio_v1.csv")

# Observation cutoff
OBSERVATION_CUTOFF = datetime(2025, 12, 31)

# Minimum star count (to match human PR baseline which is from 500+ star repos)
MIN_STARS = 500

# Xiao et al. License Blacklist (Non-Software Licenses)
BAD_LICENSES = [
    'creative commons',
    'cc0',
    'cc-by',
    'attribution 4.0',
    'share alike',
    'open font',
    'ofl',
    'unlicense',
    'other',
    'none',
    'no license',
]

# Source code extensions for analysis
SOURCE_EXTENSIONS = (
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.c', '.cpp', '.cc', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt'
)


class GitHubFetcher:
    """Fetches repository metadata from GitHub API with rate limit handling."""
    
    def __init__(self, token: str = None):
        self.headers = {"Authorization": f"token {token}"} if token else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def fetch_repo_metadata(self, slug: str) -> dict:
        """
        Fetch metadata for a single repository.
        
        Args:
            slug: Repository slug (owner/repo)
            
        Returns:
            Dictionary with repo metadata or None if failed
        """
        url = f"https://api.github.com/repos/{slug}"
        
        try:
            resp = self.session.get(url, timeout=30)
            
            # Handle rate limiting
            if resp.status_code == 403:
                reset_time = int(resp.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 60)
                print(f"  [Rate Limit] Sleeping {wait_time:.0f}s...")
                time.sleep(wait_time)
                return self.fetch_repo_metadata(slug)
            
            if resp.status_code == 404:
                return None  # Repo doesn't exist or is private
            
            if resp.status_code != 200:
                print(f"  [Warning] {slug}: HTTP {resp.status_code}")
                return None
            
            data = resp.json()
            
            # Check for releases (lightweight: just check if tags exist)
            tags_url = f"https://api.github.com/repos/{slug}/tags?per_page=1"
            tags_resp = self.session.get(tags_url, timeout=10)
            has_release = (
                tags_resp.status_code == 200 and 
                len(tags_resp.json()) > 0
            )
            
            return {
                "slug": slug,
                "url": data.get("html_url"),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "license": (
                    data.get("license", {}).get("name", "None") 
                    if data.get("license") else "None"
                ),
                "is_archived": data.get("archived", False),
                "is_fork": data.get("fork", False),
                "size_kb": data.get("size", 0),
                "open_issues": data.get("open_issues_count", 0),
                "language": data.get("language") or "Unknown",
                "has_release": has_release,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
            }
            
        except requests.exceptions.Timeout:
            print(f"  [Timeout] {slug}")
            return None
        except Exception as e:
            print(f"  [Error] {slug}: {e}")
            return None


def analyze_code_ratio(repo_url: str) -> dict:
    """
    Clone repository and compute code ratio using cloc.
    
    Code ratio = lines_of_code / (lines_of_code + lines_of_comments)
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Dictionary with url, code_ratio, loc, comments, or None if failed
    """
    if not repo_url:
        return None
    
    temp_dir = tempfile.mkdtemp()
    result = None
    
    try:
        # Shallow clone (just latest commit, no blobs until needed)
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", repo_url, temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120
        )
        
        if clone_result.returncode != 0:
            return None
        
        # Run cloc
        cloc_result = subprocess.run(
            ["cloc", "--json", "--quiet", temp_dir],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if cloc_result.returncode == 0:
            data = json.loads(cloc_result.stdout)
            stats = data.get('SUM', {})
            code = stats.get('code', 0)
            comment = stats.get('comment', 0)
            
            # Xiao's formula: Code / (Code + Comment)
            denominator = code + comment
            ratio = code / denominator if denominator > 0 else 0.0
            
            result = {
                'url': repo_url,
                'code_ratio': ratio,
                'loc': code,
                'comments': comment,
                'total_lines': code + comment + stats.get('blank', 0)
            }
            
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return result


def normalize_slug(url: str) -> str:
    """
    Normalize repository URL to slug format (owner/repo).
    
    Handles various URL formats from the dataset.
    """
    s = str(url).lower().strip().replace('.git', '').rstrip('/')
    
    if 'github.com/' in s:
        return s.split('github.com/')[-1]
    if 'api.github.com/repos/' in s:
        return s.split('api.github.com/repos/')[-1]
    
    # Fallback: take last two path components
    parts = s.split('/')
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    
    return s


def is_bad_license(license_name: str) -> bool:
    """Check if license is in the blacklist (non-software licenses)."""
    lic = str(license_name).lower().strip()
    
    if lic in ('none', 'null', 'nan', ''):
        return True
    
    return any(bad in lic for bad in BAD_LICENSES)


def main():
    print("=" * 70)
    print("XIAO ET AL. FILTERING PIPELINE (v1)")
    print("=" * 70)
    print(f"Observation cutoff: {OBSERVATION_CUTOFF.date()}")
    print(f"Minimum stars: {MIN_STARS}")
    print()
    
    # =========================================================================
    # STEP 1: LOAD DATA AND IDENTIFY COHORT
    # =========================================================================
    print("STEP 1: Identifying Cohort")
    print("-" * 40)
    
    # Load PR data
    print("  Loading PR data...")
    df_agent = pd.read_parquet(AGENT_FILE)
    df_human = pd.read_parquet(HUMAN_FILE)
    
    print(f"  Agent PRs: {len(df_agent):,}")
    print(f"  Human PRs: {len(df_human):,}")
    
    # Find URL column
    url_col_a = next(
        (col for col in ['repo_url', 'url', 'repository_url'] if col in df_agent.columns),
        None
    )
    url_col_h = next(
        (col for col in ['repo_url', 'url', 'repository_url'] if col in df_human.columns),
        None
    )
    
    if not url_col_a or not url_col_h:
        print(f"  ERROR: Could not find URL column")
        print(f"  Agent columns: {df_agent.columns.tolist()}")
        print(f"  Human columns: {df_human.columns.tolist()}")
        sys.exit(1)
    
    # Normalize slugs
    df_agent['slug'] = df_agent[url_col_a].apply(normalize_slug)
    df_human['slug'] = df_human[url_col_h].apply(normalize_slug)
    
    # Get unique repos
    agent_repos = set(df_agent['slug'].unique())
    human_repos = set(df_human['slug'].unique())
    
    print(f"  Unique agent repos: {len(agent_repos):,}")
    print(f"  Unique human repos: {len(human_repos):,}")
    
    # Intersection: repos with BOTH agent and human PRs
    # This is required for valid comparison in RQ1
    cohort_slugs = agent_repos.intersection(human_repos)
    print(f"  Intersection (both agent & human PRs): {len(cohort_slugs):,}")
    
    # =========================================================================
    # STEP 2: FETCH/LOAD REPOSITORY METADATA
    # =========================================================================
    print("\nSTEP 2: Repository Metadata")
    print("-" * 40)
    
    # Try to load from AIDev's repository data first
    if REPO_FILE.exists():
        print("  Loading repository metadata from AIDev...")
        df_repo = pd.read_parquet(REPO_FILE)
        
        # Normalize slugs in repo data
        repo_url_col = next(
            (col for col in ['url', 'html_url', 'repo_url'] if col in df_repo.columns),
            None
        )
        if repo_url_col:
            df_repo['slug'] = df_repo[repo_url_col].apply(normalize_slug)
    else:
        df_repo = pd.DataFrame()
    
    # Load cached metadata
    if META_CACHE_FILE.exists():
        meta_df = pd.read_csv(META_CACHE_FILE)
        print(f"  Loaded {len(meta_df)} cached metadata entries")
    else:
        meta_df = pd.DataFrame()
    
    # Identify repos needing metadata fetch
    known_slugs = set(meta_df['slug']) if not meta_df.empty else set()
    needed_slugs = [s for s in cohort_slugs if s not in known_slugs]
    
    if needed_slugs:
        print(f"  Fetching metadata for {len(needed_slugs)} repos...")
        
        if not GITHUB_TOKEN:
            print("  WARNING: No GITHUB_TOKEN found. API rate limits will apply.")
        
        fetcher = GitHubFetcher(GITHUB_TOKEN)
        new_rows = []
        
        for i, slug in enumerate(tqdm(needed_slugs, desc="  Fetching")):
            result = fetcher.fetch_repo_metadata(slug)
            if result:
                new_rows.append(result)
            
            # Save progress periodically
            if (i + 1) % 100 == 0 and new_rows:
                temp_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)
                temp_df.to_csv(META_CACHE_FILE, index=False)
        
        if new_rows:
            meta_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)
            meta_df.to_csv(META_CACHE_FILE, index=False)
            print(f"  Cached {len(new_rows)} new metadata entries")
    
    # Filter to cohort
    meta_df = meta_df[meta_df['slug'].isin(cohort_slugs)].copy()
    print(f"  Metadata available for: {len(meta_df)} repos")
    
    # =========================================================================
    # STEP 3: APPLY STAR FILTER
    # =========================================================================
    print(f"\nSTEP 3: Star Filter (>= {MIN_STARS})")
    print("-" * 40)
    
    before_stars = len(meta_df)
    meta_df = meta_df[meta_df['stars'] >= MIN_STARS].copy()
    print(f"  Removed: {before_stars - len(meta_df)} repos below {MIN_STARS} stars")
    print(f"  Remaining: {len(meta_df)}")
    
    # =========================================================================
    # STEP 4: APPLY LICENSE FILTER
    # =========================================================================
    print("\nSTEP 4: License Filter")
    print("-" * 40)
    
    before_license = len(meta_df)
    meta_df = meta_df[~meta_df['license'].apply(is_bad_license)].copy()
    print(f"  Removed: {before_license - len(meta_df)} repos with non-software licenses")
    print(f"  Remaining: {len(meta_df)}")
    
    # =========================================================================
    # STEP 5: APPLY ARCHIVE/RELEASE FILTER
    # =========================================================================
    print("\nSTEP 5: Archive & Release Filter")
    print("-" * 40)
    
    before_archive = len(meta_df)
    meta_df = meta_df[meta_df['is_archived'] == False].copy()
    print(f"  Removed: {before_archive - len(meta_df)} archived repos")
    
    before_release = len(meta_df)
    meta_df = meta_df[meta_df['has_release'] == True].copy()
    print(f"  Removed: {before_release - len(meta_df)} repos without releases")
    print(f"  Remaining: {len(meta_df)}")
    
    # =========================================================================
    # STEP 6: CONTRIBUTOR FILTER
    # =========================================================================
    print("\nSTEP 6: Contributor Filter (>= 2)")
    print("-" * 40)
    
    # Count unique contributors per repo from PR data
    users_agent = df_agent[df_agent['slug'].isin(meta_df['slug'])][['slug', 'user']]
    users_human = df_human[df_human['slug'].isin(meta_df['slug'])][['slug', 'user']]
    all_users = pd.concat([users_agent, users_human])
    
    contrib_counts = (
        all_users.groupby('slug')['user']
        .nunique()
        .reset_index(name='contributors')
    )
    
    meta_df = meta_df.merge(contrib_counts, on='slug', how='left')
    meta_df['contributors'] = meta_df['contributors'].fillna(0).astype(int)
    
    before_contrib = len(meta_df)
    meta_df = meta_df[meta_df['contributors'] >= 2].copy()
    print(f"  Removed: {before_contrib - len(meta_df)} repos with < 2 contributors")
    print(f"  Remaining: {len(meta_df)}")
    
    # =========================================================================
    # STEP 7: PR COUNT ENRICHMENT
    # =========================================================================
    print("\nSTEP 7: PR Count Enrichment")
    print("-" * 40)
    
    # Count PRs per repo
    agent_pr_counts = (
        df_agent[df_agent['slug'].isin(meta_df['slug'])]
        .groupby('slug').size()
        .reset_index(name='agent_pr_count')
    )
    human_pr_counts = (
        df_human[df_human['slug'].isin(meta_df['slug'])]
        .groupby('slug').size()
        .reset_index(name='human_pr_count')
    )
    
    meta_df = meta_df.merge(agent_pr_counts, on='slug', how='left')
    meta_df = meta_df.merge(human_pr_counts, on='slug', how='left')
    meta_df['agent_pr_count'] = meta_df['agent_pr_count'].fillna(0).astype(int)
    meta_df['human_pr_count'] = meta_df['human_pr_count'].fillna(0).astype(int)
    meta_df['total_pr_count'] = meta_df['agent_pr_count'] + meta_df['human_pr_count']
    
    print(f"  Total PRs in cohort: {meta_df['total_pr_count'].sum():,}")
    print(f"    Agent PRs: {meta_df['agent_pr_count'].sum():,}")
    print(f"    Human PRs: {meta_df['human_pr_count'].sum():,}")
    
    # =========================================================================
    # STEP 8: Q1 STATISTICAL FILTER (Per Language)
    # =========================================================================
    print("\nSTEP 8: Q1 Statistical Filter (Per Language)")
    print("-" * 40)
    
    # Metrics for Q1 filtering: total_pr_count, open_issues, size_kb
    # Xiao: "excluded repositories in the first quartile (Q1) for each metric"
    # Interpretation: Remove repos in Q1 for ANY metric (union), not ALL metrics (intersection)
    
    filtered_dfs = []
    
    for lang in meta_df['language'].unique():
        if pd.isna(lang) or lang == 'Unknown':
            continue
        
        lang_df = meta_df[meta_df['language'] == lang].copy()
        
        if len(lang_df) < 4:
            # Too few repos to compute meaningful quartiles
            filtered_dfs.append(lang_df)
            continue
        
        # Compute Q1 thresholds
        q1_pr = lang_df['total_pr_count'].quantile(0.25)
        q1_issues = lang_df['open_issues'].quantile(0.25)
        q1_size = lang_df['size_kb'].quantile(0.25)
        
        # Keep repos ABOVE Q1 for ALL metrics
        # (This is the strict interpretation - remove bottom 25% on each dimension)
        lang_filtered = lang_df[
            (lang_df['total_pr_count'] > q1_pr) &
            (lang_df['open_issues'] > q1_issues) &
            (lang_df['size_kb'] > q1_size)
        ]
        
        removed = len(lang_df) - len(lang_filtered)
        if removed > 0:
            print(f"  {lang}: {len(lang_df)} -> {len(lang_filtered)} (-{removed})")
        
        filtered_dfs.append(lang_filtered)
    
    if not filtered_dfs:
        print("  ERROR: No repos survived Q1 filtering")
        sys.exit(1)
    
    meta_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"  Remaining after Q1 filter: {len(meta_df)}")
    
    # =========================================================================
    # STEP 9: CODE RATIO FILTER (97% CI Per Language)
    # =========================================================================
    print("\nSTEP 9: Code Ratio Filter (97% CI)")
    print("-" * 40)
    
    # Load cached ratios
    if RATIO_CACHE_FILE.exists():
        ratio_df = pd.read_csv(RATIO_CACHE_FILE)
        print(f"  Loaded {len(ratio_df)} cached code ratios")
    else:
        ratio_df = pd.DataFrame()
    
    # Identify repos needing ratio computation
    known_urls = set(ratio_df['url']) if not ratio_df.empty else set()
    to_analyze = meta_df[~meta_df['url'].isin(known_urls)].copy()
    
    if not to_analyze.empty:
        print(f"  Computing code ratios for {len(to_analyze)} repos...")
        
        # Use multiprocessing for cloning
        n_workers = min(multiprocessing.cpu_count(), 8)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(analyze_code_ratio, row['url']): row['url']
                for _, row in to_analyze.iterrows()
            }
            
            new_ratios = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="  Analyzing"):
                result = future.result()
                if result:
                    new_ratios.append(result)
            
            if new_ratios:
                ratio_df = pd.concat(
                    [ratio_df, pd.DataFrame(new_ratios)],
                    ignore_index=True
                )
                ratio_df.to_csv(RATIO_CACHE_FILE, index=False)
                print(f"  Cached {len(new_ratios)} new code ratios")
    
    # Merge ratios
    meta_df = meta_df.merge(
        ratio_df[['url', 'code_ratio', 'loc']],
        on='url',
        how='inner'
    )
    print(f"  Repos with code ratio data: {len(meta_df)}")
    
    # Apply 97% CI filter per language
    filtered_dfs = []
    
    for lang in meta_df['language'].unique():
        lang_df = meta_df[meta_df['language'] == lang].copy()
        
        if len(lang_df) < 3:
            filtered_dfs.append(lang_df)
            continue
        
        ratios = lang_df['code_ratio']
        mean = ratios.mean()
        std = ratios.std()
        
        # 97% CI: approximately ±2.17 standard deviations
        z = 2.17
        lower = mean - (z * std)
        upper = mean + (z * std)
        
        lang_filtered = lang_df[
            (lang_df['code_ratio'] >= lower) &
            (lang_df['code_ratio'] <= upper)
        ]
        
        removed = len(lang_df) - len(lang_filtered)
        if removed > 0:
            print(f"  {lang}: {len(lang_df)} -> {len(lang_filtered)} (-{removed}, CI: [{lower:.3f}, {upper:.3f}])")
        
        filtered_dfs.append(lang_filtered)
    
    meta_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"  Remaining after code ratio filter: {len(meta_df)}")
    
    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print("\n" + "=" * 70)
    print("FILTERING COMPLETE")
    print("=" * 70)
    
    # Clean up columns
    output_cols = [
        'slug', 'url', 'stars', 'forks', 'license', 'language',
        'is_archived', 'has_release', 'size_kb', 'open_issues',
        'contributors', 'agent_pr_count', 'human_pr_count', 'total_pr_count',
        'code_ratio', 'loc'
    ]
    
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in meta_df.columns]
    final_df = meta_df[output_cols].copy()
    
    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nFinal repository count: {len(final_df)}")
    print(f"Total agent PRs: {final_df['agent_pr_count'].sum():,}")
    print(f"Total human PRs: {final_df['human_pr_count'].sum():,}")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    
    # Language distribution
    print("\nLanguage distribution:")
    lang_dist = final_df['language'].value_counts()
    for lang, count in lang_dist.head(10).items():
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()