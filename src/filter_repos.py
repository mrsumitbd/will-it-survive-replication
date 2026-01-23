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
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
AGENT_FILE = Path('data/all_pull_request.parquet')
HUMAN_FILE = Path('data/human_pull_request.parquet')
REPO_FILE = Path('data/all_repository.parquet')
OUTPUT_FILE = Path('data/aidev_filtered_v1.csv')
META_CACHE_FILE = Path('data/repo_metadata_v1.csv')
RATIO_CACHE_FILE = Path('data/code_ratio_v1.csv')
OBSERVATION_CUTOFF = datetime(2025, 12, 31)
MIN_STARS = 500
BAD_LICENSES = ['creative commons', 'cc0', 'cc-by', 'attribution 4.0', 'share alike', 'open font', 'ofl', 'unlicense', 'other', 'none', 'no license']
SOURCE_EXTENSIONS = ('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt')

class GitHubFetcher:

    def __init__(self, token: str=None):
        self.headers = {'Authorization': f'token {token}'} if token else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def fetch_repo_metadata(self, slug: str) -> dict:
        url = f'https://api.github.com/repos/{slug}'
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 403:
                reset_time = int(resp.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 60)
                print(f'  [Rate Limit] Sleeping {wait_time:.0f}s...')
                time.sleep(wait_time)
                return self.fetch_repo_metadata(slug)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                print(f'  [Warning] {slug}: HTTP {resp.status_code}')
                return None
            data = resp.json()
            tags_url = f'https://api.github.com/repos/{slug}/tags?per_page=1'
            tags_resp = self.session.get(tags_url, timeout=10)
            has_release = tags_resp.status_code == 200 and len(tags_resp.json()) > 0
            return {'slug': slug, 'url': data.get('html_url'), 'stars': data.get('stargazers_count', 0), 'forks': data.get('forks_count', 0), 'license': data.get('license', {}).get('name', 'None') if data.get('license') else 'None', 'is_archived': data.get('archived', False), 'is_fork': data.get('fork', False), 'size_kb': data.get('size', 0), 'open_issues': data.get('open_issues_count', 0), 'language': data.get('language') or 'Unknown', 'has_release': has_release, 'created_at': data.get('created_at'), 'updated_at': data.get('updated_at')}
        except requests.exceptions.Timeout:
            print(f'  [Timeout] {slug}')
            return None
        except Exception as e:
            print(f'  [Error] {slug}: {e}')
            return None

def analyze_code_ratio(repo_url: str) -> dict:
    if not repo_url:
        return None
    temp_dir = tempfile.mkdtemp()
    result = None
    try:
        clone_result = subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', repo_url, temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if clone_result.returncode != 0:
            return None
        cloc_result = subprocess.run(['cloc', '--json', '--quiet', temp_dir], capture_output=True, text=True, timeout=300)
        if cloc_result.returncode == 0:
            data = json.loads(cloc_result.stdout)
            stats = data.get('SUM', {})
            code = stats.get('code', 0)
            comment = stats.get('comment', 0)
            denominator = code + comment
            ratio = code / denominator if denominator > 0 else 0.0
            result = {'url': repo_url, 'code_ratio': ratio, 'loc': code, 'comments': comment, 'total_lines': code + comment + stats.get('blank', 0)}
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
    s = str(url).lower().strip().replace('.git', '').rstrip('/')
    if 'github.com/' in s:
        return s.split('github.com/')[-1]
    if 'api.github.com/repos/' in s:
        return s.split('api.github.com/repos/')[-1]
    parts = s.split('/')
    if len(parts) >= 2:
        return f'{parts[-2]}/{parts[-1]}'
    return s

def is_bad_license(license_name: str) -> bool:
    lic = str(license_name).lower().strip()
    if lic in ('none', 'null', 'nan', ''):
        return True
    return any((bad in lic for bad in BAD_LICENSES))

def main():
    print('=' * 70)
    print('XIAO ET AL. FILTERING PIPELINE (v1)')
    print('=' * 70)
    print(f'Observation cutoff: {OBSERVATION_CUTOFF.date()}')
    print(f'Minimum stars: {MIN_STARS}')
    print()
    print('STEP 1: Identifying Cohort')
    print('-' * 40)
    print('  Loading PR data...')
    df_agent = pd.read_parquet(AGENT_FILE)
    df_human = pd.read_parquet(HUMAN_FILE)
    print(f'  Agent PRs: {len(df_agent):,}')
    print(f'  Human PRs: {len(df_human):,}')
    url_col_a = next((col for col in ['repo_url', 'url', 'repository_url'] if col in df_agent.columns), None)
    url_col_h = next((col for col in ['repo_url', 'url', 'repository_url'] if col in df_human.columns), None)
    if not url_col_a or not url_col_h:
        print(f'  ERROR: Could not find URL column')
        print(f'  Agent columns: {df_agent.columns.tolist()}')
        print(f'  Human columns: {df_human.columns.tolist()}')
        sys.exit(1)
    df_agent['slug'] = df_agent[url_col_a].apply(normalize_slug)
    df_human['slug'] = df_human[url_col_h].apply(normalize_slug)
    agent_repos = set(df_agent['slug'].unique())
    human_repos = set(df_human['slug'].unique())
    print(f'  Unique agent repos: {len(agent_repos):,}')
    print(f'  Unique human repos: {len(human_repos):,}')
    cohort_slugs = agent_repos.intersection(human_repos)
    print(f'  Intersection (both agent & human PRs): {len(cohort_slugs):,}')
    print('\nSTEP 2: Repository Metadata')
    print('-' * 40)
    if REPO_FILE.exists():
        print('  Loading repository metadata from AIDev...')
        df_repo = pd.read_parquet(REPO_FILE)
        repo_url_col = next((col for col in ['url', 'html_url', 'repo_url'] if col in df_repo.columns), None)
        if repo_url_col:
            df_repo['slug'] = df_repo[repo_url_col].apply(normalize_slug)
    else:
        df_repo = pd.DataFrame()
    if META_CACHE_FILE.exists():
        meta_df = pd.read_csv(META_CACHE_FILE)
        print(f'  Loaded {len(meta_df)} cached metadata entries')
    else:
        meta_df = pd.DataFrame()
    known_slugs = set(meta_df['slug']) if not meta_df.empty else set()
    needed_slugs = [s for s in cohort_slugs if s not in known_slugs]
    if needed_slugs:
        print(f'  Fetching metadata for {len(needed_slugs)} repos...')
        if not GITHUB_TOKEN:
            print('  WARNING: No GITHUB_TOKEN found. API rate limits will apply.')
        fetcher = GitHubFetcher(GITHUB_TOKEN)
        new_rows = []
        for i, slug in enumerate(tqdm(needed_slugs, desc='  Fetching')):
            result = fetcher.fetch_repo_metadata(slug)
            if result:
                new_rows.append(result)
            if (i + 1) % 100 == 0 and new_rows:
                temp_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)
                temp_df.to_csv(META_CACHE_FILE, index=False)
        if new_rows:
            meta_df = pd.concat([meta_df, pd.DataFrame(new_rows)], ignore_index=True)
            meta_df.to_csv(META_CACHE_FILE, index=False)
            print(f'  Cached {len(new_rows)} new metadata entries')
    meta_df = meta_df[meta_df['slug'].isin(cohort_slugs)].copy()
    print(f'  Metadata available for: {len(meta_df)} repos')
    print(f'\nSTEP 3: Star Filter (>= {MIN_STARS})')
    print('-' * 40)
    before_stars = len(meta_df)
    meta_df = meta_df[meta_df['stars'] >= MIN_STARS].copy()
    print(f'  Removed: {before_stars - len(meta_df)} repos below {MIN_STARS} stars')
    print(f'  Remaining: {len(meta_df)}')
    print('\nSTEP 4: License Filter')
    print('-' * 40)
    before_license = len(meta_df)
    meta_df = meta_df[~meta_df['license'].apply(is_bad_license)].copy()
    print(f'  Removed: {before_license - len(meta_df)} repos with non-software licenses')
    print(f'  Remaining: {len(meta_df)}')
    print('\nSTEP 5: Archive & Release Filter')
    print('-' * 40)
    before_archive = len(meta_df)
    meta_df = meta_df[meta_df['is_archived'] == False].copy()
    print(f'  Removed: {before_archive - len(meta_df)} archived repos')
    before_release = len(meta_df)
    meta_df = meta_df[meta_df['has_release'] == True].copy()
    print(f'  Removed: {before_release - len(meta_df)} repos without releases')
    print(f'  Remaining: {len(meta_df)}')
    print('\nSTEP 6: Contributor Filter (>= 2)')
    print('-' * 40)
    users_agent = df_agent[df_agent['slug'].isin(meta_df['slug'])][['slug', 'user']]
    users_human = df_human[df_human['slug'].isin(meta_df['slug'])][['slug', 'user']]
    all_users = pd.concat([users_agent, users_human])
    contrib_counts = all_users.groupby('slug')['user'].nunique().reset_index(name='contributors')
    meta_df = meta_df.merge(contrib_counts, on='slug', how='left')
    meta_df['contributors'] = meta_df['contributors'].fillna(0).astype(int)
    before_contrib = len(meta_df)
    meta_df = meta_df[meta_df['contributors'] >= 2].copy()
    print(f'  Removed: {before_contrib - len(meta_df)} repos with < 2 contributors')
    print(f'  Remaining: {len(meta_df)}')
    print('\nSTEP 7: PR Count Enrichment')
    print('-' * 40)
    agent_pr_counts = df_agent[df_agent['slug'].isin(meta_df['slug'])].groupby('slug').size().reset_index(name='agent_pr_count')
    human_pr_counts = df_human[df_human['slug'].isin(meta_df['slug'])].groupby('slug').size().reset_index(name='human_pr_count')
    meta_df = meta_df.merge(agent_pr_counts, on='slug', how='left')
    meta_df = meta_df.merge(human_pr_counts, on='slug', how='left')
    meta_df['agent_pr_count'] = meta_df['agent_pr_count'].fillna(0).astype(int)
    meta_df['human_pr_count'] = meta_df['human_pr_count'].fillna(0).astype(int)
    meta_df['total_pr_count'] = meta_df['agent_pr_count'] + meta_df['human_pr_count']
    print(f"  Total PRs in cohort: {meta_df['total_pr_count'].sum():,}")
    print(f"    Agent PRs: {meta_df['agent_pr_count'].sum():,}")
    print(f"    Human PRs: {meta_df['human_pr_count'].sum():,}")
    print('\nSTEP 8: Q1 Statistical Filter (Per Language)')
    print('-' * 40)
    filtered_dfs = []
    for lang in meta_df['language'].unique():
        if pd.isna(lang) or lang == 'Unknown':
            continue
        lang_df = meta_df[meta_df['language'] == lang].copy()
        if len(lang_df) < 4:
            filtered_dfs.append(lang_df)
            continue
        q1_pr = lang_df['total_pr_count'].quantile(0.25)
        q1_issues = lang_df['open_issues'].quantile(0.25)
        q1_size = lang_df['size_kb'].quantile(0.25)
        lang_filtered = lang_df[(lang_df['total_pr_count'] > q1_pr) & (lang_df['open_issues'] > q1_issues) & (lang_df['size_kb'] > q1_size)]
        removed = len(lang_df) - len(lang_filtered)
        if removed > 0:
            print(f'  {lang}: {len(lang_df)} -> {len(lang_filtered)} (-{removed})')
        filtered_dfs.append(lang_filtered)
    if not filtered_dfs:
        print('  ERROR: No repos survived Q1 filtering')
        sys.exit(1)
    meta_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f'  Remaining after Q1 filter: {len(meta_df)}')
    print('\nSTEP 9: Code Ratio Filter (97% CI)')
    print('-' * 40)
    if RATIO_CACHE_FILE.exists():
        ratio_df = pd.read_csv(RATIO_CACHE_FILE)
        print(f'  Loaded {len(ratio_df)} cached code ratios')
    else:
        ratio_df = pd.DataFrame()
    known_urls = set(ratio_df['url']) if not ratio_df.empty else set()
    to_analyze = meta_df[~meta_df['url'].isin(known_urls)].copy()
    if not to_analyze.empty:
        print(f'  Computing code ratios for {len(to_analyze)} repos...')
        n_workers = min(multiprocessing.cpu_count(), 8)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(analyze_code_ratio, row['url']): row['url'] for _, row in to_analyze.iterrows()}
            new_ratios = []
            for future in tqdm(as_completed(futures), total=len(futures), desc='  Analyzing'):
                result = future.result()
                if result:
                    new_ratios.append(result)
            if new_ratios:
                ratio_df = pd.concat([ratio_df, pd.DataFrame(new_ratios)], ignore_index=True)
                ratio_df.to_csv(RATIO_CACHE_FILE, index=False)
                print(f'  Cached {len(new_ratios)} new code ratios')
    meta_df = meta_df.merge(ratio_df[['url', 'code_ratio', 'loc']], on='url', how='inner')
    print(f'  Repos with code ratio data: {len(meta_df)}')
    filtered_dfs = []
    for lang in meta_df['language'].unique():
        lang_df = meta_df[meta_df['language'] == lang].copy()
        if len(lang_df) < 3:
            filtered_dfs.append(lang_df)
            continue
        ratios = lang_df['code_ratio']
        mean = ratios.mean()
        std = ratios.std()
        z = 2.17
        lower = mean - z * std
        upper = mean + z * std
        lang_filtered = lang_df[(lang_df['code_ratio'] >= lower) & (lang_df['code_ratio'] <= upper)]
        removed = len(lang_df) - len(lang_filtered)
        if removed > 0:
            print(f'  {lang}: {len(lang_df)} -> {len(lang_filtered)} (-{removed}, CI: [{lower:.3f}, {upper:.3f}])')
        filtered_dfs.append(lang_filtered)
    meta_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f'  Remaining after code ratio filter: {len(meta_df)}')
    print('\n' + '=' * 70)
    print('FILTERING COMPLETE')
    print('=' * 70)
    output_cols = ['slug', 'url', 'stars', 'forks', 'license', 'language', 'is_archived', 'has_release', 'size_kb', 'open_issues', 'contributors', 'agent_pr_count', 'human_pr_count', 'total_pr_count', 'code_ratio', 'loc']
    output_cols = [c for c in output_cols if c in meta_df.columns]
    final_df = meta_df[output_cols].copy()
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f'\nFinal repository count: {len(final_df)}')
    print(f"Total agent PRs: {final_df['agent_pr_count'].sum():,}")
    print(f"Total human PRs: {final_df['human_pr_count'].sum():,}")
    print(f'\nOutput saved to: {OUTPUT_FILE}')
    print('\nLanguage distribution:')
    lang_dist = final_df['language'].value_counts()
    for lang, count in lang_dist.head(10).items():
        print(f'  {lang}: {count}')
if __name__ == '__main__':
    main()