import pandas as pd
import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
INPUT_FILE = Path('data/aidev_filtered_v1.csv')
PR_DATA_DIR = Path('data')
OUTPUT_FILE = Path('data/survival_events_file_v1.csv')
OBSERVATION_CUTOFF = datetime(2025, 12, 31)
MAX_WORKERS = 4
SOURCE_EXTENSIONS = ('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt', '.scala', '.m', '.mm')
AGENT_KEYWORDS = {'copilot': 'GitHub Copilot', 'devin': 'Devin', 'cursor': 'Cursor', 'codex': 'OpenAI Codex', 'claude': 'Claude Code'}

def parse_git_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
    except:
        return None

def identify_agent(row: pd.Series) -> str:
    for field in ['user', 'author', 'user_login']:
        if field in row.index:
            val = str(row[field]).lower()
            for keyword, agent_name in AGENT_KEYWORDS.items():
                if keyword in val:
                    return agent_name
    if 'agent' in row.index:
        return str(row['agent'])
    return 'Unknown Agent'

def find_merge_commit(repo_dir: str, pr_number: int) -> tuple:
    cmd = ['git', 'log', '--all', '--grep', f'Merge pull request #{pr_number}', '--format=%H|%ci', '-n', '1']
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        parts = result.stdout.strip().split('|')
        if len(parts) >= 2:
            return (parts[0], parse_git_date(parts[1]))
    cmd = ['git', 'log', '--all', '--grep', f'(#{pr_number})', '--format=%H|%ci', '-n', '1']
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        parts = result.stdout.strip().split('|')
        if len(parts) >= 2:
            return (parts[0], parse_git_date(parts[1]))
    cmd = ['git', 'log', '--all', '--grep', f'#{pr_number}[^0-9]', '--format=%H|%ci', '-n', '1']
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        parts = result.stdout.strip().split('|')
        if len(parts) >= 2:
            return (parts[0], parse_git_date(parts[1]))
    return (None, None)

def get_files_in_commit(repo_dir: str, commit_sha: str) -> list:
    cmd = ['git', 'show', '--name-only', '--format=', commit_sha]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    source_files = [f for f in files if f.endswith(SOURCE_EXTENSIONS)]
    return source_files

def find_first_modification(repo_dir: str, merge_sha: str, filepath: str) -> tuple:
    cmd = ['git', 'log', '--reverse', '--format=%H|%ci', f'{merge_sha}..HEAD', '--', filepath]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    if lines:
        parts = lines[0].split('|')
        if len(parts) >= 2:
            return (parts[0], parse_git_date(parts[1]))
    return (None, None)

def process_repo(slug: str, pr_df: pd.DataFrame) -> list:
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = f'https://github.com/{slug}.git'
    try:
        clone_result = subprocess.run(['git', 'clone', '--filter=blob:none', clone_url, temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if clone_result.returncode != 0:
            return results
        for _, row in pr_df.iterrows():
            pr_number = row['number']
            is_agent = row['is_agent']
            agent_name = identify_agent(row) if is_agent else 'Human'
            merge_sha, merge_date = find_merge_commit(temp_dir, pr_number)
            if not merge_sha or not merge_date:
                continue
            files = get_files_in_commit(temp_dir, merge_sha)
            for filepath in files:
                death_sha, death_date = find_first_modification(temp_dir, merge_sha, filepath)
                if death_date:
                    duration = (death_date - merge_date).total_seconds() / 86400.0
                    event = 1
                else:
                    duration = (OBSERVATION_CUTOFF - merge_date).total_seconds() / 86400.0
                    event = 0
                    death_sha = None
                    death_date = None
                duration = max(0.001, duration)
                results.append({'slug': slug, 'pr_number': pr_number, 'is_agent': is_agent, 'agent_name': agent_name, 'unit_id': filepath, 'event': event, 'duration_days': duration, 'birth_sha': merge_sha, 'birth_date': merge_date.isoformat() if merge_date else None, 'death_sha': death_sha, 'death_date': death_date.isoformat() if death_date else None, 'granularity': 'file'})
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results

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

def main():
    print('=' * 70)
    print('SURVIVAL MEASUREMENT - FILE LEVEL (v1)')
    print('=' * 70)
    print(f'Observation cutoff: {OBSERVATION_CUTOFF.date()}')
    print()
    if not INPUT_FILE.exists():
        print(f'ERROR: {INPUT_FILE} not found. Run filter_repos_v1.py first.')
        return
    df_repos = pd.read_csv(INPUT_FILE)
    slugs = set(df_repos['slug'].unique())
    print(f'Loaded {len(slugs)} filtered repositories')
    print('Loading PR data...')
    df_agent = pd.read_parquet(PR_DATA_DIR / 'all_pull_request.parquet')
    df_human = pd.read_parquet(PR_DATA_DIR / 'human_pull_request.parquet')
    url_col_a = next((col for col in ['repo_url', 'url', 'repository_url'] if col in df_agent.columns), None)
    url_col_h = next((col for col in ['repo_url', 'url', 'repository_url'] if col in df_human.columns), None)
    df_agent['slug'] = df_agent[url_col_a].apply(normalize_slug)
    df_agent['is_agent'] = True
    df_human['slug'] = df_human[url_col_h].apply(normalize_slug)
    df_human['is_agent'] = False
    all_prs = pd.concat([df_agent, df_human], ignore_index=True)
    all_prs = all_prs[all_prs['slug'].isin(slugs)]
    print(f'PRs to process: {len(all_prs):,}')
    print(f"  Agent PRs: {all_prs['is_agent'].sum():,}")
    print(f"  Human PRs: {(~all_prs['is_agent']).sum():,}")
    repo_groups = list(all_prs.groupby('slug'))
    print(f'\nProcessing {len(repo_groups)} repositories...')
    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_repo, slug, group): slug for slug, group in repo_groups}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Mining'):
            results = future.result()
            if results:
                all_results.extend(results)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Total survival events: {len(df_results):,}')
    if not df_results.empty:
        deaths = df_results['event'].sum()
        censored = len(df_results) - deaths
        print(f'  Deaths (modified): {deaths:,} ({100 * deaths / len(df_results):.1f}%)')
        print(f'  Censored (survived): {censored:,} ({100 * censored / len(df_results):.1f}%)')
        agent_events = df_results[df_results['is_agent'] == True]
        human_events = df_results[df_results['is_agent'] == False]
        if len(agent_events) > 0:
            agent_death_rate = agent_events['event'].mean() * 100
            print(f'\n  Agent files: {len(agent_events):,} (death rate: {agent_death_rate:.1f}%)')
        if len(human_events) > 0:
            human_death_rate = human_events['event'].mean() * 100
            print(f'  Human files: {len(human_events):,} (death rate: {human_death_rate:.1f}%)')
        print(f"\n  Median duration: {df_results['duration_days'].median():.1f} days")
        print(f"  Mean duration: {df_results['duration_days'].mean():.1f} days")
    print(f'\nOutput saved to: {OUTPUT_FILE}')
if __name__ == '__main__':
    main()