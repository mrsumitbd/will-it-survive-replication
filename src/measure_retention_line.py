"""
measure_retention_line_v1.py

Measures code survival at LINE granularity using git blame.

Definition of "death": A line introduced by the PR is attributed to a different 
commit when running git blame at a later point.

This is the most rigorous granularity - tracks individual lines of code.
More computationally expensive but provides precise survival measurement.

Methodology:
1. For each merged PR, identify the merge commit SHA
2. Get the diff to find lines ADDED by this PR
3. For each added line, use git blame to check if it still exists
4. If blame shows different SHA -> line died (find when using log)
5. If blame shows same SHA -> line survived (censored)

Usage:
    python src/measure_retention_line_v1.py

Input:
    data/aidev_filtered_v1.csv
    data/all_pull_request.parquet
    data/human_pull_request.parquet

Output:
    data/survival_events_line_v1.csv

Schema:
    slug, pr_number, is_agent, agent_name, unit_id, event, duration_days,
    birth_sha, birth_date, death_sha, death_date, granularity
    
Where unit_id = "filepath::line_number" (line number as introduced)
"""

import pandas as pd
import os
import re
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

# --- CONFIGURATION ---
INPUT_FILE = Path("data/aidev_filtered_v1.csv")
PR_DATA_DIR = Path("data")
OUTPUT_FILE = Path("data/survival_events_line_v1.csv")

# Observation cutoff
OBSERVATION_CUTOFF = datetime(2025, 12, 31)

# Parallelization
MAX_WORKERS = 4

# Source code extensions
SOURCE_EXTENSIONS = (
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.c', '.cpp', '.cc', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt',
    '.scala', '.m', '.mm'
)

# Sampling: For large PRs, sample lines to avoid excessive computation
MAX_LINES_PER_FILE = 100  # Sample at most this many lines per file
MAX_FILES_PER_PR = 20     # Process at most this many files per PR

# Agent identification
AGENT_KEYWORDS = {
    'copilot': 'GitHub Copilot',
    'devin': 'Devin',
    'cursor': 'Cursor',
    'codex': 'OpenAI Codex',
    'claude': 'Claude Code',
}


def parse_git_date(date_str: str) -> Optional[datetime]:
    """Parse git date string to datetime."""
    try:
        return datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
    except:
        return None


def identify_agent(row: pd.Series) -> str:
    """Identify which agent authored the PR."""
    for field in ['user', 'author', 'user_login']:
        if field in row.index:
            val = str(row[field]).lower()
            for keyword, agent_name in AGENT_KEYWORDS.items():
                if keyword in val:
                    return agent_name
    
    if 'agent' in row.index:
        return str(row['agent'])
    
    return 'Unknown Agent'


def find_merge_commit(repo_dir: str, pr_number: int) -> Tuple[Optional[str], Optional[datetime]]:
    """Find the merge commit for a PR."""
    strategies = [
        f"Merge pull request #{pr_number}",
        f"(#{pr_number})",
        f"#{pr_number}[^0-9]",
    ]
    
    for pattern in strategies:
        cmd = [
            "git", "log", "--all",
            "--grep", pattern,
            "--format=%H|%ci",
            "-n", "1"
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        
        if result.stdout.strip():
            parts = result.stdout.strip().split('|')
            if len(parts) >= 2:
                return parts[0], parse_git_date(parts[1])
    
    return None, None


def get_added_lines(repo_dir: str, commit_sha: str) -> Dict[str, List[int]]:
    """
    Get lines ADDED by a commit.
    
    Returns:
        Dictionary mapping filepath -> list of line numbers added
    """
    # Get the diff showing only additions
    cmd = [
        "git", "show", 
        "--format=",  # No commit message
        "--unified=0",  # No context lines
        commit_sha
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        return {}
    
    added_lines = {}
    current_file = None
    
    for line in result.stdout.splitlines():
        # New file header
        if line.startswith('+++ b/'):
            current_file = line[6:]
            if current_file.endswith(SOURCE_EXTENSIONS):
                added_lines[current_file] = []
            else:
                current_file = None
        
        # Hunk header: @@ -old_start,old_count +new_start,new_count @@
        elif line.startswith('@@') and current_file:
            match = re.search(r'\+(\d+)(?:,(\d+))?', line)
            if match:
                start_line = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                # This hunk adds lines starting at start_line
                # We'll track actual additions below
        
        # Actual added line (starts with +, not +++)
        elif line.startswith('+') and not line.startswith('+++') and current_file:
            # We need to track line numbers properly
            # This simplified approach assumes sequential additions
            pass
    
    # Better approach: use git show with --numstat and parse diff more carefully
    # Let's use git diff-tree for cleaner parsing
    
    return get_added_lines_v2(repo_dir, commit_sha)


def get_added_lines_v2(repo_dir: str, commit_sha: str) -> Dict[str, List[int]]:
    """
    Get lines ADDED by a commit using blame on the commit itself.
    
    More reliable approach: get files changed, then for each file,
    run blame at that commit and find lines attributed to that commit.
    """
    added_lines = {}
    
    # Get files changed in commit
    cmd = ["git", "show", "--name-only", "--format=", commit_sha]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    
    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    source_files = [f for f in files if f.endswith(SOURCE_EXTENSIONS)]
    
    # Limit files per PR
    if len(source_files) > MAX_FILES_PER_PR:
        source_files = source_files[:MAX_FILES_PER_PR]
    
    for filepath in source_files:
        # Run blame at this commit to find lines introduced by this commit
        cmd = [
            "git", "blame",
            "--line-porcelain",
            commit_sha,
            "--", filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            continue
        
        lines_from_this_commit = []
        current_line_num = 0
        
        for line in result.stdout.splitlines():
            # Line format: <sha> <orig_line> <final_line> [<num_lines>]
            if re.match(r'^[0-9a-f]{40}', line):
                parts = line.split()
                line_sha = parts[0]
                if len(parts) >= 3:
                    current_line_num = int(parts[2])
                
                # Check if this line was introduced by our commit
                if line_sha.startswith(commit_sha[:8]):
                    lines_from_this_commit.append(current_line_num)
        
        if lines_from_this_commit:
            # Sample if too many lines
            if len(lines_from_this_commit) > MAX_LINES_PER_FILE:
                import random
                lines_from_this_commit = sorted(
                    random.sample(lines_from_this_commit, MAX_LINES_PER_FILE)
                )
            added_lines[filepath] = lines_from_this_commit
    
    return added_lines


def check_line_survival(
    repo_dir: str, 
    filepath: str, 
    line_number: int,
    birth_sha: str,
    birth_date: datetime
) -> Tuple[int, float, Optional[str], Optional[datetime]]:
    """
    Check if a specific line survived to the observation cutoff.
    
    Returns:
        Tuple of (event, duration_days, death_sha, death_date)
        event = 1 if line died, 0 if survived (censored)
    """
    # First check: does the file still exist at HEAD?
    cmd = ["git", "ls-files", "--", filepath]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    
    if not result.stdout.strip():
        # File was deleted - find when
        cmd = [
            "git", "log",
            "--reverse",
            "--format=%H|%ci",
            "--diff-filter=D",
            f"{birth_sha}..HEAD",
            "--", filepath
        ]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
        
        if result.stdout.strip():
            parts = result.stdout.strip().splitlines()[0].split('|')
            death_sha = parts[0]
            death_date = parse_git_date(parts[1])
            duration = (death_date - birth_date).total_seconds() / 86400.0
            return 1, max(0.001, duration), death_sha, death_date
        
        # File deleted but couldn't find when - treat as died at cutoff
        duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
        return 1, max(0.001, duration), None, None
    
    # File exists - check blame for the specific line
    cmd = [
        "git", "blame",
        "-L", f"{line_number},{line_number}",
        "--porcelain",
        "HEAD",
        "--", filepath
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    
    if result.returncode != 0 or not result.stdout.strip():
        # Line no longer exists at this position (file may have shrunk)
        # Find when this line was last modified
        return find_line_death(repo_dir, filepath, birth_sha, birth_date)
    
    # Parse blame output to get current SHA
    first_line = result.stdout.splitlines()[0]
    current_sha = first_line.split()[0]
    
    # Check if line is still attributed to birth commit
    if current_sha.startswith(birth_sha[:8]) or birth_sha.startswith(current_sha[:8]):
        # Line survived!
        duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
        return 0, max(0.001, duration), None, None
    else:
        # Line was modified - find when
        return find_line_death(repo_dir, filepath, birth_sha, birth_date)


def find_line_death(
    repo_dir: str,
    filepath: str,
    birth_sha: str,
    birth_date: datetime
) -> Tuple[int, float, Optional[str], Optional[datetime]]:
    """Find when a line in a file was first modified after birth."""
    cmd = [
        "git", "log",
        "--reverse",
        "--format=%H|%ci",
        f"{birth_sha}..HEAD",
        "--", filepath
    ]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    
    if result.stdout.strip():
        # First modification after birth
        parts = result.stdout.strip().splitlines()[0].split('|')
        death_sha = parts[0]
        death_date = parse_git_date(parts[1])
        
        if death_date:
            duration = (death_date - birth_date).total_seconds() / 86400.0
            return 1, max(0.001, duration), death_sha, death_date
    
    # No modifications found - survived
    duration = (OBSERVATION_CUTOFF - birth_date).total_seconds() / 86400.0
    return 0, max(0.001, duration), None, None


def process_repo(slug: str, pr_df: pd.DataFrame) -> List[dict]:
    """Process all PRs for a single repository at line granularity."""
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = f"https://github.com/{slug}.git"
    
    try:
        # Clone repository
        clone_result = subprocess.run(
            ["git", "clone", "--filter=blob:none", clone_url, temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300
        )
        
        if clone_result.returncode != 0:
            return results
        
        for _, row in pr_df.iterrows():
            pr_number = row['number']
            is_agent = row['is_agent']
            agent_name = identify_agent(row) if is_agent else 'Human'
            
            # Find merge commit
            merge_sha, merge_date = find_merge_commit(temp_dir, pr_number)
            
            if not merge_sha or not merge_date:
                continue
            
            # Get lines added by this PR
            added_lines = get_added_lines_v2(temp_dir, merge_sha)
            
            # Track survival for each line
            for filepath, line_numbers in added_lines.items():
                for line_num in line_numbers:
                    event, duration, death_sha, death_date = check_line_survival(
                        temp_dir, filepath, line_num, merge_sha, merge_date
                    )
                    
                    results.append({
                        'slug': slug,
                        'pr_number': pr_number,
                        'is_agent': is_agent,
                        'agent_name': agent_name,
                        'unit_id': f"{filepath}::{line_num}",
                        'event': event,
                        'duration_days': duration,
                        'birth_sha': merge_sha,
                        'birth_date': merge_date.isoformat(),
                        'death_sha': death_sha,
                        'death_date': death_date.isoformat() if death_date else None,
                        'granularity': 'line'
                    })
    
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def normalize_slug(url: str) -> str:
    """Normalize repository URL to slug format."""
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
    print("SURVIVAL MEASUREMENT - LINE LEVEL (v1)")
    print("=" * 70)
    print(f"Observation cutoff: {OBSERVATION_CUTOFF.date()}")
    print(f"Max lines per file: {MAX_LINES_PER_FILE}")
    print(f"Max files per PR: {MAX_FILES_PER_PR}")
    print()
    
    # Load filtered repos
    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found. Run filter_repos_v1.py first.")
        return
    
    df_repos = pd.read_csv(INPUT_FILE)
    slugs = set(df_repos['slug'].unique())
    print(f"Loaded {len(slugs)} filtered repositories")
    
    # Load PR data
    print("Loading PR data...")
    df_agent = pd.read_parquet(PR_DATA_DIR / "all_pull_request.parquet")
    df_human = pd.read_parquet(PR_DATA_DIR / "human_pull_request.parquet")
    
    # Find URL column
    url_col_a = next(
        (col for col in ['repo_url', 'url', 'repository_url'] if col in df_agent.columns),
        None
    )
    url_col_h = next(
        (col for col in ['repo_url', 'url', 'repository_url'] if col in df_human.columns),
        None
    )
    
    # Normalize and tag
    df_agent['slug'] = df_agent[url_col_a].apply(normalize_slug)
    df_agent['is_agent'] = True
    
    df_human['slug'] = df_human[url_col_h].apply(normalize_slug)
    df_human['is_agent'] = False
    
    # Combine and filter to cohort
    all_prs = pd.concat([df_agent, df_human], ignore_index=True)
    all_prs = all_prs[all_prs['slug'].isin(slugs)]
    
    print(f"PRs to process: {len(all_prs):,}")
    print(f"  Agent PRs: {all_prs['is_agent'].sum():,}")
    print(f"  Human PRs: {(~all_prs['is_agent']).sum():,}")
    
    # Process by repository
    repo_groups = list(all_prs.groupby('slug'))
    print(f"\nProcessing {len(repo_groups)} repositories...")
    print("(This may take a while - line-level analysis is intensive)")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_repo, slug, group): slug
            for slug, group in repo_groups
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mining"):
            results = future.result()
            if results:
                all_results.extend(results)
    
    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total survival events (lines): {len(df_results):,}")
    
    if not df_results.empty:
        deaths = df_results['event'].sum()
        censored = len(df_results) - deaths
        print(f"  Deaths (modified): {deaths:,} ({100*deaths/len(df_results):.1f}%)")
        print(f"  Censored (survived): {censored:,} ({100*censored/len(df_results):.1f}%)")
        
        # By agent vs human
        agent_events = df_results[df_results['is_agent'] == True]
        human_events = df_results[df_results['is_agent'] == False]
        
        if len(agent_events) > 0:
            agent_death_rate = agent_events['event'].mean() * 100
            print(f"\n  Agent lines: {len(agent_events):,} (death rate: {agent_death_rate:.1f}%)")
        
        if len(human_events) > 0:
            human_death_rate = human_events['event'].mean() * 100
            print(f"  Human lines: {len(human_events):,} (death rate: {human_death_rate:.1f}%)")
        
        # Duration stats
        print(f"\n  Median duration: {df_results['duration_days'].median():.1f} days")
        print(f"  Mean duration: {df_results['duration_days'].mean():.1f} days")
        
        # Unique PRs and files
        print(f"\n  Unique PRs: {df_results['pr_number'].nunique():,}")
        unique_files = df_results['unit_id'].apply(lambda x: x.split('::')[0]).nunique()
        print(f"  Unique files: {unique_files:,}")
    
    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()