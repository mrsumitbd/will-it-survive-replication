import pandas as pd
import os
import re
import subprocess
import tempfile
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
DATA_DIR = Path('data')
MAX_WORKERS = 4
BOT_KEYWORDS = ['bot', 'dependabot', 'renovate', 'copilot', 'devin', 'sweep', 'codeium', 'snyk', 'action', 'runner', 'greenkeeper', 'semantic-release', 'auto-merge', 'mergify', 'kodiak', 'github-actions', 'netlify', 'vercel', 'circleci', 'travis', 'jenkins', 'cursor', 'claude']
CLASSIFICATION_RULES = {'Corrective (Bug Fix)': ['fix', 'bug', 'error', 'issue', 'crash', 'fail', 'patch', 'resolve', 'correct', 'hotfix', 'defect', 'fault', 'repair', 'wrong', 'broken', 'problem', 'regression', 'typo'], 'Perfective (Refactor)': ['refactor', 'clean', 'tidy', 'lint', 'format', 'style', 'optimize', 'simplify', 'reorganize', 'restructure', 'improve', 'enhance', 'polish', 'beautify', 'modernize', 'dead code', 'rename', 'extract', 'inline', 'move', 'split'], 'Adaptive (New Feature)': ['feat', 'add', 'new', 'implement', 'support', 'feature', 'introduce', 'create', 'enable', 'allow', 'extend'], 'Preventive (Testing)': ['test', 'spec', 'coverage', 'benchmark', 'assert', 'mock', 'stub', 'fixture', 'e2e', 'integration', 'unit test'], 'Documentation': ['doc', 'comment', 'readme', 'changelog', 'license', 'contributing', 'example', 'tutorial', 'guide', 'api doc'], 'Chore/Maintenance': ['chore', 'bump', 'update', 'upgrade', 'merge', 'sync', 'release', 'version', 'dependency', 'deps', 'ci', 'cd', 'build', 'config', 'setup', 'init'], 'Security': ['security', 'vuln', 'cve', 'xss', 'csrf', 'injection', 'auth', 'permission', 'sanitize', 'escape', 'encrypt'], 'Performance': ['perf', 'performance', 'speed', 'fast', 'slow', 'memory', 'cache', 'lazy', 'async', 'parallel', 'concurrent', 'optimize']}

def is_bot(name: str, email: str) -> bool:
    combined = (str(name) + str(email)).lower()
    return any((keyword in combined for keyword in BOT_KEYWORDS))

def classify_message(msg: str) -> Tuple[str, str]:
    msg_lower = str(msg).lower()
    matches = []
    for category, keywords in CLASSIFICATION_RULES.items():
        for keyword in keywords:
            if keyword in msg_lower:
                matches.append((category, keyword))
                break
    if not matches:
        return ('Other', 'unknown')
    priority = ['Security', 'Corrective (Bug Fix)', 'Performance', 'Adaptive (New Feature)', 'Perfective (Refactor)', 'Preventive (Testing)', 'Documentation', 'Chore/Maintenance']
    for cat in priority:
        for match_cat, keyword in matches:
            if match_cat == cat:
                return (cat, keyword)
    return (matches[0][0], matches[0][1])

def get_commit_details(repo_dir: str, commit_sha: str) -> Optional[Dict]:
    cmd = ['git', 'show', '--format=%H|%an|%ae|%s|%b', '--numstat', '-n', '1', commit_sha]
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().split('\n')
    if not lines:
        return None
    header = lines[0]
    parts = header.split('|', 4)
    if len(parts) < 4:
        return None
    sha = parts[0]
    author_name = parts[1]
    author_email = parts[2]
    subject = parts[3]
    body = parts[4] if len(parts) > 4 else ''
    lines_added = 0
    lines_deleted = 0
    for line in lines[1:]:
        if line.strip() and '\t' in line:
            stats = line.split('\t')
            if len(stats) >= 2:
                try:
                    added = int(stats[0]) if stats[0] != '-' else 0
                    deleted = int(stats[1]) if stats[1] != '-' else 0
                    lines_added += added
                    lines_deleted += deleted
                except ValueError:
                    pass
    return {'sha': sha, 'author_name': author_name, 'author_email': author_email, 'subject': subject, 'body': body, 'full_message': f'{subject}\n{body}'.strip(), 'lines_added': lines_added, 'lines_deleted': lines_deleted}

def mine_details_for_repo(slug: str, deaths_df: pd.DataFrame) -> List[Dict]:
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = f'https://github.com/{slug}.git'
    try:
        clone_result = subprocess.run(['git', 'clone', '--filter=blob:none', clone_url, temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if clone_result.returncode != 0:
            return results
        for _, row in deaths_df.iterrows():
            death_sha = row.get('death_sha')
            if not death_sha or pd.isna(death_sha):
                continue
            commit_info = get_commit_details(temp_dir, death_sha)
            if not commit_info:
                continue
            reason, reason_keyword = classify_message(commit_info['full_message'])
            modifier_is_bot = is_bot(commit_info['author_name'], commit_info['author_email'])
            results.append({'slug': slug, 'pr_number': row['pr_number'], 'unit_id': row['unit_id'], 'is_agent': row['is_agent'], 'agent_name': row.get('agent_name', 'Unknown'), 'granularity': row.get('granularity', 'unknown'), 'duration_days': row['duration_days'], 'reason': reason, 'reason_keyword': reason_keyword, 'modifier_is_bot': modifier_is_bot, 'modifier_author': commit_info['author_name'], 'modifier_email': commit_info['author_email'], 'death_message': commit_info['subject'], 'death_message_full': commit_info['full_message'][:500], 'killer_hash': death_sha, 'lines_added': commit_info['lines_added'], 'lines_deleted': commit_info['lines_deleted']})
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results

def main():
    parser = argparse.ArgumentParser(description='Mine death details for survival events')
    parser.add_argument('--granularity', choices=['file', 'hunk', 'line', 'all'], default='all', help='Which granularity to process (default: all)')
    args = parser.parse_args()
    print('=' * 70)
    print('DEATH DETAILS MINING (v1)')
    print('=' * 70)
    print()
    granularities = ['file', 'hunk', 'line'] if args.granularity == 'all' else [args.granularity]
    for granularity in granularities:
        input_file = DATA_DIR / f'survival_events_{granularity}_v1.csv'
        output_file = DATA_DIR / f'death_details_{granularity}_v1.csv'
        if not input_file.exists():
            print(f'Skipping {granularity}: {input_file} not found')
            continue
        print(f'\n--- Processing {granularity.upper()} level ---')
        df = pd.read_csv(input_file)
        deaths = df[df['event'] == 1].copy()
        print(f'Death events to analyze: {len(deaths):,}')
        if deaths.empty:
            print('No deaths to analyze')
            continue
        repo_groups = list(deaths.groupby('slug'))
        print(f'Repositories: {len(repo_groups)}')
        all_results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(mine_details_for_repo, slug, group): slug for slug, group in repo_groups}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Mining'):
                results = future.result()
                if results:
                    all_results.extend(results)
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_file, index=False)
        print(f'\nResults for {granularity} level:')
        print(f'  Total death details: {len(df_results):,}')
        if not df_results.empty:
            print(f'\n  Modification reasons:')
            reason_counts = df_results['reason'].value_counts()
            for reason, count in reason_counts.items():
                pct = 100 * count / len(df_results)
                print(f'    {reason}: {count:,} ({pct:.1f}%)')
            print(f'\n  By author type:')
            agent_deaths = df_results[df_results['is_agent'] == True]
            human_deaths = df_results[df_results['is_agent'] == False]
            if len(agent_deaths) > 0:
                agent_corrective = (agent_deaths['reason'] == 'Corrective (Bug Fix)').mean() * 100
                print(f'    Agent code: {len(agent_deaths):,} deaths, {agent_corrective:.1f}% corrective')
            if len(human_deaths) > 0:
                human_corrective = (human_deaths['reason'] == 'Corrective (Bug Fix)').mean() * 100
                print(f'    Human code: {len(human_deaths):,} deaths, {human_corrective:.1f}% corrective')
            bot_modifications = df_results['modifier_is_bot'].sum()
            print(f'\n  Modifications by bots: {bot_modifications:,} ({100 * bot_modifications / len(df_results):.1f}%)')
        print(f'\nOutput saved to: {output_file}')
    print('\n' + '=' * 70)
    print('DEATH DETAILS MINING COMPLETE')
    print('=' * 70)
if __name__ == '__main__':
    main()