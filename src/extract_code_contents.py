import pandas as pd
import numpy as np
import argparse
import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import hashlib
DATA_DIR = Path('data')
OUTPUT_DIR = Path('data')
MAX_WORKERS = 4
SOURCE_EXTENSIONS = ('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.h', '.hpp', '.cs', '.go', '.rb', '.php', '.rs', '.swift', '.kt', '.scala', '.m', '.mm')
PYTHON_KEYWORDS = {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'import', 'from', 'return', 'yield', 'raise', 'pass', 'break', 'continue', 'lambda', 'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False', 'async', 'await', 'global', 'nonlocal', 'assert', 'del'}
JAVA_KEYWORDS = {'public', 'private', 'protected', 'static', 'final', 'abstract', 'class', 'interface', 'extends', 'implements', 'new', 'this', 'super', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 'throws', 'import', 'package', 'void', 'int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short', 'null'}
JS_KEYWORDS = {'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 'new', 'this', 'class', 'extends', 'import', 'export', 'default', 'async', 'await', 'yield', 'null', 'undefined', 'true', 'false', 'typeof', 'instanceof', 'in', 'of', 'delete', 'void'}
C_KEYWORDS = {'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while', 'include', 'define', 'ifdef', 'ifndef', 'endif', 'pragma'}
GO_KEYWORDS = {'break', 'case', 'chan', 'const', 'continue', 'default', 'defer', 'else', 'fallthrough', 'for', 'func', 'go', 'goto', 'if', 'import', 'interface', 'map', 'package', 'range', 'return', 'select', 'struct', 'switch', 'type', 'var', 'nil', 'true', 'false', 'iota', 'append', 'cap', 'close', 'copy', 'delete', 'len', 'make', 'new', 'panic', 'print', 'println', 'recover'}
ALL_KEYWORDS = PYTHON_KEYWORDS | JAVA_KEYWORDS | JS_KEYWORDS | C_KEYWORDS | GO_KEYWORDS
OPERATORS = {'==', '!=', '<=', '>=', '&&', '||', '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '->', '=>', '::', '...', '**', '//', '<<', '>>', '&', '|', '^', '~', '<', '>', '+', '-', '*', '/', '%', '=', '!', '?', ':', '.', ',', ';', '(', ')', '[', ']', '{', '}'}

def tokenize_code(code: str, preserve_case: bool=False) -> List[str]:
    if not code or not isinstance(code, str):
        return []
    tokens = []
    code = re.sub('\\"\\"\\".*?\\"\\"\\"', ' STRING_LITERAL ', code, flags=re.DOTALL)
    code = re.sub("\\'\\'\\'.*?\\'\\'\\'", ' STRING_LITERAL ', code, flags=re.DOTALL)
    code = re.sub('\\"[^\\"]*\\"', ' STRING_LITERAL ', code)
    code = re.sub("\\'[^\\']*\\'", ' STRING_LITERAL ', code)
    code = re.sub('//.*?$', ' COMMENT ', code, flags=re.MULTILINE)
    code = re.sub('#.*?$', ' COMMENT ', code, flags=re.MULTILINE)
    code = re.sub('/\\*.*?\\*/', ' COMMENT ', code, flags=re.DOTALL)
    for op in sorted(OPERATORS, key=len, reverse=True):
        if op in code:
            code = code.replace(op, f' {op} ')
    raw_tokens = code.split()
    for token in raw_tokens:
        if not token:
            continue
        if token in OPERATORS:
            tokens.append(token)
            continue
        if token in ('STRING_LITERAL', 'COMMENT'):
            tokens.append(token)
            continue
        if re.match('^[0-9]+\\.?[0-9]*$', token):
            tokens.append('NUM_LITERAL')
            continue
        split_camel = re.sub('([a-z])([A-Z])', '\\1 \\2', token)
        split_snake = split_camel.replace('_', ' ')
        subtokens = re.split('[^a-zA-Z0-9]+', split_snake)
        for subtoken in subtokens:
            if not subtoken:
                continue
            if subtoken.isdigit():
                continue
            if not preserve_case:
                subtoken = subtoken.lower()
            if len(subtoken) < 2:
                continue
            tokens.append(subtoken)
    return tokens

def get_file_content_at_commit(repo_dir: str, filepath: str, commit_sha: str) -> Optional[str]:
    cmd = ['git', 'show', f'{commit_sha}:{filepath}']
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout

def get_line_content(file_content: str, line_number: int) -> Optional[str]:
    if not file_content:
        return None
    lines = file_content.splitlines()
    if line_number < 1 or line_number > len(lines):
        return None
    return lines[line_number - 1]

def get_hunk_content(file_content: str, hunk_id: str) -> Optional[str]:
    if not file_content:
        return None
    try:
        if '::' in hunk_id:
            line_part = hunk_id.split('::')[1]
        else:
            line_part = hunk_id
        if '-' in line_part:
            start, end = map(int, line_part.split('-'))
        else:
            start = end = int(line_part)
    except (ValueError, IndexError):
        return None
    lines = file_content.splitlines()
    if start < 1 or end > len(lines):
        return None
    return '\n'.join(lines[start - 1:end])

def parse_unit_id(unit_id: str, granularity: str, row: pd.Series=None) -> Tuple[str, Optional[int], Optional[int]]:
    if granularity == 'file':
        return (unit_id, None, None)
    elif granularity == 'line':
        if '::' in unit_id:
            parts = unit_id.rsplit('::', 1)
            filepath = parts[0]
            try:
                line_num = int(parts[1])
                return (filepath, line_num, line_num)
            except ValueError:
                return (filepath, None, None)
        return (unit_id, None, None)
    elif granularity == 'hunk':
        if '::' in unit_id:
            filepath = unit_id.rsplit('::', 1)[0]
        else:
            filepath = unit_id
        if row is not None:
            try:
                start = int(row.get('hunk_start', 0))
                end = int(row.get('hunk_end', 0))
                if start > 0 and end > 0:
                    return (filepath, start, end)
            except (ValueError, TypeError):
                pass
        return (filepath, None, None)
    return (unit_id, None, None)

def extract_code_for_unit(repo_dir: str, unit_id: str, birth_sha: str, granularity: str, row: pd.Series=None) -> Optional[str]:
    filepath, start_line, end_line = parse_unit_id(unit_id, granularity, row)
    file_content = get_file_content_at_commit(repo_dir, filepath, birth_sha)
    if file_content is None:
        return None
    if granularity == 'file':
        return file_content
    elif granularity == 'line':
        if start_line is None:
            return None
        return get_line_content(file_content, start_line)
    elif granularity == 'hunk':
        if start_line is None or end_line is None:
            return None
        lines = file_content.splitlines()
        if start_line < 1 or end_line > len(lines):
            return None
        return '\n'.join(lines[start_line - 1:end_line])
    return None

def find_merge_commit(repo_dir: str, pr_number: int) -> Optional[str]:
    cmd = ['git', 'log', '--all', '--grep', f'Merge pull request #{pr_number}', '--format=%H', '-n', '1']
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        return result.stdout.strip()
    cmd = ['git', 'log', '--all', '--grep', f'(#{pr_number})', '--format=%H', '-n', '1']
    result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        return result.stdout.strip()
    return None

def process_repo(slug: str, events_df: pd.DataFrame, granularity: str) -> List[Dict]:
    results = []
    temp_dir = tempfile.mkdtemp()
    clone_url = f'https://github.com/{slug}.git'
    try:
        clone_result = subprocess.run(['git', 'clone', '--filter=blob:none', clone_url, temp_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        if clone_result.returncode != 0:
            return results
        has_birth_sha = 'birth_sha' in events_df.columns and events_df['birth_sha'].notna().any()
        if has_birth_sha:
            for birth_sha, group in events_df.groupby('birth_sha'):
                for _, row in group.iterrows():
                    unit_id = row['unit_id']
                    code_content = extract_code_for_unit(temp_dir, unit_id, birth_sha, granularity, row)
                    if code_content is None:
                        code_content = ''
                    tokens = tokenize_code(code_content)
                    results.append({'slug': slug, 'pr_number': row['pr_number'], 'unit_id': unit_id, 'birth_sha': birth_sha, 'code_content': code_content, 'code_tokens': ' '.join(tokens), 'num_tokens': len(tokens)})
        else:
            pr_sha_cache = {}
            for _, row in events_df.iterrows():
                pr_number = row['pr_number']
                unit_id = row['unit_id']
                if pr_number not in pr_sha_cache:
                    pr_sha_cache[pr_number] = find_merge_commit(temp_dir, pr_number)
                birth_sha = pr_sha_cache[pr_number]
                if birth_sha is None:
                    continue
                code_content = extract_code_for_unit(temp_dir, unit_id, birth_sha, granularity, row)
                if code_content is None:
                    code_content = ''
                tokens = tokenize_code(code_content)
                results.append({'slug': slug, 'pr_number': pr_number, 'unit_id': unit_id, 'birth_sha': birth_sha, 'code_content': code_content, 'code_tokens': ' '.join(tokens), 'num_tokens': len(tokens)})
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract code content for survival events')
    parser.add_argument('--granularity', type=str, default='file', choices=['file', 'hunk', 'line'], help='Granularity level (default: file)')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help=f'Maximum parallel workers (default: {MAX_WORKERS})')
    parser.add_argument('--sample', type=int, default=None, help='Sample N repositories for testing (default: all)')
    args = parser.parse_args()
    print('=' * 70)
    print('CODE CONTENT EXTRACTION FOR BOW FEATURES')
    print('=' * 70)
    print(f'\nGranularity: {args.granularity}')
    print(f'Max workers: {args.max_workers}')
    input_file = DATA_DIR / f'survival_events_{args.granularity}_v1.csv'
    output_file = OUTPUT_DIR / f'code_content_{args.granularity}_v1.csv'
    if not input_file.exists():
        print(f'\nERROR: {input_file} not found.')
        print('Run the survival measurement script first.')
        return
    print(f'\nLoading survival events from {input_file}...')
    df_events = pd.read_csv(input_file)
    print(f'  Total events: {len(df_events):,}')
    print(f"  Unique repositories: {df_events['slug'].nunique()}")
    print(f"  Unique PRs: {df_events['pr_number'].nunique()}")
    if 'unit_id' not in df_events.columns and 'file' in df_events.columns:
        df_events['unit_id'] = df_events['file']
        print("  Note: Using 'file' column as 'unit_id'")
    if 'birth_sha' not in df_events.columns:
        print("  Note: 'birth_sha' not found - will find merge commits during extraction")
    required_cols = ['slug', 'pr_number', 'unit_id']
    missing_cols = [c for c in required_cols if c not in df_events.columns]
    if missing_cols:
        print(f'\nERROR: Missing required columns: {missing_cols}')
        print(f'Available columns: {df_events.columns.tolist()}')
        return
    if args.sample:
        sampled_slugs = df_events['slug'].drop_duplicates().sample(n=min(args.sample, df_events['slug'].nunique()), random_state=42)
        df_events = df_events[df_events['slug'].isin(sampled_slugs)]
        print(f'\n  Sampled {args.sample} repositories: {len(df_events):,} events')
    repo_groups = list(df_events.groupby('slug'))
    print(f'\nProcessing {len(repo_groups)} repositories...')
    all_results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_repo, slug, group, args.granularity): slug for slug, group in repo_groups}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting'):
            results = future.result()
            if results:
                all_results.extend(results)
    df_output = pd.DataFrame(all_results)
    df_output.to_csv(output_file, index=False)
    print('\n' + '=' * 70)
    print('EXTRACTION SUMMARY')
    print('=' * 70)
    print(f'\nTotal code units extracted: {len(df_output):,}')
    if len(df_output) > 0:
        token_counts = df_output['num_tokens']
        print(f'\nToken statistics:')
        print(f'  Mean tokens per unit: {token_counts.mean():.1f}')
        print(f'  Median tokens per unit: {token_counts.median():.1f}')
        print(f'  Min tokens: {token_counts.min()}')
        print(f'  Max tokens: {token_counts.max()}')
        empty_count = (df_output['num_tokens'] == 0).sum()
        print(f'\n  Units with no tokens: {empty_count:,} ({100 * empty_count / len(df_output):.1f}%)')
        original_count = len(df_events)
        extracted_count = len(df_output)
        print(f'\nCoverage: {extracted_count:,} / {original_count:,} ({100 * extracted_count / original_count:.1f}%)')
    print(f'\nOutput saved to: {output_file}')
    if len(df_output) > 0:
        print('\n' + '-' * 70)
        print('SAMPLE TOKENS (first 3 units with tokens)')
        print('-' * 70)
        sample = df_output[df_output['num_tokens'] > 0].head(3)
        for _, row in sample.iterrows():
            print(f"\nUnit: {row['unit_id'][:60]}...")
            tokens = row['code_tokens'].split()[:20]
            print(f"Tokens ({row['num_tokens']} total): {' '.join(tokens)}...")
if __name__ == '__main__':
    main()