"""download_data.py [--optional]: download the AIDev release (Hugging Face, hao-li/AIDev) into rebuild/aidev/
(REBUILD_DIR/aidev). --optional also downloads the optional tables. Needs network access and the requests package
(pip install requests). Only rebuild_data.sh needs it."""
import os
import sys
import requests
from pathlib import Path
DATA_DIR = Path(os.environ.get('REBUILD_DIR', 'rebuild')) / 'aidev'
HF_BASE_URL = 'https://huggingface.co/datasets/hao-li/AIDev/resolve/main'
FILES_TO_DOWNLOAD = {'all_pull_request.parquet': f'{HF_BASE_URL}/all_pull_request.parquet', 'human_pull_request.parquet': f'{HF_BASE_URL}/human_pull_request.parquet', 'all_repository.parquet': f'{HF_BASE_URL}/all_repository.parquet', 'repository.parquet': f'{HF_BASE_URL}/repository.parquet', 'pr_commits.parquet': f'{HF_BASE_URL}/pr_commits.parquet', 'pr_commit_details.parquet': f'{HF_BASE_URL}/pr_commit_details.parquet', 'pr_task_type.parquet': f'{HF_BASE_URL}/pr_task_type.parquet', 'human_pr_task_type.parquet': f'{HF_BASE_URL}/human_pr_task_type.parquet', 'all_user.parquet': f'{HF_BASE_URL}/all_user.parquet', 'pull_request.parquet': f'{HF_BASE_URL}/pull_request.parquet', 'pr_review_comments_v2.parquet': f'{HF_BASE_URL}/pr_review_comments_v2.parquet', 'pr_timeline.parquet': f'{HF_BASE_URL}/pr_timeline.parquet', 'issue.parquet': f'{HF_BASE_URL}/issue.parquet', 'related_issue.parquet': f'{HF_BASE_URL}/related_issue.parquet'}
OPTIONAL_FILES = {'pr_comments.parquet': f'{HF_BASE_URL}/pr_comments.parquet', 'pr_reviews.parquet': f'{HF_BASE_URL}/pr_reviews.parquet', 'user.parquet': f'{HF_BASE_URL}/user.parquet'}

def download_file(url: str, dest_path: Path, timeout: int=300) -> bool:
    print(f'Downloading {dest_path.name}...')
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f'\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
                    sys.stdout.flush()
        print('\n  ✓ Done')
        return True
    except requests.exceptions.HTTPError as e:
        print(f'\n  ✗ HTTP Error: {e}')
        if dest_path.exists():
            dest_path.unlink()
        return False
    except requests.exceptions.Timeout:
        print(f'\n  ✗ Timeout after {timeout}s')
        if dest_path.exists():
            dest_path.unlink()
        return False
    except Exception as e:
        print(f'\n  ✗ Error: {e}')
        if dest_path.exists():
            dest_path.unlink()
        return False

def main():
    print('=' * 60)
    print('AIDev Dataset Download Script (v1)')
    print('=' * 60)
    print(f'Target directory: {DATA_DIR.absolute()}')
    print()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    successful = []
    failed = []
    skipped = []
    print('--- Required Files ---')
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = DATA_DIR / filename
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f'Skipping {filename} (exists, {size_mb:.1f} MB)')
            skipped.append(filename)
            continue
        if download_file(url, dest_path):
            successful.append(filename)
        else:
            failed.append(filename)
    print('\n--- Optional Files ---')
    download_optional = '--optional' in sys.argv
    if download_optional:
        for filename, url in OPTIONAL_FILES.items():
            dest_path = DATA_DIR / filename
            if dest_path.exists():
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                print(f'Skipping {filename} (exists, {size_mb:.1f} MB)')
                skipped.append(filename)
                continue
            if download_file(url, dest_path):
                successful.append(filename)
            else:
                failed.append(filename)
    print('\n' + '=' * 60)
    print('DOWNLOAD SUMMARY')
    print('=' * 60)
    print(f'\n✓ Successfully downloaded: {len(successful)}')
    for f in successful:
        size_mb = (DATA_DIR / f).stat().st_size / (1024 * 1024)
        print(f'  - {f} ({size_mb:.1f} MB)')
    print(f'\n○ Skipped (already exist): {len(skipped)}')
    for f in skipped:
        print(f'  - {f}')
    if failed:
        print(f'\n✗ Failed: {len(failed)}')
        for f in failed:
            print(f'  - {f}')
        print('\nSome files failed to download. Check your internet connection')
        print('or try again later. HuggingFace may have rate limits.')
        sys.exit(1)
    print('\n' + '=' * 60)
    print('DATA DIRECTORY CONTENTS')
    print('=' * 60)
    total_size = 0
    for f in sorted(DATA_DIR.glob('*.parquet')):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f'  {f.name}: {size_mb:.1f} MB')
    print(f'\nTotal: {total_size:.1f} MB')
if __name__ == '__main__':
    main()