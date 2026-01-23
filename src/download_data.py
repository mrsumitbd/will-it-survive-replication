"""
download_data_v1.py

Downloads all required AIDev dataset files for the AI code survival study.

Files downloaded:
- Core PR data (agent + human baseline)
- Repository metadata (for filtering and RQ3 predictors)
- Commit-level data (for tracking code changes)
- Task type classifications (for RQ2 validation)
- User data (developer features)

Usage:
    python src/download_data.py

Output:
    data/*.parquet files

Note: 
    This script requires internet access to HuggingFace.
    Some files are large (pr_commit_details.parquet ~2GB+).
"""

import os
import sys
import requests
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")
HF_BASE_URL = "https://huggingface.co/datasets/hao-li/AIDev/resolve/main"

# All files needed for the survival analysis study
FILES_TO_DOWNLOAD = {
    # ===========================================
    # CORE PR DATA
    # ===========================================
    # All agent-authored PRs (~933K PRs across ~116K repos)
    "all_pull_request.parquet": f"{HF_BASE_URL}/all_pull_request.parquet",
    
    # Human PRs sampled from 500+ star repos (baseline for comparison)
    "human_pull_request.parquet": f"{HF_BASE_URL}/human_pull_request.parquet",
    
    # ===========================================
    # REPOSITORY METADATA
    # ===========================================
    # All repos metadata (license, language, stars, forks)
    "all_repository.parquet": f"{HF_BASE_URL}/all_repository.parquet",
    
    # AIDev-pop subset repos (100+ stars) - enriched metadata
    "repository.parquet": f"{HF_BASE_URL}/repository.parquet",
    
    # ===========================================
    # COMMIT-LEVEL DATA (for tracking code changes)
    # ===========================================
    # Commit metadata linked to PRs (SHAs, timestamps)
    "pr_commits.parquet": f"{HF_BASE_URL}/pr_commits.parquet",
    
    # File-level diffs (patches) - LARGE FILE
    # Note: Large patches not included due to GitHub API limits
    "pr_commit_details.parquet": f"{HF_BASE_URL}/pr_commit_details.parquet",
    
    # ===========================================
    # TASK TYPE CLASSIFICATION (for RQ2)
    # ===========================================
    # LLM-based classification of PR purpose (Conventional Commits categories)
    "pr_task_type.parquet": f"{HF_BASE_URL}/pr_task_type.parquet",
    
    # Human PR task types (for baseline comparison)
    "human_pr_task_type.parquet": f"{HF_BASE_URL}/human_pr_task_type.parquet",
    
    # ===========================================
    # USER DATA (developer features for RQ3)
    # ===========================================
    "all_user.parquet": f"{HF_BASE_URL}/all_user.parquet",
    
    # ===========================================
    # AIDEV-POP ENRICHED DATA (100+ stars subset)
    # ===========================================
    # PRs from 100+ star repos with full metadata
    "pull_request.parquet": f"{HF_BASE_URL}/pull_request.parquet",
    
    # Review comments (use v2, v1 is incomplete per README)
    "pr_review_comments_v2.parquet": f"{HF_BASE_URL}/pr_review_comments_v2.parquet",
    
    # PR timeline events
    "pr_timeline.parquet": f"{HF_BASE_URL}/pr_timeline.parquet",
    
    # Related issues
    "issue.parquet": f"{HF_BASE_URL}/issue.parquet",
    "related_issue.parquet": f"{HF_BASE_URL}/related_issue.parquet",
}

# Optional files (not strictly required but useful)
OPTIONAL_FILES = {
    "pr_comments.parquet": f"{HF_BASE_URL}/pr_comments.parquet",
    "pr_reviews.parquet": f"{HF_BASE_URL}/pr_reviews.parquet",
    "user.parquet": f"{HF_BASE_URL}/user.parquet",
}


def download_file(url: str, dest_path: Path, timeout: int = 300) -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: Source URL
        dest_path: Destination file path
        timeout: Request timeout in seconds
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Downloading {dest_path.name}...")
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(block_size):
                f.write(chunk)
                downloaded += len(chunk)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(
                        f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                    )
                    sys.stdout.flush()
        
        print("\n  ✓ Done")
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"\n  ✗ HTTP Error: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False
        
    except requests.exceptions.Timeout:
        print(f"\n  ✗ Timeout after {timeout}s")
        if dest_path.exists():
            dest_path.unlink()
        return False
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    print("=" * 60)
    print("AIDev Dataset Download Script (v1)")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR.absolute()}")
    print()
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    # Download required files
    print("--- Required Files ---")
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = DATA_DIR / filename
        
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"Skipping {filename} (exists, {size_mb:.1f} MB)")
            skipped.append(filename)
            continue
        
        if download_file(url, dest_path):
            successful.append(filename)
        else:
            failed.append(filename)
    
    # Ask about optional files
    print("\n--- Optional Files ---")
    download_optional = input("Download optional files? (y/N): ").strip().lower() == 'y'
    
    if download_optional:
        for filename, url in OPTIONAL_FILES.items():
            dest_path = DATA_DIR / filename
            
            if dest_path.exists():
                size_mb = dest_path.stat().st_size / (1024 * 1024)
                print(f"Skipping {filename} (exists, {size_mb:.1f} MB)")
                skipped.append(filename)
                continue
            
            if download_file(url, dest_path):
                successful.append(filename)
            else:
                failed.append(filename)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    
    print(f"\n✓ Successfully downloaded: {len(successful)}")
    for f in successful:
        size_mb = (DATA_DIR / f).stat().st_size / (1024 * 1024)
        print(f"  - {f} ({size_mb:.1f} MB)")
    
    print(f"\n○ Skipped (already exist): {len(skipped)}")
    for f in skipped:
        print(f"  - {f}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        print("\nSome files failed to download. Check your internet connection")
        print("or try again later. HuggingFace may have rate limits.")
        sys.exit(1)
    
    # Final inventory
    print("\n" + "=" * 60)
    print("DATA DIRECTORY CONTENTS")
    print("=" * 60)
    total_size = 0
    for f in sorted(DATA_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"\nTotal: {total_size:.1f} MB")


if __name__ == "__main__":
    main()