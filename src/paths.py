"""paths.py: every path of the package, relative to the package root (no absolute paths)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
TABLES = os.environ.get('TABLES_DIR', os.path.join(DATA, 'tables'))  # TABLES_DIR: rebuilt tables (rebuild_data.sh)
COHORT = os.path.join(DATA, 'cohort')
AIDEV = os.path.join(DATA, 'aidev')
AUDIT = os.path.join(DATA, 'audit')
FILE_COV = os.path.join(DATA, 'file_covariates')
RQ2_DATA = os.path.join(DATA, 'rq2')
RQ3_DATA = os.path.join(DATA, 'rq3')
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results'))
SRC = os.path.join(ROOT, 'src')


def out(*parts):
    """Path inside results/; creates the parent directory."""
    p = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def quick():
    """True when reproduce.sh runs with --quick (fewer repetitions and bootstrap replicates)."""
    return os.environ.get('REPRO_QUICK', '0') == '1'
