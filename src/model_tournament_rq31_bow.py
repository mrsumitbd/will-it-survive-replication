"""
model_tournament_rq31_bow_v1.py

RQ3.1: Can we predict whether AI-generated code will be modified?
Approach: BOW features + LIME localization (file-level only)

Methodology:
1. Load code content (tokens) and survival events
2. Build BOW features using CountVectorizer or TfidfVectorizer
3. Run 30x10 RepeatedGroupKFold CV on FULL data (no train/test split)
4. Select best model using AUC-PR (primary metric)
5. Train final model on all data for LIME explanations
6. Apply LIME with stratified sampling (TP, FP, TN, FN) for qualitative analysis

Note: CV provides unbiased performance estimates. No held-out test set needed
since we're not doing model selection based on test performance.

Usage:
    # Combined (all 6 languages), CountVectorizer
    python src/model_tournament_rq31_bow.py --n-reps 30 --n-folds 10 --rebalance smote --vectorizer count
    
    # Python only, TF-IDF
    python src/model_tournament_rq31_bow.py --n-reps 30 --n-folds 10 --rebalance smote --vectorizer tfidf --language py

Input:
    data/code_content_file_v1.csv
    data/survival_events_file_v1.csv

Output (suffix = {language}_{vectorizer}, e.g., "combined_count", "py_tfidf"):
    results/rq31_bow_tournament_scores_{suffix}.csv   - Raw CV scores
    results/rq31_bow_tournament_summary_{suffix}.csv  - Summary statistics
    results/rq31_bow_final_metrics_{suffix}.csv       - Full-data model metrics (reference only)
    results/rq31_lime_explanations_{suffix}.csv       - LIME with line localization
    models/rq31_bow_best_model_{suffix}.joblib
    models/rq31_bow_vectorizer_{suffix}.joblib
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import joblib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ML imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler

# Boosting libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# For SMOTE and Under-sampling (XAI4SE approach for imbalanced data)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# For LIME
from lime.lime_text import LimeTextExplainer

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# CV Configuration
N_REPETITIONS = 30
N_FOLDS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# BOW Configuration
MAX_FEATURES = 1000
MIN_DF = 5
MAX_DF = 0.9

# Supported languages (n >= 376 for 95% CI with ±5% margin of error, for 15990 total files (pop))
# TypeScript includes .ts and .tsx (same language, different file types)
SUPPORTED_LANGUAGES = {'py', 'ts', 'tsx', 'rs', 'cs', 'java', 'go'}
MIN_SAMPLES_THRESHOLD = 376  # Statistical justification for language inclusion

# LIME Configuration
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 1000
MAX_LIME_EXPLANATIONS = 100  # Limit for computational reasons


# Custom tokenizer function (must be module-level for pickling)
def whitespace_tokenizer(text):
    """Simple whitespace tokenizer for pre-tokenized text."""
    return text.split()


# =============================================================================
# REGEX-BASED COMMENT REMOVAL
# Remove comments from source code before tokenization
# Supports: Python, TypeScript/JavaScript, Rust, C#, Java, Go
# =============================================================================

def remove_comments(code: str, extension: str = None) -> str:
    """
    Remove comments from source code using regex.
    
    Supports only the 6 languages with sufficient data (n >= 385):
    - Python (.py)
    - TypeScript (.ts, .tsx)
    - Rust (.rs)
    - C# (.cs)
    - Java (.java)
    - Go (.go)
    
    Args:
        code: Raw source code string or space-joined tokens
        extension: File extension
    
    Returns:
        Code with comments removed
    """
    if not code or not isinstance(code, str):
        return ""
    
    # Normalize extension
    if extension:
        extension = extension.lower().strip('.')
    
    # Python-style comments
    if extension == 'py':
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', ' ', code)
        code = re.sub(r"'''[\s\S]*?'''", ' ', code)
        code = re.sub(r'#.*$', ' ', code, flags=re.MULTILINE)
    
    # C-style comments (TypeScript, Rust, C#, Java, Go)
    elif extension in ['ts', 'tsx', 'rs', 'cs', 'java', 'go']:
        code = re.sub(r'/\*[\s\S]*?\*/', ' ', code)
        code = re.sub(r'//.*$', ' ', code, flags=re.MULTILINE)
    
    return code


def remove_comments_from_tokens(tokens_string: str, extension: str = None) -> str:
    """
    Remove comments from tokenized code.
    
    Process: join tokens -> apply regex comment removal -> split back
    
    Args:
        tokens_string: Space-separated tokens
        extension: File extension for language-specific handling
    
    Returns:
        Space-separated tokens with comments removed
    """
    if not tokens_string or not isinstance(tokens_string, str):
        return ""
    
    # Join tokens into pseudo-code string
    code = tokens_string
    
    # Apply regex-based comment removal
    code = remove_comments(code, extension)
    
    # Clean up whitespace and return
    tokens = code.split()
    return ' '.join(tokens)
    if not code or not isinstance(code, str):
        return ""
    
    # Normalize extension
    if extension:
        extension = extension.lower().strip('.')
    
    # Python-style comments (# and docstrings)
    if extension in ['py']:
        # Remove docstrings first (must be before single-line)
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', ' ', code)
        code = re.sub(r"'''[\s\S]*?'''", ' ', code)
        # Remove single-line comments
        code = re.sub(r'#.*$', ' ', code, flags=re.MULTILINE)
    
    # C-style comments (// and /* */)
    # Applies to: TypeScript, JavaScript, Rust, C#, Java, Go, C, C++
    elif extension in ['ts', 'tsx', 'js', 'jsx', 'rs', 'cs', 'java', 'go', 'c', 'cpp', 'cc', 'h', 'hpp', 'kt', 'scala', 'swift']:
        # Remove multi-line comments first
        code = re.sub(r'/\*[\s\S]*?\*/', ' ', code)
        # Remove single-line comments
        code = re.sub(r'//.*$', ' ', code, flags=re.MULTILINE)
    
    # If extension unknown, try both styles
    else:
        # Python-style
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', ' ', code)
        code = re.sub(r"'''[\s\S]*?'''", ' ', code)
        code = re.sub(r'#.*$', ' ', code, flags=re.MULTILINE)
        # C-style
        code = re.sub(r'/\*[\s\S]*?\*/', ' ', code)
        code = re.sub(r'//.*$', ' ', code, flags=re.MULTILINE)
    
    return code


# =============================================================================
# SYNTAX TOKEN FILTER
# Based on "Natural Software Revisited" (Rahman, Palani, Rigby - ICSE 2019)
# 
# Key insight: "Separators account for 44% of all tokens... We suggest that
# SyntaxTokens be filtered to reduce noise in code recommenders."
#
# We filter: operators, punctuation, keywords, and placeholders
# We keep: API calls, identifiers, meaningful names
# =============================================================================

# Operators (all languages)
SYNTAX_OPERATORS = {
    # Arithmetic
    '+', '-', '*', '/', '%', '**', '//',
    # Comparison
    '==', '!=', '<', '>', '<=', '>=', '<>', '===', '!==',
    # Assignment
    '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=',
    '&=', '|=', '^=', '>>=', '<<=',
    # Logical
    '&&', '||', '!', 'and', 'or', 'not',
    # Bitwise
    '&', '|', '^', '~', '<<', '>>',
    # Other
    '++', '--', '->', '=>', '::', '...', '?.', '??',
    '@', '#', '$', '\\',
}

# Punctuation / Separators
SYNTAX_PUNCTUATION = {
    '(', ')', '[', ']', '{', '}',
    ',', ';', ':', '.', '?', '!',
    '"', "'", '`', 
}

# Keywords for the 6 supported languages only
SYNTAX_KEYWORDS = {
    # Python
    'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
    'finally', 'with', 'import', 'from', 'return', 'yield', 'raise', 'pass',
    'break', 'continue', 'lambda', 'in', 'is', 'none', 'true', 'false',
    'async', 'await', 'global', 'nonlocal', 'assert', 'del', 'as',
    
    # TypeScript (includes JavaScript keywords)
    'function', 'var', 'let', 'const', 'new', 'this', 'typeof', 'instanceof',
    'delete', 'void', 'null', 'undefined', 'of', 'export', 'default',
    'extends', 'implements', 'interface', 'type', 'enum', 'declare',
    'module', 'namespace', 'abstract', 'readonly', 'keyof', 'infer',
    
    # Rust
    'fn', 'let', 'mut', 'pub', 'crate', 'mod', 'use', 'impl', 'trait',
    'where', 'move', 'ref', 'self', 'unsafe', 'dyn', 'box', 'match',
    'loop', 'extern', 'macro', 'macro_rules',
    
    # C#
    'public', 'private', 'protected', 'static', 'final', 'abstract',
    'internal', 'sealed', 'override', 'partial', 'event', 'delegate',
    'checked', 'unchecked', 'fixed', 'lock', 'ref', 'out', 'params',
    'stackalloc', 'base', 'get', 'set', 'value', 'where', 'select',
    'orderby', 'ascending', 'descending', 'group', 'into', 'join',
    'virtual', 'using', 'namespace',
    
    # Java
    'public', 'private', 'protected', 'static', 'final', 'abstract',
    'synchronized', 'volatile', 'transient', 'native', 'strictfp',
    'throws', 'throw', 'extends', 'implements', 'super', 'package',
    
    # Go
    'func', 'chan', 'defer', 'fallthrough', 'range', 'map', 'iota',
    'append', 'cap', 'close', 'copy', 'len', 'make', 'panic', 'print',
    'println', 'recover', 'nil', 'package', 'import', 'struct', 'interface',
    'select', 'case', 'default', 'go', 'goto', 'const', 'type', 'var',
    
    # Common primitives (6 languages)
    'int', 'integer', 'long', 'short', 'byte', 'float', 'double', 'decimal',
    'char', 'string', 'bool', 'boolean', 'void', 'object', 'any', 'never',
    'number', 'symbol', 'bigint',
}

# Our tokenizer placeholders
SYNTAX_PLACEHOLDERS = {
    'string_literal', 'num_literal', 'comment',
}

# Comment indicators to remove (multi-language)
COMMENT_INDICATORS = {
    '#', '//', '/*', '*/', '"""', "'''", '<!--', '-->',
    'todo', 'fixme', 'xxx', 'hack', 'note', 'bug',
}

# Combine all syntax tokens
ALL_SYNTAX_TOKENS = (
    SYNTAX_OPERATORS | 
    SYNTAX_PUNCTUATION | 
    SYNTAX_KEYWORDS | 
    SYNTAX_PLACEHOLDERS |
    COMMENT_INDICATORS
)


def filter_syntax_tokens(tokens_string: str, min_length: int = 3) -> str:
    """
    Filter out syntax tokens, keeping only API-like identifiers.
    
    Based on "Natural Software Revisited" (Rahman et al., ICSE 2019):
    "We suggest that SyntaxTokens be filtered to reduce noise in code recommenders."
    
    Note: Comments should be removed BEFORE tokenization using remove_comments().
    
    Args:
        tokens_string: Space-separated tokens
        min_length: Minimum token length to keep (default 3)
    
    Returns:
        Filtered space-separated tokens
    """
    if not tokens_string or not isinstance(tokens_string, str):
        return ""
    
    tokens = tokens_string.split()
    filtered = []
    
    for token in tokens:
        token_lower = token.lower()
        
        # Skip syntax tokens
        if token_lower in ALL_SYNTAX_TOKENS:
            continue
        
        # Skip short tokens (likely syntax remnants)
        if len(token) < min_length:
            continue
        
        # Skip pure numbers
        if token.isdigit():
            continue
        
        filtered.append(token)
    
    return ' '.join(filtered)


def extract_extension(filepath: str) -> str:
    """Extract file extension from filepath."""
    if not filepath or not isinstance(filepath, str):
        return 'unknown'
    parts = filepath.rsplit('.', 1)
    if len(parts) == 2:
        return parts[1].lower()
    return 'unknown'


def get_models() -> Dict:
    """
    Get dictionary of models to evaluate.
    
    8 classifiers representing fundamental learning paradigms:
    - Linear & Probabilistic: LR, NB
    - Distance & Kernel: KNN, SVM
    - Ensemble Trees: RF, XGB, CB
    - Neural Networks: MLP
    """
    return {
        # Linear & Probabilistic
        'LogisticRegression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'NaiveBayes': MultinomialNB(
            alpha=1.0  # Laplace smoothing
        ),
        
        # Distance & Kernel
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        # LinearSVC is much faster than SVC(kernel='rbf') for large datasets
        # Wrapped in CalibratedClassifierCV to get probability estimates for AUC
        'SVM': CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                max_iter=2000,
                dual='auto'
            ),
            cv=3
        ),
        
        # Ensemble Trees
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        ),
        'CatBoost': CatBoostClassifier(
            n_estimators=100,
            max_depth=5,
            auto_class_weights='Balanced',
            random_state=RANDOM_STATE,
            verbose=False
        ),
        
        # Neural Networks
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        )
    }


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost based on class imbalance."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    return n_neg / n_pos if n_pos > 0 else 1.0


def load_and_prepare_data(language_filter: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load code content and survival events, join them.
    
    Args:
        language_filter: If provided, only include files with this extension (e.g., 'py', 'java')
    
    Returns:
        df: Merged DataFrame
        X_tokens: Array of token strings
        y: Target array (event)
        groups: Group array (repository indices)
    """
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load code content
    code_file = DATA_DIR / "code_content_file_v1.csv"
    if not code_file.exists():
        raise FileNotFoundError(f"{code_file} not found. Run extract_code_content_v1.py first.")
    
    df_code = pd.read_csv(code_file)
    print(f"\nCode content: {len(df_code):,} rows")
    
    # Load survival events
    events_file = DATA_DIR / "survival_events_file_v1.csv"
    if not events_file.exists():
        raise FileNotFoundError(f"{events_file} not found.")
    
    df_events = pd.read_csv(events_file)
    print(f"Survival events: {len(df_events):,} rows")
    
    # Standardize column names for merge
    if 'unit_id' not in df_events.columns and 'file' in df_events.columns:
        df_events['unit_id'] = df_events['file']
    
    # Merge on slug, pr_number, unit_id
    df = pd.merge(
        df_code,
        df_events[['slug', 'pr_number', 'unit_id', 'event', 'duration_days', 'is_agent']],
        on=['slug', 'pr_number', 'unit_id'],
        how='inner'
    )
    print(f"Merged data: {len(df):,} rows")
    
    # Extract file extension for language analysis
    df['extension'] = df['unit_id'].apply(extract_extension)
    
    # Filter to supported languages only (6 languages with n >= 385)
    before_lang_filter = len(df)
    df = df[df['extension'].isin(SUPPORTED_LANGUAGES)].reset_index(drop=True)
    print(f"\nFiltered to supported languages: {sorted(SUPPORTED_LANGUAGES)}")
    print(f"  Before: {before_lang_filter:,} rows")
    print(f"  After: {len(df):,} rows")
    print(f"  Removed: {before_lang_filter - len(df):,} rows from unsupported languages")
    
    # Additional language filtering (if specific language requested)
    if language_filter:
        language_filter = language_filter.lower().strip('.')
        if language_filter not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language .{language_filter} not in supported languages: {SUPPORTED_LANGUAGES}")
        before_filter = len(df)
        df = df[df['extension'] == language_filter].reset_index(drop=True)
        print(f"\nLanguage filter: .{language_filter}")
        print(f"  Before: {before_filter:,} rows")
        print(f"  After: {len(df):,} rows")
        
        if len(df) == 0:
            raise ValueError(f"No files found with extension .{language_filter}")
    else:
        print(f"\nLanguage distribution:")
        lang_counts = df['extension'].value_counts()
        for ext, count in lang_counts.items():
            print(f"  .{ext}: {count:,} ({100*count/len(df):.1f}%)")
    
    # Remove rows with no tokens (can't predict)
    df = df[df['num_tokens'] > 0].reset_index(drop=True)
    print(f"After removing empty tokens: {len(df):,} rows")
    
    # Step 1: Remove comments using regex (join -> regex -> split)
    print("\nRemoving comments...")
    df['code_tokens_no_comments'] = df.apply(
        lambda row: remove_comments_from_tokens(row['code_tokens'], row['extension']),
        axis=1
    )
    
    # Step 2: Filter syntax tokens (Natural Software Revisited - Rahman et al. ICSE 2019)
    print("Filtering syntax tokens (keeping API/identifiers only)...")
    df['code_tokens_filtered'] = df['code_tokens_no_comments'].apply(filter_syntax_tokens)
    df['num_tokens_filtered'] = df['code_tokens_filtered'].apply(lambda x: len(x.split()) if x else 0)
    
    # Stats on filtering
    original_tokens = df['num_tokens'].sum()
    filtered_tokens = df['num_tokens_filtered'].sum()
    reduction_pct = 100 * (1 - filtered_tokens / original_tokens) if original_tokens > 0 else 0
    print(f"  Original tokens: {original_tokens:,}")
    print(f"  After filtering: {filtered_tokens:,}")
    print(f"  Reduction: {reduction_pct:.1f}%")
    
    # Remove rows with no tokens after filtering
    df = df[df['num_tokens_filtered'] > 0].reset_index(drop=True)
    print(f"  Rows with tokens after filtering: {len(df):,}")
    
    # Use filtered tokens
    X_tokens = df['code_tokens_filtered'].values
    y = df['event'].values
    
    # Create group indices for repositories
    df['group_idx'] = pd.factorize(df['slug'])[0]
    groups = df['group_idx'].values
    
    # Summary
    print(f"\nTarget distribution:")
    print(f"  Modified (event=1): {np.sum(y == 1):,} ({100 * np.mean(y):.1f}%)")
    print(f"  Survived (event=0): {np.sum(y == 0):,} ({100 * (1 - np.mean(y)):.1f}%)")
    print(f"\nRepositories: {df['slug'].nunique()}")
    print(f"PRs: {df['pr_number'].nunique()}")
    
    return df, X_tokens, y, groups


def run_single_fold(
    X_train_tokens: np.ndarray,
    X_val_tokens: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    models: Dict,
    fold_idx: int,
    rebalance_strategy: str = "none",
    vectorizer_type: str = "tfidf"
) -> List[Dict]:
    """
    Run a single CV fold: fit vectorizer, train models, evaluate.
    
    Args:
        rebalance_strategy: "none", "smote", or "under"
        vectorizer_type: "count" or "tfidf"
    
    Returns:
        List of score dictionaries with all metrics for CI computation
    """
    import time
    from sklearn.metrics import matthews_corrcoef
    
    results = []
    
    # Fit vectorizer on training data only (no leakage)
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=(1, 3),
            tokenizer=whitespace_tokenizer,
            lowercase=False
        )
    else:
        vectorizer = CountVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=(1, 3),
            tokenizer=whitespace_tokenizer,
            lowercase=False
        )
    
    X_train_bow = vectorizer.fit_transform(X_train_tokens)
    X_val_bow = vectorizer.transform(X_val_tokens)
    
    # Apply class rebalancing (XAI4SE approach)
    if rebalance_strategy == "smote":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_bow_resampled, y_train_resampled = smote.fit_resample(
            X_train_bow.toarray(), y_train
        )
        from scipy.sparse import csr_matrix
        X_train_bow_sparse = csr_matrix(X_train_bow_resampled)
        X_train_dense = X_train_bow_resampled
    elif rebalance_strategy == "under":
        undersample = RandomUnderSampler(random_state=RANDOM_STATE)
        X_train_bow_resampled, y_train_resampled = undersample.fit_resample(
            X_train_bow, y_train
        )
        X_train_bow_sparse = X_train_bow_resampled
        X_train_dense = X_train_bow_resampled.toarray()
    else:
        X_train_bow_sparse = X_train_bow
        X_train_bow_resampled = X_train_bow.toarray()
        y_train_resampled = y_train
        X_train_dense = X_train_bow_resampled
    
    X_val_dense = X_val_bow.toarray()
    
    # Compute class weight for XGBoost
    scale_pos_weight = compute_scale_pos_weight(y_train)
    
    for name, model in models.items():
        try:
            start_time = time.time()
            
            # Clone model to avoid state leakage
            if name == 'XGBoost':
                model_clone = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    eval_metric='logloss',
                    verbosity=0
                )
            else:
                from sklearn.base import clone
                model_clone = clone(model)
            
            # Select input format based on model
            if name in ['NaiveBayes']:
                X_train_input = X_train_bow_sparse
                X_val_input = X_val_bow
            elif name in ['KNN', 'SVM', 'MLP']:
                X_train_input = X_train_dense
                X_val_input = X_val_dense
            else:
                X_train_input = X_train_bow_sparse
                X_val_input = X_val_bow
            
            # Train
            model_clone.fit(X_train_input, y_train_resampled)
            
            # Predict probabilities
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_val_input)[:, 1]
            else:
                y_prob = model_clone.decision_function(X_val_input)
            
            elapsed = time.time() - start_time
            
            # Binary predictions for classification metrics
            y_pred = (y_prob >= 0.5).astype(int)
            
            # Compute all metrics for paper reporting with CI
            auc_roc = roc_auc_score(y_val, y_prob)
            auc_pr = average_precision_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            
            results.append({
                'fold': fold_idx,
                'model': name,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'f1': f1,
                'mcc': mcc,
                'precision': precision,
                'recall': recall,
                'time_seconds': elapsed
            })
            
        except Exception as e:
            print(f"    Warning: {name} failed in fold {fold_idx}: {str(e)[:50]}")
            results.append({
                'fold': fold_idx,
                'model': name,
                'auc_roc': np.nan,
                'auc_pr': np.nan,
                'f1': np.nan,
                'mcc': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'time_seconds': np.nan
            })
    
    return results


def run_tournament(
    X_tokens: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_repetitions: int = N_REPETITIONS,
    n_folds: int = N_FOLDS,
    rebalance_strategy: str = "none",
    vectorizer_type: str = "tfidf"
) -> pd.DataFrame:
    """
    Run the full model tournament with repeated GroupKFold CV.
    
    Returns:
        DataFrame with all scores
    """
    from tqdm import tqdm
    
    print("\n" + "=" * 70)
    print("MODEL TOURNAMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Repetitions: {n_repetitions}")
    print(f"  Folds: {n_folds}")
    print(f"  Total evaluations per model: {n_repetitions * n_folds}")
    print(f"  Vectorizer: {vectorizer_type.upper()}")
    print(f"  max_features: {MAX_FEATURES}")
    print(f"  min_df: {MIN_DF}, max_df: {MAX_DF}")
    print(f"  Class rebalancing: {rebalance_strategy}")
    
    models = get_models()
    print(f"\nModels: {list(models.keys())}")
    
    all_results = []
    total_folds = n_repetitions * n_folds
    
    with tqdm(total=total_folds, desc="Tournament Progress") as pbar:
        for rep in range(n_repetitions):
            # Different random state for each repetition
            gkf = GroupKFold(n_splits=n_folds)
            
            # Shuffle groups for this repetition
            np.random.seed(RANDOM_STATE + rep)
            unique_groups = np.unique(groups)
            shuffled_group_map = {g: i for i, g in enumerate(np.random.permutation(unique_groups))}
            shuffled_groups = np.array([shuffled_group_map[g] for g in groups])
            
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_tokens, y, shuffled_groups)):
                X_train_tokens = X_tokens[train_idx]
                X_val_tokens = X_tokens[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                fold_results = run_single_fold(
                    X_train_tokens, X_val_tokens,
                    y_train, y_val,
                    models, fold_idx,
                    rebalance_strategy,
                    vectorizer_type
                )
                
                for res in fold_results:
                    res['repetition'] = rep
                
                all_results.extend(fold_results)
                pbar.update(1)
                pbar.set_postfix({'rep': f"{rep+1}/{n_repetitions}", 'fold': f"{fold_idx+1}/{n_folds}"})
    
    print("\n")
    return pd.DataFrame(all_results)


def compute_summary_statistics(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics with 95% CI for all metrics.
    
    Metrics reported (for paper):
    - AUC-PR (primary metric for ranking - better for imbalanced data)
    - AUC-ROC
    - F1
    - MCC
    - Precision
    - Recall
    
    All metrics include mean, std, and 95% CI from percentile bootstrap.
    """
    summary = []
    
    metrics = ['auc_pr', 'auc_roc', 'f1', 'mcc', 'precision', 'recall']
    
    for model in scores_df['model'].unique():
        model_scores = scores_df[scores_df['model'] == model]
        
        row = {'model': model}
        
        for metric in metrics:
            scores = model_scores[metric].dropna()
            
            if len(scores) > 0:
                row[f'{metric}_mean'] = scores.mean()
                row[f'{metric}_std'] = scores.std()
                row[f'{metric}_ci_lower'] = np.percentile(scores, 2.5)
                row[f'{metric}_ci_upper'] = np.percentile(scores, 97.5)
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_ci_lower'] = np.nan
                row[f'{metric}_ci_upper'] = np.nan
        
        row['n_scores'] = len(model_scores['auc_pr'].dropna())
        summary.append(row)
    
    summary_df = pd.DataFrame(summary)
    
    # Rank by AUC-PR (primary metric, descending)
    summary_df = summary_df.sort_values('auc_pr_mean', ascending=False)
    summary_df['rank'] = range(1, len(summary_df) + 1)
    
    return summary_df


def train_final_model(
    X_tokens: np.ndarray,
    y: np.ndarray,
    best_model_name: str,
    rebalance_strategy: str = "none",
    vectorizer_type: str = "tfidf"
) -> Tuple[Dict, object, object, np.ndarray]:
    """
    Train best model on full dataset for LIME explanations.
    
    Returns:
        metrics: Dictionary of training metrics (for reference)
        model: Trained model
        vectorizer: Fitted vectorizer
        y_prob: Predicted probabilities for all samples
    """
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON FULL DATA")
    print("=" * 70)
    
    # Fit vectorizer on all data
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=(1, 3),
            tokenizer=whitespace_tokenizer,
            lowercase=False
        )
    else:
        vectorizer = CountVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=(1, 3),
            tokenizer=whitespace_tokenizer,
            lowercase=False
        )
    
    X_bow = vectorizer.fit_transform(X_tokens)
    
    # NOTE: For final model, we use CLASS WEIGHTS instead of SMOTE
    # SMOTE on full data can cause the model to predict everything as one class
    # Class weights give more realistic probability estimates for LIME analysis
    
    print(f"\nVectorizer: {vectorizer_type.upper()}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Samples: {X_bow.shape[0]:,}")
    print(f"NOTE: Using class weights (not {rebalance_strategy}) for final model to get realistic predictions")
    
    # Compute class weights
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    weight_neg = len(y) / (2 * n_neg)
    weight_pos = len(y) / (2 * n_pos)
    
    print(f"Class weights: survived={weight_neg:.3f}, modified={weight_pos:.3f}")
    
    # Get model with class weights
    if best_model_name == 'XGBoost':
        scale_pos_weight = n_neg / n_pos  # XGBoost uses scale_pos_weight
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
    elif best_model_name == 'CatBoost':
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            class_weights={0: weight_neg, 1: weight_pos},
            random_state=RANDOM_STATE,
            verbose=False
        )
    elif best_model_name == 'LogisticRegression':
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif best_model_name == 'SVM':
        model = SVC(
            kernel='linear',
            probability=True,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
    elif best_model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif best_model_name == 'MLP':
        # MLP doesn't support class_weight, so we'll use sample_weight
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=RANDOM_STATE
        )
    elif best_model_name == 'KNN':
        # KNN doesn't support class weights
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    else:
        # NaiveBayes - use priors
        model = MultinomialNB()
    
    # Prepare input format
    if best_model_name in ['KNN', 'SVM', 'MLP']:
        X_train_input = X_bow.toarray()
    else:
        X_train_input = X_bow
    
    print(f"\nTraining {best_model_name} on full data (with class weights)...")
    
    # For MLP, compute sample weights
    if best_model_name == 'MLP':
        sample_weights = np.where(y == 0, weight_neg, weight_pos)
        # MLP doesn't support sample_weight in fit, train normally
        model.fit(X_train_input, y)
    else:
        model.fit(X_train_input, y)
    
    # Get predictions
    if best_model_name in ['KNN', 'SVM', 'MLP']:
        X_predict = X_bow.toarray()
    else:
        X_predict = X_bow
    
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_predict)[:, 1]
    else:
        decision = model.decision_function(X_predict)
        y_prob = 1 / (1 + np.exp(-decision))
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Compute metrics on full data (for reference only - CV metrics are primary)
    metrics = {
        'model': best_model_name,
        'vectorizer': vectorizer_type,
        'rebalance': f'{rebalance_strategy}_cv_only',  # Note: class weights used for final
        'n_samples': len(y),
        'auc_roc': roc_auc_score(y, y_prob),
        'auc_pr': average_precision_score(y, y_prob),
        'f1': f1_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'n_predicted_modified': int(np.sum(y_pred)),
        'n_predicted_survived': int(np.sum(y_pred == 0)),
    }
    
    print(f"\nFull Data Metrics (for reference - CV metrics are primary):")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Predicted: {metrics['n_predicted_modified']:,} modified, {metrics['n_predicted_survived']:,} survived")
    
    return metrics, model, vectorizer, y_prob


def localize_tokens_to_lines(code_content: str, important_tokens: set) -> Dict:
    """
    Map important tokens to actual line numbers in original code.
    
    Args:
        code_content: Original raw code (with newlines)
        important_tokens: Set of tokens identified as important by LIME
    
    Returns:
        Dict with line numbers and token occurrences
    """
    import re
    
    if not code_content or pd.isna(code_content):
        return {
            'lines_with_important_tokens': [],
            'line_numbers': '',
            'token_line_map': '',
            'num_lines_affected': 0,
            'total_lines': 0,
            'line_coverage': 0
        }
    
    lines = code_content.split('\n')
    important_tokens_lower = set(t.lower() for t in important_tokens)
    
    lines_with_tokens = []
    token_line_map = {}  # token -> list of line numbers
    
    for line_num, line in enumerate(lines, start=1):
        # Tokenize line by splitting on non-alphanumeric
        line_tokens = set(re.findall(r'\b\w+\b', line.lower()))
        
        # Check which important tokens appear in this line
        found_tokens = line_tokens & important_tokens_lower
        
        if found_tokens:
            lines_with_tokens.append({
                'line_num': line_num,
                'tokens_found': list(found_tokens),
                'line_preview': line.strip()[:100]
            })
            
            # Track which lines each token appears on
            for tok in found_tokens:
                if tok not in token_line_map:
                    token_line_map[tok] = []
                token_line_map[tok].append(line_num)
    
    # Summarize
    line_numbers = sorted(set(lt['line_num'] for lt in lines_with_tokens))
    
    return {
        'lines_with_important_tokens': lines_with_tokens,
        'line_numbers': ','.join(map(str, line_numbers[:30])),  # First 30 line numbers
        'token_line_map': str({k: v[:5] for k, v in list(token_line_map.items())[:10]}),
        'num_lines_affected': len(line_numbers),
        'total_lines': len(lines),
        'line_coverage': round(len(line_numbers) / len(lines), 4) if lines else 0
    }


def apply_lime_explanations(
    model,
    vectorizer,
    df: pd.DataFrame,
    X_tokens: np.ndarray,
    y: np.ndarray,
    y_prob: np.ndarray,
    max_explanations: int = MAX_LIME_EXPLANATIONS
) -> pd.DataFrame:
    """
    Apply LIME to explain predictions on full dataset.
    
    Enhanced for qualitative analysis with LINE-LEVEL LOCALIZATION:
    - Token weights for interpretability  
    - Maps important tokens to actual line numbers in original code
    - Samples stratified by prediction type (TP, FP, TN, FN)
    - Includes line previews for manual inspection
    
    Returns:
        DataFrame with LIME explanations and line localization
    """
    print("\n" + "=" * 70)
    print("LIME EXPLANATIONS")
    print("=" * 70)
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(
        class_names=['survived', 'modified'],
        split_expression=r'\s+',
        bow=True
    )
    
    # Create prediction function for LIME
    def predict_proba_fn(texts):
        """Prediction function for LIME."""
        X_bow = vectorizer.transform(texts)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_bow)
        else:
            decision = model.decision_function(X_bow)
            prob_pos = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - prob_pos, prob_pos])
    
    # Categorize predictions: TP, FP, TN, FN
    y_pred = (y_prob >= 0.5).astype(int)
    
    tp_idx = np.where((y == 1) & (y_pred == 1))[0]  # True Positive
    fp_idx = np.where((y == 0) & (y_pred == 1))[0]  # False Positive
    tn_idx = np.where((y == 0) & (y_pred == 0))[0]  # True Negative
    fn_idx = np.where((y == 1) & (y_pred == 0))[0]  # False Negative
    
    print(f"\nPrediction breakdown:")
    print(f"  True Positives (correctly predicted modified): {len(tp_idx):,}")
    print(f"  False Positives (incorrectly predicted modified): {len(fp_idx):,}")
    print(f"  True Negatives (correctly predicted survived): {len(tn_idx):,}")
    print(f"  False Negatives (incorrectly predicted survived): {len(fn_idx):,}")
    
    # Stratified sampling: get examples from each category
    samples_per_category = max_explanations // 4
    np.random.seed(RANDOM_STATE)
    
    selected_indices = []
    categories = []
    
    for cat_name, cat_idx in [('TP', tp_idx), ('FP', fp_idx), ('TN', tn_idx), ('FN', fn_idx)]:
        if len(cat_idx) > 0:
            n_samples = min(samples_per_category, len(cat_idx))
            sampled = np.random.choice(cat_idx, n_samples, replace=False)
            selected_indices.extend(sampled)
            categories.extend([cat_name] * n_samples)
    
    print(f"\nGenerating LIME explanations for {len(selected_indices)} files (stratified)...")
    
    explanations = []
    
    for i, (idx, category) in enumerate(zip(selected_indices, categories)):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(selected_indices)}...")
        
        try:
            # Get the filtered token string (used for prediction)
            token_string = X_tokens[idx]
            
            # Get file info including original code
            file_info = df.iloc[idx]
            original_code = file_info.get('code_content', '')
            
            # Get explanation
            exp = explainer.explain_instance(
                token_string,
                predict_proba_fn,
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES
            )
            
            # Extract top tokens with weights
            token_weights = exp.as_list()
            
            # Separate positive (push toward modified) and negative tokens
            positive_tokens = [(t, w) for t, w in token_weights if w > 0]
            negative_tokens = [(t, w) for t, w in token_weights if w < 0]
            
            # --- LINE LOCALIZATION using original code ---
            important_token_set = set([t for t, w in token_weights])
            
            # Map important tokens to actual line numbers
            line_info = localize_tokens_to_lines(original_code, important_token_set)
            
            # Also compute position stats in filtered token stream
            filtered_tokens = token_string.split()
            total_filtered = len(filtered_tokens)
            
            # Token concentration in filtered stream
            important_in_filtered = sum(1 for t in filtered_tokens if t.lower() in set(t.lower() for t in important_token_set))
            token_concentration = important_in_filtered / total_filtered if total_filtered > 0 else 0
            
            # Store detailed explanation with line localization
            explanations.append({
                'slug': file_info['slug'],
                'pr_number': file_info['pr_number'],
                'unit_id': file_info['unit_id'],
                'extension': extract_extension(file_info['unit_id']),
                'prediction_category': category,
                'actual_label': int(y[idx]),
                'predicted_prob': float(y_prob[idx]),
                'predicted_label': int(y_pred[idx]),
                # Token stats
                'total_tokens': total_filtered,
                'important_token_count': len(token_weights),
                'token_concentration': round(token_concentration, 4),
                # Line localization (from original code)
                'total_lines': line_info['total_lines'],
                'num_lines_affected': line_info['num_lines_affected'],
                'line_coverage': line_info['line_coverage'],
                'affected_line_numbers': line_info['line_numbers'],
                'token_line_map': line_info['token_line_map'],
                # Token weights
                'num_positive_tokens': len(positive_tokens),
                'num_negative_tokens': len(negative_tokens),
                'top_positive_tokens': ','.join([f"{t}:{w:.4f}" for t, w in sorted(positive_tokens, key=lambda x: -x[1])[:5]]),
                'top_negative_tokens': ','.join([f"{t}:{w:.4f}" for t, w in sorted(negative_tokens, key=lambda x: x[1])[:5]]),
                'all_positive_tokens': ','.join([t for t, w in positive_tokens]),
                'all_negative_tokens': ','.join([t for t, w in negative_tokens]),
                'all_token_weights': str(token_weights),
                # Line previews for qualitative analysis (first 5 affected lines)
                'line_previews': str([
                    f"L{lt['line_num']}: {lt['line_preview']}" 
                    for lt in line_info['lines_with_important_tokens'][:5]
                ]),
            })
            
        except Exception as e:
            print(f"  Warning: LIME failed for index {idx}: {str(e)[:50]}")
    
    print(f"\nGenerated {len(explanations)} explanations")
    
    # Summary by category
    if explanations:
        exp_df = pd.DataFrame(explanations)
        print("\nExplanations by category:")
        for cat in ['TP', 'FP', 'TN', 'FN']:
            cat_count = len(exp_df[exp_df['prediction_category'] == cat])
            if cat_count > 0:
                avg_prob = exp_df[exp_df['prediction_category'] == cat]['predicted_prob'].mean()
                avg_lines = exp_df[exp_df['prediction_category'] == cat]['num_lines_affected'].mean()
                print(f"  {cat}: {cat_count} samples, avg prob: {avg_prob:.3f}, avg lines affected: {avg_lines:.1f}")
    
    return pd.DataFrame(explanations)


def main():
    parser = argparse.ArgumentParser(
        description="RQ3.1 BOW Model Tournament with LIME Localization"
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=N_REPETITIONS,
        help=f"Number of CV repetitions (default: {N_REPETITIONS})"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help=f"Number of CV folds (default: {N_FOLDS})"
    )
    parser.add_argument(
        "--skip-lime",
        action="store_true",
        help="Skip LIME explanations"
    )
    parser.add_argument(
        "--max-lime",
        type=int,
        default=MAX_LIME_EXPLANATIONS,
        help=f"Maximum LIME explanations (default: {MAX_LIME_EXPLANATIONS})"
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="none",
        choices=["none", "smote", "under"],
        help="Class rebalancing strategy: none, smote, or under (default: none)"
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default="tfidf",
        choices=["count", "tfidf"],
        help="Vectorizer type: count (CountVectorizer) or tfidf (TfidfVectorizer). Default: tfidf"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter by file extension (e.g., 'py', 'java', 'js'). Default: all languages"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("RQ3.1: WILL AI-GENERATED CODE BE MODIFIED?")
    print("Approach: BOW Features + LIME Localization (File-Level)")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df, X_tokens, y, groups = load_and_prepare_data(language_filter=args.language)
    
    # Run tournament on FULL data (no train/test split)
    # CV provides unbiased performance estimates via grouped cross-validation
    scores_df = run_tournament(
        X_tokens, y, groups,
        n_repetitions=args.n_reps,
        n_folds=args.n_folds,
        rebalance_strategy=args.rebalance,
        vectorizer_type=args.vectorizer
    )
    
    # Build filename suffix: {language}_{vectorizer}
    # e.g., "combined_count", "py_tfidf", "go_count"
    lang_suffix = args.language if args.language else "combined"
    file_suffix = f"{lang_suffix}_{args.vectorizer}"
    
    # Save raw scores
    scores_file = RESULTS_DIR / f"rq31_bow_tournament_scores_{file_suffix}.csv"
    scores_df.to_csv(scores_file, index=False)
    print(f"\nScores saved to: {scores_file}")
    
    # Print timing summary
    if 'time_seconds' in scores_df.columns:
        print("\n" + "-" * 70)
        print("MODEL TIMING (avg seconds per fold)")
        print("-" * 70)
        timing = scores_df.groupby('model')['time_seconds'].mean().sort_values()
        for model, avg_time in timing.items():
            print(f"  {model:20s}: {avg_time:6.2f}s")
    
    # Compute summary
    summary_df = compute_summary_statistics(scores_df)
    
    print("\n" + "=" * 70)
    print("TOURNAMENT RESULTS (CV)")
    print("=" * 70)
    print("\nModel Rankings by AUC-PR (primary metric):")
    print("-" * 70)
    
    for _, row in summary_df.iterrows():
        print(f"  {row['rank']}. {row['model']:20s} "
              f"AUC-PR: {row['auc_pr_mean']:.4f} [{row['auc_pr_ci_lower']:.4f} - {row['auc_pr_ci_upper']:.4f}] "
              f"AUC-ROC: {row['auc_roc_mean']:.4f}")
    
    # Save summary
    summary_file = RESULTS_DIR / f"rq31_bow_tournament_summary_{file_suffix}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # Get best model (ranked by AUC-PR)
    best_model_name = summary_df.iloc[0]['model']
    print(f"\nBest model: {best_model_name}")
    
    # Train final model on FULL data (for LIME explanations)
    final_metrics, model, vectorizer, y_prob = train_final_model(
        X_tokens, y,
        best_model_name,
        rebalance_strategy=args.rebalance,
        vectorizer_type=args.vectorizer
    )
    
    # Save final model metrics (for reference - CV metrics are primary)
    metrics_file = RESULTS_DIR / f"rq31_bow_final_metrics_{file_suffix}.csv"
    pd.DataFrame([final_metrics]).to_csv(metrics_file, index=False)
    print(f"\nFinal model metrics saved to: {metrics_file}")
    
    # Save model and vectorizer
    model_file = MODELS_DIR / f"rq31_bow_best_model_{file_suffix}.joblib"
    vectorizer_file = MODELS_DIR / f"rq31_bow_vectorizer_{file_suffix}.joblib"
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print(f"Model saved to: {model_file}")
    print(f"Vectorizer saved to: {vectorizer_file}")
    
    # LIME explanations on full dataset
    if not args.skip_lime:
        lime_df = apply_lime_explanations(
            model, vectorizer, df,
            X_tokens, y, y_prob,
            max_explanations=args.max_lime
        )
        
        # Save LIME explanations
        lime_file = RESULTS_DIR / f"rq31_lime_explanations_{file_suffix}.csv"
        lime_df.to_csv(lime_file, index=False)
        print(f"\nLIME explanations saved to: {lime_file}")
        
        # Show sample explanations by category with line info
        if len(lime_df) > 0:
            print("\n" + "-" * 70)
            print("SAMPLE LIME EXPLANATIONS (one per category)")
            print("-" * 70)
            
            for cat in ['TP', 'FP', 'TN', 'FN']:
                cat_samples = lime_df[lime_df['prediction_category'] == cat]
                if len(cat_samples) > 0:
                    row = cat_samples.iloc[0]
                    print(f"\n[{cat}] File: {row['unit_id'][:55]}...")
                    print(f"  Actual: {'Modified' if row['actual_label'] == 1 else 'Survived'}, "
                          f"Predicted: {'Modified' if row['predicted_label'] == 1 else 'Survived'} "
                          f"(prob: {row['predicted_prob']:.3f})")
                    print(f"  Lines: {row['num_lines_affected']}/{row['total_lines']} affected "
                          f"({row['line_coverage']:.1%} coverage)")
                    print(f"  Affected lines: {row['affected_line_numbers'][:50]}...")
                    print(f"  Top positive: {row['top_positive_tokens'][:55]}...")
    
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Best model (CV): {best_model_name}")
    print(f"Vectorizer: {args.vectorizer.upper()}")
    print(f"Language: {lang_suffix}")
    print(f"Samples: {len(y):,}")
    print(f"CV AUC-PR (primary): {summary_df.iloc[0]['auc_pr_mean']:.4f} "
          f"[{summary_df.iloc[0]['auc_pr_ci_lower']:.4f} - {summary_df.iloc[0]['auc_pr_ci_upper']:.4f}]")
    print(f"CV AUC-ROC (secondary): {summary_df.iloc[0]['auc_roc_mean']:.4f} "
          f"[{summary_df.iloc[0]['auc_roc_ci_lower']:.4f} - {summary_df.iloc[0]['auc_roc_ci_upper']:.4f}]")


if __name__ == "__main__":
    main()