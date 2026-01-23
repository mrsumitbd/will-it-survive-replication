# Replication Package: Will It Survive? Deciphering the Fate of AI-Generated Code in Open Source

This repository contains the replication package for the paper "Will It Survive? Deciphering the Fate of AI-Generated Code in Open Source" submitted to EASE 2026.

## Overview

This study presents a survival analysis tracking individual AI-generated code units from birth through modification in open-source repositories. We analyze 201 repositories and over 200,000 code units to investigate:

- **RQ1 (Survival):** Does agent-authored code survive longer than human-authored code?
- **RQ2 (Intent):** When agent-authored code is modified, what is the intent?
- **RQ3a (Localization):** Can we localize modification-prone lines?
- **RQ3b (Temporal):** Can we predict when code will be modified?

## Repository Structure

```
.
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── data/
│   ├── aidev_filtered_v1.csv          # Filtered repository cohort (201 repos)
│   ├── survival_events_line_v1.csv    # Line-level survival events
│   ├── survival_events_file_v1.csv    # File-level survival events
│   ├── death_details_line_v1.csv      # Line-level modification details
│   ├── death_details_file_v1.csv      # File-level modification details
│   ├── code_content_file_v1.csv       # Code content for BOW features
│   ├── process_features_rq32.csv      # Process features for RQ3b
│   ├── prediction_features_v1.csv     # Prediction features
│   └── repo_metadata_v1.csv           # Repository metadata
└── src/
    ├── download_data.py               # Download AIDev dataset
    ├── filter_repos.py                # Repository filtering pipeline
    ├── extract_code_contents.py       # Extract code content
    ├── extract_features.py            # Extract prediction features
    ├── extract_process_features_rq32.py # Extract process features
    ├── measure_retention_line.py      # Line-level survival measurement
    ├── measure_retention_file.py      # File-level survival measurement
    ├── mine_death_details.py          # Extract modification intent
    ├── analyze_rq1.py                 # RQ1: Survival analysis
    ├── analyze_rq1-check_ph_assumption.py # Check proportional hazards
    ├── analyze_rq2.py                 # RQ2: Modification intent analysis
    ├── model_tournament_rq31_bow.py   # RQ3a: BOW-based localization
    ├── model_tournament_rq32_binned_process.py # RQ3b: Temporal prediction
    ├── lime_analysis_rq32_binned_process.py # LIME explanations for RQ3b
    ├── convert_to_wide_format_for_scott-and-knott.py # Prepare Scott-Knott input
    └── analyze_scott_knott.R          # Scott-Knott ESD test
```

## Requirements

### Software Dependencies

- Python 3.10+
- R 4.0+ (for Scott-Knott ESD test)
- Git (for repository cloning and blame analysis)

### Python Environment Setup

We recommend using Conda to manage the environment:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ai-code-survival
```

Alternatively, using pip:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### R Dependencies

Install required R packages:

```r
install.packages(c("ScottKnottESD", "tidyverse"))
```

## Data Description

### Input Data

| File | Description | Rows | Used In |
|------|-------------|------|---------|
| `aidev_filtered_v1.csv` | Filtered cohort of 201 repositories with both agent and human PRs | 201 | All RQs |
| `survival_events_line_v1.csv` | Line-level survival events (birth, death, censoring) | ~210K | RQ1 |
| `survival_events_file_v1.csv` | File-level survival events | ~16K | RQ1 |
| `death_details_line_v1.csv` | Modification intent for each line death | ~129K | RQ2 |
| `death_details_file_v1.csv` | Modification intent for each file death | ~13K | RQ2 |
| `code_content_file_v1.csv` | Source code content for BOW feature extraction | ~15K | RQ3a |
| `process_features_rq32.csv` | Process-level features (commit velocity, file age, etc.) | ~12K | RQ3b |
| `prediction_features_v1.csv` | Combined prediction features | ~15K | RQ3a |
| `repo_metadata_v1.csv` | Repository metadata (stars, contributors, etc.) | 201 | RQ1 |

### Key Variables

**Survival Events:**
- `repository_slug`: Repository identifier (owner/name)
- `pr_number`: Pull request number
- `author_type`: `Agent` or `Human`
- `agent_name`: Specific agent (e.g., `GitHub Copilot`, `Devin`)
- `birth_date`: Date code was merged
- `death_date`: Date code was modified (null if censored)
- `survival_days`: Time from birth to death/censoring
- `is_dead`: Binary indicator (1 = modified, 0 = censored)

**Modification Intent:**
- `intent`: Classification (`Corrective`, `Perfective`, `Adaptive`, `Preventive`, `Other`)
- `commit_message`: Commit message of modifying commit

## Reproduction Steps

### Quick Start (Using Provided Data)

If you want to reproduce the analysis using the provided processed data:

```bash
# Activate environment
conda activate ai-code-survival

# Run RQ1 analysis
python src/analyze_rq1.py

# Run RQ2 analysis
python src/analyze_rq2.py

# Run RQ3a analysis
python src/model_tournament_rq31_bow.py

# Run RQ3b analysis
python src/model_tournament_rq32_binned_process.py

# Run Scott-Knott ESD test (R)
Rscript src/analyze_scott_knott.R
```

### Full Reproduction (From Scratch)

To reproduce the entire pipeline from the original AIDev dataset:

#### Step 1: Download AIDev Dataset

```bash
python src/download_data.py --output data/raw/
```

This downloads the AIDev dataset from the original source.

#### Step 2: Filter Repositories

```bash
python src/filter_repos.py \
    --input data/raw/aidev.parquet \
    --output data/aidev_filtered_v1.csv
```

This applies the filtering criteria described in Section 2.1.2:
- Cohort identification (repos with both agent and human PRs)
- License filter
- Repository state filter
- Statistical distribution filter (Q1 removal)
- Code ratio confidence interval filter

#### Step 3: Measure Survival Events

```bash
# Line-level (primary analysis)
python src/measure_retention_line.py \
    --input data/aidev_filtered_v1.csv \
    --output data/survival_events_line_v1.csv

# File-level (secondary analysis)
python src/measure_retention_file.py \
    --input data/aidev_filtered_v1.csv \
    --output data/survival_events_file_v1.csv
```

**Note:** This step requires cloning repositories and running `git blame`. It may take several hours depending on network speed and disk I/O.

#### Step 4: Extract Modification Details

```bash
python src/mine_death_details.py \
    --survival data/survival_events_line_v1.csv \
    --output data/death_details_line_v1.csv
```

#### Step 5: Extract Features for RQ3

```bash
# Code content for BOW features
python src/extract_code_contents.py \
    --input data/aidev_filtered_v1.csv \
    --output data/code_content_file_v1.csv

# Prediction features
python src/extract_features.py \
    --input data/aidev_filtered_v1.csv \
    --output data/prediction_features_v1.csv

# Process features for RQ3b
python src/extract_process_features_rq32.py \
    --input data/survival_events_file_v1.csv \
    --output data/process_features_rq32.csv
```

#### Step 6: Run Analyses

```bash
# RQ1: Survival Analysis
python src/analyze_rq1.py
python src/analyze_rq1-check_ph_assumption.py

# RQ2: Modification Intent
python src/analyze_rq2.py

# RQ3a: Line Localization (BOW)
python src/model_tournament_rq31_bow.py

# RQ3b: Temporal Prediction
python src/model_tournament_rq32_binned_process.py
python src/lime_analysis_rq32_binned_process.py

# Prepare for Scott-Knott
python src/convert_to_wide_format_for_scott-and-knott.py

# Scott-Knott ESD test
Rscript src/analyze_scott_knott.R
```

## Expected Output

### RQ1: Survival Analysis

Running `analyze_rq1.py` produces:

| Output File | Description | Paper Reference |
|-------------|-------------|-----------------|
| `results/rq1_survival_summary.csv` | Death rates by author type and granularity | Table 3 |
| `results/rq1_cox_results.csv` | Cox regression hazard ratios | Table 4 |
| `results/rq1_agent_analysis.csv` | Survival by agent type | Table 5 |
| `results/rq1_logrank_tests.csv` | Log-rank test results | Section 3.3 |
| `visuals/rq1_kaplan_meier_line.pdf` | Kaplan-Meier curves | Figure 1 |

**Expected Results:**
- Line-level death rate: Agent 53.9%, Human 69.3% (Δ = -15.4pp)
- Hazard Ratio: 0.842 (95% CI: 0.833–0.852, p < 0.001)

### RQ2: Modification Intent

Running `analyze_rq2.py` produces:

| Output File | Description | Paper Reference |
|-------------|-------------|-----------------|
| `results/rq2_reason_summary.csv` | Intent distribution by author type | Table 6 |
| `results/rq2_agent_analysis.csv` | Corrective rate by agent | Table 7 |
| `results/rq2_chi_square_tests.csv` | Chi-square test results | Section 4.3 |
| `results/rq2_standardized_residuals.csv` | Standardized residuals | Section 4.3 |

**Expected Results:**
- Chi-square: χ² = 1739.17, df = 4, p < 0.001
- Effect size: Cramér's V = 0.116
- Corrective rate: Agent 26.3%, Human 23.0% (Δ = +3.3pp)

### RQ3a: Line Localization

Running `model_tournament_rq31_bow.py` produces:

| Output File | Description | Paper Reference |
|-------------|-------------|-----------------|
| `results/rq31_bow_tournament_summary_combined_count.csv` | Model tournament results | Table 8 |
| `results/rq31_lime_explanations_combined_count.csv` | LIME token attributions | Figure 2 |

**Expected Results:**
- Best model: XGBoost
- AUC-ROC: 0.671 (95% CI: 0.663–0.679)
- AUC-PR: 0.903 (95% CI: 0.897–0.910)

### RQ3b: Temporal Prediction

Running `model_tournament_rq32_binned_process.py` produces:

| Output File | Description | Paper Reference |
|-------------|-------------|-----------------|
| `results/rq32_process_tournament_summary.csv` | Model tournament results | Table 9 |
| `results/rq32_process_lime_feature_importance.csv` | Feature importance | Section 5.3 |

**Expected Results:**
- Best model: Logistic Regression
- Macro F1: 0.285 (95% CI: 0.279–0.291)
- AUC-ROC: 0.563 (95% CI: 0.557–0.568)

## Troubleshooting

### Common Issues

**1. Git blame timeout:**
```
Error: Git blame operation timed out
```
Solution: Increase timeout in `measure_retention_line.py` or process repositories in smaller batches.

**2. Memory error during BOW vectorization:**
```
MemoryError: Unable to allocate array
```
Solution: Reduce `max_features` parameter in `model_tournament_rq31_bow.py` or use a machine with more RAM.

**3. R package installation fails:**
```
Error in install.packages: package 'ScottKnottESD' is not available
```
Solution: Install from GitHub:
```r
devtools::install_github("klainfo/ScottKnottESD")
```

### Hardware Requirements

- **Minimum:** 16 GB RAM, 50 GB disk space
- **Recommended:** 32 GB RAM, 100 GB disk space, SSD
- **Estimated runtime:** 
  - Using provided data: ~30 minutes
  - Full reproduction: ~8-12 hours (depending on network and disk speed)

## Citation

If you use this replication package, please cite:

```bibtex
@inproceedings{anonymous2026survival,
  title={Will It Survive? Deciphering the Fate of AI-Generated Code in Open Source},
  author={Anonymous},
  booktitle={Proceedings of the International Conference on Evaluation and Assessment in Software Engineering (EASE)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this replication package, please open an issue in this repository.