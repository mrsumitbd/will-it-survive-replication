# Scripts

All paths are relative to the package root (`src/paths.py`). `reproduce.sh` runs the scripts of `src/analysis/` and
`src/rq3/`; `rebuild_data.sh` runs those of `src/mining/`. "Paper" names the element that the output feeds.

## src/paths.py

Package paths (`data/`, `results/`; `RESULTS_DIR` and `TABLES_DIR` override them) and the quick-mode flag.

## src/analysis/ (cohort, audit, RQ1, RQ2, report)

| Script | What it does | Inputs | Outputs (in `results/`) | Paper |
|---|---|---|---|---|
| `funnel.py` | Repository filtering funnel, steps (1) to (5) | `data/aidev/`, `data/cohort/repo_metadata_v1.csv`, `prs_cohort256.csv` | `cohort/filtering_funnel.csv`, `cohort/funnel.json` | Table 1 |
| `paper_stats.py` | Merged-PR attrition, mined cohort per agent, largest PRs, follow-up | `data/cohort/prs_cohort256.csv`, `data/tables/` | `cohort/cohort_stats.json` | Table 1, Sections 2.1.3, 2.2.5 |
| `audit_summary.py` | Audit label counts with Wilson intervals (325 PRs) | `data/audit/audit_rater1.csv` | `audit/audit_summary.json` | Section 2.1.5 |
| `kappa.py` | Cohen's kappa and Gwet's AC1 between the two raters, joined on the PR | `data/audit/audit_rater1.csv`, `data/audit/second_rater/second_rater_annotation_sheet.xlsx` | `audit/kappa.json` | Section 2.1.5 |
| `xlsx_read.py` | Reads an .xlsx sheet with the standard library (used by `kappa.py`) | | | |
| `fbc.py` | Shared RQ1 helpers: loading, primary design (no vendored paths, PR weights), Cox models stratified by repository with PR-clustered SEs, pooled model, repository bootstrap, RMST | | | |
| `rq1_fb.py <unit> [B]` | Primary design and variants (with vendored paths, unweighted, sampled), log-rank, pooled Cox, RMST and modification-rate bootstrap (B = 1,000), per-agent HRs, 14-day rates, unit statistics | `data/tables/` | `rq1/rq1_<unit>.json` | Tables 2, 3, 4; Section 3.3 |
| `a4_ph.py` | Schoenfeld test and time-split HRs (0-14, 14-90, >90 days) | `data/tables/` | `rq1/a4_ph_line.json` | Section 3.3 |
| `a7_sub.py` | Substantial modification (theta = 25 to 100%) for hunks and files | `data/tables/lines.parquet` | `rq1/a7_sub.json` | Section 3.3 |
| `a9_model.py` | Primary model with file covariates (recent changes, age, new file, test path) | `data/tables/`, `data/file_covariates/` | `rq1/a9.json` | Sections 3.3, 6.2 |
| `a_misc.py` | Audit vs. cohort and RQ1 without audited PRs (A8), activity terciles (A10), descriptives (A11), Kaplan-Meier figure (A13) | `data/tables/`, `data/audit/audit_rater1.csv`, `data/cohort/repo_meta_v6.csv` | `rq1/a_misc.json`, `figures/rq1_km_line_primary.{pdf,png}` | Figure 1; Sections 2.1.5, 3.3 |
| `c1_calendar.py` | Calendar-time checks: PRs merged on or after 2025-05-16, strata repository x merge month, time split | `data/tables/` | `rq1/c1_calendar_line.json` | Section 3.3 |
| `rate_14d.py` | 14-day modification rates in further variants | `data/tables/` | `rq1/rate_14d.txt` | (supplementary) |
| `a14_tasktype.py` | RQ1 with the AIDev task type of each PR: task-type mix, strata repository x task type (PR-weighted and unweighted), task-type covariates, feat and fix PRs only | `data/tables/`, `data/aidev/pr_task_type_816.csv.gz` | `rq1/a14_tasktype.json` | Table 3, Section 3.3 |
| `plot_hr_forest.py [<dir>]` | Forest plot of the HRs across the RQ1 design choices | `results/rq1/` | `figures/rq1_hr_forest.{pdf,png}` | Figure 2 |
| `rq2_fb.py [--from-clones <dir>]` | Intent of the modifying commits (whole-word and substring classifiers), chi-square, Cramer's V, residuals, sensitivity analyses | `data/tables/`, `data/rq2/commit_parents.csv` | `rq2/rq2_fb.json` | Table 5, Section 4.3 |
| `a15_rq2_aidev_coverage.py` | Share of the RQ2 modifying commits that belong to an AIDev PR (PR commit or merge commit of a cohort PR) | `data/tables/`, `data/aidev/pr_commit_shas_816.csv.gz` | `rq2/a15_rq2_aidev_coverage.json` | Section 3.2 |
| `ease_src/mine_death_details.py` | Keyword rules of the intent classifier (`CLASSIFICATION_RULES`) and the substring classifier | | | Section 4.2 |
| `ease_src/extract_features.py` | Mapping of the eight fine-grained categories to the five classes (`REASON_MAPPING`) | | | Section 4.2 |
| `make_summary.py [<dir>]` | Long-form summary of every output | `results/` | `SUMMARY.md` | (all) |
| `paper_numbers.py [--results <dir>]` | Compares every number of the paper with the outputs in `<dir>` (default `results/`) | `results/` or `expected_results/` | `PAPER_NUMBERS.md` | (all) |

## src/rq3/ (RQ3; run from `results/rq3/<subset>/`, which holds `data/`)

| Script | What it does | Outputs (in `results/rq3/<subset>/results/`) | Paper |
|---|---|---|---|
| `rq3a_tournament.py` | RQ3a bag-of-words tournament of eight classifiers, 30 x 10 grouped cross-validation, SMOTE in the training folds (`RQ3_N_REPS` sets the repetitions) | `rq31_bow_tournament_{scores,summary}_combined_count.csv` | Section 5.3 |
| `rq3a_sub.py` | Runs `rq3a_tournament.py` with a subset of the models (all files: random forest, CatBoost, XGBoost) | as above | Section 5.3 |
| `rq3a_wide.py` | Scores per fold to wide format for Scott-Knott ESD | `rq3a_wide_*.csv` | |
| `sk_esd.py` | Scott-Knott ESD test (port of the R package, parametric version) | `rq3a_sk_*.csv`, `rq3b_sk_*.csv` | Section 5.3 |
| `d1_lr.py` | Logistic-regression baseline on file size and changes in the previous three months, same cross-validation | `d1_lr_size_history.json` | Section 5.3, Abstract |
| `rq3_explain.py` | Out-of-sample SHAP and LIME of the best model (five repository folds) | `rq3a_explain_summary.json`, `rq3b_explain_summary.json` | Section 5.3 |
| `lime_timed.py` | Runs `rq3_explain.py` and times the LIME explanations | `lime_runtime.json` | Section 5.3 |
| `rq3a_localize.py` | Line localization: scores each added line by the SHAP values of its tokens, compares with line length and random choice (`MAXF` caps the files per fold) | `rq3a_line_localization_summary.json` | Section 5.3 |
| `rq3b_tournament.py` | RQ3b tournament on the process features with AutoSpearman, four time bins, Macro F1 and AUC-ROC | `rq32_process_tournament_{scores,summary}.csv`, `rq32_process_selected_features.json` | Section 5.3 |
| `lime_analysis_rq32_binned_process.py` | LIME for the RQ3b models (used by `rq3_explain.py`) | | |
| `extract_code_contents.py` | Tokenizer of the code (used by `rq3a_localize.py` and `src/mining/extract_rq3_git.py`) | | |

## src/mining/ (data collection; `rebuild_data.sh`)

| Script | What it does | Output |
|---|---|---|
| `download_data.py` | Downloads the AIDev release | `rebuild/aidev/` |
| `aidev_extract.py` | AIDev totals and the PRs of the 816 repositories with agent and human PRs | `data/aidev/` |
| `aidev_task_types.py` | AIDev task types and PR commits of the 816 repositories | `data/aidev/pr_task_type_816.csv.gz`, `data/aidev/pr_commit_shas_816.csv.gz` |
| `filter_repos.py` | Filtering of Xiao et al.: metadata from the GitHub API, license, archived, release, contributors, bottom quartile, code ratio | repository metadata, code ratios, filtered list |
| `code_ratio.py` | Code ratio of one repository with cloc, without timeouts | one JSON line |
| `cohort_prs.py` | Merged PRs of the filtered repositories | `data/cohort/prs_cohort256.csv` |
| `mine_survival.py` | Mines one repository: merge commit, added lines, `git blame --reverse` line events, file events | per-repository parquet files |
| `build_tables.py` | Assembles the line, hunk, file, PR and commit tables; flags vendored paths; excludes the PRs listed in both AIDev tables | `data/tables/` |
| `build_substantial.py` | Substantial-modification tables (earlier version of `a7_sub.py`) | |
| `filter_prs.py` | Copies the tables without a list of PRs (earlier audit sensitivity run) | |
| `sanitize_commits.py` | Removes author names and e-mail addresses from the commit table | `data/tables/commits.parquet` |
| `commit_parents.py` | Parent counts of the modifying commits from bare clones | `data/rq2/commit_parents.csv` |
| `a9_extract.py` | File covariates at merge time from bare clones | `data/file_covariates/` |
| `extract_rq3_git.py` | File content and tokens at the merge commit and the git process features | RQ3 inputs |
| `extract_rq3_pr_features.py` | PR-history features from the GitHub search API (original definition) | `pr_features.jsonl` |
| `extract_rq3_pr_features_fixed.py` | Same with the leakage fix used in the paper (excludes the PR itself; merged before the merge time) | `data/rq3/pr_features/pr_features_fixed.jsonl` |
| `build_rq3_inputs.py` | RQ3a and RQ3b inputs in the classifier format | |
| `d_prep.py` | Removes vendored files from the RQ3 inputs | `data/rq3/rq3a_*` |
| `d_full_prep.py` | RQ3b inputs with the leakage-fixed PR history | `data/rq3/rq3b_*` |
| `funnel_replay.py` | Replays the funnel with the cached files of an earlier run (diagnostic) | `data/cohort/filtering_funnel.csv` |
| `funnel_diagnostics/rebuild.py`, `rows5.py` | Diagnostics of the funnel: repositories that pass step (4) but are not in the cohort, PR counts of the 256 repositories | `data/cohort/q1_not_in_246.csv` |
| `xlsx2csv.py` | Converts one sheet of an .xlsx file to CSV (`audit_rater1.csv`) | |
