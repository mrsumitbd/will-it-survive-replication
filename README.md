# Replication package: "Will It Survive? Deciphering the Fate of AI Agent-Authored Code in Open Source"

This package contains the data, the analysis scripts and the outputs behind the paper. It follows the lines that
2,794 merged pull requests (PRs) from the AIDev dataset added to 215 open-source repositories, compares the survival
of agent-attributed and human code (RQ1), the intent of the commits that modify it (RQ2), and the prediction of whether
and when it will be modified (RQ3).

One command runs every analysis from the shipped data and writes a report that maps each number of the paper to the
reproduced value:

```bash
bash reproduce.sh
```

## Contents

| Path | What it is |
|---|---|
| `reproduce.sh` | Runs all analyses from `data/` into `results/`, then writes `results/PAPER_NUMBERS.md` |
| `rebuild_data.sh` | Rebuilds `data/` from the AIDev release and GitHub (network, several days); not needed to reproduce the paper |
| `requirements.txt` | Pinned Python packages (Python 3.11) |
| `data/` | All inputs of the analyses (see `data/README.md`) |
| `src/analysis/` | Cohort, audit, RQ1 and RQ2 analyses, and the paper-number report (see `src/README.md`) |
| `src/rq3/` | RQ3 models: tournaments, Scott-Knott ESD, baselines, SHAP and LIME, line localization |
| `src/mining/` | Data collection and preparation, used only by `rebuild_data.sh` |
| `expected_results/` | The outputs of the run behind the paper, in the same layout as `results/` |
| `results/` | Created by `reproduce.sh` (not versioned) |
| `LICENSE` | MIT license |

## Requirements

* macOS or Linux with bash, gzip and Python 3.11. The paper's results were produced with Python 3.11.8 on macOS (Apple silicon).
* About 3 GB of free disk space (the RQ3 inputs are unpacked into `results/`) and 8 GB of memory.
* No network access is needed after the packages are installed. No GPU is needed.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash reproduce.sh --quick      # smoke test of every script (see "Runtimes")
bash reproduce.sh              # the paper's settings
```

Open `results/PAPER_NUMBERS.md` for the comparison with the paper and `results/figures/rq1_km_line_primary.pdf` for Figure 1 (`rq1_hr_forest.pdf` for Figure 2).

To compare the shipped outputs of the paper run with the paper without running anything:

```bash
python src/analysis/paper_numbers.py --results expected_results
```

This writes `expected_results/PAPER_NUMBERS.md`.

## Step by step

`reproduce.sh` runs the steps below in this order. `bash reproduce.sh --list` lists them, and
`bash reproduce.sh --only <step>` runs a single one. Each step writes its log to `results/logs/<step>.log` and prints
its runtime. The script stops at the first failing step and shows the end of its log. A finished step is skipped when
the script is run again, unless `--force` is given or the step last ran in the other mode (quick or full).

| Step | Script | Output in `results/` | Paper |
|---|---|---|---|
| cohort_funnel | `src/analysis/funnel.py` | `cohort/filtering_funnel.csv`, `cohort/funnel.json` | Table 1, steps (1) to (5) |
| cohort_stats | `src/analysis/paper_stats.py` | `cohort/cohort_stats.json` | Table 1 (merged PRs), Sections 2.1.3 and 2.2 |
| audit_summary | `src/analysis/audit_summary.py` | `audit/audit_summary.json` | Section 2.1.5 |
| audit_kappa | `src/analysis/kappa.py` | `audit/kappa.json` | Section 2.1.5 (Cohen's kappa) |
| rq1_line, rq1_hunk, rq1_file | `src/analysis/rq1_fb.py` | `rq1/rq1_{line,hunk,file}.json` | Tables 2, 3 and 4; Section 3.3 |
| rq1_ph | `src/analysis/a4_ph.py` | `rq1/a4_ph_line.json` | Section 3.3 (proportional hazards, time split) |
| rq1_substantial | `src/analysis/a7_sub.py` | `rq1/a7_sub.json` | Section 3.3 (substantial modification) |
| rq1_file_history | `src/analysis/a9_model.py` | `rq1/a9.json` | Section 3.3 (file covariates), Section 6.2 |
| rq1_misc_figure | `src/analysis/a_misc.py` | `rq1/a_misc.json`, `figures/rq1_km_line_primary.pdf` | Figure 1; Section 2.1.5 (cohort audit); Section 3.3 |
| rq1_calendar | `src/analysis/c1_calendar.py` | `rq1/c1_calendar_line.json` | Section 3.3 (calendar time) |
| rq1_rate_14d | `src/analysis/rate_14d.py` | `rq1/rate_14d.txt` | 14-day rates (additional variants) |
| rq1_task_type | `src/analysis/a14_tasktype.py` | `rq1/a14_tasktype.json` | Table 3 (repository x task type), Section 3.3 (task type) |
| rq1_hr_figure | `src/analysis/plot_hr_forest.py` | `figures/rq1_hr_forest.pdf` | Figure 2 |
| rq2 | `src/analysis/rq2_fb.py` | `rq2/rq2_fb.json` | Table 5, Section 4.3 |
| rq2_aidev_coverage | `src/analysis/a15_rq2_aidev_coverage.py` | `rq2/a15_rq2_aidev_coverage.json` | Section 3.2 (share of modifying commits in AIDev PRs) |
| rq3_unpack | (shell) | `rq3/*/data/` | inputs of RQ3 |
| rq3a_agent_* | `src/rq3/rq3a_tournament.py`, `rq3a_wide.py`, `sk_esd.py`, `d1_lr.py`, `lime_timed.py`, `rq3a_localize.py` | `rq3/rq3a_agent/results/` | Section 5.3 (RQ3a, line localization) |
| rq3b_agent_* | `src/rq3/rq3b_tournament.py`, `sk_esd.py`, `lime_timed.py` | `rq3/rq3b_agent/results/` | Section 5.3 (RQ3b) |
| rq3b_all_* | `src/rq3/rq3b_tournament.py`, `sk_esd.py` | `rq3/rq3b_all/results/` | Section 5.3 (RQ3b, all files) |
| rq3a_all_* | `src/rq3/rq3a_sub.py`, `rq3a_wide.py`, `sk_esd.py`, `d1_lr.py`, `rq3a_localize.py` | `rq3/rq3a_all/results/` | Section 5.3 (RQ3a, all files) |
| summary | `src/analysis/make_summary.py` | `SUMMARY.md` | long-form summary of every output |
| paper_numbers | `src/analysis/paper_numbers.py` | `PAPER_NUMBERS.md` | every number of the paper |

The default mode uses the paper's settings: 30 repetitions of 10-fold cross-validation grouped by repository,
1,000 bootstrap replicates, all files for the line localization and fixed seeds. `--quick` fits the RQ1 models on a fixed random quarter of the PRs with
50 bootstrap replicates, and runs RQ3 with one repetition of 10-fold cross-validation and at most 40 files per fold for
the line localization. It checks that every script runs, and its numbers are not expected to match the paper. The report header and the console say so.

## Paper-to-output map

`results/PAPER_NUMBERS.md` lists every empirical number of the paper, section by section and table by table, with
the paper's value, the reproduced value, and the status. The main sources are:

| Paper element | Output |
|---|---|
| Abstract, Introduction | collected from the outputs below |
| Table 1 (funnel) | `cohort/funnel.json` (steps 1 to 5), `cohort/cohort_stats.json` (merged PRs) |
| Section 2.1.3 (cohort) | `cohort/cohort_stats.json` |
| Section 2.1.5 (audit) | `audit/audit_summary.json` (325 PRs), `rq1/a_misc.json` key `A8` (221 PRs in the cohort) |
| Table 2 (units and events) | `rq1/rq1_{line,hunk,file}.json` key `A12` |
| Table 3 (designs) | `rq1/rq1_{line,hunk,file}.json` keys `A1_*` |
| Figure 1 | `figures/rq1_km_line_primary.pdf`, `rq1/a_misc.json` key `A13` |
| Figure 2 | `figures/rq1_hr_forest.pdf` (from `rq1/rq1_line.json`, `rq1/a4_ph_line.json`, `rq1/c1_calendar_line.json`, `rq1/a14_tasktype.json`) |
| Section 3.3 (RQ1 text) | `rq1/rq1_line.json` (`A2` pooled, `A3` RMST and bootstrap, `A6` 14-day rates), `rq1/a4_ph_line.json`, `rq1/c1_calendar_line.json`, `rq1/a7_sub.json`, `rq1/a9.json`, `rq1/a_misc.json` (`A8`, `A10`, `A11`), `rq1/a14_tasktype.json` (task type), `rq2/a15_rq2_aidev_coverage.json` |
| Table 4 (per agent) | `rq1/rq1_line.json` key `A5` |
| Table 5 and Section 4.3 (RQ2) | `rq2/rq2_fb.json` |
| Table 6 and Section 5.3 (RQ3a) | `rq3/rq3a_{agent,all}/results/` (`rq31_bow_tournament_summary_combined_count.csv`, `rq3a_sk_*.csv`, `d1_lr_size_history.json`, `rq3a_explain_summary.json`, `rq3a_line_localization_summary.json`, `lime_runtime.json`) |
| Table 6 and Section 5.3 (RQ3b) | `rq3/rq3b_{agent,all}/results/` (`rq32_process_tournament_summary.csv`, `rq3b_sk_*.csv`, `rq32_process_selected_features.json`, `rq3b_explain_summary.json`) |

On the shipped `expected_results/`, the report gives 648 of 651 numbers matched (206 of 207 statements):

* One statement is partial. Step (5) of Table 1 (code ratio) is reproduced from the cohort list, because the
  per-repository code ratios of the final filtering run are not part of the package (see `data/README.md`).
* Numbers from the AIDev release, the literature and the design (for example the observation window) are listed as
  external. The AIDev numbers are checked against the shipped AIDev summary.
* Cohen's kappa of the attribution audit (0.85) is computed by `src/analysis/kappa.py` from the first rater's labels
  (`data/audit/audit_rater1.csv`) and the second rater's labels
  (`data/audit/second_rater/second_rater_annotation_sheet.xlsx`), joined on the PR.

## Runtimes

Measured on a laptop with an Apple M2 processor (8 cores) and 24 GB of memory.

| Run | Time |
|---|---|
| `pip install -r requirements.txt` | about 1 minute |
| `bash reproduce.sh --quick` | about 30 minutes (26 minutes on the machine above; the first start of Python takes a few extra minutes) |
| `bash reproduce.sh` (paper settings) | about 9 hours: RQ1 about 4 hours (the line-level Cox models take most of it), RQ3 about 5 hours (RQ3a agent 75 minutes, RQ3a all files 85 minutes, RQ3b agent 25 minutes, RQ3b all files 110 minutes) |
| `python src/analysis/paper_numbers.py --results expected_results` | a few seconds |

The full-run times are those of the run behind the paper. The RQ3 models use all processor cores.

## Reproducibility notes

* Every random step has a fixed seed (42 for the cross-validation, SMOTE, the models and the explanation folds; 1 and
  5 for the bootstraps). With the pinned packages, a full run reproduces `expected_results/`. The LIME runtime per
  explanation depends on the machine.
* The RQ3 tournaments write a checkpoint per fold (`results/rq3/*/results/ckpt*.jsonl`) and resume from it after an
  interruption. `reproduce.sh` removes the checkpoint when it reruns a tournament step.
* The commit table (`data/tables/commits.parquet`) ships without author names and e-mail addresses, and personal
  e-mail addresses in commit messages are replaced by `<email>`. `src/mining/sanitize_commits.py` checks that this
  leaves every RQ2 intent label unchanged.

## Full data rebuild

`rebuild_data.sh` documents and runs the pipeline that produced `data/`, step by step. It needs network access, `git`,
`cloc`, the `requests` package, an authenticated GitHub CLI (`gh`, for the search API) and a GitHub token for the
metadata step, about 100 GB of disk and several days. It writes everything to `rebuild/` and never overwrites `data/`.

| Step | Script | Output in `rebuild/` |
|---|---|---|
| download | `src/mining/download_data.py` | `aidev/*.parquet` (AIDev release, Hugging Face `hao-li/AIDev`) |
| aidev_extract | `src/mining/aidev_extract.py` | `aidev_extract/` (same files as `data/aidev/`) |
| aidev_task_types | `src/mining/aidev_task_types.py` | `aidev_extract/pr_task_type_816.csv.gz`, `aidev_extract/pr_commit_shas_816.csv.gz` |
| filter | `src/mining/filter_repos.py` (with `code_ratio.py` for repositories that time out) | `repo_metadata_v1.csv`, `code_ratio_v1.csv`, `aidev_filtered_v1.csv` |
| cohort_prs | `src/mining/cohort_prs.py` | `prs_cohort256.csv` |
| clone | `git clone --bare` | `clones/<owner>__<name>.git` |
| mine | `src/mining/mine_survival.py` (`git blame --reverse` from each merge commit to the last commit on or before December 31, 2025) | `mine/` |
| tables | `src/mining/build_tables.py` | `tables/{lines,hunks,files,prs,commits}.parquet` |
| sanitize | `src/mining/sanitize_commits.py` | `tables/commits.parquet` |
| parents | `src/mining/commit_parents.py` | `commit_parents.csv` |
| file_covariates | `src/mining/a9_extract.py` | `file_covariates/` |
| rq3_git | `src/mining/extract_rq3_git.py` | `rq3_inputs/` (file content, tokens, git process features) |
| rq3_pr_features | `src/mining/extract_rq3_pr_features.py`, `extract_rq3_pr_features_fixed.py` | `rq3_inputs/pr_features*.jsonl` |
| rq3_inputs | `src/mining/build_rq3_inputs.py` | `rq3_256/` |
| rq3_prep | `src/mining/d_prep.py`, `d_full_prep.py` | `rq3/`, `rq3full/` (vendored files removed, PR-history leakage fix) |
| compare | (inline) | prints whether each rebuilt file equals the shipped one |

Repositories change over time, so a rebuild will not give exactly the shipped data. Two cohort repositories already
could not be mined: `dust-tt/dust` is no longer public, and `elizaos/eliza` rewrote its history after the observation
window. The repository metadata in `data/cohort/repo_metadata_v1.csv` was collected in January 2026, and a later
collection can change the filtering funnel (for example, a repository whose license or release status changed).

## License

The code is released under the MIT license (`LICENSE`). The data derived from AIDev keeps the license of the AIDev
dataset.
