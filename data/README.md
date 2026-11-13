# Data

Every input of `reproduce.sh`. `rebuild_data.sh` rebuilds these files from the AIDev release and GitHub
(see the main README, section "Full data rebuild"). Slugs are `owner/name` in lower case. Times are UTC.

## aidev/ (Table 1, steps up to (4); RQ1 task type)

| File | Content | Produced by |
|---|---|---|
| `aidev_summary.json` | Totals of the full AIDev release: agent PRs (932,791) per agent, repositories by `repo_id` (116,211) and by URL (116,423), human PRs (6,618) and their repositories (818), the 816 repositories with agent and human PRs | `src/mining/aidev_extract.py` |
| `aidev_prs_816.csv.gz` | All 16,395 PRs (agent and human table, merged or not) of the 816 repositories with agent and human PRs. Columns: `table` (agent or human), `id`, `slug`, `number`, `agent` (agent name, or Human), `user_hash` (SHA-1 prefix of the GitHub login, used only to count distinct authors), `created_at`, `merged_at` | `src/mining/aidev_extract.py` |
| `pr_task_type_816.csv.gz` | The conventional-commit task type that AIDev assigns to each PR from its title and description (AIDev files `pr_task_type.parquet` and `human_pr_task_type.parquet`), for the PRs of `aidev_prs_816.csv.gz`. Columns: `id`, `type`. Used by the RQ1 task-type analysis | `src/mining/aidev_task_types.py` |
| `pr_commit_shas_816.csv.gz` | The commits of the AIDev agent PRs of the 816 repositories (AIDev file `pr_commits.parquet`, without messages and authors). Columns: `pr_id`, `sha`. Used to count the RQ2 modifying commits that belong to an AIDev PR | `src/mining/aidev_task_types.py` |

## cohort/

| File | Content | Produced by |
|---|---|---|
| `prs_cohort256.csv` | The 4,688 merged PRs (2,302 agent, 2,386 human) of the 246 filtered repositories that have a merged PR. Columns: `slug`, `number`, `merged_at`, `author_type` (Agent or Human), `agent_name`, `pr_id` (AIDev id). Five PRs appear twice (once in each AIDev table); `build_tables.py` excludes them | `src/mining/cohort_prs.py` |
| `repo_metadata_v1.csv` | GitHub metadata of 813 of the 816 candidate repositories (three had no metadata), collected in January 2026 (stars, license, archived, release, size, open issues, language). This is the version used for the paper's funnel | `src/mining/filter_repos.py` |
| `repo_meta_v6.csv` | Metadata of the 246 cohort repositories with merged PRs, from the final filtering run, including contributors, AIDev PR counts and code ratio. Used for the stars, language and contributors covariates and the activity terciles (RQ1) | `src/mining/filter_repos.py` |
| `code_ratio_v1.csv` | Code ratios (cloc) of an earlier filtering run (214 repositories). Kept for transparency; the per-repository code ratios of the final run are not available, see below | `src/mining/filter_repos.py`, `code_ratio.py` |
| `q1_not_in_246.csv` | The repositories that pass step (4) but are not among the 246, with their PR counts (from a rebuild with the newer metadata, which has one extra repository, felix3322/potplayer_chatgpt_translate) | `src/mining/funnel_diagnostics/rebuild.py` |
| `filtering_funnel.csv` | Funnel of the earlier replay with `code_ratio_v1.csv` (diagnostic; its last two rows differ from Table 1) | `src/mining/funnel_replay.py` |

Step (5) of Table 1 (code ratio within the 97% interval of the language) cannot be recomputed from the shipped
files, because the code ratios of the 266 repositories of the final run are not available. `src/analysis/funnel.py`
therefore reproduces the row from the cohort: the 256 repositories are the 246 repositories of `prs_cohort256.csv`
plus the 10 step-(4) repositories without any merged PR. The report marks this row as partial.

## tables/ (RQ1, RQ2)

Survival tables of the mined cohort (`src/mining/mine_survival.py`, then `build_tables.py`). The observation window
ends on 2025-12-31 23:59:59 UTC.

`lines.parquet`: one row per line added by a merged PR to a tracked source file (854,503 rows; 854,139 with a
resolved event).

| Column | Meaning |
|---|---|
| `slug`, `pr_number`, `file`, `line`, `hunk` | PR, file path, line number in the merge commit, diff hunk index |
| `birth_sha`, `birth_date` | merge commit of the PR on the default branch and its time |
| `event` | 1 = modified or deleted by the cutoff, 0 = censored, empty = no resolved event (`kind` = `death_commit_not_found`, 364 lines) |
| `kind` | `died`, `censored` or `death_commit_not_found` |
| `last_seen_sha`, `death_sha`, `death_date` | last commit in which the line still exists, the modifying commit and its time |
| `blank` | the line is blank |
| `duration_days` | days from the merge to the modification or to the cutoff |
| `vendored` | the path is vendored or generated (regular expression in `build_tables.py`; excluded from the primary design) |
| `is_agent`, `author_type`, `agent_name`, `group` | attribution of the PR; `group` is Supervised assistants (Copilot, Codex, Cursor, Claude Code), Devin (autonomous) or Human |
| `n_added_lines`, `n_source_files` | lines added and source files changed by the PR (Cox covariates) |
| `stars`, `contributors`, `language` | repository covariates (`repo_meta_v6.csv`) |

`hunks.parquet`: one row per hunk (62,529). A hunk is modified at the earliest modification of any of its lines
(`n_lines`, `n_dead`, `first_death`, plus the PR columns above).

`files.parquet`: one row per source file changed by a PR (18,308). `unit_id` is the path, `file_added_lines` the lines
the PR added to it. A file is modified by the first later commit that touches it (`death_sha`, `death_date`).

`prs.parquet`: one row per merged PR of the 244 mined repositories (4,607). `status` is `ok` when the merge commit was
located (4,248) or `merge_not_found` (359); `match_method` says how; `base_sha` is the base of the PR.

`commits.parquet`: the 8,896 commits that modify a tracked unit (`sha`, `message`). Author names and e-mail addresses
were removed and personal e-mail addresses in messages replaced by `<email>` (`src/mining/sanitize_commits.py`; the
RQ2 intent labels are unchanged).

## rq2/

`commit_parents.csv`: parent count of each of the 8,893 commits that modify a line or a file (`slug`, `sha`,
`n_parents`), from bare clones of the repositories with `git rev-list --no-walk --parents`
(`src/mining/commit_parents.py`). A commit with more than one parent is a merge commit (RQ2 sensitivity).
`src/analysis/rq2_fb.py --from-clones <dir>` recomputes the counts from local clones instead.

## file_covariates/ (RQ1 sensitivity, RQ3a baseline)

One parquet file per repository (`<owner>__<name>.parquet`), one row per file unit, from the bare clones
(`src/mining/a9_extract.py`): `slug`, `pr_number`, `file`, `new_file` (the path does not exist at the PR base),
`prior_mod_3m` (commits reachable from the PR base that touch the path in the 90 days before the merge),
`file_age_days` (days since the first commit that added the path; 0 for new files).

## rq3/

| Path | Content |
|---|---|
| `rq3a_agent/`, `rq3a_all/` | RQ3a inputs, agent files (7,954) and all files (18,120), without vendored files. `survival_events_file_v1.csv` (`slug`, `pr_number`, `unit_id`, `event`, `duration_days`, `is_agent`) and `code_content_file_v1.csv.gz` (file content at the merge commit and its tokens: `birth_sha`, `code_content`, `code_tokens`, `num_tokens`). `reproduce.sh` unpacks them into `results/rq3/` |
| `rq3b_agent/`, `rq3b_all/` | RQ3b inputs, modified agent files (6,278) and all modified files (14,886), without vendored files and with the full PR-history leakage fix. `process_features_rq32.csv`: the eight process features, `birth_date`, `contributor`, `duration_days` and `duration_bin` (0: within one day, 1: one day to one week, 2: one week to one month, 3: more than one month) |
| `pr_features/pr_features_fixed.jsonl` | PR-history features per PR from the GitHub search API, with the leakage fix (`src/mining/extract_rq3_pr_features_fixed.py`) |
| `prep_log.txt`, `prep_log_full_fix.txt` | logs of the input preparation (`src/mining/d_prep.py`, `d_full_prep.py`) |

## audit/

| File | Content |
|---|---|
| `attribution_audit_sheet_rater1.xlsx` | The first rater's labels of the 325 audited agent PRs (sheet `audit`) and the labeling guide (sheet `guide`). The package keeps only the labels; the optional evidence and notes columns are omitted |
| `audit_rater1.csv` | The `audit` sheet as CSV (`sample_id`, `agent_name`, `group`, `slug`, `pr_number`, PR details, `label`). Labels: agent, mixed, human, unclear. E-mail addresses in the PR descriptions are replaced by `<email>` |
| `attribution_audit_labelling_instructions.md` | The written labeling guide |
| `exclude_human.csv`, `exclude_human_unclear.csv`, `human_labelled_prs.csv` | PRs labeled human (or unclear) for the RQ1 sensitivity analysis |
| `second_rater/second_rater_annotation_sheet.xlsx` | The second rater's labels of 100 of the 325 PRs (sheet `annotate`, column `label`; the optional evidence and notes columns are omitted). The rater worked independently with the same guide; disagreements were resolved by discussion. `src/analysis/kappa.py` computes the agreement |
