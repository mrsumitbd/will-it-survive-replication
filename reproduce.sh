#!/usr/bin/env bash
# reproduce.sh: run every analysis of the paper from data/ into results/.
#
#   bash reproduce.sh            paper settings (30 x 10 cross-validation, 1,000 bootstrap replicates); several hours
#   bash reproduce.sh --quick    smoke test: RQ1 on a random quarter of the PRs, 50 bootstrap replicates, RQ3 with
#                                1 x 10 cross-validation and line localization on at most 40 files per fold.
#                                The numbers differ from the paper. Checks that every script runs.
#   bash reproduce.sh --force    rerun every step, even the finished ones (combine with --quick if needed)
#   bash reproduce.sh --only <step>   run a single step (see the list printed by --list)
#   bash reproduce.sh --list     list the steps
#
# Uses `python` from the active environment (see README.md). Each step writes a log to results/logs/<step>.log.
# A finished step is skipped on a rerun unless --force is given or it last ran in the other mode (quick or full).
# The script stops at the first failing step. At the end, src/analysis/paper_numbers.py writes results/PAPER_NUMBERS.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
QUICK=0; FORCE=0; ONLY=""; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK=1 ;;
    --force) FORCE=1 ;;
    --only) ONLY="${2:?--only needs a step name}"; shift ;;
    --list) LIST=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see bash reproduce.sh --help)"; exit 2 ;;
  esac
  shift
done

if [ "$QUICK" = 1 ]; then MODE=quick; B=50; export RQ3_N_REPS=1; MAXF=40
else MODE=full; B=1000; export RQ3_N_REPS=30; MAXF=0; fi
export REPRO_QUICK=$QUICK MAXF
export PYTHONHASHSEED=0 MPLBACKEND=Agg

RES="$ROOT/results"; LOGS="$RES/logs"; DONE="$RES/.done"; SRC="$ROOT/src"; DATA="$ROOT/data"
RQ3="$RES/rq3"

# ---------------------------------------------------------------- steps: name | clean-before-run | command
STEPS=()
add() { STEPS+=("$1|$2|$3"); }
add cohort_funnel      ""  "python src/analysis/funnel.py"
add cohort_stats       ""  "python src/analysis/paper_stats.py > results/cohort/paper_stats.txt"
add audit_summary      ""  "python src/analysis/audit_summary.py > results/audit/audit_summary.txt"
add audit_kappa        ""  "python src/analysis/kappa.py > results/audit/kappa.txt"
add rq1_line   "results/rq1/rq1_line.json"  "python src/analysis/rq1_fb.py line $B"
add rq1_hunk   "results/rq1/rq1_hunk.json"  "python src/analysis/rq1_fb.py hunk $B"
add rq1_file   "results/rq1/rq1_file.json"  "python src/analysis/rq1_fb.py file $B"
add rq1_ph             ""  "python src/analysis/a4_ph.py line"
add rq1_substantial    ""  "python src/analysis/a7_sub.py"
add rq1_file_history   ""  "python src/analysis/a9_model.py"
add rq1_misc_figure    "results/rq1/a_misc.json"  "python src/analysis/a_misc.py"
add rq1_calendar       ""  "python src/analysis/c1_calendar.py line"
add rq1_rate_14d       ""  "python src/analysis/rate_14d.py"
add rq2                ""  "python src/analysis/rq2_fb.py"
add rq3_unpack         ""  "unpack_rq3"
# RQ3a, agent files: 8-model tournament, Scott-Knott ESD, size/history baseline, out-of-sample SHAP/LIME, line localization
add rq3a_agent_tournament "results/rq3/rq3a_agent/results results/rq3/rq3a_agent/models" \
    "cd $RQ3/rq3a_agent && mkdir -p results models && python $SRC/rq3/rq3a_tournament.py --vectorizer count --rebalance smote --skip-lime"
add rq3a_agent_scott_knott "" "cd $RQ3/rq3a_agent && python $SRC/rq3/rq3a_wide.py results/rq31_bow_tournament_scores_combined_count.csv results && for m in auc_pr auc_roc; do python $SRC/rq3/sk_esd.py results/rq3a_wide_\$m.csv results/rq3a_sk_\$m.csv; done"
add rq3a_agent_baseline "" "cd $RQ3/rq3a_agent && python $SRC/rq3/d1_lr.py $SRC/rq3 $DATA/file_covariates"
add rq3a_agent_explain  "" "cd $RQ3/rq3a_agent && python $SRC/rq3/lime_timed.py rq3a $SRC/rq3"
add rq3a_agent_localize "" "cd $RQ3/rq3a_agent && python $SRC/rq3/rq3a_localize.py $SRC/rq3 $SRC/rq3 $DATA/tables/lines.parquet"
# RQ3b, agent files: tournament, Scott-Knott ESD, out-of-sample SHAP/LIME
add rq3b_agent_tournament "results/rq3/rq3b_agent/results results/rq3/rq3b_agent/models" \
    "cd $RQ3/rq3b_agent && mkdir -p results models && python $SRC/rq3/rq3b_tournament.py"
add rq3b_agent_scott_knott "" "cd $RQ3/rq3b_agent && for m in macro_f1 auc_roc; do python $SRC/rq3/sk_esd.py results/rq32_process_wide_\$m.csv results/rq3b_sk_\$m.csv; done"
add rq3b_agent_explain  "" "cd $RQ3/rq3b_agent && python $SRC/rq3/lime_timed.py rq3b $SRC/rq3"
# RQ3b, all files (comparison): tournament, Scott-Knott ESD
add rq3b_all_tournament "results/rq3/rq3b_all/results results/rq3/rq3b_all/models" \
    "cd $RQ3/rq3b_all && mkdir -p results models && python $SRC/rq3/rq3b_tournament.py"
add rq3b_all_scott_knott "" "cd $RQ3/rq3b_all && for m in macro_f1 auc_roc; do python $SRC/rq3/sk_esd.py results/rq32_process_wide_\$m.csv results/rq3b_sk_\$m.csv; done"
# RQ3a, all files (comparison): random forest, CatBoost and XGBoost, baseline, line localization
add rq3a_all_tournament "results/rq3/rq3a_all/results results/rq3/rq3a_all/models" \
    "cd $RQ3/rq3a_all && mkdir -p results models && python $SRC/rq3/rq3a_sub.py $SRC/rq3 RandomForest,CatBoost,XGBoost --vectorizer count --rebalance smote --skip-lime"
add rq3a_all_scott_knott "" "cd $RQ3/rq3a_all && python $SRC/rq3/rq3a_wide.py results/rq31_bow_tournament_scores_combined_count.csv results && for m in auc_pr auc_roc; do python $SRC/rq3/sk_esd.py results/rq3a_wide_\$m.csv results/rq3a_sk_\$m.csv; done"
add rq3a_all_baseline "" "cd $RQ3/rq3a_all && python $SRC/rq3/d1_lr.py $SRC/rq3 $DATA/file_covariates"
add rq3a_all_localize "" "cd $RQ3/rq3a_all && python $SRC/rq3/rq3a_localize.py $SRC/rq3 $SRC/rq3 $DATA/tables/lines.parquet"
add summary            ""  "python src/analysis/make_summary.py results"

unpack_rq3() {
  for s in rq3a_agent rq3a_all; do
    mkdir -p "$RQ3/$s/data"
    gzip -dc "$DATA/rq3/$s/code_content_file_v1.csv.gz" > "$RQ3/$s/data/code_content_file_v1.csv"
    cp "$DATA/rq3/$s/survival_events_file_v1.csv" "$RQ3/$s/data/"
  done
  for s in rq3b_agent rq3b_all; do
    mkdir -p "$RQ3/$s/data"; cp "$DATA/rq3/$s/process_features_rq32.csv" "$RQ3/$s/data/"
  done
  cp "$DATA/rq3/prep_log.txt" "$DATA/rq3/prep_log_full_fix.txt" "$RQ3/"
  echo "RQ3 inputs unpacked into results/rq3/"
}

if [ "$LIST" = 1 ]; then for s in "${STEPS[@]}"; do echo "${s%%|*}"; done; exit 0; fi

# ---------------------------------------------------------------- checks
command -v python >/dev/null || { echo "ERROR: no 'python' on PATH. Activate the environment first (see README.md)."; exit 1; }
python - <<'EOF' || { echo "ERROR: the Python environment is incomplete. Run: pip install -r requirements.txt"; exit 1; }
import sys
if sys.version_info[:2] != (3, 11):
    print(f"WARNING: Python {sys.version.split()[0]}; the paper's results were produced with Python 3.11.")
import pandas, numpy, scipy, sklearn, lifelines, xgboost, catboost, imblearn, shap, lime, matplotlib, pyarrow  # noqa
EOF
mkdir -p "$RES" "$LOGS" "$DONE" "$RES/cohort" "$RES/audit"
echo "$MODE" > "$RES/RUN_MODE.txt"

fmt() { local s=$1; printf '%dh %02dm %02ds' $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }
echo "================================================================"
if [ "$MODE" = quick ]; then
  echo " QUICK MODE (smoke test): RQ1 on 25% of the PRs, 50 bootstrap replicates, RQ3 with 1 x 10 CV."
  echo " The numbers will NOT match the paper. Run without --quick for the paper's settings."
else
  echo " FULL MODE: the paper's settings (30 x 10 CV, 1,000 bootstrap replicates)."
fi
echo " python: $(command -v python) ($(python -V 2>&1))"
echo "================================================================"

T0=$(date +%s); N=0; SKIP=0
for s in "${STEPS[@]}"; do
  name="${s%%|*}"; rest="${s#*|}"; clean="${rest%%|*}"; cmd="${rest#*|}"
  [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
  marker="$DONE/$name"
  if [ "$FORCE" = 0 ] && [ -f "$marker" ] && [ "$(cat "$marker")" = "$MODE" ]; then
    echo "[skip] $name (finished earlier in $MODE mode; use --force to rerun)"; SKIP=$((SKIP + 1)); continue
  fi
  rm -f "$marker"
  for c in $clean; do rm -rf "$ROOT/$c"; done
  printf '[run ] %-26s ' "$name"
  t=$(date +%s)
  if [ "$cmd" = unpack_rq3 ]; then
    ok=0; ( unpack_rq3 ) > "$LOGS/$name.log" 2>&1 || ok=$?
  else
    ok=0; ( cd "$ROOT" && eval "$cmd" ) > "$LOGS/$name.log" 2>&1 || ok=$?
  fi
  dt=$(( $(date +%s) - t ))
  if [ "$ok" != 0 ]; then
    echo "FAILED after $(fmt $dt)"
    echo "----------------------------------------------------------------"
    echo "ERROR: step '$name' failed (exit code $ok). Last lines of results/logs/$name.log:"
    tail -n 25 "$LOGS/$name.log"
    echo "----------------------------------------------------------------"
    echo "Fix the problem and rerun 'bash reproduce.sh$( [ "$QUICK" = 1 ] && echo ' --quick')'; finished steps are skipped."
    exit 1
  fi
  echo "$MODE" > "$marker"; echo "$name $MODE $dt" >> "$LOGS/runtimes.txt"
  echo "done in $(fmt $dt)"; N=$((N + 1))
done

if [ -z "$ONLY" ] || [ "$ONLY" = paper_numbers ]; then
  printf '[run ] %-26s ' paper_numbers
  t=$(date +%s)
  python src/analysis/paper_numbers.py --results results > "$LOGS/paper_numbers.log" 2>&1 || { echo FAILED; tail -n 25 "$LOGS/paper_numbers.log"; exit 1; }
  echo "done in $(fmt $(( $(date +%s) - t )))"
  tail -n 4 "$LOGS/paper_numbers.log"
fi
echo "================================================================"
echo " $N step(s) run, $SKIP skipped, total $(fmt $(( $(date +%s) - T0 ))) ($MODE mode)."
echo " Report: results/PAPER_NUMBERS.md   Figure 1: results/figures/rq1_km_line_primary.pdf"
[ "$MODE" = quick ] && echo " QUICK MODE: the reproduced values are not expected to match the paper."
echo "================================================================"
