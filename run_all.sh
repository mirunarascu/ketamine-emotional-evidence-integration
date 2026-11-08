#!/usr/bin/env bash
# Runs the whole analysis, in order, for both cohorts.
#
#   bash run_all.sh            # full run (1000 shuffles per cohort; about 1.5 h on 8 cores)
#   bash run_all.sh --quick    # test run with 3 shuffles (writes the same files, so the null
#                              # results are NOT the reported ones; rerun without --quick)
#
# Needs the environment in environment.yml and the data files in data/ (or their folder in
# PEDUKS_DATA_DIR, see README.md). Stops at the first error.
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then QUICK="--n 3"; fi
N_JOBS="${N_JOBS:-8}"            # cores for the null shuffles, e.g. N_JOBS=4 bash run_all.sh

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }

for cohort in in_person online; do
  step "1. behavioural measures: $cohort"
  python analysis/behaviour/01_make_behaviour_tables.py --cohort "$cohort"
  python analysis/behaviour/02_task_descriptives.py --cohort "$cohort"
  python analysis/behaviour/03_null_shuffles.py --cohort "$cohort" --n-jobs "$N_JOBS" $QUICK
  python analysis/behaviour/04_null_tests.py --cohort "$cohort"

  step "2. integration kernels: $cohort"
  python analysis/kernels/01_facewise_linear.py --cohort "$cohort"
  python analysis/kernels/02_timebinned_logistic.py --cohort "$cohort"

  step "3. statistics: $cohort"
  Rscript analysis/stats/01_supplement_tables.R --cohort "$cohort"
  python analysis/stats/03_descriptives.py --cohort "$cohort"
done

step "3. statistics: in person only (Table S9, questionnaires)"
Rscript analysis/stats/02_covariates_S9.R
Rscript analysis/stats/04_questionnaire_stats.R

step "4. figures"
for cohort in in_person online; do
  python analysis/figures/01_behaviour_panels.py --cohort "$cohort"
  python analysis/figures/02_detection_curves.py --cohort "$cohort"
  python analysis/figures/03_kernel_panels.py --cohort "$cohort"
done
python analysis/figures/04_questionnaire_panels.py
python analysis/figures/05_example_blocks.py

step "done: outputs in output/"
