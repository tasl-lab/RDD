#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
EXCLUDE="--exclude-suboptimal-tasks"
seeds() { local m=$1; for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

# Gemini-2.5-Pro decomposer baseline vs RDD (both decompose, then finetune the same LLaVA planner).
echo "Gemini-2.5-Pro"; $SR $(seeds gemini_2.5_pro) --decimal 1 --exp-name "Gemini-2.5-Pro" $EXCLUDE --clear-log
echo "rdd (ours)";     $SR $(seeds rdd)            --decimal 1 --exp-name rdd              $EXCLUDE --summary
