#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
EXCLUDE="--exclude-suboptimal-tasks"
seeds() { local m=$1; for s in 0 1 2; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

# number of demos per task used to build the prior. 'rdd' (main) uses 3 demos.
echo "1 demo (rdd)";  $SR $(seeds rdd_ep1) --decimal 1 --exp-name ep1_rdd $EXCLUDE --clear-log
echo "3 demos (rdd)"; $SR $(seeds rdd)     --decimal 1 --exp-name ep3_rdd $EXCLUDE
echo "3 demos (uvd)"; $SR $(seeds uvd)     --decimal 1 --exp-name ep3_uvd $EXCLUDE --summary
