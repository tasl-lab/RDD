#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
EXCLUDE="--exclude-suboptimal-tasks"
seeds() { local m=$1; for s in 0 1 2; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

# vector-database sampling rate. 1.0 is the main RDD config.
echo "sr 1.0";  $SR $(seeds rdd_sr_1.0)  --decimal 1 --exp-name sr_1.0  $EXCLUDE --clear-log
echo "sr 0.5";  $SR $(seeds rdd_sr_0.5)  --decimal 1 --exp-name sr_0.5  $EXCLUDE
echo "sr 0.25"; $SR $(seeds rdd_sr_0.25) --decimal 1 --exp-name sr_0.25 $EXCLUDE --summary
