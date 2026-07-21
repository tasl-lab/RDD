#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
EXCLUDE="--exclude-suboptimal-tasks"
seeds() { local m=$1; for s in 0 1 2; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

# alpha weights on the retrieval prior. alpha_1.0 is the main RDD config.
first="--clear-log"
for a in 0.0 0.5 1.0 2.0; do
	last=""; [ "$a" = "2.0" ] && last="--summary"
	$SR $(seeds rdd_alpha_${a}) --decimal 1 --exp-name "alpha_${a}" $EXCLUDE $first $last
	first=""
done
