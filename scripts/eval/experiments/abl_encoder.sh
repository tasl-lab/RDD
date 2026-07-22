#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
EXCLUDE="--exclude-suboptimal-tasks"
seeds() { local m=$1; for s in 0 1 2; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

# visual encoder used to build the retrieval prior. liv is the main config.
encoders=(liv r3m vip vc1 clip dinov2 resnet)
first="--clear-log"
for i in "${!encoders[@]}"; do
	enc=${encoders[$i]}
	last=""; [ "$i" = "$(( ${#encoders[@]} - 1 ))" ] && last="--summary"
	$SR $(seeds rdd_${enc}) --decimal 1 --exp-name "$enc" $EXCLUDE $first $last
	first=""
done
