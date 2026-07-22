#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
# generalization to unseen tasks: exclude the planner's finetuning tasks.
FLAGS="--exclude-train-tasks --exclude-suboptimal-tasks"
seeds() { local m=$1; for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

echo "heuristic (train-set)"; $SR $(seeds heuristic_trainset) --decimal 1 --exp-name trainset $FLAGS --clear-log
echo "rdd (ours)";            $SR $(seeds rdd)                --decimal 1 --exp-name rdd      $FLAGS --summary
