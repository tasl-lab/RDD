#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rdd

############## EDIT HERE ##############
RUNS_DIR=3rdparty/RACER/racer/runs
MODEL=racer-visuomotor-policy-rich
EXCLUDE="--exclude-suboptimal-tasks"   # set to "" to include every task
######################################

seeds() { local m=$1; for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "${RUNS_DIR}/${MODEL}/${m}-${s} "; done; }
SR="python scripts/eval/success_rate.py"

echo "w/o finetune";      $SR $(seeds vanilla_llava)  --decimal 1 --exp-name "w/o finetune" $EXCLUDE --clear-log
echo "fixed_interval";    $SR $(seeds fixed_interval) --decimal 1 --exp-name fixed_interval $EXCLUDE
echo "uvd";               $SR $(seeds uvd)            --decimal 1 --exp-name uvd            $EXCLUDE
echo "expert (heuristic)";$SR $(seeds heuristic)      --decimal 1 --exp-name expert         $EXCLUDE
echo "rdd (ours)";        $SR $(seeds rdd)            --decimal 1 --exp-name rdd            $EXCLUDE --summary
