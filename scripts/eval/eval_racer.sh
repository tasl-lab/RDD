#!/bin/bash
set -e -x

# Run the RVT2 visuomotor rollout eval over 10 demo seeds for one decomposition
# method. Writes per-task success_/failure_ logs to
#   <RUNS_DIR>/<MODEL>/<EXP_NAME>-<seed>
# which scripts/eval/success_rate.py then aggregates.
#
#   $1 EVALUATOR_NUM – total evaluators distributed across DEVICES
#   $2 EXP_NAME      – method tag: vanilla_llava | fixed_interval | uvd | heuristic | rdd (+ ablation tags)

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

EVALUATOR_NUM=$1
if ! is_int "$EVALUATOR_NUM"; then echo "Error: evaluator_num must be an integer."; exit 1; fi
EXP_NAME=$2

############## EDIT HERE FOR YOUR OWN SETUP ##############
DEVICES=(6 7)                         # GPU devices for the RVT2 evaluators
MODEL=racer-visuomotor-policy-rich    # visuomotor policy checkpoint folder
RUNS_DIR=racer/runs                   # output root, relative to the RACER repo
VNC_DISPLAY=:2                        # your VNC display
##########################################################

export DISPLAY=$VNC_DISPLAY
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6  # fix for ubuntu 22.04

eval "$(conda shell.bash hook)"
conda activate racer
cd 3rdparty/RACER

echo "${MODEL}"
start=$(date +%s)

for seed in 0 1 2 3 4 5 6 7 8 9; do
python racer/evaluation/rollout.py \
	--device ${DEVICES[*]} \
	--evaluator-num $EVALUATOR_NUM \
	--model-folder ${RUNS_DIR}/${MODEL} \
	--eval-datafolder racer/data/rlbench/test \
	--tasks all \
	--start-episode 0 \
	--eval-episodes 25 \
	--episode-chunk-length 3 \
	--episode-length 30 \
	--log-name test \
	--model-name model_17.pth \
	--demo-seed-bias $seed \
	--eval-log-dir ${RUNS_DIR}/${MODEL}/${EXP_NAME}-$seed \
	--lm-address http://localhost:20001/encode/ \
	--vlm-address http://localhost:21002 http://localhost:21003 \
	--use-vlm
done

end=$(date +%s)
printf "Execution time: %d min %d s\n" $(( (end-start)/60 )) $(( (end-start)%60 ))
