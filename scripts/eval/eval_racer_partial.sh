#!/bin/bash
set -e -x

# Same as eval_racer.sh but restricted to a 16-task subset (train-set /
# generalization experiment). Output dir convention is unchanged.
#   $1 EVALUATOR_NUM   $2 EXP_NAME

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

EVALUATOR_NUM=$1
if ! is_int "$EVALUATOR_NUM"; then echo "Error: evaluator_num must be an integer."; exit 1; fi
EXP_NAME=$2

############## EDIT HERE FOR YOUR OWN SETUP ##############
DEVICES=(1 6 7)
MODEL=racer-visuomotor-policy-rich
RUNS_DIR=racer/runs
VNC_DISPLAY=:2
##########################################################

export DISPLAY=$VNC_DISPLAY
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

eval "$(conda shell.bash hook)"
conda activate racer
cd 3rdparty/RACER

start=$(date +%s)
for seed in 0 1 2 3 4 5 6 7 8 9; do
python racer/evaluation/rollout.py \
	--device ${DEVICES[*]} \
	--evaluator-num $EVALUATOR_NUM \
	--model-folder ${RUNS_DIR}/${MODEL} \
	--eval-datafolder racer/data/rlbench/test \
	--tasks meat_off_grill open_drawer place_cups place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap \
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
printf "Execution time: %.2f minutes\n" "$(echo "($end - $start) / 60" | bc -l)"
