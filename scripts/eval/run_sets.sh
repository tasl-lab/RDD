#!/bin/bash
# Orchestrate evaluation set(s): bring up two LLaVA planner workers, run the
# RVT2 rollout eval, then tear the workers down. Run the LM tokenizer server
# (scripts/eval/serve_tokenizer.sh <port> <device>, port 20001) separately first.

############## EDIT HERE FOR YOUR OWN SETUP ##############
LLAVA_DEVICES=(4 5)            # GPUs for the two planner workers
LLAVA_PORTS=(21002 21003)     # must match --vlm-address in eval_racer.sh
EVALUATOR_NUM=6               # evaluators passed to eval_racer.sh
WARMUP_SECONDS=240           # time to let the planner servers load
# Each set name is a finetuned planner checkpoint, or 'vanilla_llava'
# for the vanilla / w-o-finetune baseline. Rollout logs land in
# <RUNS_DIR>/<MODEL>/<set_name>-<seed>.
sets=("rdd")
##########################################################

run_set() {
	local set_name="$1"
	echo "Starting set: $set_name"

	./scripts/eval/serve_llava.sh "${LLAVA_PORTS[0]}" "${LLAVA_DEVICES[0]}" "${set_name}" &
	local pid_a=$!
	./scripts/eval/serve_llava.sh "${LLAVA_PORTS[1]}" "${LLAVA_DEVICES[1]}" "${set_name}" &
	local pid_b=$!
	echo "Planner workers started (PIDs $pid_a $pid_b); warming up ${WARMUP_SECONDS}s"
	sleep "$WARMUP_SECONDS"

	./scripts/eval/eval_racer.sh "$EVALUATOR_NUM" "${set_name}"

	echo "Terminating planner workers"
	kill -9 "$pid_a" "$pid_b" 2>/dev/null
	wait "$pid_a" "$pid_b" 2>/dev/null
	sleep 30
	echo "Set $set_name completed"
}

for set in "${sets[@]}"; do
	run_set "$set"
done
