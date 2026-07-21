#!/bin/bash
set -e -x

# Serve a LLaVA-NeXT planner worker.
#   $1 PORT    – planner port (eval expects 21002 / 21003)
#   $2 DEVICE  – GPU index
#   $3 MODEL_NAME – checkpoint name under Open-LLaVA-NeXT/checkpoints,
#                   or 'llama3-llava-next-8b' for the vanilla (w/o-finetune) baseline.

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

PORT=$1
if ! is_int "$PORT"; then echo "Error: Port must be an integer."; exit 1; fi
DEVICE=$2
if ! is_int "$DEVICE"; then echo "Error: Device must be an integer."; exit 1; fi
MODEL_NAME=$3
if [[ -z "$MODEL_NAME" ]]; then echo "Error: MODEL_NAME (\$3) is required (a checkpoint name, or 'vanilla_llava'/'llama3-llava-next-8b' for the baseline)."; exit 1; fi

eval "$(conda shell.bash hook)"
conda activate llava-next
cd 3rdparty/Open-LLaVA-NeXT

if [[ "$MODEL_NAME" == "llama3-llava-next-8b" || "$MODEL_NAME" == "vanilla_llava" ]]; then
	echo "Using original (vanilla) llava model"
	exec env CUDA_VISIBLE_DEVICES=$DEVICE python deploy/llava_server.py \
		--model-path ../models/llama3-llava-next-8b \
		--model-name llava_llama3_lora \
		--port $PORT
else
	echo "Using finetuned llava model"
	exec env CUDA_VISIBLE_DEVICES=$DEVICE python deploy/llava_server.py \
		--model-path ./checkpoints/$MODEL_NAME \
		--model-base ../models/llama3-llava-next-8b \
		--model-name llava_llama3_lora \
		--port $PORT
fi
