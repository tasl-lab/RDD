#!/bin/bash
set -e -x

# Serve the T5 language-tokenizer / encoder used by the RACER rollout
# (the eval convention is PORT 20001; eval connects to http://localhost:<PORT>/encode/).
#   $1 PORT   $2 DEVICE

is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

PORT=$1
if ! is_int "$PORT"; then echo "Error: Port must be an integer."; exit 1; fi
DEVICE=$2
if ! is_int "$DEVICE"; then echo "Error: Device must be an integer."; exit 1; fi

eval "$(conda shell.bash hook)"
conda activate llava-next
cd 3rdparty/Open-LLaVA-NeXT

exec env CUDA_VISIBLE_DEVICES=$DEVICE uvicorn deploy.lm_server:app --port $PORT
