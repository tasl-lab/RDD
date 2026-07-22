#!/bin/bash
set -e -x

# Finetune the LLaVA-NeXT planner (LoRA) on a RACER-DataGen output.
#   $1 datagen_run_dir – RACER-DataGen run dir with the LLaVA-format json
#   $2 exp_name        – checkpoint name, saved to Open-LLaVA-NeXT/checkpoints/<exp_name>

cd 3rdparty/Open-LLaVA-NeXT
eval "$(conda shell.bash hook)"
conda activate llava-next
./scripts/finetune_task_lora_local_mytrain.sh "$1" "$2"
cd ../..
