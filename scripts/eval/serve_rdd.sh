#!/bin/bash
set -e -x

eval "$(conda shell.bash hook)"
conda activate rdd

# Serve the RDD demonstration-decomposer API (POST /decompose) on port 8001.
uvicorn rdd_server:app --port 8001 --workers 8
