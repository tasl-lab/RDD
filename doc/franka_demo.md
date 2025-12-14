<!-- 
python scripts/dataset/video_to_frames.py resources/franka/IMG_8900_4x_compressed.mp4 data/raw_data/franka/IMG_8900 --fps 10 --height 720 
python scripts/dataset/video_to_frames.py resources/franka/IMG_8908_4x_compressed.mp4 data/raw_data/franka/IMG_8908 --fps 10 --height 720 

cp resources/franka/IMG_8900_4x_compressed.txt data/raw_data/franka/IMG_8900/info.txt
cp resources/franka/IMG_8908_4x_compressed.txt data/raw_data/franka/IMG_8908/info.txt
python scripts/dataset/frames_dataset_proc.py data/raw_data/franka data/datasets/franka --task-name demo_task

python build_vec_database.py 0 liv 1.0 data/datasets/franka/splits/train \
	--name-suffix franka \
	--views front_rgb \
	--embed-mode ood

uvicorn rdd_server:app --port 8001 --workers 8

python eval_rdd.py \
	data/datasets/franka \
	data/eval_out/franka \
	--worker-num=4 -->

# Example: Franka Demonstrations (RDD Decomposition)

This guide mirrors **Example #1: Kitchen Cleaning**, but targets the **Franka** videos/text metadata you listed.

---

## 1) Cut the Videos into Image Frames

Convert each video to a folder of frames (10 FPS, height 720).

```bash
python scripts/dataset/video_to_frames.py resources/franka/IMG_8900_4x_compressed.mp4 data/raw_data/franka/IMG_8900 --fps 10 --height 720
python scripts/dataset/video_to_frames.py resources/franka/IMG_8908_4x_compressed.mp4 data/raw_data/franka/IMG_8908 --fps 10 --height 720
```

Expected outputs:
- `data/raw_data/franka/IMG_8900/` (frame images)
- `data/raw_data/franka/IMG_8908/` (frame images)

---

## 2) Convert the Frames into RLBench Format

Copy per-trajectory metadata into the corresponding `info.txt`, then run the dataset processing script.

```bash
cp resources/franka/IMG_8900_4x_compressed.txt data/raw_data/franka/IMG_8900/info.txt
cp resources/franka/IMG_8908_4x_compressed.txt data/raw_data/franka/IMG_8908/info.txt

python scripts/dataset/frames_dataset_proc.py data/raw_data/franka data/datasets/franka --task-name demo_task
```

Expected outputs:
- `data/datasets/franka/` (processed dataset)
- `data/datasets/franka/splits/train/` (train split used below)

---

## 3) Build Vector Datasets (Embedding Database)

Build the vector database for retrieval/scoring (views: `front_rgb`, embed mode: `ood`).

```bash
python build_vec_database.py 0 liv 1.0 data/datasets/franka/splits/train \
  --name-suffix franka \
  --views front_rgb \
  --embed-mode ood
```

Expected outputs (typical):
- `data/vec_databases/franka/train/` (or a similarly named directory under `data/vec_databases/`)

---

## 4) Configure and Start the RDD Server

1. Edit `configs/rdd_server.yaml` to point to the Franka vector database path and set similarity scoring mode:

```yaml
vec_database_path: "data/vec_databases/franka/train"
mode: "ood"
```

2. Start the service:

```bash
uvicorn rdd_server:app --port 8001 --workers 8
```

Notes:
- Keep this terminal running while you evaluate.
- If `8001` is in use, choose a different port and pass it consistently to your evaluator (if your evaluator supports a flag/env var).

---

## 5) Decompose and Evaluate

Run decomposition + evaluation on the processed Franka dataset:

```bash
python eval_rdd.py \
  data/datasets/franka \
  data/eval_out/franka \
  --worker-num=4
```

Expected outputs:
- `data/eval_out/franka/` (evaluation artifacts, logs, and decomposition outputs)

---
