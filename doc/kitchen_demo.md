# Example #1: Kitchen Cleaning

In this simple example, we'll explore the decomposition for two tasks about putting away kitechenwares.

### Cut the .mov into Image Frames

```
python scripts/dataset/video_to_frames.py resources/clean_kitchen/IMG_0600.MOV data/raw_data/clean_kitchen/IMG_0600 --fps 10 --height 720 
python scripts/dataset/video_to_frames.py resources/clean_kitchen/IMG_0601.MOV data/raw_data/clean_kitchen/IMG_0601 --fps 10 --height 720
```

### Convert the Frames into RLBench Format

```
cp resources/clean_kitchen/IMG_0600.txt data/raw_data/clean_kitchen/IMG_0600/info.txt
cp resources/clean_kitchen/IMG_0601.txt data/raw_data/clean_kitchen/IMG_0601/info.txt
python scripts/dataset/frames_dataset_proc.py data/raw_data/clean_kitchen data/datasets/clean_kitchen --task-name demo_task
```

### Build Vector Datasets

```
python build_vec_database.py 0 liv 1.0 data/datasets/clean_kitchen/splits/train \
	--name-suffix clean_kitchen \
	--views front_rgb \
	--embed-mode ood
```

### Config and Start RDD server

1. Set path to vector databases & interval similarity scroing mode in  [configs/rdd_server.yaml](configs/rdd_server.yaml)

   ```
   vec_database_path: "data/vec_databases/clean_kitchen/train"
   mode: "ood"
   ```
2. Start service

   ```
   uvicorn rdd_server:app --port 8001 --workers 8
   ```

### Decompose and Evaluate

```
python eval_rdd.py \
	data/datasets/clean_kitchen \
	data/eval_out/clean_kitchen \
	--worker-num=4
```

You can then view the starting frames of each subtask at `data/eval_out/clean_kitchen/visualization/demo_task/ep_0/rdd`.