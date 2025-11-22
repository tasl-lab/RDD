# Build Dataset

```
python scripts/dataset/video_to_frames.py data/raw_data/clean_kitchen/IMG_0600.MOV --fps 10 --height 720
python scripts/dataset/video_to_frames.py data/raw_data/clean_kitchen/IMG_0601.MOV --fps 10 --height 720

rm -r data/datasets/clean_kitchen
python scripts/dataset/frames_dataset_proc.py data/raw_data/clean_kitchen data/datasets/clean_kitchen --task-name demo_task

rm -r /data4/mingxuan/workspace/RDD/data/vec_databases/clean_kitchen
python build_vec_database.py 0 liv 1.0 data/datasets/clean_kitchen/splits/train \
	--name-suffix clean_kitchen \
	--views front_rgb \
	--embed-mode ood

python eval_rdd.py \
	data/datasets/clean_kitchen \
	data/eval_out/clean_kitchen \
	--worker-num=4
```
