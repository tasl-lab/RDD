# Example #3: AgiBotWorld & RoboCerebra

We now explore using RDD and the publich datasets.

### Prepare Datasets

This section will format the raw datasets to a unified RLBench-like structure:

```bash
├── metadata
│   └── metadata.pkl
├── raw
│   └── ...
├── seg
│   └── ...
└── splits
    ├── train
        ├── <task>
            └── <episode>
                └── <view>
        └── ...
    ├── val
        └── ...
    └── val_gt
        └── ...
```

**AgiBotWorld**

Download from [agibot-world/AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha#download-the-dataset) to `RDD/data/datasets`. the file structure should look like

```
RDD/data/datasets/AgiBot-World/AgiBotWorld-Alpha
├── observations
│   ├──...
│   └── 327
└── task_info
    ├──...
    └── task_327.json
```

Format to RLBench-like dataset

```
python scripts/dataset/agibotworld2rlbench.py \
	data/datasets/agibotworld/AgiBot-World/AgiBotWorld-Alpha \
	data/datasets/agibotworld/AgiBot-World/AgiBotWorld-Alpha_rlbench \
	16 \
	--num-episodes 190
```

**RoboCerebra**

Download the training set from [qiukingballball/RoboCerebra](https://huggingface.co/datasets/qiukingballball/RoboCerebra/tree/main/RoboCerebra_trainset) to `RDD/data/datasets`. the file structure should look like

```
RDD/data/datasets/RoboCerebra/homerobo_trainingset/
├── coffee_table
├── kitchen_table
├── study_table
|...
```

Format to RLBench-like dataset

```
python scripts/dataset/robocerebra_proc.py \
	data/datasets/RoboCerebra/homerobo_trainingset \
	data/datasets/RoboCerebra/homerobo_trainingset_rlbench \
	16 \
	--num-episodes 700
```

## Build Vector Datasets

**AgiBotWorld**

```bash
python build_vec_database.py 0 liv 1.0 data/datasets/AgiBot-World/AgiBotWorld-Alpha_rlbench/splits/train \
	--name-suffix agibotworld_train \
	--views front_rgb \
	--embed-mode default
```

**RoboCerebra (OOD)**

```bash
python build_vec_database.py 0 liv 1.0 data/datasets/RoboCerebra/homerobo_trainingset_rlbench/splits/train \
	--name-suffix robocerebra_train \
	--views front_rgb \
	--embed-mode ood
```

The generated vector database will be saved at `/data/vec_databases/<name-suffix>/<split>`

## Demonstration Decomposition

**Config and Start RDD server**

1. Set path to vector databases & interval similarity scroing mode in  [configs/rdd_server.yaml](configs/rdd_server.yaml)

   ```yaml
   # agibotworld
   vec_database_path: "data/vec_databases/agibotworld_train/train"
   mode: "default" # * will set beta to 0.0
   ```

   ```yaml
   # robocerebra
   vec_database_path: "data/vec_databases/robocerebra_train/train"
   mode: "ood" # * will set alpha to 0.0
   ```
2. Start service

   ```
   uvicorn rdd_server:app --port 8001 --workers 8
   ```

**Decompose and Evaluate**

```bash
# agibotworld
python eval_rdd.py \
	data/datasets/AgiBot-World/AgiBotWorld-Alpha_rlbench \
	--ep-num=8 \
	--worker-num=8
```

```bash
# robocerebra
python eval_rdd.py \
	data/datasets/RoboCerebra/homerobo_trainingset_rlbench \
	--ep-num=8 \
	--worker-num=8
```