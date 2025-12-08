# **RDD**: **R**etrieval-Based **D**emonstration **D**ecomposer

[![arXiv](https://img.shields.io/badge/arXiv-2510.14968-red)](https://arxiv.org/pdf/2510.14968)
[![Website](https://img.shields.io/badge/Website-RDD-blue)](https://rdd-neurips.github.io/)
![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-purple)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**RDD** is an retrieval-based visual demonstration decomposer that automatically identifies sub-tasks visually similar to a set of existing expert-labeled sub-tasks.

[Mingxuan Yan](https://waterhyacinthinnanhu.github.io/)<sup>1</sup>,
[Yuping Wang](https://www.linkedin.com/in/yuping-wang-5a7178185/)<sup>1,2</sup>,
[Zechun Liu](https://zechunliu.com/)<sup>3</sup>,
[Jiachen Li](https://jiachenli94.github.io/)<sup>1</sup>

<sup>1</sup> University of California, Riverside &nbsp; <sup>2</sup> University of Michigan &nbsp; <sup>3</sup> Meta AI


<p align="center">
  <img src="https://rdd-neurips.github.io/static/images/method.png" style="max-width: 75%; height: auto;">
  <br>
<span style="display: inline-block; text-align: center; width: 100%;">
  <em>RDD formulates demonstration decomposition as an optimal partitioning problem, using retrieval with approximate nearest neighbor search (ANNS) and dynamic programming to efficiently find the optimal decomposition strategy.</em>
  </span>
</p>

## Applications

- **Sub-task Discovery with Prior:** Different to non-prior heuristic sub-task discovery algorithms such as [UVD](https://arxiv.org/abs/2310.08581), RDD identifies sub-tasks that are **visually similar to ones in a given expert labeled sub-task dataset**. This is specially useful when generating additional sub-tasks for fine-tuning or data augmentation, which encourages the policy to reuse learned skills from the original dataset.
- **Planner-visuomotor Alignment:** In hierarchical VLAs, the planner, often a powerful VLM, performs task planning and reasoning to break down complex tasks into simpler sub-tasks with step-by-step language instructions. Conditioned on the generated sub-task instructions, a learning-based visuomotor policy, trained on datasets with short-horizon sub-tasks, performs precise manipulation to complete the sub-tasks one by one, thereby completing long-horizon tasks. RDD automatically decomposes demonstrations into sub-tasks by **aligning the visual features of the decomposed sub-task intervals with those from the training data of the low-level visuomotor policies.**

# Installation

Set up python environment:

```
conda create -n rdd python==3.9 -y && conda activate rdd
./scripts/setup/setup_rdd_env.sh
```

This default installation only supports encoders `LIV`, `ClIP`, `ResNet`. To install full support for other encoders (  `VIP `, `R3M `, `VC-1`) please follow [setup_rdd_env.sh](./scripts/setup/setup_rdd_env.sh).

# Example #1: Kitchen Cleaning

In this simple example, we'll explore the decomposition for two tasks about putting away kitechenwares.

### Cut the .mov into Image Frames

```
python scripts/dataset/video_to_frames.py resources/clean_kitchen/IMG_0600.MOV data/raw_data/clean_kitchen/IMG_0600 --fps 10 --height 720 
python scripts/dataset/video_to_frames.py resources/clean_kitchen/IMG_0601.MOV data/raw_data/clean_kitchen/IMG_0601 --fps 10 --height 720
```

### Convert the Frames into RLBench Format

```
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
   # agibotworld
   vec_database_path: "data/vec_databases/agibotworld_train/train"
   mode: "default" # * will set beta to 0.0
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

# Example #2: AgiBotWorld & RoboCerebra

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

## Checklist

* [X] Release the core algorithm and demo scripts on AgiBotWorld & RoboCerebra. (ETA: by the end of Oct.2025).
* [ ] Release use case scripts on hiearchical VLA (RACER). (ETA: expect delay, by the end of Dec.2025)

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{yan2025rdd,
  title={RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks},
  author={Yan, Mingxuan and Wang, Yuping and Liu, Zechun and Li, Jiachen},
  booktitle={Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025},
}
```
