# Example #4: RACER (Hierarchical VLA)

This example reproduces RDD's *planner–visuomotor alignment* use case on RLBench
with [RACER](https://github.com/sled-group/RACER): a LLaVA-NeXT **planner**
breaks a long-horizon task into sub-task instructions, and an RVT2 **visuomotor
policy** executes them. RDD decomposes the expert demos into sub-tasks that are
visually aligned with the policy's training data; those sub-tasks become the
planner's finetuning data, improving planner–policy alignment.

The pipeline spans three companion forks (cloned by the setup script):

| Repo | Role |
|------|------|
| [RACER](https://github.com/WaterHyacinthInNANHU/RACER) `@main` | RVT2 visuomotor policy + rollout eval |
| [RACER-DataGen](https://github.com/WaterHyacinthInNANHU/RACER-DataGen) `@release` | decompose demos + Gemini rich-language labels → planner data |
| [Open-LLaVA-NeXT](https://github.com/WaterHyacinthInNANHU/Open-LLaVA-NeXT) `@racer_llava` | LLaVA planner finetune + serving |

## 1. Setup

```bash
./scripts/setup/setup_racer_env.sh
```
Creates conda envs `racer`, `racer_datagen`, `llava-next`, and downloads the
`racer-visuomotor-policy-rich`, `llama3-llava-next-8b`, and `t5-11b` models.
Requires CoppeliaSim and CUDA 11.7 (see the RACER install guide).

## 2. Build the vector database (expert sub-task prior)

```bash
python build_vec_database.py 0 liv 1.0 data/rlbench_raw/RACER-augmented_rlbench \
	--name-suffix rlbench_aug_liv_1.0 \
	--views front_rgb
```

## 3. Serve the RDD decomposer

Uncomment the `rlbench (RACER)` block in [configs/rdd_server.yaml](../configs/rdd_server.yaml), then:
```bash
./scripts/eval/serve_rdd.sh   # POST /decompose on :8001
```

## 4. Generate planner finetuning data

Done inside the RACER-DataGen fork. Edit `pipeline.sh` (`KEYPT_METHOD ∈
{rdd, uvd, heuristic, fixed_interval, gemini-2.5-pro-preview-05-06}`, `OUT_PATH`,
`NUMBER_OF_EP`, `RLBENCH_PATH`, VNC display) and set `RDD_SERVER_ADDR` in
`racer_datagen/utils/const_utils.py` to the port-8001 server for the `rdd`
method, then:
```bash
cd 3rdparty/RACER-DataGen && bash pipeline.sh
```
Repeat per method to produce one finetuning dataset each.

## 5. Finetune the planner

```bash
./scripts/eval/finetune_llava.sh <datagen_run_dir> rdd   # -> checkpoints/rdd
```

## 6. Serve planner + tokenizer

```bash
./scripts/eval/serve_tokenizer.sh 20001 0        # T5 encoder
./scripts/eval/serve_llava.sh    21002 4 rdd     # planner worker 1
./scripts/eval/serve_llava.sh    21003 5 rdd     # planner worker 2
```

## 7. Evaluate (RVT2 rollout, 10 seeds)

```bash
./scripts/eval/run_sets.sh              # brings planner up, runs eval, tears down
# or directly, with planner/tokenizer already serving:
./scripts/eval/eval_racer.sh 6 rdd      # writes racer/runs/<MODEL>/rdd-{0..9}
```
Run once per method (`vanilla_llava`, `fixed_interval`, `uvd`, `heuristic`, `rdd`).

## 8. Aggregate results

```bash
./scripts/eval/experiments/main.sh           # main success-rate comparison
./scripts/eval/experiments/abl_alpha.sh      # ablation: retrieval-prior weight alpha
./scripts/eval/experiments/abl_encoder.sh    # ablation: visual encoder
./scripts/eval/experiments/abl_ep_num.sh     # ablation: number of demos
./scripts/eval/experiments/abl_vec_sample.sh # ablation: vector-database sampling rate
./scripts/eval/experiments/gemini_pro.sh     # Gemini-2.5-Pro planner baseline
./scripts/eval/experiments/train_set.sh      # generalization to unseen tasks
```

## Expected results (from the paper, arXiv:2510.14968)

**Main — multi-task success rate on RLBench (avg % ± std, avg rank; lower rank better):**

| Method | Success ↑ | Rank ↓ |
|--------|-----------|--------|
| w/o Finetune | 52.6 ± 8.2 | 4.5 |
| Uniform (fixed-interval) | 71.3 ± 5.4 | 3.1 |
| UVD | 71.4 ± 5.1 | 3.0 |
| **RDD (ours)** | **74.9 ± 6.9** | **2.2** |
| Expert (heuristic) | 75.1 ± 4.7 | 2.2 |

RDD matches expert-level decomposition without expert labels. Ablations
(encoder, α, #demos, vector-sampling rate) and the Gemini-2.5-Pro planner
comparison are reproduced by the `scripts/eval/experiments/*.sh` drivers; see
the paper for the full per-table numbers. Decomposition-accuracy (IoU) is
covered separately by [Example #3](agi_cerebra_demo.md) via `eval_rdd.py`.
