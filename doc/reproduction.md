# Reproducing the paper results

This guide covers reproducing the quantitative results in
[RDD (NeurIPS 2025)](https://arxiv.org/pdf/2510.14968). The core algorithm demos live in
Examples #1–#4 from the [README](../README.md); this document is the practical companion —
what each table needs, what it costs, and the pitfalls that only show up when you run it.

## Which table needs which pipeline

| Paper table | What it measures | Pipeline | Driver |
|---|---|---|---|
| Table 1 | RLBench multi-task success rate (5 decomposition methods) | RACER, full 8 stages | `scripts/eval/experiments/main.sh` |
| Table 2 | Ablation: visual encoder | RACER | `experiments/abl_encoder.sh` |
| Table 3 | Ablation: retrieval weight α | RACER | `experiments/abl_alpha.sh` |
| Table 4 | Ablation: number of demos in the prior | RACER | `experiments/abl_ep_num.sh` |
| Table 5 | Decomposition accuracy (IoU), AgiBotWorld / RoboCerebra | decomposition only | `eval_rdd.py` — see [Example #3](agi_cerebra_demo.md) |
| Table 6 | Planner without finetuning on the target task (transfer to unseen tasks) | RACER (15-task subset) | `experiments/train_set.sh` |
| Table 7 | Gemini-2.5-Pro planner baseline | RACER | `experiments/gemini_pro.sh` |

Appendix tables extend Tables 1–4 with the per-task breakdown over all 18 RLBench tasks; the
drivers above already print per-task rows alongside the average.

**Table 6** measures whether a planner finetuned on some tasks transfers to tasks it never saw.
`success_rate.py --exclude-train-tasks` drops the three finetuning tasks (`close_jar`,
`insert_onto_square_peg`, `light_bulb_in`), which is exactly the complement of the 15-task list
in `eval_racer_partial.sh` — use that script to produce the runs.

**Table 5 is by far the cheapest** — it needs only the decomposer (no simulator, no planner
finetuning). If you want to validate RDD itself, start there. Everything else routes through
the full RACER stack described in [Example #4](racer_demo.md).

## What it costs

Measured on 8×RTX 6000 Ada (49GB). Per-unit figures are real; totals are extrapolations.

| Step | Time | Resources |
|---|---|---|
| Build vector DB (1 task, ~100 eps, LIV) | ~30 s | 1 GPU |
| Build vector DB (all 18 tasks, rate 1.0) | minutes | 1 GPU, **~10 GB disk** |
| Decompose one 166-frame episode | ~10 s (RDD) / ~5 s (UVD) | 1 GPU |
| Finetune planner (LoRA, 2 epochs, 336 steps) | **~9 min** | 4 GPUs, ~26 GB each; **~750 MB** per checkpoint |
| Serve planner | ~20 s load | 1 GPU, ~18–20 GB |
| Serve T5 tokenizer | ~1 min load | 1 GPU, ~19 GB |
| Rollout, 1 episode | ~4 min | 1 GPU per evaluator + CoppeliaSim |

**Budget for the full Table 1:** 5 methods × 10 seeds × 18 tasks × 25 episodes ≈ 22.5k
episodes. This is days of wall-clock even with several parallel evaluators, and the rollout
logs for the paper's complete run occupy **~210 GB — 98% of it per-episode `.gif` files**.
If disk is tight, delete the GIFs; `success_rate.py` reads only the tiny
`success_*`/`failure_*` marker files (<1 MB total across all runs).

## End-to-end sequence

Follow [Example #4](racer_demo.md) for the full walkthrough. In brief, per decomposition
method `M ∈ {vanilla_llava, fixed_interval, uvd, heuristic, rdd}`:

```bash
# once
./scripts/setup/setup_racer_env.sh
python build_vec_database.py 0 liv 1.0 data/rlbench_raw/RACER-augmented_rlbench/train \
    --name-suffix rlbench_aug_liv_1.0 --views front_rgb --embed-mode default
./scripts/eval/serve_rdd.sh                      # :8001

# per method
cd 3rdparty/RACER-DataGen && bash pipeline.sh    # KEYPT_METHOD=<M>; needs a Gemini API key
./scripts/eval/finetune_llava.sh <datagen_run_dir> <M>
./scripts/eval/serve_tokenizer.sh 20001 <gpu>    # T5
./scripts/eval/serve_llava.sh    21002 <gpu> <M> # planner worker 1
./scripts/eval/serve_llava.sh    21003 <gpu> <M> # planner worker 2
./scripts/eval/eval_racer.sh 6 <M>               # -> <runs>/<model>/<M>-{0..9}

# once all methods are done
./scripts/eval/experiments/main.sh
```

`vanilla_llava` skips the datagen and finetune steps — it serves the base
`llama3-llava-next-8b` directly.

## Reference numbers

`experiments/main.sh` on a correct set of runs should reproduce Table 1:

| Method | Success ↑ | Rank ↓ |
|--------|-----------|--------|
| w/o Finetune | 52.6 ± 8.2 | 4.5 ± 1.2 |
| Uniform (fixed-interval) | 71.3 ± 5.4 | 3.1 ± 1.2 |
| UVD | 71.4 ± 5.1 | 3.0 ± 1.3 |
| **RDD (ours)** | **74.9 ± 6.9** | **2.2 ± 0.9** |
| Expert (heuristic) | 75.1 ± 4.7 | 2.2 ± 1.0 |

Averaged over 10 seeds and the 13-task filtered set. If your averages land far from these,
check pitfalls 1, 5 and 7 first — each produces plausible-looking but wrong aggregates.
