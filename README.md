# **RDD**: **R**etrieval-Based **D**emonstration **D**ecomposer

[![arXiv](https://img.shields.io/badge/arXiv-2510.14968-red)](https://arxiv.org/pdf/2510.14968)
[![Website](https://img.shields.io/badge/Website-RDD-blue)](https://rdd-neurips.github.io/)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/115042)
[![YouTube](https://img.shields.io/badge/YouTube-Video-white)](https://www.youtube.com/watch?v=bwCgUyqdT6s&embeds_referring_euri=https%3A%2F%2Frdd-neurips.github.io%2F&source_ve_path=OTY3MTQ)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

<p align="center">
  <img src="resources/figures/teaser.png" style="max-width: 75%; height: auto;">
  <br>
</p>

**RDD** is an retrieval-based visual demonstration decomposer that automatically identifies sub-tasks visually similar to a set of existing expert-labeled sub-tasks.

[Mingxuan Yan](https://waterhyacinthinnanhu.github.io/)`<sup>`1`</sup>`,
[Yuping Wang](https://www.linkedin.com/in/yuping-wang-5a7178185/)`<sup>`1,2`</sup>`,
[Zechun Liu](https://zechunliu.com/)`<sup>`3`</sup>`,
[Jiachen Li](https://jiachenli94.github.io/)`<sup>`1`</sup>`

`<sup>`1`</sup>` University of California, Riverside
`<sup>`2`</sup>` University of Michigan
`<sup>`3`</sup>` Meta AI

## Applications

- **Sub-task Discovery with Prior:** Different to non-prior heuristic sub-task discovery algorithms such as [UVD](https://arxiv.org/abs/2310.08581), RDD identifies sub-tasks that are **visually similar to ones in a given expert labeled sub-task dataset**. This is specially useful when generating additional sub-tasks for fine-tuning or data augmentation, which encourages the policy to reuse learned skills from the original dataset.
- **Planner-visuomotor Alignment:** ([Youtube Video](https://www.youtube.com/watch?v=bwCgUyqdT6s&embeds_referring_euri=https%3A%2F%2Frdd-neurips.github.io%2F&source_ve_path=OTY3MTQ)) In hierarchical VLAs, the planner, often a powerful VLM, performs task planning and reasoning to break down complex tasks into simpler sub-tasks with step-by-step language instructions. Conditioned on the generated sub-task instructions, a learning-based visuomotor policy, trained on datasets with short-horizon sub-tasks, performs precise manipulation to complete the sub-tasks one by one, thereby completing long-horizon tasks. RDD automatically decomposes demonstrations into sub-tasks by **aligning the visual features of the decomposed sub-task intervals with those from the training data of the low-level visuomotor policies.**

## Method

<p align="center">
  <img src="https://rdd-neurips.github.io/static/images/method.png" style="max-width: 75%; height: auto;">
  <br>
<span style="display: inline-block; text-align: center; width: 100%;">
  <em>RDD formulates demonstration decomposition as an optimal partitioning problem, using retrieval with approximate nearest neighbor search (ANNS) and dynamic programming to efficiently find the optimal decomposition strategy.</em>
  </span>
</p>

# Installation

Set up python environment:

```
conda create -n rdd python==3.9 -y && conda activate rdd
./scripts/setup/setup_rdd_env.sh
```

This default installation only supports encoders `LIV`, `ClIP`, `ResNet`. To install full support for other encoders (  `VIP `, `R3M `, `VC-1`) please follow [setup_rdd_env.sh](./scripts/setup/setup_rdd_env.sh).

# Example #1: Kitchen Cleaning

See [kitchen_demo.md](doc/kitchen_demo.md)

# Example #2: Franka Object Arranging

See [franka_demo.md](doc/franka_demo.md)

# Example #3: AgiBotWorld & RoboCerebra

See [agi_cerebra_demo.md](doc/agi_cerebra_demo.md)

# Custom Prior

RDD allows you to define your own sub-task prior.

## Custom Sub-task Feature

In RDD the sub-task is represented by the feature of of ending frame / starting frame. You can define your own sub-task feature by modifying sub-task feature generation class  ` subtask_embeds_to_feature` in  `rdd/embed.py`

```python
class subtask_embeds_to_feature(object):
	@staticmethod
	def feature_dim(embed_dim: int, mode: str) -> int:
		<your feature dim>
	def __call__(self, embeds: Union[np.ndarray, torch.Tensor], mode: str) -> np.ndarray:
		"""
		input: embeds (L, N)
		output: feature (2N,) or (N,) if mode=='ood'
		"""
		<your implementation>
```

## Custom Sub-task Prior Score

You can also define a completely custimized sub-task prior score by modifying `rdd_score` in `rdd/algorithms.py`

```python
def rdd_score(
	subarray: List[int], 
	searcher: AnnoySearcher, 
	embeds: np.ndarray, 
	...
	):
	<your implementation>
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
