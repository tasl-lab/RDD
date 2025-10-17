# **RDD**: **R**etrieval-Based **D**emonstration **D**ecomposer for Planner Alignment in Long-Horizon Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2510.14968-red)](https://arxiv.org/pdf/2510.14968)
[![Website](https://img.shields.io/badge/Website-RDD-blue)](https://rdd-neurips.github.io/)
![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-purple)
![License](https://img.shields.io/badge/license-MIT-blue.svg)



**RDD** is an retrieval-based visual demonstration decomposer that automatically identifies sub-tasks visually similar to a set of existing expert-labeled sub-tasks.

[Mingxuan Yan<sup>1</sup>](https://waterhyacinthinnanhu.github.io/),
[Yuping Wang<sup>1,2</sup>](https://www.linkedin.com/in/yuping-wang-5a7178185/),
[Zechun Liu<sup>3</sup>](https://zechunliu.com/),
[Jiachen Li<sup>1</sup>](https://jiachenli94.github.io/)

<sup>1</sup> University of California, Riverside; <sup>2</sup> University of Michigan; <sup>3</sup> Meta AI

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


## Checklist

* [ ] Release the core algorithm and demo scripts on AgiBotWorld & RoboCerebra. (ETA: by the end of Oct.2025).
* [ ] Release use case scripts on hiearchical VLA (RACER). (ETA: by the end of Nov.2025)


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
