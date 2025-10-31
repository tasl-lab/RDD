from typing import Callable, List, NamedTuple, Tuple, Any, Literal, Union

import torch
import numpy as np
from scipy.signal import savgol_filter, argrelextrema
import uvd.utils as U
from uvd.decomp.kernel_reg import KernelRegression
from uvd.decomp import decomp_trajectories


DEFAULT_DECOMP_KWARGS = dict(
    embed=dict(
        normalize_curve=False,
        min_interval=18,
        smooth_method="kernel",
        gamma=0.08,
    ),
    embed_no_robot=dict(
        window_length=8,
        derivative_order=1,
        derivative_threshold=1e-3,
        threshold_subgoal_passing=None,
    ),
    embed_no_robot_extended=dict(
        window_length=3,
        derivative_order=1,
        derivative_threshold=1e-3,
        threshold_subgoal_passing=None,
    ),
    oracle=dict(),
    random=dict(num_milestones=(3, 6)),
    equally=dict(num_milestones=(3, 6)),
    near_future=dict(advance_steps=5),
)



class DecompMeta(NamedTuple):
    milestone_indices: list
    milestone_starts: list = None
    iter_curves: list[np.ndarray] = None


def embedding_decomp(
	embeddings: Union[np.ndarray, torch.Tensor],
	keypoint_num: int = None,
	normalize_curve: bool = True,
	min_interval: int = 18,
	window_length: int = None,
	smooth_method: Literal["kernel", "savgol"] = "kernel",
	extrema_comparator: Callable = np.greater,
	fill_embeddings: bool = True,
	return_intermediate_curves: bool = False,
	**kwargs,
) -> tuple[Union[torch.Tensor, np.ndarray], DecompMeta]:
	if torch.is_tensor(embeddings):
		device = embeddings.device
		embeddings = U.any_to_numpy(embeddings)
	else:
		device = None
	# L, N
	assert embeddings.ndim == 2, embeddings.shape
	traj_length = embeddings.shape[0]

	cur_goal_idx = traj_length - 1
	goal_indices = [cur_goal_idx]
	cur_embeddings = embeddings[
		max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
	]
	iterate_num = 0
	iter_curves = [] if return_intermediate_curves else None
	while cur_goal_idx > (window_length or min_interval):
		if keypoint_num is not None and iterate_num >= keypoint_num:
			break
		iterate_num += 1
		# get goal embedding
		goal_embedding = cur_embeddings[-1]
		distances = np.linalg.norm(cur_embeddings - goal_embedding, axis=1)
		if normalize_curve:
			distances = distances / np.linalg.norm(cur_embeddings[0] - goal_embedding)

		x = np.arange(
			max(0, cur_goal_idx - (window_length or cur_goal_idx)), cur_goal_idx + 1
		)

		if smooth_method == "kernel":
			smooth_kwargs = dict(kernel="rbf", gamma=0.08)
			smooth_kwargs.update(kwargs or {})
			kr = KernelRegression(**smooth_kwargs)
			kr.fit(x.reshape(-1, 1), distances)
			distance_smoothed = kr.predict(x.reshape(-1, 1))
		elif smooth_method == "savgol":
			smooth_kwargs = dict(window_length=85, polyorder=2, mode="nearest")
			smooth_kwargs.update(kwargs or {})
			distance_smoothed = savgol_filter(distances, **smooth_kwargs)
		elif smooth_method is None:
			distance_smoothed = distances
		else:
			raise NotImplementedError(smooth_method)

		if iter_curves is not None:
			iter_curves.append(distance_smoothed)

		extrema_indices = argrelextrema(distance_smoothed, extrema_comparator)[0]
		x_extrema = x[extrema_indices]

		update_goal = False
		for i in range(len(x_extrema) - 1, -1, -1):
			if cur_goal_idx < min_interval:
				break
			if (
				cur_goal_idx - x_extrema[i] > min_interval
				and x_extrema[i] > min_interval
			):
				cur_goal_idx = x_extrema[i]
				update_goal = True
				goal_indices.append(cur_goal_idx)
				break

		if not update_goal or cur_goal_idx < min_interval:
			break
		cur_embeddings = embeddings[
			max(0, cur_goal_idx - (window_length or cur_goal_idx)) : cur_goal_idx + 1
		]

	goal_indices = goal_indices[::-1]
	if fill_embeddings:
		milestone_embeddings = np.concatenate(
			[embeddings[goal_indices[0], ...][None]]
			+ [
				np.full((end - start, *embeddings.shape[1:]), embeddings[end, ...])
				for start, end in zip([0] + goal_indices[:-1], goal_indices)
			],
		)
		if device is not None:
			milestone_embeddings = U.any_to_torch_tensor(
				milestone_embeddings, device=device
			)
	else:
		milestone_embeddings = None
	return milestone_embeddings, DecompMeta(
		milestone_indices=goal_indices, iter_curves=iter_curves
	)


def uvd_decompose(
	embeds: np.ndarray,
	**kwargs
) -> List[List[int]]:
	"""Quick API for UVD decomposition."""
	vid_len, feat_len = embeds.shape
	configs = DEFAULT_DECOMP_KWARGS['embed']
	configs.update(kwargs)
	_, decomp_meta = embedding_decomp(embeddings=embeds, **configs)
	indices = decomp_meta.milestone_indices
	# indices to segments
	if indices[0] != 0:
		indices = [0] + indices
	if indices[-1] != vid_len - 1:
		indices = indices + [vid_len-1]
	indices = [list(range(indices[i], indices[i + 1])) for i in range(len(indices) - 1)]
	return indices


def _uvd_decompose(
	embeds: np.ndarray,
) -> List[List[int]]:
	"""Quick API for UVD decomposition."""
	vid_len, feat_len = embeds.shape
	_, decomp_meta = decomp_trajectories("embed", embeds)
	indices = decomp_meta.milestone_indices
	# indices to segments
	if indices[0] != 0:
		indices = [0] + indices
	if indices[-1] != vid_len - 1:
		indices = indices + [vid_len-1]
	indices = [list(range(indices[i], indices[i + 1])) for i in range(len(indices) - 1)]
	return indices