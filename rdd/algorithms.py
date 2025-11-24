from typing import Callable, List, Tuple, Any, Literal
from itertools import combinations

import numpy as np

from rdd.ann import AnnoySearcher
from rdd.embed import subtask_embeds_to_feature
from rdd.uvd_wrapper import uvd_decompose




def check_positive(x: int):
	assert x > 0, "x must be a positive value"


def max_n_partition(u: List[Any], condition: Callable, max_len: int = float('inf'), min_len: int = 1, **kwargs):
	"""
	Compute the maximum number of partitions of string u such that
	for each substring in the partition, property P holds.
	
	Parameters:
		u (str): The input string.
		
	Returns:
		int: The maximum number of partitions (substrings) satisfying P.
			 Returns -1 if no valid partitioning exists.
	"""
	check_positive(max_len)
	check_positive(min_len)
	assert min_len <= max_len, "min_len must be less than or equal to max_len"
	n = len(u)
	# dp[i] stores the maximum number of valid substrings for u[0:i]
	dp = [-1] * (n + 1)
	dp[0] = 0 # Base case: empty string has 0 partitions
	parts = [[]] * (n + 1)  # Store the partitioning
	
	# Iterate over each position i in the string
	for i in range(1, n + 1):
		# Try every possible previous breakpoint j
		for j in range(max(i-max_len, 0), min(i+1-min_len, i)):
			# If substring u[j:i] satisfies P
			if condition(u[j:i], **kwargs):
				dp[i] = max(dp[i], dp[j] + 1)
				# Store the partitioning. Note when dp[i]=dp[j]+1, we use parts[j] + [u[j:i]] as the partition
				if dp[i] == dp[j] + 1:
					parts[i] = parts[j] + [u[j:i]]
		if dp[i] == -1:
			dp[i] = dp[i - 1]
			parts[i] = parts[i - 1]
	return dp[n], parts[n]


def max_sum_partition(u: List[Any], score_func: Callable, max_len: int = None, min_len: int = 1, **kwargs):
	"""
	Compute the maximum score sum of partitions of string u such that
	for each substring in the partition, property P holds.
	Solved by dynamic programing as proposed in https://arxiv.org/pdf/math/0309285
	Parameters:
		u (str): The input string.
		score_func (Callable): A function that takes a substring and returns its score.
		max_len (int): Maximum length of a substring.
		min_len (int): Minimum length of a substring.
	Returns:
		int: The maximum score sum of partitions (substrings) satisfying P.
			 Returns -1 if no valid partitioning exists.x
		list: The list of substrings that form the maximum score sum.
	"""
	n = len(u)
	if max_len is None: max_len = n
	check_positive(max_len)
	check_positive(min_len)
	assert min_len < max_len, "min_len must be less than to max_len"
	# assert max_len <= n, "max_len must be less than or equal to the length of u"
	max_len = min(max_len, n)  # Ensure max_len does not exceed the length of u
	dp = [-float('inf')] * (n + 1) # dp[i] stores the maximum score sum of substrings u[0:i]
	parts = [[]] * (n + 1)  # Store the partitioning

	# Initialize the base cases
	dp[0] = 0
	call_cnt = 0
	# Iterate over each position i in the string
	for i in range(min_len + 1, n + 1):
		# Try every possible previous breakpoint j
		_dp = []
		_parts = []
		for j in range(0, i + 1):
			if i - j < min_len or i - j > max_len:
				continue
			score = score_func(u[j:i], **kwargs)
			call_cnt += 1
			_dp.append(dp[j] + score)
			_parts.append(parts[j] + [u[j:i]])
		if _dp != []:
			dp[i] = max(_dp)
			parts[i] = _parts[_dp.index(dp[i])]
		else:
			dp[i] = dp[i - 1]
			parts[i] = parts[i - 1]
	# v = max_len - min_len + 1
	# print(f"call_cnt: {call_cnt}, expect: {(v-1)*(v+2)/2 + (n-max_len)*v}")
	return dp[n], parts[n]


def rdd_score(
	subarray: List[int], 
	searcher: AnnoySearcher, 
	embeds: np.ndarray, 
	alpha: float = 1.0, 
	beta: float = 1.0,
	dist_list: List[float] = None, 
	nn_list: dict = None,
	mode: Literal['default', 'ood'] = 'default',
	uvd_cache: dict = None
	):
	s, e = subarray[0], subarray[-1]
	assert s < e, "subarray must be a valid segment"

	# retrieval score
	query_vec = subtask_embeds_to_feature()(embeds[s:e+1], mode=mode)
	nn, min_dist = searcher.query(query_vec, num_neighbors=1)
	nn, min_dist = nn[0], min_dist[0]
 
	# heuristic length score
	length_div_penalty = abs(1 - (e - s) / nn[0].duration)
 
	# uvd score: assume the last segment is the goal frame, evaluate the accuracy of the beginning of the last segment
	if beta != 0.:
		if uvd_cache is not None and e in uvd_cache:
			last_uvd_seg_starts_at = uvd_cache[e]
		else:
			uvd_segs = uvd_decompose(embeds[:e+1], keypoint_num=1)
			last_uvd_seg_starts_at = uvd_segs[-1][0]
		if uvd_cache is not None:
			uvd_cache[e] = last_uvd_seg_starts_at
		uvd_penalty = abs(s - last_uvd_seg_starts_at) / (e - s)
	else:
		uvd_penalty = 0.0


	# alpha
	if mode == 'ood':
		print("\033[31m[WARNING]: Setting alpha to 0.0 since mode is ood.\033[0m")
		alpha = 0.0
	
	# log
	if dist_list is not None:
		dist_list.append(min_dist)
	if nn_list is not None:
		nn_list[str(subarray)] = nn
	return - (min_dist + alpha * length_div_penalty + beta * uvd_penalty) * len(subarray)




# Example usage:
if __name__ == "__main__":
	from time import time

	# Example usage of max_sum_partition
	u = np.random.randint(-10, -2, size=500).tolist()
	min_len = 2
	max_len = 100
	print("Input array:", u)
	# score_func = lambda x: np.prod(x) # product of the elements
	score_func = lambda x: 1/(np.sum(x)+1e-5)+np.sum(x) # product of the elements

	def brute_force_partition(u, score_func, min_len, max_len):
		segments = []
		for i in range(1, len(u) + 1):
			for indices in combinations(range(1, len(u)), i - 1):
				indices = (0,) + indices + (len(u),)
				segments.append([u[indices[j]:indices[j + 1]] for j in range(len(indices) - 1)])
		segments = [segment for segment in segments if all(len(s) >= min_len for s in segment)]
		segments = [segment for segment in segments if all(len(s) <= max_len for s in segment)]
		solution_score = lambda segment: sum(score_func(s) for s in segment)
		if not segments:
			raise ValueError("No valid partitioning found.")
		max_score_segment = max(segments, key=solution_score)
		return solution_score(max_score_segment), max_score_segment

	ts = time()
	print(f"Brute Force : {brute_force_partition(u, score_func, min_len=min_len, max_len=max_len)}")
	print(f"Time: {time() - ts:.4f}s")
	ts = time()
	print(f"Our Solution: {max_sum_partition(u, score_func, min_len=min_len, max_len=max_len)}")
	print(f"Time: {time() - ts:.4f}s")
