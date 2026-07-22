from pathlib import Path
import sys, os
sys.path.append(os.getcwd())
from typing import List, Dict

import typer
import numpy as np
from scipy.stats import rankdata

from utils.file_sys import list_dir, ensure_path_exists, save_as_pickle, load_from_pickle, remove_path


# Tasks on which the RVT2 visuomotor policy is itself sub-optimal; excluded from
# the aggregate with --exclude-suboptimal-tasks so the planner comparison is fair.
SUB_OPTIMAL_TASKS = [
	"insert_onto_square_peg",
	"place_shape_in_shape_sorter",
	"place_cups",
	"stack_blocks",
	"stack_cups",
]

# Tasks whose sub-tasks are in the planner's finetuning set; excluded with
# --exclude-train-tasks to measure generalization to unseen tasks.
TRAIN_FINETUNE_TASKS = [
	"close_jar",
	"insert_onto_square_peg",
	"light_bulb_in",
]


def get_stat(results: Dict[str, List[float]]):
	means = {k: np.array(v).mean() for k, v in results.items()}
	stds = {k: np.array(v).std() for k, v in results.items()}
	return means, stds


def get_success_rate(exp_path, exclude_suboptimal_tasks: bool = False, exclude_train_tasks: bool = False):
	results = {}
	for task_path in list_dir(exp_path):
		task_name = task_path.name
		if exclude_suboptimal_tasks and any(task_name.startswith(t) for t in SUB_OPTIMAL_TASKS):
			continue
		if exclude_train_tasks and any(task_name.startswith(t) for t in TRAIN_FINETUNE_TASKS):
			continue
		if task_name.startswith("test"):
			continue
		if not task_path.is_dir():
			continue
		results[task_name] = []
		for ep_path in list_dir(task_path):
			log_files = []
			for log_file_path in list_dir(ep_path):
				if log_file_path.name.startswith("failure_") or log_file_path.name.startswith("success_"):
					log_files.append(log_file_path)
			if len(log_files) > 1:
				raise ValueError(f"More than one log file found in {ep_path}: {log_files}")
			if len(log_files) == 0:
				raise ValueError(f"No log file found in {ep_path}")
			log_file = log_files[0]
			if log_file.name.startswith("failure_"):
				results[task_name].append(0)
			elif log_file.name.startswith("success_"):
				results[task_name].append(1)
			else:
				raise ValueError(f"Unknown log file found in {ep_path}: {log_file}")
	results = {k: results[k] for k in sorted(results.keys())}
	return get_stat(results), results


def get_latex_tab_format(decimal: int = 1):
	num_format = "{:." + str(decimal) + "f}"
	pattern = '& {} \\text{{{{\\tiny ± {} }}}}'.format(num_format, num_format)
	return num_format, pattern


def print_results(all_res: dict, decimal: int = 1, n_lines: int = 2):
	lengths = {task: len(v) for task, v in all_res.items()}
	if len(set(lengths.values())) > 1:
		majority = max(set(lengths.values()), key=list(lengths.values()).count)
		odd = sorted(t for t, n in lengths.items() if n != majority)
		raise ValueError(
			f"Runs cover different task sets ({sorted(set(lengths.values()))} runs per task); "
			f"aggregate would be meaningless. Inconsistent tasks: {odd}"
		)
	all_res_array = np.array(list(all_res.values()))  # (task_n, exp_n)
	means, stds = get_stat(all_res)
	num_format, pattern = get_latex_tab_format(decimal)
	for task, success_rate in means.items():
		print("Task: {}, Success Rate: {}".format('{}', num_format).format(task, success_rate * 100))
	print("Average Success Rate: {}".format(num_format).format(np.mean(list(means.values())) * 100))
	print("latex format:")
	mean_list = np.array(list(means.values()))
	std_list = np.array(list(stds.values()))
	print(pattern.format(all_res_array.mean(0).mean() * 100, all_res_array.mean(0).std() * 100), end=" ")
	print("& RANK ", end=" ")
	break_points = np.linspace(0, len(mean_list) + 2 - 1, n_lines + 1).astype(int)[1:] - 2
	for i, (m, v) in enumerate(zip(mean_list, std_list)):
		print(pattern.format(m * 100, v * 100), end=" ")
		if i in break_points:
			print("\\\\")
	print("")


def main(
	exp_path: List[str] = typer.Argument(..., help="Path(s) to experiment run directories to aggregate"),
	decimal: int = typer.Option(1, help="Number of decimal places to round to"),
	n_lines: int = typer.Option(2, help="Number of lines in latex table"),
	exp_name: str = typer.Option(None, help="Experiment name (key in the summary log)"),
	summary: bool = typer.Option(False, help="Print cross-experiment ranking summary"),
	clear_log: bool = typer.Option(False, help="Clear the summary log before logging this run"),
	exclude_suboptimal_tasks: bool = typer.Option(False, help="Exclude tasks where the visuomotor policy is sub-optimal"),
	exclude_train_tasks: bool = typer.Option(False, help="Exclude tasks in the planner's finetuning set"),
):
	if isinstance(exp_path, str):
		exp_path = [exp_path]
	all_res = {}
	for exp_p in exp_path:
		(means, stds), res = get_success_rate(exp_p, exclude_suboptimal_tasks, exclude_train_tasks)
		print(f"Experiment Path: {exp_p}")
		for k, v in means.items():
			all_res.setdefault(k, [])
			all_res[k] += [v]
	print("=" * 20)
	print_results(all_res, decimal, n_lines)
	save_dir = Path('tmp/eval/')
	save_path = save_dir / "success_rate.pkl"
	ensure_path_exists(save_dir)
	if clear_log:
		remove_path(save_path)
	if save_path.exists():
		log_res = load_from_pickle(save_path)
		log_res[exp_name] = all_res
	else:
		log_res = {exp_name: all_res}
	save_as_pickle(save_path, log_res)
	if summary:
		all_means = []
		for _, res in log_res.items():
			means, _ = get_stat(res)
			all_means.append(list(means.values()))
		try:
			all_means = np.stack(all_means, axis=0)  # (exp_n, task_n)
		except ValueError:
			print("all_means dimensions:")
			for i, m in enumerate(all_means):
				print(f"all_means[{i}]: {len(m)}")
			raise ValueError("all_means dimensions are not consistent across experiments")
		all_ranks = rankdata(-all_means, axis=0)
		all_ranks_mean = all_ranks.mean(axis=1)
		all_ranks_std = all_ranks.std(axis=1)
		print("\nRanking:")
		_, latex_pattern = get_latex_tab_format(decimal)
		for i, k in enumerate(log_res.keys()):
			print(latex_pattern.format(all_ranks_mean[i], all_ranks_std[i]))
		print("\nrelative performance boost (last experiment is treated as ours):")
		all_means = np.concatenate([all_ranks.mean(axis=1, keepdims=True), all_means], axis=1)
		all_means = np.concatenate([all_means[:, 1:].mean(axis=1, keepdims=True), all_means], axis=1)
		for i, k in enumerate(log_res.keys()):
			boost = (all_means[-1, :] - all_means[i, :]) / all_means[i, :] * 100
			print(f"{k}: ")
			for b in boost.tolist():
				print(f"{b:>{decimal + 6}.{decimal}f}", end=" ")
			print("")


if __name__ == "__main__":
	typer.run(main)
