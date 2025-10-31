from pathlib import Path
from os import PathLike
import sys, os
sys.path.append(os.getcwd())
from typing import List, Tuple, Any, Dict
import requests
import pprint

import typer
from sqlmodel import Field
from sqlalchemy import JSON, Column
import numpy as np

from rdd.datasets.rlbench import RLBenchVecDataset, RLBenchAnnoySearcher
from rdd.embed import EmbedPreprocs
from utils.file_sys import list_dir, ensure_path_exists, remove_path, symlink
from utils.concurrent import AsyncWorkerPool, imap_tqdm
from utils.database import Database, BaseEntry
from utils.images import frames_to_video


EP_NUM = None
TASKS = None
METHODS = ['rdd', 'uvd']


def load_split(dataset_path: PathLike, views: List[str]):
	"""
	returns: {task: (ep, frame, view)}
	"""
	vids = {}
	for task_path in list_dir(dataset_path):
		task_name = task_path.name
		task_vids = []
		episode_dirs = list_dir(task_path)
		for ep_path in sorted(episode_dirs, key=lambda x: int(x.name.split('case')[-1])):
			ep_vid_frames = []
			for view_path in list_dir(ep_path):
				if view_path.name not in views:
					continue
				view_frames = list_dir(view_path)
				view_frames = [f for f in view_frames if f.suffix == '.png']
				view_frames = sorted(view_frames, key=lambda x: int(x.name.split('.')[0]))
				view_frames = [str(f) for f in view_frames]  # convert to str for JSON serialization
				ep_vid_frames.append(view_frames)
			ep_vid_frames = np.array(ep_vid_frames, dtype=object)
			ep_vid_frames = ep_vid_frames.T.tolist()
			task_vids.append(ep_vid_frames)
		vids[task_name] = task_vids
	return vids
			

class Results(BaseEntry, table=True):
	"""Database entry for results."""
	task: str
	ep_idx: int
	method: str
	starts: List[int] = Field(default=None, sa_column=Column(JSON))
	ends: List[int] = Field(default=None, sa_column=Column(JSON))


def _get_segments(args) -> List[List[int]]:
	ids, server_addr, vid, method, preprocessor = args
	response = requests.post(f"{server_addr}/decompose", json={
		"paths": vid,
		"preprocessor": preprocessor,
		"method": method,
	})
	if response.status_code != 200:
		raise Exception(f"Failed to get segments: {response.text}")
	return ids, response.json()['segments']


def load_gt_segments(gt_dataset_path: PathLike) -> Dict[str, List[List[int]]]:
	gt_segments = {}
	for task_path in list_dir(gt_dataset_path):
		task_name = task_path.name
		if TASKS is not None and task_name not in TASKS:
			continue
		episode_dirs = list_dir(task_path)
		task_ep_segments = []
		for ep_path in sorted(episode_dirs, key=lambda x: int(x.name.split('case')[-1]))[:EP_NUM]:
			keypoint_path = sorted(list((ep_path / 'front_rgb').glob('*_expert.png')), key=lambda x: int(x.name.split('_')[0]))
			keypoints = [k.name.split('_')[0] for k in keypoint_path]
			task_ep_segments.append([[int(b), int(e)] for b, e in zip(keypoints[:-1], keypoints[1:])])
		gt_segments[task_name] = task_ep_segments
	return gt_segments


def calculate_iou(seg1, seg2):
	"""Calculate IoU between two segments [start, end]"""
	# Find intersection
	intersection_start = max(seg1[0], seg2[0])
	intersection_end = min(seg1[1], seg2[1])
	# If no intersection
	if intersection_start >= intersection_end:
		return 0.0
	intersection = intersection_end - intersection_start
	# Calculate union
	union = (seg1[1] - seg1[0]) + (seg2[1] - seg2[0]) - intersection
	return intersection / union if union > 0 else 0.0


def find_nearest_neighbors(A, B):
	"""Find nearest neighbor in B for each segment in A based on max IoU"""
	nearest_neighbors = []
	for seg_a in A:
		max_iou = -1
		nearest_idx = -1
		nearest_seg = None
		for i, seg_b in enumerate(B):
			iou = calculate_iou(seg_a, seg_b)
			if iou > max_iou:
				max_iou = iou
				nearest_idx = i
				nearest_seg = seg_b
		nearest_neighbors.append(nearest_seg)
	return nearest_neighbors


def get_accuracy(seg_pred, seg_gt):
	assert len(seg_pred) > 0, "Ground truth segments cannot be empty"
	ious = []
	pred_nn = find_nearest_neighbors(seg_pred, seg_gt)
	for gt, pred in zip(seg_pred, pred_nn):
		ious.append(calculate_iou(gt, pred))
	return np.mean(ious)


def main(
	dataset_path: str = typer.Argument('data/rlbench_raw/robocerebra/'),
	results_save_path: str = typer.Argument('data/eval_out/robocerebra/'),
	preprocessor: str = typer.Argument('liv'),
	worker_num: int = typer.Option(4, help='Number of workers for concurrent processing.'),
	rdd_port: int = typer.Option(8001, help='Port number for the RDD server.'),
	eval_only: bool = typer.Option(False, help='eval only.'),
	views: List[str] = typer.Option(['front_rgb'], help='Views to use for the dataset.'),
	tasks: List[str] = typer.Option(None, help='Tasks to evaluate. If None, evaluate all tasks.'),
	ep_num: int = typer.Option(None, help='Number of episodes to evaluate per task. If None, evaluate all episodes.'),
):
	TASKS = tasks
	EP_NUM = ep_num
	dataset_path = Path(dataset_path) / 'splits/val'
	gt_dataset_path = dataset_path.parent / 'val_gt'
	ensure_path_exists(results_save_path)
	if not eval_only:
		if (Path(results_save_path) / 'results.db').exists():
			remove_path(Path(results_save_path) / 'results.db')
			input('Romoving old results database? press Enter to continue...')
			print(f"Removed old results database at {results_save_path}/results.db")
		results = Database(Path(results_save_path) / 'results.db', entry_class=Results, auto_commit=True)
		# pool = AsyncWorkerPool(worker_num=worker_num, worker_type='thread')
		vids = load_split(dataset_path, views)
		tasks = []
		for task_name, task_vids in vids.items():
			if TASKS is not None and task_name not in TASKS:
				continue
			for ep_idx, vid in list(enumerate(task_vids))[:EP_NUM]:
				for method in METHODS:
					tasks.append(
						(
							(task_name, ep_idx, method), # id
							f"http://localhost:{rdd_port}", 
							vid, 
							method, 
							preprocessor
						)
					)
		mp_results = imap_tqdm(_get_segments, tasks, processes=worker_num, desc="Decomposing...")
		
		viz_save_path = Path(results_save_path) / 'visualization'
		if viz_save_path.exists():
			remove_path(viz_save_path)
		# for task_name, ep_idx, method, token in tasks:
		for (task_name, ep_idx, method), segments in mp_results:
			viz_save_path_ep = viz_save_path / task_name / f"ep_{ep_idx}" / method
			ensure_path_exists(viz_save_path_ep)
			for s in [seg[0] for seg in segments]:
				src_image_path = Path(vids[task_name][ep_idx][s][0])  # Assuming the first view is the main one
				dst_image_path = viz_save_path_ep / src_image_path.name
				symlink(src_image_path, dst_image_path)
			results.add(Results(
				task=task_name,
				ep_idx=ep_idx,
				method=method,
				starts= [seg[0] for seg in segments],
				ends= [seg[-1] for seg in segments],
			))
		results.commit()
	else:
		if not (Path(results_save_path) / 'results.db').exists():
			raise ValueError(f"Results database not found at {results_save_path}/results.db")
		results = Database(Path(results_save_path) / 'results.db', entry_class=Results)
  
	# eval
	gt_segments = load_gt_segments(gt_dataset_path)
	accs = {}
	for method in METHODS:
		accs[method] = {}
		for task in gt_segments.keys():
			accs[method][task] = []
			for ep_idx, gt_ep_segs in list(enumerate(gt_segments[task]))[:EP_NUM]:
				pred_segs: List[Results] = results.select(task=task, method=method, ep_idx=ep_idx)
				if len(pred_segs) == 0:
					raise ValueError(f"empty segment prediction for task {task}, method {method}, episode {ep_idx}")
				pred_segs = [[int(s), int(e)] for s, e in zip(pred_segs[0].starts, pred_segs[0].ends)]
				# accs[method][task].append(get_accuracy(gt_ep_segs, pred_segs))
				accs[method][task].append(get_accuracy(pred_segs, gt_ep_segs))
			accs[method][task] = np.mean(accs[method][task])
	pprint.pprint(accs)

	
if __name__ == "__main__":
	typer.run(main)
    