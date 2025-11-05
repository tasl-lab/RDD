import sys, os
sys.path.append(os.getcwd())

import typer
import numpy as np

from utils.file_sys import *
from utils.concurrent import AsyncWorkerPool
from utils.images import extract_frames_from_vid
import re


def convert_robocerebra_to_rlbench(
	robocerebra_path: str = typer.Argument('data/robocerebra'),
	save_to: str = typer.Argument('data/rlbench_raw/robocerebra'),
	worker_num: int = typer.Argument(4),
	num_episodes: int = typer.Option(None),
	downsample_rate: int = typer.Option(12, help='Downsample rate for frames extraction. 1 means no downsampling.'),
	trainset_ratio: float = typer.Option(0.8, help='Ratio of episodes to use for training set. The rest will be used for validation.'),
	intermediate_steps: int = typer.Option(2, help='Number of intermediate frames per segment.'),
):
	assert intermediate_steps >= 0, "number of intermediate frames must be prositive"
	# output paths
	raw_dataset_path = Path(save_to) / 'raw'
	seg_dataset_path = Path(save_to) / 'seg'
	seg_dataset_split_path = Path(save_to) / 'splits'
	metadata_path = Path(save_to) / 'metadata'
	ensure_path_exists(metadata_path)
	ensure_path_exists(raw_dataset_path)
	ensure_path_exists(seg_dataset_path)
	if (metadata_path / 'metadata.pkl').exists():
		old_metadata = load_from_pickle(metadata_path / 'metadata.pkl')
	else:
		old_metadata = {}
	# read dataset
	ep_iterator = []
	for task_dir in list_dir(robocerebra_path):
		if not task_dir.is_dir():
			continue
		task_name = task_dir.name
		ep_n = 0
		episode_dirs = [p for p in list_dir(task_dir) if p.is_dir() and p.name.startswith('case')]
		for episode_dir in sorted(episode_dirs, key=lambda x: int(x.name.split('case')[-1])):
			ep_iterator.append((task_name, episode_dir))
			ep_n += 1
			if num_episodes is not None and ep_n >= num_episodes:
				break

	# build raw_dataset by extracting frames from videos
	tasks = []
	for task_name, episode_dir in ep_iterator:
		if (task_name, episode_dir) in old_metadata.get('ep_iterator', []):
			continue
		episode_name = episode_dir.name
		video_path = episode_dir / f'{str(episode_name)}.mp4'
		frames_dir = raw_dataset_path / task_name / episode_name / 'front_rgb'
		tasks.append((video_path, frames_dir, downsample_rate))
	pool = AsyncWorkerPool(worker_num)
	for args in tasks:
		pool.add_task(extract_frames_from_vid, *args)
	pool.wait_for_results()

	# build seg_dataset by symlink frames and videos
	seg_ep_dict = {}
	for task_name, episode_dir in ep_iterator:
		if task_name not in seg_ep_dict: seg_ep_dict[task_name] = []
		if (task_name, episode_dir) in old_metadata.get('ep_iterator', []):
			continue
		task_desc_path = episode_dir / 'task_description.txt'
		with open(task_desc_path, "r") as file:
			docstring = file.read()
		boundaries = re.findall(r'\[(\d+),\s*(\d+)\]', docstring)
		step_boundaries = [(int(start), int(end)) for start, end in boundaries]
		step_boundaries = [(s, e) for s, e in step_boundaries if e - s - 1 > intermediate_steps]
		keyframe_steps = [start for start, _ in step_boundaries] + [step_boundaries[-1][1]]
		for step in keyframe_steps:
			if downsample_rate is not None:
				step = step // downsample_rate if downsample_rate > 1 else step
			src = raw_dataset_path / task_name / episode_dir.name / 'front_rgb' / f'{step:06d}.png'
			dst = seg_dataset_path / task_name / episode_dir.name / 'front_rgb' / f'{step}_expert.png'
			if not src.exists() or dst.exists():
				continue
			ensure_path_exists(dst.parent)
			symlink(relative_path(src, dst.parent), dst)
		# intermediate frames
		for b_step, e_step in step_boundaries:
			cur_intermediate_steps = np.linspace(b_step, e_step, intermediate_steps + 2)[1:-1].astype(int).tolist()
			b_step = b_step // downsample_rate if downsample_rate > 1 else b_step
			for step in cur_intermediate_steps:
				if downsample_rate is not None:
					step = step // downsample_rate if downsample_rate > 1 else step
				src = raw_dataset_path / task_name / episode_dir.name / 'front_rgb' / f'{step:06d}.png'
				dst = seg_dataset_path / task_name / episode_dir.name / 'front_rgb' / f'{b_step}_expert-{step}.png'
				if not src.exists() or dst.exists():
					continue
				ensure_path_exists(dst.parent)
				symlink(relative_path(src, dst.parent), dst)
		seg_ep_dict[task_name].append(seg_dataset_path / task_name / episode_dir.name)

	# split dataset
	ensure_path_exists(seg_dataset_split_path)
	trainset_path = seg_dataset_split_path / 'train'
	valset_gt_path = seg_dataset_split_path / 'val_gt'
	valset_path = seg_dataset_split_path / 'val'
	ensure_path_exists(trainset_path)
	ensure_path_exists(valset_gt_path)
	ensure_path_exists(valset_path)
	for task_name, eps in seg_ep_dict.items():
		split_idx = int(len(eps) * trainset_ratio)
		train_eps = eps[:split_idx]
		val_eps = eps[split_idx:]
		for split_eps, split_path in [(train_eps, trainset_path), (val_eps, valset_gt_path)]:
			for ep_path in split_eps:
				dst_ep_path = split_path / task_name / ep_path.name
				ensure_path_exists(dst_ep_path.parent)
				symlink(relative_path(ep_path, dst_ep_path.parent), dst_ep_path)
		for ep_path in val_eps:
			src_ep_path = raw_dataset_path / task_name / ep_path.name
			dst_ep_path = valset_path / task_name / ep_path.name
			ensure_path_exists(dst_ep_path.parent)
			symlink(relative_path(src_ep_path, dst_ep_path.parent), dst_ep_path)

	# save metadata
	metadata = {
		'num_episodes': len(ep_iterator),
		'downsample_rate': downsample_rate,
		'robocerebra_path': robocerebra_path,
		'save_to': save_to,
	}
	save_as_pickle(metadata_path / 'metadata.pkl', metadata)


if __name__ == "__main__":
	typer.run(convert_robocerebra_to_rlbench)

