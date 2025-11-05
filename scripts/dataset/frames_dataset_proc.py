import sys, os
sys.path.append(os.getcwd())

import typer
import numpy as np
from pathlib import Path

from utils.file_sys import *


def convert_frames_to_rlbench(
	raw_data_path: str,
	save_to: str,
	task_name: str = typer.Option('custom_task', help='Name of the task'),
	downsample_rate: int = typer.Option(1, help='Downsample rate for frames. 1 means no downsampling.'),
	trainset_ratio: float = typer.Option(0.8, help='Ratio of episodes to use for training set. The rest will be used for validation.'),
	intermediate_steps: int = typer.Option(2, help='Number of intermediate frames per segment.'),
):
	"""
	Convert raw frame data with info.txt to RLBench-style dataset format.

	Expected raw data structure:
	- data/raw_data/IMG_XXXX/
	  - 000000.png, 000001.png, ... (sequential frames)
	  - info.txt (contains ending frame numbers of each sub-task, one per line)
	"""
	assert intermediate_steps >= 0, "number of intermediate frames must be positive"

	# Output paths
	raw_dataset_path = Path(save_to) / 'raw'
	seg_dataset_path = Path(save_to) / 'seg'
	seg_dataset_split_path = Path(save_to) / 'splits'
	metadata_path = Path(save_to) / 'metadata'
	ensure_path_exists(metadata_path)
	ensure_path_exists(raw_dataset_path)
	ensure_path_exists(seg_dataset_path)

	# Load existing metadata if available
	if (metadata_path / 'metadata.pkl').exists():
		old_metadata = load_from_pickle(metadata_path / 'metadata.pkl')
	else:
		old_metadata = {}

	# Collect all episode directories from raw_data_path
	raw_data_root = Path(raw_data_path)
	ep_iterator = []
	episode_dirs = []

	for episode_dir in sorted(list_dir(raw_data_root)):
		if not episode_dir.is_dir():
			continue
		# Check if info.txt exists
		info_file = episode_dir / 'info.txt'
		if not info_file.exists():
			print(f"Skipping {episode_dir.name}: no info.txt found")
			continue

		episode_dirs.append(episode_dir)
		ep_iterator.append((task_name, episode_dir))

	print(f"Found {len(episode_dirs)} episodes to process")

	# Process raw dataset - copy/symlink frames
	for task_name_iter, episode_dir in ep_iterator:
		if (task_name_iter, episode_dir) in old_metadata.get('ep_iterator', []):
			print(f"Skipping {episode_dir.name}: already processed")
			continue

		episode_name = episode_dir.name
		source_frames_dir = episode_dir
		dest_frames_dir = raw_dataset_path / task_name_iter / episode_name / 'front_rgb'
		ensure_path_exists(dest_frames_dir)

		# Get all PNG frames
		frame_files = sorted([f for f in list_dir(source_frames_dir) if f.suffix == '.png'])

		# Copy/symlink frames with downsampling
		for idx, frame_file in enumerate(frame_files):
			if downsample_rate > 1 and idx % downsample_rate != 0:
				continue

			new_idx = idx // downsample_rate if downsample_rate > 1 else idx
			src = frame_file
			dst = dest_frames_dir / f'{new_idx:06d}.png'

			if dst.exists():
				continue

			# Use symlink for efficiency
			symlink(relative_path(src, dst.parent), dst)

	print("Raw dataset created")

	# Build seg_dataset by creating symlinks to keyframes and intermediate frames
	seg_ep_dict = {}
	for task_name_iter, episode_dir in ep_iterator:
		if task_name_iter not in seg_ep_dict:
			seg_ep_dict[task_name_iter] = []

		if (task_name_iter, episode_dir) in old_metadata.get('ep_iterator', []):
			continue

		episode_name = episode_dir.name
		info_file = episode_dir / 'info.txt'

		# Parse info.txt to get sub-task ending frames
		with open(info_file, 'r') as f:
			ending_frames = [int(line.strip()) for line in f if line.strip()]

		# Convert ending frames to step boundaries
		# If ending_frames = [43, 73, 148, 191], then boundaries are:
		# [(0, 43), (43, 73), (73, 148), (148, 191)]
		step_boundaries = []
		start_frame = 0
		for end_frame in ending_frames:
			step_boundaries.append((start_frame, end_frame))
			start_frame = end_frame

		# Filter out boundaries that are too short for intermediate steps
		step_boundaries = [(s, e) for s, e in step_boundaries if e - s > intermediate_steps]

		if not step_boundaries:
			print(f"Warning: No valid step boundaries for {episode_name}")
			continue

		# Keyframes: start of each segment + final end frame
		keyframe_steps = [start for start, _ in step_boundaries] + [step_boundaries[-1][1]]

		# Create symlinks for keyframes
		for step in keyframe_steps:
			if downsample_rate > 1:
				step = step // downsample_rate

			src = raw_dataset_path / task_name_iter / episode_name / 'front_rgb' / f'{step:06d}.png'
			dst = seg_dataset_path / task_name_iter / episode_name / 'front_rgb' / f'{step}_expert.png'

			if not src.exists():
				print(f"Warning: Source frame not found: {src}")
				continue
			if dst.exists():
				continue

			ensure_path_exists(dst.parent)
			symlink(relative_path(src, dst.parent), dst)

		# Create symlinks for intermediate frames
		for b_step, e_step in step_boundaries:
			# Generate intermediate frame indices
			cur_intermediate_steps = np.linspace(b_step, e_step, intermediate_steps + 2)[1:-1].astype(int).tolist()
			b_step_ds = b_step // downsample_rate if downsample_rate > 1 else b_step

			for step in cur_intermediate_steps:
				if downsample_rate > 1:
					step = step // downsample_rate

				src = raw_dataset_path / task_name_iter / episode_name / 'front_rgb' / f'{step:06d}.png'
				dst = seg_dataset_path / task_name_iter / episode_name / 'front_rgb' / f'{b_step_ds}_expert-{step}.png'

				if not src.exists():
					continue
				if dst.exists():
					continue

				ensure_path_exists(dst.parent)
				symlink(relative_path(src, dst.parent), dst)

		seg_ep_dict[task_name_iter].append(seg_dataset_path / task_name_iter / episode_name)

	print(f"Segmented dataset created with {sum(len(eps) for eps in seg_ep_dict.values())} episodes")

	# Split dataset into train and validation
	ensure_path_exists(seg_dataset_split_path)
	trainset_path = seg_dataset_split_path / 'train'
	valset_gt_path = seg_dataset_split_path / 'val_gt'
	valset_path = seg_dataset_split_path / 'val'
	ensure_path_exists(trainset_path)
	ensure_path_exists(valset_gt_path)
	ensure_path_exists(valset_path)

	for task_name_iter, eps in seg_ep_dict.items():
		split_idx = int(len(eps) * trainset_ratio)
		train_eps = eps[:split_idx]
		val_eps = eps[split_idx:]

		print(f"Task '{task_name_iter}': {len(train_eps)} train episodes, {len(val_eps)} val episodes")

		# Create symlinks for train and val_gt (ground truth)
		for split_eps, split_path in [(train_eps, trainset_path), (val_eps, valset_gt_path)]:
			for ep_path in split_eps:
				dst_ep_path = split_path / task_name_iter / ep_path.name
				ensure_path_exists(dst_ep_path.parent)
				if not dst_ep_path.exists():
					symlink(relative_path(ep_path, dst_ep_path.parent), dst_ep_path)

		# Create symlinks for val (raw frames for validation)
		for ep_path in val_eps:
			src_ep_path = raw_dataset_path / task_name_iter / ep_path.name
			dst_ep_path = valset_path / task_name_iter / ep_path.name
			ensure_path_exists(dst_ep_path.parent)
			if not dst_ep_path.exists():
				symlink(relative_path(src_ep_path, dst_ep_path.parent), dst_ep_path)

	# Save metadata
	metadata = {
		'num_episodes': len(ep_iterator),
		'downsample_rate': downsample_rate,
		'raw_data_path': raw_data_path,
		'save_to': save_to,
		'task_name': task_name,
		'ep_iterator': ep_iterator,
	}
	save_as_pickle(metadata_path / 'metadata.pkl', metadata)

	print(f"\nDataset construction complete!")
	print(f"  Raw dataset: {raw_dataset_path}")
	print(f"  Segmented dataset: {seg_dataset_path}")
	print(f"  Splits: {seg_dataset_split_path}")
	print(f"  Metadata: {metadata_path / 'metadata.pkl'}")


if __name__ == "__main__":
	typer.run(convert_frames_to_rlbench)
