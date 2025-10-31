from os import PathLike

import imageio
import numpy as np
import cv2

from .file_sys import ensure_path_exists


def frames_to_video(frames, path, fps=1, **kwargs):
    frames = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_RGB2BGR) for frame in frames]
    imageio.mimsave(str(path), frames, fps=fps, **kwargs)


def extract_frames_from_vid(video_path: PathLike, frames_dir: PathLike, downsample_rate: int = None):
	sampled_frame_ids = []
	ensure_path_exists(frames_dir)
	cap = cv2.VideoCapture(str(video_path))
	frame_idx = 0
	sampled_frame_idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if downsample_rate is not None and frame_idx % downsample_rate != 0:
			frame_idx += 1
			continue
		frame_idx += 1
		sampled_frame_ids.append(frame_idx)
		frame_path = frames_dir / f"{sampled_frame_idx:06d}.png"
		cv2.imwrite(str(frame_path), frame)
		sampled_frame_idx += 1
	cap.release()
	return sampled_frame_ids

