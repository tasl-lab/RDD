
from functools import lru_cache
from typing import Literal, List, Union
from os import PathLike
from enum import Enum
from dataclasses import dataclass

import numpy as np
import torch
import imageio
import cv2
from PIL import Image

from uvd.decomp import decomp_trajectories
from uvd.models import Preprocessor
import uvd


@dataclass
class PreprocType:
	name: str
	dim: int


class EmbedPreprocs:
	"""Enum for UVD preprocessors."""
	VIP = PreprocType(name="vip", dim=1024)
	R3M = PreprocType(name="r3m", dim=2048)
	LIV = PreprocType(name="liv", dim=1024)
	CLIP = PreprocType(name="clip", dim=1024)
	VC1 = PreprocType(name="vc1", dim=768)
	DINO_V2 = PreprocType(name="dinov2", dim=1024)
	RESNET = PreprocType(name="resnet", dim=1000)

	@staticmethod
	def get_preprocessor(name: str) -> PreprocType:
		"""Get the preprocessor for UVD."""
		if name == "vip":
			return EmbedPreprocs.VIP
		elif name == "r3m":
			return EmbedPreprocs.R3M
		elif name == "liv":
			return EmbedPreprocs.LIV
		elif name == "clip":
			return EmbedPreprocs.CLIP
		elif name == "vc1":
			return EmbedPreprocs.VC1
		elif name == "dinov2":
			return EmbedPreprocs.DINO_V2
		elif name == "resnet":
			return EmbedPreprocs.RESNET
		else:
			raise ValueError(f"Unknown preprocessor: {name}")



@lru_cache(maxsize=1)
def get_preprocessor(name: str, **kwargs) -> Preprocessor:
	"""Get the preprocessor for UVD."""
	return uvd.models.get_preprocessor(name, **kwargs)


def uvd_embed(
	frames: Union[np.ndarray, str, List[PathLike]],
	preprocessor: Union[Literal["vip", "r3m", "liv", "clip", "vc1", "dinov2"], Preprocessor] = "r3m",
	device: Union[torch.device, str, None] = "cuda",
	to_dim: int = None,
	to_numpy: bool = True
) -> np.ndarray :
	if isinstance(frames, str):
		from decord import VideoReader
		vr = VideoReader(frames, height=224, width=224)
		frames = vr[:].asnumpy()
	elif isinstance(frames, list):
		target_res = cv2.imread(frames[0]).shape[:2]
		# Set target_res to a square (use the larger of height/width)
		h, w = target_res
		side = max(h, w)
		target_res = (side, side)
		frames_np = []
		for frame in frames:
			img = imageio.imread(frame)
			if img.shape[1] != target_res[0] or img.shape[0] != target_res[1]:
				img = Image.fromarray(img)
				img = img.resize(target_res, Image.BICUBIC)
				img = np.array(img)
			frames_np.append(img)
		frames = np.array(frames_np)
		# frames = np.array([imageio.imread(frame) for frame in frames])
	if isinstance(preprocessor, str):
		preprocessor = get_preprocessor(preprocessor, device=device)
	embeds = preprocessor.process(frames, return_numpy=False) # (L, N)
	# pool embeds with torch 1D pooling
	if to_dim is not None:
		embeds = torch.nn.functional.adaptive_avg_pool1d(
			embeds, to_dim
		)
	if to_numpy:
		embeds = embeds.cpu().numpy()
	return embeds


class subtask_embeds_to_feature(object):
	@staticmethod
	def feature_dim(embed_dim: int, mode: Literal['default', 'ood'] = 'defualt') -> int:
		return embed_dim * 2 if mode == 'default' else embed_dim

	def __call__(self, embeds: Union[np.ndarray, torch.Tensor], mode: Literal['default', 'ood'] = 'defualt') -> np.ndarray:
		"""
		input: embeds (L, N)
		output: feature (2N,) or (N,) if mode=='ood'
		"""
		if isinstance(embeds, torch.Tensor):
			if mode == 'ood':
				embeds = embeds[-1]	 # (N,)
			else:
				embeds = torch.concatenate([embeds[0], embeds[-1]])  # (2N,)
			embeds = embeds.cpu().numpy()
		elif isinstance(embeds, np.ndarray):
			if mode == 'ood':
				embeds = embeds[-1]  # (N,)
			else:
				embeds = np.concatenate([embeds[0], embeds[-1]])  # (2N,)
		else:
			raise ValueError(f"Unsupported embeds type: {type(embeds)}")
		return embeds
