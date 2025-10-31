from pathlib import Path
from os import PathLike, symlink
from typing import List, Literal, Dict, Union, Tuple
import random
import contextlib, os

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from sqlmodel import Field
from sqlalchemy import JSON, Column
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from joblib import parallel_backend, parallel_config, Parallel, delayed
import joblib

from utils.file_sys import ensure_path_exists, remove_path, relative_path, load_from_pickle, save_as_pickle, list_dir
from utils.images import frames_to_video
from utils.database import Database, BaseEntry
from utils.concurrent import AsyncWorkerPool
from utils.func_tools import suppress_stdout, flatten_list
from rdd.embed import uvd_embed, PreprocType, EmbedPreprocs, subtask_embeds_to_feature
from rdd.ann import AnnoySearcher


RGB_VIEWS = [
	'front_rgb', 'wrist_rgb', 'left_shoulder_rgb', 'right_shoulder_rgb',
]


def _load_task(
	data_path, 
	max_ep_num: int = None, 
	view: str = 'front_rgb'
	) -> dict:
	task_demos = []
	for ep_path in sorted(Path(data_path).glob('./episode*'), key=lambda x: int(x.stem.split("episode")[-1]))[:max_ep_num]:
		ep_demos = (ep_path / view).glob('*.png')
		ep_demos = sorted(ep_demos, key=lambda x: int(x.stem.split(".")[0]))
		task_demos.append(ep_demos)
	return task_demos # [ep_idx, img_idx]


def load_rlbench(dataset_path, **kwargs) -> dict:
	demos = {}
	for task_path in Path(dataset_path).glob('*'):
		if not task_path.is_dir():
			continue
		demos[task_path.stem] = _load_task(Path(task_path) / 'all_variations' / 'episodes', **kwargs)
	return demos # {task_idx: [ep_idx, img_idx]}


def _load_racer_aug_task(
	data_path, 
	rand_select_eps_num: int = None,
	selected_eps_idexes: List[int] = None, 
	view: str = 'front_rgb', 
	index_only: bool = False
	) -> dict:
	task_demos = []
	ep_paths = sorted(Path(data_path).glob('*'))
	ep_indexes = list(range(len(ep_paths)))
	if rand_select_eps_num is not None:
		assert rand_select_eps_num <= len(ep_paths), f"rand_select_eps_num {rand_select_eps_num} is larger than the number of episodes {len(ep_paths)} in {data_path}"
		selected_eps_idexes = random.sample(ep_indexes, rand_select_eps_num)
		ep_paths = [ep_paths[i] for i in selected_eps_idexes]
	elif selected_eps_idexes is not None:
		ep_paths = [ep_paths[i] for i in selected_eps_idexes]
	else:
		selected_eps_idexes = ep_indexes
	for ep_path in ep_paths:
		assert (ep_path / view).exists(), f"View {view} not found in episode {ep_path}"
		ep_demos = (ep_path / view).glob('*_expert.png')
		if index_only:
			ep_demos = sorted([int(ep.zstem.split("_")[0]) for ep in ep_demos])
		else:
			ep_demos = sorted(ep_demos, key=lambda x: int(x.stem.split("_")[0]))
		# add intermediate steps
		ep_demos_with_inter = []
		for b_step, e_step in zip(ep_demos[:-1], ep_demos[1:]):
			_intersteps = list((ep_path / view).glob(f'{str(b_step.stem)}-*.png'))
			# if not _intersteps:
				# logger.warning(f"No intermediate steps found between {b_step} and {e_step} in {ep_path / view}, skipping.")
				# continue
			if _intersteps:
				if index_only:
					_intersteps = sorted([int(ep.stem.split("-")[-1]) for ep in _intersteps])
				else:
					_intersteps = sorted(_intersteps, key=lambda x: int(x.stem.split("-")[-1]))
			ep_demos_with_inter.append([b_step] + _intersteps + [e_step])
		task_demos.append(ep_demos_with_inter)
	return task_demos, selected_eps_idexes # [ep_idx, [img_idx]]


def load_racer_aug_rlbench(dataset_path, include_tasks: List[str] = None, **kwargs) -> dict:
	demos = {}
	for task_path in Path(dataset_path).glob('*'):
		if not task_path.is_dir():
			continue
		if task_path.stem in ['log', 'retry']:
			continue
		if include_tasks is not None:
			if task_path.stem not in include_tasks:
				continue
		demos[task_path.stem] = _load_racer_aug_task(Path(task_path), **kwargs)
	if include_tasks is not None:
		if len(demos) != len(include_tasks):
			raise ValueError(f"Loaded {len(demos)} tasks {demos.keys()} from {dataset_path}, but expected {len(include_tasks)} tasks {include_tasks}. Make sure the include_tasks list is correct.")
	return demos # {task_name: ([ep_idx, img_idx],[ori_ep_idx])}



class RLBenchVecEntry(BaseEntry, table=True):
	frame_id: str
	view: str
	vec_path: str
	task_name: str
	task_index: int
	task_path: str
	episode_index: int
	origin_episode_index: int
	episode_path: str
	frame_paths: List[str] = Field(default=None, sa_column=Column(JSON))
	frame_indexes: List[int] =Field(default=None, sa_column=Column(JSON))
	vid_path: str
 
	@property
	def duration(self) -> int:
		frame_timestamps = sorted([int(Path(f).stem.split("_")[0]) for f in self.frame_paths])
		if len(frame_timestamps) < 2:
			raise RuntimeError(f"Invaid frame {self}, only {len(frame_timestamps)} timestamps found: {frame_timestamps}, expected at least 2")
		return frame_timestamps[-1] - frame_timestamps[0]


class RLBenchVecDataset(Database):
	def __init__(self,
	vec_dataset_path: PathLike,
	rlbench_dataset_path: PathLike = None,
	selected_eps_idexes: List[int] = None,
	verbose: bool = False,
	raise_if_not_exists: bool = False,
	preprocessor: PreprocType = EmbedPreprocs.R3M,
	device: Union[torch.device, str, int] = 'cuda',
	embed_dim: int = None,
	embed_worker_num: int = 1,
	embed_mode: Literal['default', 'ood'] = 'defualt',
	cpu_worker_num: int = 4,
	sample_rate: float = 1.0,
	views: List[str] = RGB_VIEWS,
	include_tasks: List[str] = None
	):
		self.vec_dataset_path = Path(vec_dataset_path).absolute()
		if raise_if_not_exists and not self.vec_dataset_path.exists():
			raise FileNotFoundError(f"Vec dataset path {self.vec_dataset_path} does not exist.")
		# load basic info
		if embed_dim is None:
			self.embed_dim = preprocessor.dim
		else:
			assert embed_dim > 0 and embed_dim <= preprocessor.dim, f"embed_dim {embed_dim} must be in (0, {preprocessor.dim}]"
			self.embed_dim = embed_dim
		# self.dim = self.embed_dim * 2 # tobe inferred
		self.dim = None # tobe inferred
		self.embed_mode = embed_mode
		self.preprocessor_name = preprocessor.name
		self.rlbench_dataset_path = Path(rlbench_dataset_path).absolute() if rlbench_dataset_path else None
		self.selected_eps_idexes = selected_eps_idexes
		self.verbose = verbose
		self.device = device
		self.embed_worker_num = embed_worker_num
		self.cpu_worker_num = cpu_worker_num
		self.sample_rate = sample_rate
		self.views = views
		self.include_tasks = include_tasks
		for v in self.views: assert v in RGB_VIEWS, f"Invalid view {v}, must be one of {RGB_VIEWS}"
		self._init_vec_database_on_disk()

	def _init_vec_database_on_disk(self):
		if self.verbose: logger.info(f"Finding vec dataset at {self.vec_dataset_path}")
		ensure_path_exists(self.vec_dataset_path)
		database_path = self.vec_dataset_path / 'index.db'	
		raw_data_path = self.vec_dataset_path / 'data'
		meta_data_path = self.vec_dataset_path / 'meta.pkl'	
		new_database = not database_path.exists()
		super(RLBenchVecDataset, self).__init__(database_path, entry_class=RLBenchVecEntry, auto_commit=False)
		if new_database:
			if self.verbose: logger.info(f"Creating new vec dataset at {self.vec_dataset_path}")
			assert self.rlbench_dataset_path is not None, "RLBench dataset path is not set."
			assert self.rlbench_dataset_path.exists(), f"RLBench dataset path {self.rlbench_dataset_path} does not exist."
			embed_worker = AsyncWorkerPool(self.embed_worker_num, worker_type='process', mp_method='spawn')
			cpu_worker = AsyncWorkerPool(self.cpu_worker_num, worker_type='process')
			for rgb_view in self.views:
				demos = load_racer_aug_rlbench(self.rlbench_dataset_path, include_tasks=self.include_tasks, selected_eps_idexes=self.selected_eps_idexes, view=rgb_view)
				for task_idx, (task_name, episodes) in enumerate(tqdm(demos.items(), desc=rgb_view)) if self.verbose else enumerate(demos.items()):
					# subsample episodes by sampling rate
					task_save_path = raw_data_path / task_name
					episodes, ori_eps_idxs = episodes
					if self.sample_rate < 1.0:
						# subsample the first few episodes
						_selected_indexes = list(range(int(len(episodes)*self.sample_rate)))
						episodes = [episodes[i] for i in _selected_indexes]
						ori_eps_idxs = [ori_eps_idxs[i] for i in _selected_indexes]
					for eps_idx, eps in enumerate(tqdm(episodes, desc=task_name)) if self.verbose else enumerate(episodes):
						subgoal_frames = eps
						
						if self.dim is None:
							self.dim = subtask_embeds_to_feature.feature_dim(self.embed_dim, self.embed_mode)
							if self.verbose: logger.info(f"Detected {len(subgoal_frames)} frames / subtask, setting vec dataset dimension to {self.dim}")

						eps_save_path = raw_data_path / task_name / str(eps_idx) / rgb_view
						ensure_path_exists(eps_save_path)
						# link subgoal frames
						# linked_subgoal_frames = []
						for i, frame in enumerate(flatten_list(subgoal_frames)):
							frame_save_path = eps_save_path / frame.name
							remove_path(frame_save_path)
							symlink(frame, frame_save_path)
							# linked_subgoal_frames.append(frame_save_path)
						linked_subtask_frames = eps
						# save subgoal
						ep_embed_save_paths = []
						# assert len(linked_subgoal_frames) >= 2, f"Episode {eps_idx} in task {task_name} has less than 2 keyframes: {linked_subgoal_frames}"
						for i in range(len(linked_subtask_frames)):
							# fs = [linked_subtask_frames[i], linked_subtask_frames[i+1]]
							fs = linked_subtask_frames[i]
							vid_path = eps_save_path / f'subgoals_{i}.gif'
							embed_path = eps_save_path / f'embed_{i}.npy'
							ep_embed_save_paths.append(embed_path)
							cpu_worker.add_task(frames_to_video, fs, vid_path, fps=2, loop=0)
							# frames_to_video(fs, vid_path, fps=2, loop=0)
							self.add(RLBenchVecEntry(
								frame_id=f'{task_name}_{eps_idx}_{i}',
								view=str(rgb_view),
								vec_path=str(relative_path(embed_path, self.vec_dataset_path)),
								task_name=task_name,
								task_index=task_idx,
								task_path=str(relative_path(task_save_path, self.vec_dataset_path)),
								episode_index=eps_idx,
								origin_episode_index=ori_eps_idxs[eps_idx],
								episode_path=str(relative_path(eps_save_path, self.vec_dataset_path)),
								frame_paths=[str(relative_path(f, self.vec_dataset_path)) for f in fs],
								frame_indexes=[i, i+1],
								vid_path=str(relative_path(vid_path, self.vec_dataset_path)),
							))
						# batch embed
						embed_worker.add_task(
		  					self._subtask_batch_embed, 
			   				linked_subtask_frames, 
				   			ep_embed_save_paths, 
					  		self.preprocessor_name, 
							self.embed_dim,
							self.embed_mode,
							self.device
						)
			self.commit()
			embed_worker.wait_for_results()
			embed_worker.close()
			cpu_worker.wait_for_results()
			cpu_worker.close()
			# save meta data
			save_as_pickle(meta_data_path, {
				'dim': self.dim,
				'preprocessor_name': self.preprocessor_name,
			})
		else:
			# load meta data
			meta_data = load_from_pickle(meta_data_path)
			self.dim = meta_data['dim']
			self.preprocessor_name = meta_data['preprocessor_name']
			if self.verbose: logger.info(f"Loaded meta data from {meta_data_path}: {meta_data}")
			if self.verbose: logger.info(f"Loaded vec dataset from {self.vec_dataset_path}")

	@staticmethod
	def _subtask_batch_embed(frame_paths, embed_save_paths, preprocessor_name, embed_dim, embed_mode, device):
		for idx, ep in enumerate(frame_paths):
			embeddings = uvd_embed(ep, preprocessor=preprocessor_name, device=device, to_dim=embed_dim, to_numpy=False)
			np.save(embed_save_paths[idx], subtask_embeds_to_feature()(embeddings, embed_mode))  # average-pooling over timesteps


def np_adaptive_pooling(x: np.ndarray, to_dim: int) -> np.ndarray:
	x = torch.from_numpy(x)
	x = torch.nn.functional.adaptive_avg_pool1d(
		x, to_dim
	).numpy()
	return x


class RLBenchBaseSearcher:
	def __init__(self, 
		searcher_path: PathLike, 
		vec_database_path: PathLike,
		include_views: List[str] = ['front_rgb', 'wrist_rgb'],
		rand_sample_num: int = None,
		seed = 0,
		use_cached_index: bool = True,
		verbose: bool = False
	):
		self.searcher_path = Path(searcher_path)
		self.meta_data_path = Path(str(self.searcher_path) + '.meta.pkl')
		self.rand_sample_num = rand_sample_num
		self.rng = np.random.default_rng(seed)
		self.seed = seed
		self.use_cached = use_cached_index
		for v in include_views: assert v in RGB_VIEWS, f"Invalid view {v}, must be one of {RGB_VIEWS}"
		self.include_views = include_views
		self.verbose = verbose
		self.vec_database_path = Path(vec_database_path)
		self.vec_database = RLBenchVecDataset(self.vec_database_path, raise_if_not_exists=True)
		self.dim = self.vec_database.dim * len(self.include_views)
		self._get_vecs()
   
	def _get_vecs(self):
		if self.use_cached:
			assert self.meta_data_path.exists(), f"Meta data path {self.meta_data_path} does not exist."
			if self.verbose: logger.info(f"Loading from cached index at {self.searcher_path}")
			meta_data = load_from_pickle(self.meta_data_path)
			self.vecs_all_frames = meta_data['vecs_all_frames']
		else:
			# collect multi-view vectors for each frame
			all_vecs: List[RLBenchVecEntry] = self.vec_database.select_all()
			frame_ids = sorted(set([v.frame_id for v in all_vecs]))
			task_names = sorted(set([v.task_name for v in all_vecs]))
			vecs_all_frames: List[List[RLBenchVecEntry]] = []
			for fid in frame_ids:
				vecs_of_frame: List[RLBenchVecEntry] = self.vec_database.select(frame_id=fid)
				vecs_sorted_by_views: List[RLBenchVecEntry] = []
				for view in self.include_views:
					if view not in [v.view for v in vecs_of_frame]:
						raise RuntimeError(f"View {view} not found in entries {vecs_of_frame}. [frame_id: {fid}]")
					vecs_sorted_by_views.append([v for v in vecs_of_frame if v.view == view][0])
				vecs_all_frames.append(vecs_sorted_by_views)
			# sampling
			if self.rand_sample_num is not None:
				selected_vecs: List[List[RLBenchVecEntry]] = []
				for task in task_names:
					task_frame_vecs = [vecs_of_frame for vecs_of_frame in vecs_all_frames if vecs_of_frame[0].task_name == task]
					assert self.rand_sample_num <= len(task_frame_vecs), f"rand_sample_num {self.rand_sample_num} is larger than the number of entries {len(task_frame_vecs)}"
					selected_vecs.extend(self.rng.choice(task_frame_vecs, self.rand_sample_num, replace=False))
				vecs_all_frames = selected_vecs
			self.vecs_all_frames = vecs_all_frames
			save_as_pickle(self.meta_data_path, {
				'vecs_all_frames': self.vecs_all_frames,
			})
			if self.verbose: logger.info(f"Loaded {len(self.vecs_all_frames)} frames from {self.vec_database_path}")	

	def _vec_loader(self):
		for idx, vecs_of_frame in enumerate(self.vecs_all_frames):
			_vecs = [np.load(self.vec_database_path / v.vec_path) for v in vecs_of_frame]
			yield idx, np.concatenate(_vecs, axis=0)


class RLBenchAnnoySearcher(RLBenchBaseSearcher):
	def __init__(self, 
		*args,
		n_trees = 10, 
		distance_measure = "angular",
		pca_dim: int = None,
		**kwargs,
	):
		super(RLBenchAnnoySearcher, self).__init__(*args, **kwargs)
		self.n_trees = n_trees
		self.distance_measure = distance_measure
		self.pca_dim = pca_dim
		self._build_ann_database()
   
	def _build_ann_database(self):
		if self.use_cached:
			assert self.searcher_path.exists(), f"ANN database path {self.searcher_path} does not exist."
			pca_path = self.searcher_path.with_suffix('.pca.pkl')
			self.pca: PCA = joblib.load(pca_path) if pca_path.exists() else None
			self.pca_dim = self.pca.n_components_ if self.pca is not None else None
			if self.verbose and self.pca_dim is not None:
				logger.info(f"Loaded PCA with dimension {self.pca_dim} from {pca_path}")
			if self.pca_dim is not None:
				if self.verbose: logger.info(f"Applying PCA to reduce dimension from {self.dim} to {self.pca_dim}")
				self.dim = self.pca_dim
			self.ann_database = AnnoySearcher(self.dim, self.searcher_path, verbose=self.verbose)
		else:
			if self.verbose: logger.info(f"Building ANN index at {self.searcher_path}")
			# build ann database
			remove_path(self.searcher_path)
			if self.pca_dim is not None:
				if self.verbose: logger.info(f"Applying PCA to reduce dimension from {self.dim} to {self.pca_dim}")
				self.dim = self.pca_dim
			self.ann_database = AnnoySearcher(self.dim, self.searcher_path, n_trees=self.n_trees, distance_measure=self.distance_measure, seed=self.seed, verbose=self.verbose)
			# apply PCA if specified
			if self.pca_dim is not None:
				assert self.pca_dim > 0 and self.pca_dim <= self.dim, f"pca_dim {self.pca_dim} must be in (0, {self.dim}]"
				idxes, vecs = [], []
				for idx, vec in self._vec_loader():
					idxes.append(idx)
					vecs.append(vec)
				pca = PCA(n_components=self.pca_dim, random_state=self.seed)
				vecs_reduced = pca.fit_transform(np.stack(vecs))
				joblib.dump(pca, self.searcher_path.with_suffix('.pca.pkl'))
				self.pca = pca
				for i, vec in enumerate(vecs_reduced):
					self.ann_database.add(vec, idxes[i])
			else:
				for idx, vec in self._vec_loader():
					self.ann_database.add(vec, idx)
			# save the index
			self.ann_database.save()
		if self.verbose: logger.info(f"Loaded ANN index with {len(self.ann_database)} entries")

	def query(self, 
	  	x: np.ndarray, 
	   	num_neighbors: int = 1
	) -> Tuple[list, list]:
		if self.pca is not None:
			x = self.pca.transform(x.reshape(1, -1)).flatten()
		nn_idxs, dists = self.ann_database.query(x, num_neighbors)
		nn_idxs = nn_idxs.tolist()
		dists = dists.tolist()
		nn_entries = [self.vecs_all_frames[i] for i in nn_idxs]
		return nn_entries, dists

	def __len__(self):
		return len(self.ann_database)


class RLBenchIsolationForestSearcher(RLBenchBaseSearcher):
	def __init__(self, 
		*args,
		n_estimators: int = 100,
		max_samples: Union[str, float] = 'auto',
		max_features: float = 1.0,
		n_jobs: int = 16,
		**kwargs
		):
		super(RLBenchIsolationForestSearcher, self).__init__(*args, **kwargs)
		self.n_estimators = n_estimators
		self.max_samples = max_samples
		self.max_features = max_features
		self.n_jobs = n_jobs
		self._build_forest()

	def _build_forest(self):
		if self.use_cached:
			assert self.searcher_path.exists(), f"Forest database path {self.searcher_path} does not exist."
			self.forest = load_from_pickle(self.searcher_path)
		else:
			forest = IsolationForest(
				n_estimators = self.n_estimators,
				random_state = self.seed,
				max_samples = self.max_samples,
				max_features = self.max_features,
				verbose = 0
			)
			all_vecs = [vec[1] for vec in self._vec_loader()]
			forest.fit(all_vecs)
			save_as_pickle(self.searcher_path, forest)
			self.forest = forest

	def query(self,
     	x: Union[np.ndarray, List[np.ndarray]], 
	) -> list:
		if isinstance(x, list):
			x = np.concatenate(x, axis=0)
		# scores = self.forest.decision_function(x)
		# Parallel(n_jobs=self.n_jobs)(delayed(self.forest.decision_function)(x))
		# return scores.tolist()
		with parallel_config(verbose=0):
			with parallel_backend("threading", n_jobs=self.n_jobs):
				scores = suppress_stdout(self.forest.decision_function, x)
		return scores.tolist()

	def __len__(self):
		return len(self.vecs_all_frames)