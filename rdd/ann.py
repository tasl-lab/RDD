
from pathlib import Path
from os import PathLike
from typing import Any, Dict, Optional, Tuple, Literal

import numpy as np
import warnings
from loguru import logger

from annoy import AnnoyIndex


class AnnoySearcher:
	"""
	Approximate Nearest Neighbour searcher.
	"""
	def __init__(self, 
		dim: int, 
   		database_path: PathLike, 
		n_trees: int = 10, 
		distance_measure: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
		seed: int = 0,
		verbose: bool = False
	):
		self.database_path = Path(database_path)
		self.dim = dim
		self.verbose = verbose
		self.n_trees = n_trees
		self._init_database(
			distance_measure=distance_measure,
			seed=seed
		)
		self.is_load = False

	def _init_database(self, 
		distance_measure: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
		seed: int = 0
	):
		self.index = AnnoyIndex(self.dim, distance_measure)
		if self.database_path.exists():
			if self.verbose: logger.info(f"Loading index from existing ANN database {self.database_path}")
			self.index.load(str(self.database_path))
			self.is_load = True
		else:
			if self.verbose: logger.info(f"Building index at {self.database_path}")
			self.index.set_seed(seed)
			self.is_load = False
		
	def add(self, vec: np.ndarray, idx: int = None):
		if self.is_load:
			raise RuntimeError("Cannot add item to an reloaded ANN database.")
		self.index.add_item(len(self) if idx is None else idx, vec)

	def save(self, path: PathLike = None):
		"""
		Save the index to disk.
		"""
		if self.is_load:
			raise warnings.warn("The index is already loaded from disk. Skip saving.")
		else:
			if path is None: path = self.database_path
			self.index.build(self.n_trees)
			self.index.save(str(path))

	def query(self, query: np.ndarray, num_neighbors: int = 1) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Search for the nearest neighbors of the query.
		"""
		neighbors, distances = self.index.get_nns_by_vector(query, num_neighbors, include_distances=True)
		neighbors = np.array(neighbors)
		distances = np.array(distances)
		return neighbors, distances

	def __len__(self):
		return self.index.get_n_items()