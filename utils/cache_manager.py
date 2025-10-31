from os import PathLike
import os
import pickle
from pathlib import Path
from typing import List, Any, Iterable, Union, Iterator, Dict, Tuple, Literal
from subprocess import run
import threading
import multiprocessing as mp
import shutil
import contextlib
from dataclasses import dataclass

import elara
import elara.exceptions

from .file_sys import ensure_path_exists, remove_path, move_path, copy_path
from .func_tools import DelayedKeyboardInterrupt
from .concurrent import ProxyFuncExecutor, ProxyFunc


def _is_ge_zero(val):
    return isinstance(val, int) and val >= 0

def _is_positive(val):
	return isinstance(val, int) and val > 0

class WrappedElara(elara.Elara):
	def __init__(self, path, commitdb, key_path=None, cache_param=None):
		super().__init__(path, commitdb, key_path, cache_param)
		# re-init cull_freq
		if cache_param == None:
			self.cull_freq = 0
		else:
			if "cull_freq" in cache_param:
				if _is_ge_zero(cache_param["cull_freq"]) and cache_param["cull_freq"] <= 100:
					self.cull_freq = cache_param["cull_freq"]
				else:
					raise elara.exceptions.InvalidCacheParams("cull_freq")

	def set(self, key, value, max_age=None):
		value = pickle.dumps(value)
		return super().set(key, value, max_age)

	def get(self, key):
		value = super().get(key)
		if value is None:
			return None
		return pickle.loads(value)


@dataclass
class CacheManagerProxy:
	get_func_proxy: ProxyFunc
	put_func_proxy: ProxyFunc

	def get(self, *args, **kwargs) -> Any:
		return self.get_func_proxy(*args, **kwargs)

	def put(self, *args, **kwargs) -> Any:
		return self.put_func_proxy(*args, **kwargs)


class CacheManager:
	def __init__(self, cache_dir: PathLike, commit_interval: int = None) -> None:
		"""A simple cache manager that caches data in a directory

		Args:
			cache_dir (PathLike): path to cache directory
			commit_interval (int, optional): commit to disk every <commit_interval> puts. If set to None, changes will ONLY BE COMMIT WHEN CLOSE. Defaults to None.

		Raises:
			e: _description_
		"""
		self.cache_dir = Path(cache_dir).resolve()
		self._manager = mp.Manager()
		self.index_lock = self._manager.Lock()
		self.commit_interval = commit_interval
		self.num_unsave_puts = 0
		self._cache_data_path = self.cache_dir / 'data'
		self._index_path = self.cache_dir / 'index.db'
		self._index_path_bak = self.cache_dir / 'index.db.bak'
		if not self.cache_dir.exists():
			ensure_path_exists(self.cache_dir)
			ensure_path_exists(self._cache_data_path)
		try:
			self.index = self._get_index()
			if not self._index_path.exists(): # if index file does not exist, create it
				self.commit()
		except Exception as e:
			print(f'failed to load index from {self._index_path}, maybe the index file is corrupted. Please consider using the backup index file at {self._index_path_bak}')
			raise e
		# if successfully load the index, make a backup of it
		shutil.copyfile(self._index_path, self._index_path_bak)
		# proxy
		self._get_func_proxy = ProxyFuncExecutor(self.get)
		self._put_func_proxy = ProxyFuncExecutor(self.put)

	def _get_index(self, autocommit: bool = False) -> elara.Elara:
		return WrappedElara(str(self._index_path), commitdb=autocommit, key_path=None, cache_param=None)

	def get(self, key: str, raise_keyerror: bool = False) -> Any:
		# read from index file
		with self.index_lock:
			index = self.index
			v = index.get(key)
			if v is None:
				if raise_keyerror:
					raise KeyError(f'key {key} not found in {self}')
				else:
					return None
		# return
		v['data'] = Path(v['data'])
		if v['dtype'] == 'path': # this is a path to cached data, just return it
			return v['data']
		elif v['dtype'] == 'pkl':
			with open(v['data'], 'rb') as f:
				return pickle.load(f)
		else:
			raise ValueError(f'unknown dtype {v["dtype"]}')

	def put(self, 	key: str, 
					data: Any, force: bool = False, 
					data_as_path: bool = False, 
					cache_method: Literal['move', 'copy'] = 'copy', 
					delay_keyboard_intrpt: bool = False
		) -> Path:
		"""cache data

		Args:
			key (str): cache key
			data (Any): data to cache
			force (bool, optional): overwrite existing keys if set to True. Defaults to False.
			data_as_path (bool, optional): interpret &#39;data&#39; as a path and will cache the contents of the path. Defaults to False.
			cache_method (Literal[&#39;move&#39;, &#39;copy&#39;], optional): when data_as_path is True, select the method to move contents into cache folder. Defaults to 'copy'.
			delay_keyboard_intrpt (bool, optional): prevent caching operation from being interrupted by keyboardInterrupt. Note that this will only work in main thread of the main interpreter. Defaults to False.

		Raises:
			KeyError: _description_
			NotImplementedError: _description_

		Returns:
			Path: the path to cached data
		"""
		context = DelayedKeyboardInterrupt() if delay_keyboard_intrpt else contextlib.nullcontext()
		with context:
			cache_path = self._cache_data_path / key
			# check if key already exists
			with self.index_lock:
				index = self.index
				# check if key already exists
				if index.exists(key):
					if force:
						index.rem(key)
					else:
						if key in index:
							raise KeyError(f'key {key} already exists in {self}')
			# build cache on disk
			remove_path(cache_path)
			if data_as_path: # if set, interpret data as a path and copy it to cache
				data_path = Path(data)
				if not data_path.exists():
					raise FileNotFoundError(f'path {data_path} does not exist')
				if cache_method == 'move':
					move_path(data_path, cache_path)
				elif cache_method == 'copy':
					copy_path(data_path, cache_path)
				else:
					raise NotImplementedError(f'cache_method {cache_method} not implemented')
			else: # else, dump as pkl
				with open(cache_path, 'wb') as f:
					pickle.dump(data, f)
			# update index after cache is built
			with self.index_lock:
				index = self.index
				index.set(key, {'dtype': 'path' if data_as_path else 'pkl', 'data': str(cache_path)})
			# check if need to commit
			if self.commit_interval is not None:
				self.num_unsave_puts += 1
				if self.num_unsave_puts >= self.commit_interval:
					self.commit()
					self.num_unsave_puts = 0
			return cache_path

	def commit(self):
		with self.index_lock:
			self.index.commit()

	def get_proxy(self) -> CacheManagerProxy:
		return CacheManagerProxy(self._get_func_proxy.get_proxy(), self._put_func_proxy.get_proxy())

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(cache_path={self.cache_dir})'

	def close(self):
		self.commit()
		self._get_func_proxy.close()
		self._put_func_proxy.close()
