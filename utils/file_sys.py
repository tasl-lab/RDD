from functools import partial, lru_cache
from os import PathLike, scandir, symlink
import os
from os.path import relpath, isfile, commonpath
import pickle
from pathlib import Path
from typing import List, Any, Iterable, Union, Iterator, Dict, Tuple, Literal
from subprocess import run
import sys
import multiprocessing as mp
import shutil
import shlex
import yaml
import json

from tqdm import tqdm


def add_path_to_sys(path: PathLike) -> None:
	path = Path(path)
	if str(path) not in sys.path:
		sys.path.append(str(path))


def get_script_dir(_file_) -> Path:
	return Path(os.path.realpath(_file_)).parent


def list_files(dir_path: PathLike, relative: bool = False) -> List[Path]:
	"""
	list all files in a dictionary
	"""
	files = [f for f in Path(dir_path).rglob("*") if f.is_file()]
	if relative: files = [f.relative_to(dir_path) for f in files]
	return files


def relative_path(path: PathLike, base: PathLike) -> Path:
	"""
	return relative path
	"""
	rel_path = os.path.relpath(str(path), str(base))
	return Path(rel_path)


def ensure_path_exists(path: PathLike) -> bool:
	"""
	ensure path exists
	
	return:
		True if path exists, otherwise False
	"""
	if not Path(path).exists(): 
		Path(path).mkdir(parents=True, exist_ok=True)
		return False
	else:
		return True


def remove_path(path: PathLike) -> None:
	"""
	remove path
	"""
	path = Path(path)
	if path.exists():
		if path.is_symlink():
			path.unlink()
		elif path.is_dir():
			shutil.rmtree(path)
		else:
			os.remove(path)


def copy_path(src: PathLike, dst: PathLike) -> None:
	"""
	copy path
	"""
	src = Path(src)
	dst = Path(dst)
	if src.is_dir():
		shutil.copytree(src, dst)
	else:
		shutil.copy(src, dst)


def move_path(src: PathLike, dst: PathLike) -> None:
	"""
	move path
	"""
	src = Path(src)
	dst = Path(dst)
	shutil.move(src, dst)


def _walk(	path: PathLike, 
			depth: Union[int, float] = float('inf')) -> Iterator[PathLike]:
	"""Recursively list files and directories up to a certain depth"""
	if depth is None: depth = float('inf')
	depth -= 1
	with scandir(path) as p:
		for entry in p:
			if entry.is_dir() and depth > 0:
				yield from _walk(entry.path, depth)
			else:
				yield entry.path


def list_dir(	dir_path: PathLike, 
				relative: bool = False, 
				depth: Union[int, float] = 1) -> Iterator[Path]:
	"""list all files/folders in a dictionary to a certain depth

	Args:
		dir_path (PathLike): path to the directory
		relative (bool, optional): return relative path. Defaults by False (returns absolute path).
		depth (Union[int, float], optional): list to certain depth. Defaults by infinite (list all leaf nodes in a dictionary).

	Yields:
		Iterator[Path]: iterator of files/folders
	"""
	dir_path = Path(dir_path).absolute()
	for f in _walk(dir_path, depth):
		if relative: yield Path(f).relative_to(dir_path)
		else: yield Path(f)


def _fill_dir(	source_dir: Path, 
				target_dir: Path) -> None:
	for subpath in source_dir.iterdir():
		dst_path = target_dir / subpath.relative_to(source_dir)
		if dst_path.exists():
			if subpath.is_dir():
				assert dst_path.is_dir(), f'given src dictionary {subpath}, {dst_path} must also be a directory'
				_fill_dir(subpath, dst_path)
		else:
			ensure_path_exists(dst_path.parent)
			src_path = relpath(subpath, dst_path.parent)
			symlink(src_path, dst_path)


def symlink_dir(source_dir: PathLike, 
				target_dir: PathLike,
				replace_src_with: Dict[str, PathLike] = {},
				show_progress: bool = False) -> None:
	source_dir = Path(source_dir)
	target_dir = Path(target_dir)
	# first symlink all things in replace_src_with
	iterable = replace_src_with.items() if not show_progress else tqdm(replace_src_with.items())
	for srcpath, tarpath in iterable:
		dst_path = target_dir / Path(srcpath).relative_to(source_dir)
		src_path = relpath(tarpath, dst_path.parent)
		ensure_path_exists(dst_path.parent)
		symlink(src_path, dst_path)
	# fill others
	_fill_dir(source_dir, target_dir)


def starts_with_root(path, root):
    try:
        path = Path(path)
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_from_pickle(path: PathLike) -> Any:
	with open(path, 'rb') as f:
		return pickle.load(f)


@lru_cache()
def load_from_pickle_cached(path: PathLike) -> Any:
	with open(path, 'rb') as f:
		return pickle.load(f)


def save_as_pickle(path: PathLike, data: Any) -> None:
	with open(path, 'wb') as f:
		pickle.dump(data, f)


def _shell_cmd(command_line: str):
	command_line_args = shlex.split(command_line)
	exitcode = run(command_line_args).returncode
	return exitcode


class RamDisk:
	def __init__(self, disk_name: str, size_in_mega_bytes: int = 1024, if_exist_then: Literal['remove', 'keep', 'raise'] = 'raise') -> None:
		self.media_root = Path('/media')
		self.disk_root = self.media_root / disk_name
		self.disk_size_M = size_in_mega_bytes
		if self.disk_root.exists():
			if if_exist_then == 'remove':
				self.remove()
			elif if_exist_then == 'keep':
				return
			elif if_exist_then == 'raise':
				raise FileExistsError(f'RamDisk {self.disk_root} already exists!')
			else:
				raise NotImplementedError(f'disk existed handler {if_exist_then} is not implemented')
		self.init()

	def get_disk_root(self):
		return self.disk_root

	def init(self):
		_shell_cmd(f'sudo mkdir -p {str(self.disk_root)}')
		_shell_cmd(f'sudo mount -t tmpfs -o size={self.disk_size_M}M tmpfs {self.disk_root}')

	def remove(self):
		_shell_cmd(f'sudo umount {self.disk_root}') # clear contents
		_shell_cmd(f'sudo mount -t tmpfs -o size=0M tmpfs {self.disk_root}') # remove ramdisk

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}(disk_root={self.disk_root}, disk_size_M={self.disk_size_M})'


def load_yaml(path: str) -> dict:
	"""Load yaml configuration file

	Args:
		path (str): path to the yaml file

	Returns:
		dict: configuration dictionary
	"""
	with open(path, 'r') as f:
		config = yaml.safe_load(f)
	return config


def load_json(path: str) -> dict:
	with open(path, 'r') as f:
		js = json.load(f)
	return js
