from typing import List, Dict
from pathlib import Path
from os import PathLike

from confection import Config
from easydict import EasyDict as edict


def merge_configs(configs: List[Config]) -> Config:
	"""
	Merge a list of Config objects into one
	"""
	merged = Config()
	for cfg in configs:
		merged = merged.merge(cfg)
	return merged


def load_config(config_path: PathLike) -> Config:
	"""
	Load a config file to Config object
	"""
	config_path = Path(config_path)
	cfg = Config().from_disk(config_path)
	if '@control' not in cfg:
		return cfg
	if 'include' not in cfg['@control']:
		return cfg
	included_paths = cfg['@control']['include']
	assert isinstance(included_paths, list), 'include key must be a list'
	if not included_paths:
		return cfg
	heri_list = []
	for included in included_paths:
		parent_cfg_path = (config_path.parent / included).resolve()
		assert parent_cfg_path.exists(), f'Included file {parent_cfg_path} does not exist'
		parent_cfg = load_config(parent_cfg_path)
		heri_list.append(parent_cfg)
	heri_list.append(cfg) # add this config
	return merge_configs(heri_list)


def dump_config(config: Config, config_path: PathLike):
	"""
	Dump a Config object to a config file
	"""
	config_path = Path(config_path)
	config.to_disk(config_path)


def dump_dict(config_dict: Dict, config_path: PathLike):
	"""
	Dump a dict to a config file
	"""
	config_path = Path(config_path)
	config = Config(config_dict)
	config.to_disk(config_path)
