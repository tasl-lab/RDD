from pathlib import Path
from os import PathLike
from typing import List, Tuple, Any, Literal

import torch
import typer

from rdd.datasets.rlbench import RLBenchVecDataset, RLBenchAnnoySearcher
from rdd.embed import EmbedPreprocs



def build_rlbench_vec_database(
    dataset_path: PathLike,
	save_to: PathLike, 
	device=torch.device(0), 
	preprocessor=EmbedPreprocs.R3M,
	sampling_rate=1.0,
	views=['front_rgb', 'wrist_rgb'],
	include_tasks: List[str] = None,
	pca_dim: int = None,
	embed_mode: Literal['default', 'ood'] = 'defualt',
	):
	def build_split(split):
		vec_dataset_path = Path(save_to) / split
		_ = RLBenchVecDataset(vec_dataset_path, dataset_path, verbose=True, 
                        device=device, preprocessor=preprocessor, embed_worker_num=4, cpu_worker_num=16, embed_mode=embed_mode,
                        sample_rate=sampling_rate, views=views, include_tasks=include_tasks)
		build_rlbench_ann_searcher(vec_dataset_path, use_cached_index=False, views=views, pca_dim=pca_dim)
	build_split('train')


def build_rlbench_ann_searcher(
    vec_database_path, 
    use_cached_index=True, 
    pca_dim=None,
    views=['front_rgb', 'wrist_rgb']
    ):
	vec_database_path = Path(vec_database_path)
	ann_dataset_path = vec_database_path / 'index.ann'
	searcher = RLBenchAnnoySearcher(
  		ann_dataset_path,
		vec_database_path,
		verbose=True,
		n_trees=10,
		pca_dim=pca_dim,
		use_cached_index=use_cached_index,
		include_views=views
	)
	print(f'searcher with {len(searcher)} vectors')
	return searcher
	

def main(
	device: int, 
	preproc_name: str, 
	sampling_rate: float, 
	dataset_path: str = typer.Argument('data/rlbench_raw/RACER-augmented_rlbench'),
	views: List[str] = typer.Option(['front_rgb', 'wrist_rgb'], help='Views to include in the vector database.'),
	include_tasks: List[str] = typer.Option(None, help='List of tasks to include in the dataset. If None, all tasks are included.'),
	name_suffix: str = typer.Option('', help='Folder name to append to the vector database name for differentiation.'),
	pca_dim: int = typer.Option(None, help='Dimension to reduce the vectors to using PCA. If None, no PCA is applied.'),
	embed_mode: str = typer.Option('default', help='Embedding mode to use.')
):
	valid_preprocessors = [EmbedPreprocs.R3M.name, EmbedPreprocs.CLIP.name, EmbedPreprocs.VIP.name, EmbedPreprocs.LIV.name, EmbedPreprocs.VC1.name, EmbedPreprocs.DINO_V2.name, EmbedPreprocs.RESNET.name]
	assert preproc_name in valid_preprocessors, f'Invalid preprocessor: {preproc_name}. Valid options are: {valid_preprocessors}'
	assert 0 < sampling_rate <= 1, f'Invalid sampling rate: {sampling_rate}. Must be between 0 and 1.'
	vec_database_path = Path(f'data/vec_databases/{name_suffix}').resolve()
	if vec_database_path.exists():
		raise FileExistsError(f'Vector database already exists at {vec_database_path}')
	build_rlbench_vec_database(
		dataset_path,
		vec_database_path, 
		torch.device(device), 
		views=views,
		preprocessor=EmbedPreprocs.get_preprocessor(preproc_name),
		sampling_rate=sampling_rate,
		include_tasks=include_tasks,
		pca_dim=pca_dim,
		embed_mode=embed_mode
	)


if __name__ == '__main__':
	typer.run(main)
