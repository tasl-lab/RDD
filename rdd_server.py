from typing import List
from contextlib import asynccontextmanager
import io
import argparse
from typing import Dict, List, Any, Literal
from pathlib import Path
import debugpy

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import numpy as np
from loguru import logger
from easydict import EasyDict as edict

from rdd.embed import EmbedPreprocs, uvd_embed, get_preprocessor
from rdd.datasets.rlbench import RLBenchAnnoySearcher, RLBenchVecEntry
from rdd.algorithms import max_sum_partition, rdd_score
from rdd.uvd_wrapper import _uvd_decompose
from utils.concurrent import AsyncWorkerPool
from utils.file_sys import load_yaml


"""
Decomposer functions
"""


def _rdd_decompose(
	embeds: np.ndarray,
	args: argparse.Namespace,
):
	assert len(set([emb.shape for emb in embeds])) == 1, f"vids have different shapes: {[emb.shape for emb in embeds]}"
	searcher = RLBenchAnnoySearcher(
		Path(args.vec_database_path) / 'index.ann',
		args.vec_database_path,
		verbose=False,
		n_trees=10,
		use_cached_index=True,
		include_views=args.views
	)
	vid_len, feat_len = embeds.shape
	array = list(range(vid_len))
	nn_list = {}
	segments = max_sum_partition(array, rdd_score, min_len=2, max_len=args.max_len, alpha=args.alpha, beta=args.beta,
								searcher=searcher, embeds=embeds, nn_list=nn_list, mode=args.mode)
	nn_list = [nn_list[str(seg)] for seg in segments[1]] # get the nearest neighbor of each segment
	return segments[1], nn_list


"""
Applications
"""

# load yaml config
args = edict(load_yaml('configs/rdd_server.yaml'))
if args.debug:
	port = 5678
	debugpy.listen(port)
	print(f"Debug Mode: Waiting for debugger attach at port {port}...")
	debugpy.wait_for_client()


class DecomposeRequest(BaseModel):
	paths: List[List[str]]
	preprocessor: str = EmbedPreprocs.R3M.name
	method: str


global_objs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
	# Load the ML model
	global_objs['preprocessor'] = get_preprocessor(args.preprocessor, device=args.device)
	yield
	# Clean up


app = FastAPI(lifespan=lifespan)


# define a get route to return the config
@app.get("/config")
async def get_config():
	return JSONResponse(
		content=jsonable_encoder({
			"config": vars(args)
		})
	)


@app.post("/decompose")
async def rdd_decompose(request: DecomposeRequest):
	if request.method not in args.decomposer:
		return JSONResponse(
			content=jsonable_encoder({
				"error": f"Invalid decomposer: {request.method}, expected {args.decomposer}"
			}),
			status_code=400
		)
	assert len(request.paths) > 1, "At least two frames are required for decomposition"
	logger.info(f"{request.method} decomposing video of {len(request.paths)} frames: {request.paths[0]}...")
	frame_list = np.array(request.paths, dtype=object) # (vid_len, view_len)
	vid_len, view_len = frame_list.shape
	frame_list = frame_list.flatten()
	embeds = []
	for chk in np.array_split(frame_list, vid_len // args.chunk_size + 1):
		embeds.append(
			uvd_embed(
				chk.tolist(),
				preprocessor=global_objs['preprocessor'],
			)
		)
	embeds = np.concatenate(embeds, axis=0) # (all_frames_len, embed_dim)
	embeds = embeds.reshape(vid_len, view_len, -1) # (vid_len, view_len, embed_dim)
	embeds = np.concatenate([embeds[:,i,:] for i in range(view_len)], axis=1) # concatenate all views (vid_len, embed_dim*view_len)
	if request.method == "rdd":
		segments, nn_list = _rdd_decompose(embeds, args)
		nn_list = [[view.model_dump() for view in frame] for frame in nn_list]
		logger.info(f"{request.method} decomposed video: {request.paths[0]}... into {len(segments)} segments: {[s[0] for s in segments]}")
		return JSONResponse(
			content=jsonable_encoder({
				"segments": segments,
				"nn_list": nn_list
			})
		)
	elif request.method == "uvd":
		segments = _uvd_decompose(embeds)
		logger.info(f"{request.method} decomposed video: {request.paths[0]}... into {len(segments)} segments: {[s[0] for s in segments]}")
		return JSONResponse(
			content=jsonable_encoder({
				"segments": segments
			})
		)
	else:
		raise ValueError(f"Unknown decomposer: {args.decomposer}")

 
 
if __name__ == "__main__":
	uvicorn.run(app, host="localhost", port=8000)
