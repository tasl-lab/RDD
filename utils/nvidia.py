import multiprocessing as mp
from typing import List
from subprocess import check_output


def nvidia_gpu_count() -> int:
	# https://stackoverflow.com/a/4760517
	result = check_output(['nvidia-smi --query-gpu=name --format=csv,noheader | wc -l'], shell=True, text=True)
	result = int(result)
	return result


def nvidia_gpu_free_mem(gpu_idx: int = 0) -> int:
	"""return free gpu memory in MB"""
	# https://github.com/UNCode101/Tdarr_Plugins/blob/0b528d2a970f353c30f94e30e0d8e3fbc20ed16a/Community/Tdarr_Plugin_FFMPEG_All_NVIDIA_GPUs.js
	# https://stackoverflow.com/a/59571639
	result = check_output([f'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {gpu_idx}'], shell=True, text=True)
	result = int(result)
	return result


class NvcodecResource:
	def __init__(self, gpus: List[int], num_nvcodec_cores: int = 1, mp_manager = None) -> None:
		self.num_gpus = len(gpus)
		self.num_nvcodec_cores = num_nvcodec_cores # set it according to your gpus' capacities https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
		if mp_manager is not None:
			self.resource = mp_manager.dict()
		else:
			self.resource = mp.Manager().dict()
		for gpu in gpus:
			self.resource[gpu] = num_nvcodec_cores
	
	def get(self) -> int:
		while True:
			for gpu, cores in self.resource.items():
				if cores > 0:
					self.resource[gpu] -= 1
					return gpu
		
	def release(self, gpu: int) -> None:
		self.resource[gpu] += 1

	def num_total_nvcodec_cores(self):
		return self.num_nvcodec_cores * self.num_gpus
