from multiprocessing import Pool
import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from threading import Thread, Condition
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Empty
from functools import partial
from os import PathLike, scandir, symlink
from queue import Queue
import os
from os.path import relpath, isfile, commonpath
from os.path import isfile
from typing import List, Any, Iterable, Union, Iterator, Dict, Tuple, Literal, Callable
from subprocess import Popen, PIPE, STDOUT, DEVNULL, run, TimeoutExpired
import shlex
import logging
import logging.handlers
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import count
import traceback
from pathlib import Path
import time
import sys

from tqdm import tqdm
import zmq

from .structures import IDPool
from .file_sys import ensure_path_exists


def shell_cmd(command_line: str, log_to: Union[PathLike, str] = 'stdout', maxBytes: int = 10**9, timeout: float = None, **kwargs):
	command_line_args = shlex.split(command_line)
	if isinstance(log_to, str): 
		if log_to == 'stdout':
			exitcode = run(command_line_args, timeout=timeout, **kwargs).returncode
		else:
			raise NotImplementedError
	elif isinstance(log_to, PathLike):
		# create a file logger with max byte limit
		test_logger = logging.getLogger('SHELL_CMD')
		# overwrite manually https://stackoverflow.com/a/77223050
		if isfile(log_to): os.remove(log_to)
		maybe_backup_file = str(log_to) + '.1'
		if isfile(maybe_backup_file): os.remove(maybe_backup_file)
		# log cmd to file
		with open(log_to, 'w') as f: f.write(command_line + '\n')
		# limit bytes https://stackoverflow.com/a/3999638 & https://stackoverflow.com/a/48095853
		handler = logging.handlers.RotatingFileHandler(log_to, maxBytes=maxBytes, backupCount=1)
		test_logger.setLevel(logging.INFO)
		test_logger.addHandler(handler)
		# run command, merge stderr to stdout and redirect to pipe https://stackoverflow.com/a/21978778
		process = Popen(command_line_args, stdout=PIPE, stderr=STDOUT, **kwargs)
		with process.stdout:
			for line in iter(process.stdout.readline, b''): # b'\n'-separated lines
				test_logger.info(line.strip().decode('utf-8'))
		exitcode = process.wait(timeout=timeout) # 0 means success
		test_logger.removeHandler(handler)
		handler.close() # release resources
	else: # silent
		process = Popen(command_line_args, stdout=DEVNULL, stderr=DEVNULL, **kwargs)
		exitcode = process.wait(timeout=timeout) # 0 means success
	return exitcode

# def shell_cmd(command_line: str, log_to: Union[PathLike, None] = None, maxBytes: int = 10**9, timeout: float = float('inf'), **kwargs):
# 	command_line_args = shlex.split(command_line)
# 	if log_to is None: 
# 		exitcode = run(command_line_args, timeout=timeout, **kwargs).returncode
# 	elif isinstance(log_to, PathLike):
# 		# log cmd to file
# 		with open(log_to, 'w') as f: 
# 			f.write(command_line + '\n\n')
# 			init_pos = f.tell()
# 		with open(log_to, 'a') as f:
# 			# exitcode = run(command_line_args, stdout=f, stderr=f, timeout=timeout, **kwargs).returncode
# 			process = Popen(command_line_args, stdout=f, stderr=f, **kwargs)
# 			ts = time.time()
# 			while True:
# 				try:
# 					exitcode = process.wait(timeout=0.5) # 0 means success
# 					break
# 				except TimeoutExpired:
# 					te = time.time()
# 					if te - ts > timeout:
# 						process.kill()
# 						raise TimeoutExpired(command_line, timeout)
# 					else:
# 						# check if log file size exceeds the limit
# 						if f.tell() > maxBytes:
# 							# set pos back to init_pos
# 							f.seek(init_pos)
# 	else:
# 		raise ValueError(f'log_to must be None or a PathLike object, got {log_to}')
# 	return exitcode


class AsyncWorkerPool:
	def __init__(self, worker_num: int, worker_type: Literal['process', 'thread'] = 'process', mp_method: Literal['fork', 'spawn', 'forkserver'] = 'spawn', **kwargs): # TODO
		"""Create a new asyncornized process pool.

		Args:
			process_num (int): number of processes
			worker_type (Literal['process', 'thread'], optional): type of worker. Defaults to 'process'.
			kwargs (dict): additional arguments for ProcessPoolExecutor or ThreadPoolExecutor
		"""
		self.process_num = worker_num
		if worker_type == 'process':
			self.pool = mp.get_context(mp_method).Pool(processes=self.process_num, **kwargs)
			self.worker_type = 'process'
		else:
			self.pool = ThreadPoolExecutor(max_workers=self.process_num, **kwargs)
			self.worker_type = 'thread'
		self.results = []

	@staticmethod
	def _proc_err_callback(e: Exception):
		# print exception stack
		traceback.print_exception(type(e), e, e.__traceback__)
		# pass

	def add_task(self, func: callable, *args, **kwargs) -> AsyncResult:
		"""Add a new task to the task queue."""
		if self.worker_type == 'process':
			async_result = self.pool.apply_async(func, args=args, kwds=kwargs, error_callback=self._proc_err_callback)
		else:
			async_result = self.pool.submit(func, *args, **kwargs)
		self.results.append(async_result)
		return async_result

	def wait_for_results(self) -> List[Any]:
		"""Get all results and close the processes."""  
		if self.worker_type == 'process':
			results = [async_result.get() for async_result in self.results]
		else:
			results = [async_result.result() for async_result in self.results]
		self.results = []
		return results

	def wait_for_task_result(self, async_result: AsyncResult) -> Any:
		if self.worker_type == 'process':
			res = async_result.get()
		else:
			res = async_result.result()
		self.results.remove(async_result)
		return res

	def get_one_result(self, block: bool = True) -> Any:
		while True:
			for async_result in self.results:
				done = async_result.ready() if self.worker_type == 'process' else async_result.done()
				if done:
					self.results.remove(async_result)
					return async_result, async_result.get() if self.worker_type == 'process' else async_result.result()
			else:
				if not block: raise ValueError('no ready result')

	def close(self) -> None:
		"""Close the process pool."""
		if self.worker_type == 'process':
			self.pool.close()
			self.pool.join()
			self.pool.terminate()
		else:
			self.pool.shutdown()


@dataclass
class Pipeline:
	funcs: Iterable[callable]
	args_list: Iterable[tuple] = None
	kwargs_list: Iterable[dict] = None
	progress: int = -1
	id: int = None
	_defualt_id: int = field(default_factory=count().__next__, init=False)

	def __post_init__(self):
		assert len(self.funcs) > 1, 'pipeline must have at least 2 tasks'
		if self.args_list is None:
			self.args_list = [() for _ in range(len(self.funcs))]
		if self.kwargs_list is None:
			self.kwargs_list = [{} for _ in range(len(self.funcs))]
		assert len(self.funcs) == len(self.args_list) == len(self.kwargs_list), 'funcs, args_list, kwargs_list must have the same length'
		self.id = self._defualt_id if self.id is None else self.id

	def next_task(self) -> Tuple[callable, tuple, dict]:
		"""Get the next task from the pipeline."""
		self.progress += 1
		if self.progress < len(self.funcs):
			return self.progress, self.funcs[self.progress], self.args_list[self.progress], self.kwargs_list[self.progress]
		else:
			return None

	def reset(self):
		self.progress = -1


# class ConcurrentDict:
# 	def __init__(self, mp_manager):
# 		self.data = mp_manager.dict()
# 		self._update_condition = mp_manager.Condition()

# 	def set(self, key: Any, value: Any):
# 		with self._update_condition:
# 			self.data[key] = value
# 			self._update_condition.notify_all()

# 	def pop(self, key: Any, timeout: float = None):
# 		while True:
# 			with self._update_condition:
# 				if key in self.data.keys():
# 					value = self.data.pop(key)
# 					break
# 				if not self._update_condition.wait(timeout=timeout):
# 					raise TimeoutError
# 		return value

# 	def clear(self):
# 		with self._update_condition:
# 			self.data.clear()
# 			self._update_condition.notify_all()


class ConcurrentDict:
	def __init__(self, mp_manager):
		self.data = mp_manager.dict()

	def set(self, key: Any, value: Any):
		self.data[key] = value

	def pop(self, key: Any, timeout: float = None, poll_interval: float = 0.1):
		ts = time.time()
		while True:
			if key in self.data.keys():
				value = self.data.pop(key)
				break
			else:
				time.sleep(poll_interval)
				if timeout is not None:
					if time.time() - ts > timeout:
						raise TimeoutError
		return value

	def get(self, key: Any, timeout: float = None, poll_interval: float = 0.1):
		ts = time.time()
		while True:
			if key in self.data.keys():
				value = self.data.get(key)
				break
			else:
				time.sleep(poll_interval)
				if timeout is not None:
					if time.time() - ts > timeout:
						raise TimeoutError
		return value

	def values(self):
		return self.data.values()

	def keys(self):
		return self.data.keys()

	def items(self):
		return self.data.items()

	def clear(self):
		self.data.clear()


class ZmqConnection:
	def connect(self):
		NotImplemented
	
	def send(self, msg: Any):
		NotImplemented

	def recv(self):
		NotImplemented

	def close(self):
		NotImplemented

	def reset(self):
		NotImplemented

@dataclass
class ZmqConnectionClient(ZmqConnection):
	addr: str
	def __post_init__(self):
		self.socket = None

	def connect(self):
		self.socket = zmq.Context().socket(zmq.PAIR)
		self.socket.connect(self.addr)
	
	def send(self, msg: Any):
		self.socket.send_pyobj(msg)

	def recv(self):
		return self.socket.recv_pyobj()

	def close(self):
		if self.socket is not None:
			self.socket.close()
		self.socket = None

	def reset(self):
		self.close()
		self.connect()


@dataclass
class ZmqConnectionHost(ZmqConnection):
	socket: zmq.Socket
	addr: str
	def __post_init__(self):
		self.poller = zmq.Poller()
		self.poller.register(self.socket, zmq.POLLIN)

	def recv(self):
		return self.socket.recv_pyobj()

	def send(self, msg: Any):
		while True:
			try:
				self.socket.send_pyobj(msg, flags=zmq.NOBLOCK)
				break
			except zmq.Again:
				time.sleep(0.001)

	def close(self):
		self.socket.close()

	def reset(self):
		while True:
			socks = dict(self.poller.poll(timeout=0.0))
			if self.socket in socks and socks[self.socket] == zmq.POLLIN:
				_ = self.socket.recv(flags=zmq.NOBLOCK)
			else:
				break

	def new_client(self):
		return ZmqConnectionClient(self.addr)

class ZmqSessionManager:
	def __init__(self, socket_dir: PathLike) -> None:
		self.socket_dir = Path(socket_dir).resolve()
		ensure_path_exists(self.socket_dir)
		self.session_id_pool = IDPool()
		self.sessions = []

	def new_host(self):
		self.zmq_context = zmq.Context()
		context_id = self.session_id_pool.get_id()
		session_addr = 'ipc://' + str(self.socket_dir / f'{context_id}.ipc')
		service_socket = self.zmq_context.socket(zmq.PAIR)
		service_socket.bind(session_addr)
		conn_host = ZmqConnectionHost(service_socket, session_addr)
		self.sessions.append(conn_host)
		return conn_host


def clear_queue(q: Union[Queue, mp.Queue]):
	# https://stackoverflow.com/a/18873213/14792316
	# with q.mutex:
	# 	q.queue.clear()
	# 	q.all_tasks_done.notify_all()
	# 	q.unfinished_tasks = 0
	try:
		while True:
			q.get_nowait()
	except Empty:
		pass


def clear_pipe(conn):
	while True:
		# Use select to check if there is data available for reading
		if conn.poll():
			_ = conn.recv()
		else:
			# If no data is available, break the loop
			break


class PipelineRunner:
	def __init__(self, process_nums: Iterable[int], worker_type: Literal['process', 'thread'] = 'process', **kwargs):
		self.process_nums = process_nums
		# if maxtasksperchild is None: maxtasksperchild = [None for _ in process_nums]
		self.pools = [AsyncWorkerPool(process_num, worker_type=worker_type, **kwargs) for process_num in self.process_nums]
		self.pipelines = {pool: {} for pool in self.pools}
		self.n_running = 0
		self.scheduler = None
		self.is_scheduler_running = False
		self.results = ConcurrentDict(mp.Manager())

	def add_pipeline(self, pipeline: Pipeline, last_res: Any = None) -> bool:
		"""Add a new pipeline to the task queue."""
		next_task = pipeline.next_task()
		if next_task is None: return False
		progress, func, args, kwargs = next_task
		pool = self.pools[progress]
		if last_res is not None: kwargs['PIPLINE_LAST_RESULTS'] = last_res
		async_result = pool.add_task(func, *args, **kwargs)
		self.pipelines[pool][async_result] = pipeline
		self.n_running += 1
		return True

	def wait_for_results(self, show_progress: bool = False, use_scheduler: bool = False, return_results: bool = True) -> List[Tuple[Any, int]]:
		if show_progress: pbar = tqdm(total=self.n_running)
		while self.n_running > 0 or (use_scheduler and self.is_scheduler_running):
			for pool in self.pools:
				try:
					async_result, res = pool.get_one_result(block=False)
				except ValueError:
					continue
				self.n_running -= 1
				pipeline = self.pipelines[pool].pop(async_result)
				if not self.add_pipeline(pipeline, res):
					if return_results: 
						self.results.set(pipeline.id, res)
					if show_progress: pbar.update()
		return self.results

	def start_scheduler(self, return_results: bool = True) -> None:
		self.is_scheduler_running = True
		self.scheduler = Thread(target=self.wait_for_results, args=(False, True, return_results), daemon=True)
		self.scheduler.start()
		return self.scheduler

	def close_scheduler(self) -> None:
		self.is_scheduler_running = False
		self.scheduler.join()

	def get_result_by_id(self, pipeline_id: Any, timeout: float = None):
		return self.results.pop(pipeline_id, timeout=timeout)

	def num_running_tasks(self) -> int:
		return self.n_running

	def close(self):
		for pool in self.pools:
			pool.close()
		if self.scheduler is not None: self.close_scheduler()


def imap_tqdm(  function, 
				iterable: Iterable, 
				processes: int, 
				chunksize: int = 1, 
				desc: bool = None, 
				disable: bool = False, 
				maxtasksperchild: int = None,
				mp_method: Literal['fork', 'spawn', 'forkserver'] = 'fork',
			  **kwargs) -> List[Any]:
	"""
	https://stackoverflow.com/a/73635314/14792316
	Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
	Results are always ordered and the performance should be the same as of Pool.map.
	:param function: The function that should be parallelized.
	:param iterable: The iterable passed to the function.
	:param processes: The number of processes used for the parallelization.
	:param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
	:param desc: The description displayed by tqdm in the progress bar.
	:param disable: Disables the tqdm progress bar.
	:param kwargs: Any additional arguments that should be passed to the function.
	"""
	if kwargs:
		function_wrapper = partial(_wrapper, function=function, **kwargs)
	else:
		function_wrapper = partial(_wrapper, function=function)

	results = [None] * len(iterable)
	with mp.get_context(mp_method).Pool(processes=processes, maxtasksperchild=maxtasksperchild) as p:
	# with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as p:
		with tqdm(desc=desc, total=len(iterable), disable=disable) as pbar:
			for i, result in p.imap_unordered(function_wrapper, enumerate(iterable), chunksize=chunksize):
				results[i] = result
				pbar.update()
	return results


def _wrapper(enum_iterable, function, **kwargs):
	i = enum_iterable[0]
	result = function(enum_iterable[1], **kwargs)
	return i, result


def _error_handle_wrapper(function, params):
	try:
		return function(params)
	except Exception as e:
		print(traceback.format_exc())
		raise e

def imap_tqdm_open3d(  function, 
						iterable: Iterable, 
						processes: int, 
						maxtasksperchild: int = None,
						**kwargs) -> List[Any]:
	ctx = mp.get_context('forkserver')
	with ProcessPoolExecutor(max_workers=processes, mp_context=ctx, max_tasks_per_child=maxtasksperchild, **kwargs) as p:
		results = list(p.map(_error_handle_wrapper, [function] * len(iterable), iterable))
	return results


@dataclass
class ProxyFunc:
	id_release_queue: Queue
	id_recycle_queue: Queue
	call_queue: Queue
	res_dict: ConcurrentDict

	def __call__(self, *args, **kwargs):
		job_id = self.id_release_queue.get()
		self.call_queue.put((job_id, args, kwargs))
		res, exception = self.res_dict.pop(job_id)
		self.id_recycle_queue.put(job_id)
		if exception:
			raise exception
		return res


class ProxyFuncExecutor:
	def __init__(self, func: Callable, base_args: Tuple[Any] = (), mp_manager = None, id_backup_num: int = 100) -> None:
		self._manager = mp_manager if mp_manager is not None else mp.Manager()
		self.func = func
		self.base_args = base_args
		# structures
		self.id_pool = IDPool()
		self.id_release_queue = self._manager.Queue(maxsize=id_backup_num)
		self.id_recycle_queue = self._manager.Queue()
		self.call_queue = self._manager.Queue()
		self.res_dict = ConcurrentDict(self._manager)
		# threads
		self._are_job_threads_running = True
		self.job_func_thread = Thread(target=self._job_func, daemon=True)
		self.job_id_thread = Thread(target=self._job_id, daemon=True)
		self.job_func_thread.start()
		self.job_id_thread.start()

	def get_proxy(self):
		return ProxyFunc(self.id_release_queue, 
						self.id_recycle_queue, 
						self.call_queue, 
						self.res_dict)

	def _job_func(self):
		while True:
			if not self._are_job_threads_running:
				break
			job = self.call_queue.get()
			if job is None: # exit signal
				if not self._are_job_threads_running:
					return
				else:
					raise ValueError('exit signal received but job thread is still running')
			job_id, args, kwargs = job
			args = list(self.base_args) + list(args)
			exception = None
			res = None
			try:
				res = self.func(*args, **kwargs)
			except Exception as e:
				e.args = (traceback.format_exc(),) + e.args
				exception = e
			self.res_dict.set(job_id, (res, exception))

	def _job_id(self):
		while True:
			if not self._are_job_threads_running:
				return
			# release ids
			if not self.id_release_queue.full():
				self.id_release_queue.put(self.id_pool.get_id())
			# recycle ids
			if not self.id_recycle_queue.empty():
				self.id_pool.release_id(self.id_recycle_queue.get())

	def close(self):
		self._are_job_threads_running = False
		self.call_queue.put(None)
		self.job_func_thread.join()
		self.job_id_thread.join()
