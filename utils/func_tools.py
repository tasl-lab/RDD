import os
import contextlib
from os.path import relpath, isfile, commonpath
from os.path import isfile
import pickle
from pathlib import Path
from typing import List, Any, Iterable, Union, Iterator, Dict, Tuple, Literal
import signal
import logging
import logging
import logging.handlers
from time import localtime, strftime, sleep
from dataclasses import dataclass, field
from itertools import count
import traceback
import sys
import random


def suppress_stdout(func, *args, **kwargs):
	with open(os.devnull, 'w') as devnull:
		with contextlib.redirect_stdout(devnull):
			return func(*args, **kwargs)


def flatten_list(arg):
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten_list(sub)] # recurse and collect


def chunks(lst: List, n: int) -> Iterator[List]:
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def shuffle_dict(d: Dict, seed: int = None, return_new_shuffled_idx: bool = False) -> Union[Dict, Tuple[Dict, List[int]]]:
	"""Shuffle a dictionary"""
	indexs = list(range(len(d)))
	if seed is not None:
		random.seed(seed)
	random.shuffle(indexs)
	keys = list(d.keys())
	values = list(d.values())
	new_d = {}
	for i in indexs:
		new_d[keys[i]] = values[i]
	if return_new_shuffled_idx:
		return new_d, indexs
	else:
		return new_d



def add_logger(logger: Any = None, alias: str = None):
	def decorator(cls):
		_logger = logging.getLogger(cls.__class__.__name__) if logger is None else logger
		cls.__logger = _logger
		cls.verbose = False
		def log(self, msg: str, level: Literal['info', 'debug', 'warning', 'error', 'critical'] = 'info'):
			# logger_msg_fmt = f'[{self.__class__.__name__}]' + r' {}'
			frame = traceback.extract_stack(limit=2)[0]
			logger_msg_fmt = f'| {frame.filename} line {frame.lineno} [{self.__class__.__name__}] | - ' + r'{}'
			if self.verbose:
				if level == 'info':
					self.__logger.info(logger_msg_fmt.format(msg))
				elif level == 'debug':
					self.__logger.debug(logger_msg_fmt.format(msg))
				elif level == 'warning':
					self.__logger.warning(logger_msg_fmt.format(msg))
				elif level == 'error':
					self.__logger.error(logger_msg_fmt.format(msg))
				elif level == 'critical':
					self.__logger.critical(logger_msg_fmt.format(msg))
				else:
					raise ValueError(f'Unknown logging level {level}')
		if alias is not None:
			setattr(cls, alias, log)
		else:
			setattr(cls, 'log', log)
		return cls
	return decorator


def get_time_str(format: str = "%Y-%m-%d~%H.%M.%S") -> str:
	s = strftime(format, localtime())
	return s


def clear_last_printed_line():
	"""https://stackoverflow.com/a/5291044/14792316"""
	sys.stdout.write("\033[F") # Cursor up one line
	sys.stdout.write("\033[K") # Clear to the end of line


class DelayedKeyboardInterrupt:
	"""When KeyboardInterrupt is received, it will be delayed until the end of the with block.
		```python
		with DelayedKeyboardInterrupt():
			# do something
		```
		https://stackoverflow.com/a/21919644/14792316
	"""
	def __enter__(self):
		self.signal_received = False
		self.old_handler = signal.signal(signal.SIGINT, self.handler)
				
	def handler(self, sig, frame):
		self.signal_received = (sig, frame)
		logging.debug('SIGINT received. Delaying KeyboardInterrupt.')
	
	def __exit__(self, type, value, traceback):
		signal.signal(signal.SIGINT, self.old_handler)
		if self.signal_received:
			self.old_handler(*self.signal_received)


def exception_printer(logger = None):
	if logger is None:
		logger = print
	def decorator(func):
		def wrapper(*args, **kwargs):
			try:
				return func(*args, **kwargs)
			except Exception as e:
				logger(traceback.format_exc())
				raise e # Re-raise the exception after logging it
		return wrapper
	return decorator


def retry(times, exceptions):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(
                        'Exception thrown when attempting to run %s, attempt '
                        '%d of %d' % (func, attempt, times)
                    )
                    attempt += 1
            return func(*args, **kwargs)
        return newfn
    return decorator
