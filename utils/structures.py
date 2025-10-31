import itertools
from contextlib import nullcontext



class bidict(dict):
	"""bidirectional dict
	https://stackoverflow.com/a/21894086
	usage: 
	bd = bidict({'a': 1, 'b': 2})  
	print(bd)                     # {'a': 1, 'b': 2}                 
	print(bd.inverse)             # {1: ['a'], 2: ['b']}
	bd['c'] = 1                   # Now two keys have the same value (= 1)
	print(bd)                     # {'a': 1, 'c': 1, 'b': 2}
	print(bd.inverse)             # {1: ['a', 'c'], 2: ['b']}
	del bd['c']
	print(bd)                     # {'a': 1, 'b': 2}
	print(bd.inverse)             # {1: ['a'], 2: ['b']}
	del bd['a']
	print(bd)                     # {'b': 2}
	print(bd.inverse)             # {2: ['b']}
	bd['b'] = 3
	print(bd)                     # {'b': 3}
	print(bd.inverse)             # {2: [], 3: ['b']}
	"""
	def __init__(self, *args, **kwargs):
		super(bidict, self).__init__(*args, **kwargs)
		self.inverse = {}
		for key, value in self.items():
			self.inverse.setdefault(value, []).append(key) 

	def __setitem__(self, key, value):
		if key in self:
			self.inverse[self[key]].remove(key) 
		super(bidict, self).__setitem__(key, value)
		self.inverse.setdefault(value, []).append(key)        

	def __delitem__(self, key):
		self.inverse.setdefault(self[key], []).remove(key)
		if self[key] in self.inverse and not self.inverse[self[key]]: 
			del self.inverse[self[key]]
		super(bidict, self).__delitem__(key)


class IDPool(object):
	'''
	Create a pool of IDs to allow reuse. The "new_id" function generates the next
	valid ID from the previous one. If not given, defaults to incrementing an integer.
	'''

	def __init__(self, new_id=None, lock=None):
		if new_id is None: new_id = lambda x: x + 1
		self.lock = lock if lock is not None else nullcontext
		self.ids_in_use = set()
		self.ids_free = set()
		self.new_id = new_id
		self.last_id = 0

	def get_id(self):
		with self.lock:
			if len(self.ids_free) > 0:
				return self.ids_free.pop()
			self.last_id = id = self.new_id(self.last_id)
			self.ids_in_use.add(id)
			return id

	def release_id(self, the_id):
		with self.lock:
			if the_id in self.ids_in_use:
				self.ids_in_use.remove(the_id)
				self.ids_free.add(the_id)


class Identifiable(object):
	id_iter = itertools.count()
	def __init__(self) -> None:
		self.id = next(Identifiable.id_iter)
