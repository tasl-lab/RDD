from typing import Any, List, Union, Dict, Tuple, Literal, Optional
from os import PathLike
from pathlib import Path
import pprint

from easydict import EasyDict as edict

from sqlmodel import Field, SQLModel, create_engine, Session, select


class BaseEntry(SQLModel):
	id: Optional[int] = Field(default=None, primary_key=True)


def dump_entry(entry: BaseEntry) -> edict:
	return edict(entry.model_dump())


class Database:
	def __init__(self, path_to_database: PathLike, entry_class: type, auto_commit: bool = True) -> None:
		assert 'id' in entry_class.__fields__.keys(), "entry_class must have 'id' as primary key. Example: id: Optional[int] = Field(default=None, primary_key=True)"
		self.entry_class = entry_class
		self.path_to_database = Path(path_to_database)
		self.auto_commit = auto_commit
		self.engine = create_engine(f"sqlite:///{path_to_database}")
		if not self.path_to_database.exists():
			SQLModel.metadata.create_all(self.engine)

	def _auto_commit(self, session) -> None:
		if self.auto_commit: session.commit()

	def _query(self, **kwargs):
		query = select(self.entry_class)
		for key, value in kwargs.items():
			assert hasattr(self.entry_class, key), f"Entry does not have attribute {key}"
			query = query.where(getattr(self.entry_class, key) == value)
		return query

	def _parse_entry_to_query_dict(self, entry: Any) -> Dict[str, Any]:
		entry_query = dump_entry(entry)
		entry_query.pop('id')
		return entry_query

	def add(self, entry: Any) -> None:
		with Session(self.engine) as session:
			session.add(entry)
			if self._auto_commit:
				session.commit()

	def exists(self, entry: Any, **kwargs) -> bool:
		if entry is not None:
			kwargs = {**self._parse_entry_to_query_dict(entry), **kwargs}
		else:
			assert kwargs, "No query parameters"
		with Session(self.engine) as session:
			query = select(self.entry_class)
			for key, value in kwargs.items():
				assert hasattr(self.entry_class, key), f"Entry does not have attribute {key}"
				query = query.where(getattr(self.entry_class, key) == value)
			ret = session.exec(query).first() is not None
		return ret

	def select(self, **kwargs) -> List[Any]:
		with Session(self.engine) as session:
			query = self._query(**kwargs)
			ret = session.exec(query).all()
		return ret

	def delete(self, **kwargs) -> List[Any]:
		with Session(self.engine) as session:
			query = self._query(**kwargs)
			deleted_entries = session.exec(query).all()
			for ret in deleted_entries:
				session.delete(ret)
			self._auto_commit(session)
		return deleted_entries

	def clear(self) -> None:
		with Session(self.engine) as session:
			query = select(self.entry_class)
			deleted_entries = session.exec(query).all()
			for entry in deleted_entries:
				session.delete(entry)
			self._auto_commit(session)
		return deleted_entries

	def commit(self) -> None:
		with Session(self.engine) as session:
			session.commit()

	def delete_duplicates(self, **kwargs) -> List[Any]:
		with Session(self.engine) as session:
			# find the id to preserve
			query = self._query(**kwargs)
			id_preserved = session.exec(query).first().id
			query = self._query(**kwargs).where(self.entry_class.id != id_preserved)
			deleted_entries = session.exec(query).all()
			for entry in deleted_entries:
				session.delete(entry)
			self._auto_commit(session)
		return deleted_entries

	def select_all(self) -> List[Any]:
		with Session(self.engine) as session:
			query = select(self.entry_class)
			ret = session.exec(query).all()
		return ret

	def iter_entries(self, **kwargs):
		with Session(self.engine) as session:
			query = self._query(**kwargs)
			results = session.exec(query)
			for entry in results:
				yield entry
    
	def __repr__(self) -> str:
		ret = self.select_all()
		return pprint.pformat(ret, indent=4)

	def __len__(self):
		with self.engine.begin() as session:
			return session.exec(select(self.entry_class)).count()
