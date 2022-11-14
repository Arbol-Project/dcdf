from __future__ import annotations

import abc
import typing

from numpy import typing as npt

Cid = str
Path = str


class Resolver(abc.ABC):
    @abc.abstractmethod
    def new_folder(self) -> Folder:
        """Create an empty folder"""

    @abc.abstractmethod
    def get_folder(self, cid: Cid) -> typing.Optional[Folder]:
        """Get folder"""

    @abc.abstractmethod
    def get_commit(self, cid: Cid) -> typing.Optional[Commit]:
        """Get commit"""

    @abc.abstractmethod
    def get_superchunk(self, cid: Cid) -> typing.Optional[Superchunk]:
        """Get superchunk"""

    @abc.abstractmethod
    def insert(self, root: Cid, path: Path, object: Cid) -> Cid:
        """Insert an object into the DAG."""

    @abc.abstractmethod
    def load_object(self, cid: Cid) -> typing.Optional[bytes]:
        """Load an object from the DAG."""

    @abc.abstractmethod
    def store_object(self, object: bytes) -> typing.Optional[Cid]:
        """Store object in the DAG."""


class Folder(abc.ABC):
    @abc.abstractmethod
    def update(self, path: str, object: Cid) -> Folder:
        """Add an object to this folder.

        Returns a new folder with updated folder contents.
        """

    @abc.abstractmethod
    def get(self, name: str) -> typing.Optional[FolderItem]:
        """Get an item from this folder."""

    @abc.abstractmethod
    def __getitem__(self, name: str) -> FolderItem:
        """Get an item from this folder.

        Raises `KeyError` if `name` is not in folder.
        """

    @abc.abstractmethod
    def __contains__(self, name: str) -> bool:
        """Return whether this folder contains `name`."""

    @property
    @abc.abstractmethod
    def cid(self) -> Cid:
        """Get CID of this folder."""


class FolderItem(abc.ABC):
    @property
    @abc.abstractmethod
    def cid(self) -> Cid:
        """Get CID of this folder item."""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Get size, in bytes, of folder item."""


class Commit(abc.ABC):
    @property
    @abc.abstractmethod
    def message(self) -> str:
        """Get commit message."""

    @property
    @abc.abstractmethod
    def prev(self) -> typing.Optional[Commit]:
        """Get previous commit."""

    @property
    @abc.abstractmethod
    def root(self) -> Folder:
        """Get root of this commit's file system tree."""


class Superchunk(abc.ABC):
    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int, int]:
        """Get shape of this superchunk."""

    @abc.abstractmethod
    def get(self, instant: int, row: int, col: int) -> float:
        """Get a cell's value."""

    @abc.abstractmethod
    def cell(self, start: int, end: int, row: int, col: int) -> npt.NdArray:
        """Get a cell's value across a span of time."""

    @abc.abstractmethod
    def window(
        self, start: int, end: int, top: int, bottom: int, left: int, right: int
    ) -> npt.NdArray:
        """Get a subarray of superchunk."""

    @abc.abstractmethod
    def search(
        self,
        start: int,
        end: int,
        top: int,
        bottom: int,
        left: int,
        right: int,
        lower: float,
        upper: float,
    ) -> list[tuple[int, int, int]]:
        """Search for cells in window with values within upper and lower bounds."""
