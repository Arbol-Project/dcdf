"""
The concept of a dataset that spans multiple encoded files arranged in a DAG.

Eventually some implementation of this will be moved into the core library as Rust code,
but for now it is easier/faster to explore this idea as a Python example. The Rust core
has everything needed to create a DAG, but in this layer we map from real world time and
space coordinates to files in the DAG and logical coordinates.

Conceptually, you can think of the DAG is a filesystem tree containing data in
superchunks, which in turn, are composed of subchunks. A superchunk decomposes a chunk
of time series raster data across geographic space, and a chunk's placement in the
filesystem tree encodes its position in time.

For example, for a daily global precipitation dataset, you might have one superchunk
encode a year's worth of data, using a folder structure like:

    root
    |_ 19
       |_ 1970   <-- Each year is a superchunk
       |_ 1971
       |_ etc...
    |_ 20
       |_ 2000
       |_ 2001
       |_ 2002

This tree structure is relatively flat, but datasets with measurements taken much more
frequently might have deeper structures. Each superchunk will represent a time series
raster as a 3 dimensional array where the first dimension encodes the time of a time
instant. In this example, each year would be composed of 365 or 366 time instants. Time
instants are simple indices and are zero indexed, so the index 0 in a superchunk would
contain data collected on January 1st of that year.

The object that for particular dataset performs the mapping of a real-world time
coordinate to a path to a superchunk plus index in that superchunk, is called a
`Layout`. `Layout` is an abstact base class that defines the contract a concrete
`Layout` must implement in order to be used in a dataset.

Within each superchunk, mapping from the real world geographic space (latitude,
longitude), to the logical index space of the underlying raster (row, column) is done by
means of a `GeoSpace` object. `GeoSpace` is an abstract base class that defines the
contract a contcrete `GeoSpace` must implement in order to be used in a dataset.
"""
from __future__ import annotations

import abc
import typing

import numpy

import dcdf


Shape = typing.Tuple[int, int]


class Layout(abc.ABC):
    """Interface definition for `Layout` implementors.

    A Layout defines a mapping between a real world date/time and the path to a
    superchunk and the index within that superchunk that encodes that date/time. The
    layout will use a path naming convention to encode the starting date/time of a
    superchunk and interior time instants will be located using the `step` property.
    """

    @property
    @abc.abstractmethod
    def step(self) -> numpy.timedelta64:
        """The time interval between time instants in a superchunk."""

    @abc.abstractmethod
    def locate(self, instant: numpy.datetime64) -> tuple[str, int, numpy.datetime64]:
        """Convert date and time to path and index.

        Finds the nearest representable time instant in the current layout and returns
        the path to the superchunk and the index in that superchunk of the nearest time
        instant. It also returns the precise time represented by that instant since it
        may differ from the requested instant.

        The return value only says where that instant would be represented in the
        dataset if it existed. It doesn't say anything about whether that data exists in
        the dataset.
        """

    @abc.abstractmethod
    def time_at(self, path: str, index: int) -> numpy.datetime64:
        """Convert path and index to date and time.

        Converts a superchunk's path and an index for a time instant in that superchunk
        into the datetime represented by that path and index.

        The return value only says what the time would be for that path and index. It
        doesn't say anything about whether there is any data in the dataset at that
        location.
        """


class GeoSpace(abc.ABC):
    """Interface definition for `GeoSpace` implementors.

    A GeoSpace defines the mapping between real world geographic coordinates (latitude
    and longitude) and logical array coordinates (row, column).
    """

    @abc.abstractmethod
    def locate(self, lat: float, lon: float
              ) -> tuple[tuple[int, int], tuple[float, float]]:
        """Convert geographic coordinate to array coordinate.

        Finds the nearest array cell to the specified lat/lon coordinate and returns the
        integer row and column of that cell. Also returns the precise lat/lon
        coordinates represented by that cell, since those may differ slightly from the
        requested lat/lon.
        """

    @abc.abstractmethod
    def coord_at(self, row: int, col: int) -> tuple[float, float]:
        """Convert logical array coordinate to geographic coordinate."""


class Dataset:
    """A dataset stored in a DAG."""

    def __init__(
        self,
        cid: dcdf.Cid,
        resolver: dcdf.Resolver,
        layout: Layout,
        geo: GeoSpace,
        shape: Shape,
        dtype: numpy.dtype = numpy.float32,
    ):
        self.cid = cid
        self.resolver = resolver
        self.layout = layout
        self.geo = geo
        self.shape = shape
        self.dtype = numpy.dtype(dtype)

    def add_chunk(
        self,
        time: numpy.datetime64,
        data: typing.Iterable[numpy.ndarray],
        levels: int,
        k: int = 2,
        fraction: int = 24,
        round: bool = False,
        local_threshold: int = 4096,
    ) -> Dataset:
        path, index, actual = self.layout.locate(time)
        if index != 0 or actual != time:
            raise ValueError("time passed to add_chunk must be at a superchunk boundary")

        # TODO: Make sure shape for each instant matches shape of datastream
        chunk = dcdf.build_superchunk(data, levels, self.resolver, k=k,
                                      fraction=fraction, round=round,
                                      local_threshold=local_threshold)
        prev = self.resolver.get_commit(self.cid)
        new_root = prev.root.update(path, chunk)

        message = f"Added data at {time}"
        commit = dcdf.commit(message, new_root, prev, self.resolver)

        return self._update(commit.cid)

    def _update(self, cid: dcdf.Cid):
        return type(self)(cid, self.resolver, self.layout, self.geo, self.dtype)

    def get(self, time: numpy.datetime64, lat: float, lon: float
           ) -> tuple[tuple[numpy.datetime64, float, float], float]:
        path, index, time = self.layout.locate(time)
        (row, col), (lat, lon) = self.geo.locate(lat, lon)
        chunk = self.resolver.get_superchunk(path)

        return (time, lat, lon), chunk.get(index, row, col)
