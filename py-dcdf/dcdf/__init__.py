import functools
from typing import List
import numpy

from dcdf import _dcdf


class MMEncoding:
    Time = 0
    I32 = 4
    I64 = 8
    F32 = 32
    F64 = 64

    from_dtype = {
        numpy.datetime64: Time,
        numpy.int32: I32,
        numpy.int64: I64,
        numpy.float32: F32,
        numpy.float64: F64,
    }

    to_dtype = {
        Time: numpy.datetime64,
        I32: numpy.int32,
        I64: numpy.int64,
        F32: numpy.float32,
        F64: numpy.float64,
    }


ONE_GIGABYTE = 1 << 30

_PRIVATE = object()


class Resolver:
    """
    Used to save/load datasets from IPFS
    """

    def __init__(self, cache_bytes=ONE_GIGABYTE):
        self._inner = _dcdf.PyResolver(cache_bytes)

    def get_dataset(self, cid):
        return Dataset(self._inner.get_dataset(cid), _PRIVATE)

    def ls(self, cid):
        return self._inner.ls(cid)


class Dataset:
    @classmethod
    def new(
        self, coordinates: List["Coordinate"], shape: List[int], resolver: Resolver
    ):
        (t, y, x) = coordinates

        return Dataset(
            _dcdf.PyDataset(t._inner, y._inner, x._inner, shape, resolver._inner),
            _PRIVATE,
        )

    def __init__(self, inner, private=None):
        if private is not _PRIVATE:
            raise RuntimeError("Create new Datasets using the 'new' class method.")

        self._inner = inner

    @property
    def coordinates(self):
        return [Coordinate(coord, _PRIVATE) for coord in self._inner.coordinates]

    @property
    def variables(self):
        return [Variable(var, _PRIVATE) for var in self._inner.variables]

    @property
    def shape(self):
        return self._inner.shape

    @property
    def prev(self):
        return self._inner.prev

    @property
    def cid(self):
        return self._inner.cid

    def add_variable(
        self,
        name: str,
        span_size: int,
        chunk_size: int,
        k2_levels: tuple[int],
        round=False,
        fractional_bits=0,
        dtype=numpy.float32,
    ) -> "Dataset":
        encoding = MMEncoding.from_dtype[dtype]
        return type(self)(
            self._inner.add_variable(
                name, span_size, chunk_size, k2_levels, round, fractional_bits, encoding
            ),
            _PRIVATE,
        )

    def append(self, name: str, data: numpy.ndarray) -> "Dataset":
        if data.dtype == numpy.int32:
            dataset = self._inner.append_i32(name, data)
        elif data.dtype == numpy.int64:
            dataset = self._inner.append_i64(name, data)
        elif data.dtype == numpy.float32:
            dataset = self._inner.append_f32(name, data)
        elif data.dtype == numpy.float64:
            dataset = self._inner.append_f64(name, data)
        else:
            raise ValueError(f"Unsupported dtype: {data.dtype}")

        return type(self)(dataset, _PRIVATE)

    def commit(self):
        return self._inner.commit()

    def get_coordinate(self, name):
        coord = self._inner.get_coordinate(name)
        if coord is not None:
            coord = Coordinate(coord, _PRIVATE)

        return coord

    def get_variable(self, name):
        var = self._inner.get_variable(name)
        if var is not None:
            var = Variable(var, _PRIVATE)

        return var

    def __getattr__(self, name):
        for coord in self._inner.coordinates:
            if coord.name == name:
                return Coordinate(coord, _PRIVATE)

        for var in self._inner.variables:
            if var.name == name:
                return Variable(var, _PRIVATE)

        return self.__getattribute__(name)


class Coordinate:
    @classmethod
    def time(cls, name: str, start, step):
        if isinstance(start, numpy.datetime64):
            start = int((start - numpy.datetime64(0, "s")).item().total_seconds())
        if isinstance(step, numpy.timedelta64):
            step = int(step.item().total_seconds())
        return cls(_dcdf.PyCoordinate.time(name, start, step), _PRIVATE)

    @classmethod
    def range(cls, name: str, start, step, steps, dtype=numpy.float64):
        if dtype == numpy.int32:
            return cls(_dcdf.PyCoordinate.range_i32(name, start, step, steps), _PRIVATE)
        elif dtype == numpy.int64:
            return cls(_dcdf.PyCoordinate.range_i64(name, start, step, steps), _PRIVATE)
        elif dtype == numpy.float32:
            return cls(_dcdf.PyCoordinate.range_f32(name, start, step, steps), _PRIVATE)
        elif dtype == numpy.float64:
            return cls(_dcdf.PyCoordinate.range_f64(name, start, step, steps), _PRIVATE)
        else:
            raise ValueError(f"unsupported dtype for Coordinate {dtype}")

    def __init__(self, inner: _dcdf.PyCoordinate, private=None):
        if private is not _PRIVATE:
            raise RuntimeError(
                "Please instantiate Coordinate using one of the constructor class "
                "methods like 'range' or 'time'"
            )

        self._inner = inner

    @property
    def name(self):
        return self._inner.name

    @property
    def dtype(self):
        return MMEncoding.to_dtype[self._inner.encoding]

    @property
    def _data(self):
        encoding = self._inner.encoding
        if encoding == MMEncoding.Time:
            return self._inner.data_time()
        elif encoding == MMEncoding.I32:
            return self._inner.data_i32()
        elif encoding == MMEncoding.I64:
            return self._inner.data_i64()
        elif encoding == MMEncoding.F32:
            return self._inner.data_f32()
        elif encoding == MMEncoding.F64:
            return self._inner.data_f64()
        else:  # pragma: NO COVER
            raise ValueError(f"Unexpected MMEncoding {encoding}")

    def _converter(self):
        converter = MMEncoding.to_dtype[self._inner.encoding]
        if converter is numpy.datetime64:
            converter = _to_datetime64

        return converter

    def get(self, index):
        return self._converter()(self._data.get(index))

    def slice(self, start, end):
        return self._data.slice(start, end)

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.step is not None:
                raise ValueError("step not supported for slice")
            start = 0 if i.start is None else i.start
            end = len(self) if i.stop is None else i.stop
            return self.slice(start, end)

        return self.get(i)

    def __len__(self):
        return self._inner.len()


class Variable:
    def __init__(self, inner: _dcdf.PyVariable, private=None):
        if private is not _PRIVATE:
            raise RuntimeError("Variable cannot be instantiated")

        self._inner = inner

    @property
    def name(self):
        return self._inner.name

    @property
    def span_size(self):
        return self._inner.span_size

    @property
    def chunk_size(self):
        return self._inner.chunk_size

    @property
    def k2_levels(self):
        return tuple(self._inner.k2_levels)

    @property
    def round(self):
        return self._inner.round

    @property
    def fractional_bits(self):
        return self._inner.fractional_bits

    @property
    def dtype(self):
        return MMEncoding.to_dtype[self._inner.encoding]

    @property
    def shape(self):
        return self._data.shape()

    def get(self, instant, row, col):
        return self._data.get(instant, row, col)

    def cell(self, start, stop, row, col):
        return self._data.cell(start, stop, row, col)

    def window(self, start, stop, top, bottom, left, right):
        return self._data.window(start, stop, top, bottom, left, right)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = [indices]
        else:
            indices = list(indices)

        n_indices = len(indices)
        if n_indices > 3:
            raise IndexError(
                f"too many indices for array: array is 3-dimensional, "
                f"but {len(indices)} were indexed"
            )

        while len(indices) < 3:
            indices.append(slice(0, None))

        fixed_indices = []
        for index, stop in zip(indices, self.shape):
            if _is_int(index):
                fixed_indices.append(index)
                continue

            if index.start is None:
                index = slice(0, index.stop)

            if index.stop is None:
                index = slice(index.start, stop)

            fixed_indices.append(index)

        instant, row, col = indices = fixed_indices
        scalars = tuple(map(_is_int, indices))

        def realize(instant=instant, row=row, col=col, indices=indices):
            if all(scalars):
                return self.get(instant, row, col)

            if scalars == (False, True, True):
                return self.cell(instant.start, instant.stop, row, col)

            indices = list(map(_as_slice, indices))
            instant, row, col = indices
            array = self.window(
                instant.start, instant.stop, row.start, row.stop, col.start, col.stop
            )

            mask = tuple(
                (0 if scalar else slice(None, None) for scalar in scalars[:n_indices])
            )
            if len(mask) == 1:
                mask = mask[0]
            array = array.__getitem__(mask)

            return array

        return _Slice(realize)

    @property
    def _data(self):
        encoding = self._inner.encoding
        if encoding == MMEncoding.I32:
            return self._inner.data_i32()
        elif encoding == MMEncoding.I64:
            return self._inner.data_i64()
        elif encoding == MMEncoding.F32:
            return self._inner.data_f32()
        elif encoding == MMEncoding.F64:
            return self._inner.data_f64()
        else:  # pragma: NO COVER
            raise ValueError(f"Unexpected MMEncoding {encoding}")


class _Slice:
    def __init__(self, realize):
        self.realize = realize

    @functools.cached_property
    def data(self):
        return self.realize()

    def __getitem__(self, arg):
        return self.data.__getitem__(arg)


def _to_datetime64(i):
    return numpy.datetime64(i, "s")


def _is_int(n):
    return isinstance(n, int)


def _as_slice(n):
    if _is_int(n):
        n = slice(n, n + 1)

    return n
