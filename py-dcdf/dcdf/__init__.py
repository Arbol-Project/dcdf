import numpy

from dcdf._dcdf import PySnapshot32
from dcdf._dcdf import PySnapshotU32
from dcdf._dcdf import PySnapshot64
from dcdf._dcdf import PySnapshotU64


def Snapshot(data, k=2):
    if data.dtype == numpy.int32:
        return PySnapshot32(data, k)

    elif data.dtype == numpy.uint32:
        return PySnapshotU32(data, k)

    elif data.dtype == numpy.int64:
        return PySnapshot64(data, k)

    elif data.dtype == numpy.uint64:
        return PySnapshotU64(data, k)

    else:
        raise ValueError(f"Data type {data.dtype} not supported.")
