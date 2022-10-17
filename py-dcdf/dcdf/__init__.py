import itertools
import numpy

from dcdf import _dcdf
from dcdf.interfaces import (
    Cid,
    Commit,
    Folder,
    FolderItem,
    Resolver,
    Superchunk,
)

Resolver.register(_dcdf.PyResolverF32)
Folder.register(_dcdf.PyFolderF32)
FolderItem.register(_dcdf.PyFolderItem)
Commit.register(_dcdf.PyCommitF32)
Superchunk.register(_dcdf.PySuperchunkF32)


_BUILDERS = {
    "int32": _dcdf.PyBuilderI32,
    "float32": _dcdf.PyBuilderF32,
}

_IPFS_RESOLVER_FACTORIES = {
    "float32": _dcdf.new_ipfs_resolver_f32,
}

_SUPERCHUNK_BUILDERS = {
    "float32": _dcdf.PySuperchunkBuilderF32,
}

_COMMIT_FUNCTIONS = {
    _dcdf.PyFolderF32: _dcdf.commit_f32,
}

_256_MB = 1 << 28


def build(instants, k=2, fraction=24, round=False):
    instants = iter(instants)
    first = next(instants)
    dtype = first.dtype.name
    Builder = _BUILDERS.get(dtype)
    if Builder is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if dtype in ("float32", "float64"):
        builder = Builder(first, k, fraction, round)
    else:
        builder = Builder(first, k)

    for instant in instants:
        builder.push(instant)

    return builder.finish()


_SUGGESTERS = {
    "float32": _dcdf.PyFractionSuggesterF32,
}


def suggest_fraction(instants, max_value):
    instants = iter(instants)
    first = next(instants)
    dtype = first.dtype.name
    Suggester = _SUGGESTERS.get(dtype)
    if Suggester is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    instants = itertools.chain((first,), instants)
    suggester = Suggester(max_value)
    for instant in instants:
        if not suggester.push(instant):
            break

    return Suggestion(*suggester.finish())


class Suggestion:
    def __init__(self, fractional_bits, round):
        self.fractional_bits = fractional_bits
        self.round = round


def load(file_or_path):
    if hasattr(file_or_path, "read"):
        return _dcdf.load(file_or_path)

    return _dcdf.load_from(file_or_path)


def new_ipfs_resolver(cache_bytes=_256_MB, dtype=numpy.float32):
    dtype = numpy.dtype(dtype).name
    factory = _IPFS_RESOLVER_FACTORIES.get(dtype)
    if factory is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return factory(cache_bytes)


def build_superchunk(
    instants, levels, resolver, k=2, fraction=24, round=False, local_threshold=512
):
    instants = iter(instants)
    first = next(instants)
    dtype = first.dtype.name
    Builder = _SUPERCHUNK_BUILDERS.get(dtype)
    if Builder is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    builder = Builder(first, k, fraction, round, levels, resolver, local_threshold)

    for instant in instants:
        builder.push(instant)

    return builder.finish()


def commit(message, root, prev, resolver):
    func = _COMMIT_FUNCTIONS[type(root)]
    return func(message, root.cid, prev, resolver)
