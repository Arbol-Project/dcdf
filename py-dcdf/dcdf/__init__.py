from dcdf import _dcdf
from dcdf._dcdf import load

_BUILDERS = {
    "int32": _dcdf.PyBuilderI32,
    "float32": _dcdf.PyBuilderF32,
}


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
