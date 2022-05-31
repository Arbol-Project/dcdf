from dcdf import _dcdf
from dcdf._dcdf import load

_BUILDERS = {
    "int32": _dcdf.PyBuilderI32,
}


def build(instants, k=2):
    instants = iter(instants)
    first = next(instants)
    Builder = _BUILDERS.get(first.dtype.name)
    if Builder is None:
        raise ValueError(f"Unsupported dtype: {first.dtype}")

    builder = Builder(first, k)
    for instant in instants:
        builder.push(instant)

    return builder.finish()
