import itertools

from dcdf import _dcdf

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
