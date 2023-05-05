import itertools
import numpy
import pytest

import dcdf


test_array = numpy.array(
    [
        [
            [9.5, 8.25, 7.75, 7.75, 6.125, 6.125, 3.375, 2.625],
            [7.75, 7.75, 7.75, 7.75, 6.125, 6.125, 3.375, 3.375],
            [6.125, 6.125, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
            [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 3.375, 5.0, 4.875, 4.875, 4.875, 4.875],
            [4.875, 4.875, 3.375, 4.875, 4.875, 4.875, 4.875, 4.875],
        ],
        [
            [9.5, 8.25, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
            [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
            [6.125, 6.125, 6.125, 6.125, 4.875, 3.375, 3.375, 3.375],
            [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 4.875, 5.0, 5.0, 4.875, 4.875, 4.875],
            [4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875],
        ],
        [
            [9.5, 8.25, 7.75, 7.75, 8.25, 7.75, 5.0, 5.0],
            [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 5.0, 5.0],
            [7.75, 7.75, 6.125, 6.125, 4.875, 3.375, 4.875, 4.875],
            [6.125, 6.125, 6.125, 6.125, 4.875, 4.875, 4.875, 4.875],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 4.875, 5.0, 6.125, 4.875, 4.875, 4.875],
            [4.875, 4.875, 4.875, 4.875, 5.0, 4.875, 4.875, 4.875],
        ],
    ]
)


def make_data(instants: int) -> numpy.ndarray:
    data = numpy.tile(test_array, [instants // 3 + 1, 2, 2])[:instants]
    assert data.shape == (instants, 16, 16)
    return data


def make_one(resolver: dcdf.Resolver, dtype: numpy.dtype):
    t = dcdf.Coordinate.time("t", 0, numpy.timedelta64(100, "s"))
    y = dcdf.Coordinate.range("y", -160, 20, 16, dtype)
    x = dcdf.Coordinate.range("x", -200, 25, 16, dtype)

    return dcdf.Dataset.new([t, y, x], [16, 16], resolver)


@pytest.fixture(scope="session")
def resolver():
    return dcdf.Resolver()


@pytest.mark.parametrize(
    "dtype", [numpy.int32, numpy.int64, numpy.float32, numpy.float64]
)
def test_new(resolver: dcdf.Resolver, dtype: numpy.dtype):
    dataset = make_one(resolver, dtype)

    assert dataset.coordinates[0].name == "t"
    assert dataset.coordinates[1].name == "y"
    assert dataset.coordinates[2].name == "x"

    assert dataset.shape == (16, 16)
    assert dataset.prev is None
    assert dataset.cid is None

    assert len(dataset.variables) == 0

    with pytest.raises(ValueError):
        assert len(dataset.t) == 0
    assert dataset.t[10] == numpy.datetime64("1970-01-01T00:16:40")
    assert dataset.t.dtype == numpy.datetime64
    expected = numpy.arange(
        numpy.datetime64("1970-01-01T00:33:20"),
        numpy.datetime64("1970-01-01T00:50:00"),
        numpy.timedelta64("100", "s"),
    )
    assert numpy.array_equal(
        dataset.t[20:30],
        expected,
    )

    assert len(dataset.get_coordinate("y")) == 16
    assert dataset.y[10] == 40
    assert dataset.y.dtype == dtype
    assert numpy.array_equal(dataset.y[10:], numpy.arange(40, 160, 20))

    assert len(dataset.get_coordinate("x")) == 16
    assert dataset.x[10] == 50
    assert dataset.x.dtype == dtype
    assert numpy.array_equal(dataset.x[:10], numpy.arange(-200, 50, 25))

    with pytest.raises(AttributeError):
        dataset.doesnotexist


VARIABLES = ("apples", "pears", "bananas", "grapes", "dates", "melons")


@pytest.fixture(scope="session")
def populated(resolver):
    dataset = make_one(resolver, numpy.float64)
    assert dataset.cid is None

    apple_data = make_data(360).astype(numpy.float32)
    dataset = dataset.add_variable("apples", 10, 20, [2, 2])
    dataset = dataset.append("apples", apple_data[:99])
    dataset = dataset.append("apples", apple_data[99:200])
    dataset = dataset.append("apples", apple_data[200:])

    pear_data = make_data(500).astype(numpy.float64)
    dataset = dataset.add_variable("pears", 10, 20, (2, 2), dtype=numpy.float64)
    dataset = dataset.append("pears", pear_data[:189])
    dataset = dataset.append("pears", pear_data[189:400])
    dataset = dataset.append("pears", pear_data[400:])

    banana_data = make_data(511).astype(numpy.int32)
    dataset = dataset.add_variable("bananas", 10, 20, (2, 2), dtype=numpy.int32)
    dataset = dataset.append("bananas", banana_data[:59])
    dataset = dataset.append("bananas", banana_data[59:300])
    dataset = dataset.append("bananas", banana_data[300:])

    grape_data = make_data(365).astype(numpy.int64)
    dataset = dataset.add_variable("grapes", 10, 20, (2, 2), dtype=numpy.int64)
    dataset = dataset.append("grapes", grape_data[:179])
    dataset = dataset.append("grapes", grape_data[179:300])
    dataset = dataset.append("grapes", grape_data[300:])

    assert dataset.prev is None
    cid = dataset.commit()
    dataset = resolver.get_dataset(cid)
    assert dataset.cid == cid

    date_data = make_data(489)
    dataset = dataset.add_variable("dates", 10, 20, [2, 2], True, 2)
    dataset = dataset.append("dates", date_data)
    date_data = (date_data * 4 + 0.001).round() / 4

    assert dataset.cid is None
    assert dataset.prev == cid

    melon_data = make_data(489)
    dataset = dataset.add_variable("melons", 10, 20, [2, 2], True, 2, numpy.float64)
    dataset = dataset.append("melons", melon_data)
    melon_data = (melon_data * 4 + 0.001).round() / 4

    assert dataset.cid is None
    assert dataset.prev == cid

    test_data = {
        "apples": apple_data,
        "pears": pear_data,
        "bananas": banana_data,
        "grapes": grape_data,
        "dates": date_data,
        "melons": melon_data,
    }
    return dataset, test_data


@pytest.fixture(scope="session")
def dataset(populated):
    return populated[0]


@pytest.fixture(scope="session")
def test_data(populated):
    return populated[1]


def test_populate(dataset, resolver):
    apples = dataset.apples
    assert apples.name == "apples"
    assert apples.round is False
    assert apples.span_size == 10
    assert apples.chunk_size == 20
    assert apples.k2_levels == (2, 2)
    assert apples.dtype is numpy.float32

    pears = dataset.pears
    assert pears.name == "pears"
    assert pears.round is False
    assert pears.span_size == 10
    assert pears.chunk_size == 20
    assert pears.k2_levels == (2, 2)
    assert pears.dtype is numpy.float64

    bananas = dataset.bananas
    assert bananas.name == "bananas"
    assert bananas.round is False
    assert bananas.span_size == 10
    assert bananas.chunk_size == 20
    assert bananas.k2_levels == (2, 2)
    assert bananas.dtype is numpy.int32

    grapes = dataset.grapes
    assert grapes.name == "grapes"
    assert grapes.round is False
    assert grapes.span_size == 10
    assert grapes.chunk_size == 20
    assert grapes.k2_levels == (2, 2)
    assert grapes.dtype is numpy.int64

    dates = dataset.dates
    assert dates.name == "dates"
    assert dates.round is True
    assert dates.fractional_bits == 2
    assert dates.span_size == 10
    assert dates.chunk_size == 20
    assert dates.k2_levels == (2, 2)
    assert dates.dtype is numpy.float32

    melons = dataset.melons
    assert melons.name == "melons"
    assert melons.round is True
    assert melons.fractional_bits == 2
    assert melons.span_size == 10
    assert melons.chunk_size == 20
    assert melons.k2_levels == (2, 2)
    assert melons.dtype is numpy.float64


@pytest.mark.parametrize("var", VARIABLES)
def test_get(dataset, test_data, var):
    data = test_data[var]
    variable = getattr(dataset, var)
    instants, rows, cols = variable.shape
    for instant in range(0, instants, 13):
        for row in range(0, rows, 4):
            for col in range(0, cols, 3):
                assert variable[instant, row, col].data == data[instant, row, col]


@pytest.mark.parametrize("var", VARIABLES)
def test_cell(dataset, test_data, var):
    data = test_data[var]
    variable = dataset.get_variable(var)
    instants, rows, cols = variable.shape
    for row in range(0, rows, 4):
        for col in range(0, cols, 3):
            start = row + col
            end = instants - start
            expected = data[start:end, row, col]
            got = variable[start:end, row, col].data
            assert numpy.array_equal(got, expected)


@pytest.mark.parametrize("var", VARIABLES)
def test_window(dataset, test_data, var):
    data = test_data[var]
    variable = dataset.get_variable(var)
    instants, rows, cols = variable.shape
    for top in range(0, rows // 2, 4):
        bottom = top + rows // 2
        for left in range(0, cols // 2, 3):
            right = left + cols // 2
            start = top + bottom
            end = instants - start
            expected = data[start:end, top:bottom, left:right]
            got = variable[start:end, top:bottom, left:right].data
            assert numpy.array_equal(got, expected)


def test_all_slice_permutations(dataset, test_data):
    data = test_data["apples"]
    variable = dataset.apples

    slice_args = [42, slice(23, 80), slice(None, 20)]
    for t, y in itertools.product([42, slice(23, 80)], [9, slice(6, None)]):
        slice_args.append((t, y))
    for t, y, x in itertools.product(
        [42, slice(23, 80)], [9, slice(6, 13)], [6, slice(3, 15)]
    ):
        slice_args.append((t, y, x))

    for arg in slice_args:
        expected = data.__getitem__(arg)
        got = variable.__getitem__(arg).data
        if isinstance(expected, (int, float, numpy.number)):
            assert got == expected
        else:
            assert got.shape == expected.shape
            assert numpy.array_equal(expected, got)


def test_append_unsupported_dtype(dataset):
    array = numpy.array(range(10), dtype=numpy.byte)
    with pytest.raises(ValueError):
        dataset.append("apples", array)


def test_new_coordinate_range_unsupported_dtype():
    with pytest.raises(ValueError):
        dcdf.Coordinate.range("foo", 0, 1, 10, numpy.byte)


def test_dataset_constructor_is_private():
    with pytest.raises(RuntimeError):
        dcdf.Dataset(None)


def test_coordinate_constructor_is_private():
    with pytest.raises(RuntimeError):
        dcdf.Coordinate(None)


def test_variable_constructor_is_private():
    with pytest.raises(RuntimeError):
        dcdf.Variable(None)


def test_coordinate_step_not_supported_for_slice():
    coord = dcdf.Coordinate.range("foo", 0, 1, 10)
    with pytest.raises(ValueError):
        coord[1:2:3]


def test_variable_too_many_indices_in_slice(dataset):
    with pytest.raises(IndexError):
        dataset.apples[1, 2, 3, 4]
