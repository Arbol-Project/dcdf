import itertools
import numpy
import pytest

import dcdf


def get_data(dtype):
    if dtype in (numpy.float32, numpy.float64):
        data = numpy.array(
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
            ],
            dtype=dtype,
        )
    else:
        data = numpy.array(
            [
                [
                    [9, 8, 7, 7, 6, 6, 3, 2],
                    [7, 7, 7, 7, 6, 6, 3, 3],
                    [6, 6, 6, 6, 3, 3, 3, 3],
                    [5, 5, 6, 6, 3, 3, 3, 3],
                    [4, 5, 5, 5, 4, 4, 4, 4],
                    [3, 3, 5, 5, 4, 4, 4, 4],
                    [3, 3, 3, 5, 4, 4, 4, 4],
                    [4, 4, 3, 4, 4, 4, 4, 4],
                ],
                [
                    [9, 8, 7, 7, 7, 7, 2, 2],
                    [7, 7, 7, 7, 7, 7, 2, 2],
                    [6, 6, 6, 6, 4, 3, 3, 3],
                    [5, 5, 6, 6, 3, 3, 3, 3],
                    [4, 5, 5, 5, 4, 4, 4, 4],
                    [3, 3, 5, 5, 4, 4, 4, 4],
                    [3, 3, 4, 5, 5, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                ],
                [
                    [9, 8, 7, 7, 8, 7, 5, 5],
                    [7, 7, 7, 7, 7, 7, 5, 5],
                    [7, 7, 6, 6, 4, 3, 4, 4],
                    [6, 6, 6, 6, 4, 4, 4, 4],
                    [4, 5, 5, 5, 4, 4, 4, 4],
                    [3, 3, 5, 5, 4, 4, 4, 4],
                    [3, 3, 4, 5, 6, 4, 4, 4],
                    [4, 4, 4, 4, 5, 4, 4, 4],
                ],
            ],
            dtype,
        )

    data = numpy.array(list(itertools.islice(itertools.cycle(data), 100)))
    assert data.shape == (100, 8, 8)

    return data


def get_chunk(tmpdir, data):
    built = dcdf.build(data, fraction=3)
    path = str(tmpdir / "tmpdata.dcdf")
    built.save_as(path)

    return dcdf.load(path)


dtypes = (
    numpy.int32,
    numpy.float32,
)


@pytest.mark.parametrize("dtype", dtypes)
def test_cell(tmpdir, dtype):
    data = get_data(dtype)
    chunk = get_chunk(tmpdir, data)
    assert chunk.shape == (100, 8, 8)

    for row in range(8):
        for col in range(8):
            for i, n in enumerate(chunk.cell(0, 100, row, col)):
                assert data[i][row, col] == n


def windows():
    for top in range(8):
        for bottom in range(top + 1, 8):
            for left in range(8):
                for right in range(left + 1, 8):
                    yield top, bottom, left, right


@pytest.mark.parametrize("dtype", dtypes)
def test_window(tmpdir, dtype):
    data = get_data(dtype)
    chunk = get_chunk(tmpdir, data)
    for top, bottom, left, right in windows():
        for i, window in enumerate(chunk.window(0, 100, top, bottom, left, right)):
            assert numpy.array_equal(data[i][top:bottom, left:right], window)


def searches():
    for top in range(5, 8):
        for bottom in range(top + 1, 8):
            for left in range(4):
                for right in range(left + 1, 4):
                    for lower in range(6, 9):
                        for upper in range(lower, 11):
                            yield top, bottom, left, right, lower, upper


def search(data, top, bottom, left, right, lower, upper):
    for instant in data:
        coords = []
        for row in range(top, bottom):
            for col in range(left, right):
                if lower <= instant[row, col] <= upper:
                    coords.append((row, col))

        yield set(coords)


@pytest.mark.parametrize("dtype", dtypes)
def test_search(tmpdir, dtype):
    data = get_data(dtype)
    chunk = get_chunk(tmpdir, data)
    for top, bottom, left, right, lower, upper in searches():
        expecteds = search(data, top, bottom, left, right, lower, upper)
        results = map(set, chunk.search(0, 100, top, bottom, left, right, lower, upper))
        for expected, result in zip(expecteds, results):
            assert expected == result


def test_suggest_fraction_3bits():
    data = get_data(numpy.float32)
    suggestion = dcdf.suggest_fraction(data, 15.0)
    assert suggestion.fractional_bits == 3
    assert suggestion.round is False


def test_suggest_fraction_4bits():
    data = numpy.array([[[16.0, 1.0 / 16.0]]], dtype=numpy.float32)
    suggestion = dcdf.suggest_fraction(data, 16.0)
    assert suggestion.fractional_bits == 4
    assert suggestion.round is False
