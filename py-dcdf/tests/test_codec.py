import numpy
import pytest

import dcdf


def get_test_data(dtype):
    return numpy.array(
        [
            [9, 8, 7, 7, 6, 6, 3, 2],
            [7, 7, 7, 7, 6, 6, 3, 3],
            [6, 6, 6, 6, 3, 3, 3, 3],
            [5, 5, 6, 6, 3, 3, 3, 3],
            [4, 5, 5, 5, 4, 4, 4, 4],
            [3, 3, 5, 5, 4, 4, 4, 4],
            [3, 3, 4, 5, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4],
        ],
        dtype=dtype,
    )


dtypes = (numpy.int32, numpy.uint32, numpy.int64, numpy.uint64)


class TestSnapshot:
    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_get(dtype):
        data = get_test_data(dtype)
        snapshot = dcdf.Snapshot(data)

        for i in range(8):
            for j in range(8):
                assert snapshot.get(i, j) == data[i, j]

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_get_window(dtype):
        data = get_test_data(dtype)
        snapshot = dcdf.Snapshot(data)

        def windows():
            for top in range(8):
                for bottom in range(top + 1, 8):
                    for left in range(8):
                        for right in range(left + 1, 8):
                            yield top, bottom, left, right

        for top, bottom, left, right in windows():
            assert numpy.array_equal(
                snapshot.get_window(top, bottom, left, right),
                data[top:bottom, left:right],
            )

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_search_window(dtype):
        data = get_test_data(dtype)
        snapshot = dcdf.Snapshot(data)

        def expect(top, bottom, left, right, lower, upper):
            coords = set()
            for row in range(top, bottom):
                for col in range(left, right):
                    if lower <= data[row, col] <= upper:
                        coords.add((row, col))

            return coords

        def searches():
            for top in range(0, 8):
                for bottom in range(top + 1, 8):
                    for left in range(0, 8):
                        for right in range(left + 1, 8):
                            for lower in range(4, 10):
                                for upper in range(lower, 10):
                                    yield top, bottom, left, right, lower, upper

        for top, bottom, left, right, lower, upper in searches():
            expected = expect(top, bottom, left, right, lower, upper)
            coords = snapshot.search_window(top, bottom, left, right, lower, upper)
            assert set(coords) == expected
