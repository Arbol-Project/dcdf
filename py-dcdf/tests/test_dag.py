import numpy
import pytest

import dcdf


def get_data(dtype):
    return numpy.array(
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


def array(dtype, sidelen):
    data = get_data(dtype)
    a = numpy.zeros((100, sidelen, sidelen), dtype=dtype)
    for z in range(100):
        for y in range(sidelen):
            for x in range(sidelen):
                a[z, y, x] = data[z % 3, y % 8, x % 8]

    return a


def build_superchunk(data, resolver):
    return dcdf.build_superchunk(data, 3, resolver, fraction=3, local_threshold=0)


@pytest.fixture(scope="session", params=[numpy.float32])
def dtype(request):
    return request.param


@pytest.fixture(scope="session")
def resolver(dtype):
    return dcdf.new_ipfs_resolver(dtype=dtype)


@pytest.fixture(scope="session")
def datastream(dtype, resolver):
    data1 = array(dtype, 16)
    superchunk1 = build_superchunk(data1, resolver)

    a = resolver.insert(None, "data", superchunk1)
    c = resolver.insert(None, "a", a)

    commit1 = resolver.commit("First commit", c, None)

    data2 = array(dtype, 15)
    superchunk2 = build_superchunk(data2, resolver)

    b = resolver.insert(None, "data", superchunk2)
    c = resolver.insert(c, "b", b)

    bob = resolver.store_object(b"Hi mom!\n")
    c = resolver.insert(c, "README.txt", bob)

    commit2 = resolver.commit("Second commit", c, commit1)

    print(f"HEAD ({dtype}): {commit2}")

    return commit2


def test_make_a_couple_of_commits(datastream, resolver):
    # Read DAG structure
    commit = resolver.get_commit(datastream)
    assert commit.message == "Second commit"

    c = commit.root
    bob = resolver.load_object(c["README.txt"])
    assert bob == b"Hi mom!\n"

    a = resolver.get_folder(c["a"])
    b = resolver.get_folder(c["b"])

    superchunk = resolver.get_superchunk(a["data"])
    assert superchunk.shape == (100, 16, 16)

    superchunk = resolver.get_superchunk(b["data"])
    assert superchunk.shape == (100, 15, 15)

    commit = commit.prev
    assert commit.message == "First commit"

    c = commit.root
    a = resolver.get_folder(c["a"])

    superchunk = resolver.get_superchunk(a["data"])
    assert superchunk.shape == (100, 16, 16)

    assert "b" not in c
    assert commit.prev is None


@pytest.fixture(scope="session")
def superchunk(datastream, resolver):
    commit = resolver.get_commit(datastream)
    a = resolver.get_folder(commit.root["a"])
    superchunk = resolver.get_superchunk(a["data"])
    assert superchunk.shape == (100, 16, 16)

    """
    print("")
    for instant in range(100):
        for row in range(16):
            for col in range(16):
                value = superchunk[instant, row, col]
                print(f"{value:1.3f} ", end="")
            print("")
        print("")
    """

    return superchunk


def test_get(superchunk, dtype):
    data = array(dtype, 16)
    for instant in range(100):
        for row in range(16):
            for col in range(16):
                assert superchunk[instant, row, col] == data[instant, row, col]


def test_cell(superchunk, dtype):
    data = array(dtype, 16)
    instants, rows, cols = superchunk.shape
    for row in range(rows):
        for col in range(cols):
            start = row + col
            end = instants - start
            values = superchunk.cell(start, end, row, col)
            assert len(values) == end - start
            for i in range(len(values)):
                assert values[i] == data[i + start, row, col]


def test_window(superchunk, dtype):
    data = array(dtype, 16)
    instants, rows, cols = superchunk.shape
    for top in range(rows // 2):
        bottom = top + rows // 2
        for left in range(cols // 2):
            right = left + cols // 2
            start = top + bottom
            end = instants - start
            window = superchunk.window(start, end, top, bottom, left, right)

            assert window.shape == (end - start, bottom - top, right - left)
            for i in range(end - start):
                assert numpy.array_equal(
                    window[i], data[start + i, top:bottom, left:right]
                )


def search(data, start, end, top, bottom, left, right, lower, upper):
    coords = set()
    for instant in range(start, end):
        for row in range(top, bottom):
            for col in range(left, right):
                if lower <= data[instant, row, col] <= upper:
                    coords.add((instant, row, col))

    return coords


def test_search(superchunk, dtype):
    data = array(dtype, 16)
    instants, rows, cols = superchunk.shape
    for top in range(rows // 2):
        bottom = top + rows // 2
        for left in range(cols // 2):
            right = left + cols // 2
            start = top + bottom
            end = instants - start
            lower = start // 5
            upper = end // 10

            expected = search(data, start, end, top, bottom, left, right, lower, upper)
            results = set(
                superchunk.search(start, end, top, bottom, left, right, lower, upper)
            )

            assert len(results) == len(expected)
            assert results == expected
