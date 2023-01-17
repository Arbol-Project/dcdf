import contextlib
import functools
import time

from dateutil.parser import parse as parse_date
import numpy

import dcdf
from dclimate_zarr_client import geo_utils

import dclimate
from cpc_precip import CpcPrecipDataset


DATASET = "cpc_precip_global-daily"


def query_dclimate(dataset, start, end, lat, lon):
    data = geo_utils.get_data_in_time_range(dataset, start, end)
    data = geo_utils.get_single_point(data, lat, lon)
    data = data.precip.data

    return data


def query_dcdf(dataset, start, end, lat, lon):
    (start, lat, lon), values = dataset.cell(start, end, lat, lon)
    return values


def timeit(f, n=5):
    start = time.time()
    for _ in range(n):
        f()
    elapsed = time.time() - start

    return elapsed / n


@contextlib.contextmanager
def time_this(what):
    try:
        start = time.time()
        yield
    finally:
        elapsed = time.time() - start
        print(f"{what}: {elapsed}")


if __name__ == "__main__":
    import sys

    start, end = map(parse_date, sys.argv[1:3])
    lat, lon = map(float, sys.argv[3:5])

    dclimate_root = dclimate.get_head(DATASET)
    dataset = dclimate.get_dataset(dclimate_root, instrument=True)
    production = functools.partial(query_dclimate, dataset, start, end, lat, lon)

    root_file = f".{DATASET}_head"
    root = open(root_file).read().strip()
    resolver = dcdf.new_ipfs_resolver()
    dataset = CpcPrecipDataset(root, resolver)

    np_start = numpy.datetime64(start)
    np_end = numpy.datetime64(end)
    experimental = functools.partial(query_dcdf, dataset, np_start, np_end, lat, lon)

    # Verify the two sources are giving the same results. Also allow them to prime their
    # respective caches before performing the benchmark
    with time_this("First production"):
        expected = production()
    with time_this("First experimental"):
        got = experimental()

    assert numpy.array_equal(expected, got, equal_nan=True)

    dataset = dclimate.get_dataset(dclimate_root, instrument=False)
    production = functools.partial(query_dclimate, dataset, start, end, lat, lon)
    production()   # prime again, since we reinstantiated

    print(f"production: {timeit(production, 100)}")
    print(f"experimental: {timeit(experimental, 100)}")
