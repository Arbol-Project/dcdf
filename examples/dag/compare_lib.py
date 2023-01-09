import contextlib
import functools
import json
import subprocess
import time

from dateutil.parser import parse as parse_date
import ipldstore
import numpy
import requests
import xarray

import dcdf
from dclimate_zarr_client import geo_utils

from cpc_precip import CpcPrecipDataset


DCLIMATE_TOKEN = open(".dclimate_token").read().strip()
DATASET = "cpc_precip_global-daily"
global HEADS


def get_dclimate_heads():
    url = "https://api.dclimate.net/apiv4/get_heads"
    headers = {"Authorization": DCLIMATE_TOKEN}
    response = requests.get(url, headers=headers)
    return response.json()


def get_dclimate_head(dataset):
    heads = get_dclimate_heads()
    name = heads[dataset]

    process = subprocess.run(["ipfs", "name", "resolve", name], capture_output=True)
    metadata_path = process.stdout.decode("utf8").strip()
    assert metadata_path.startswith("/ipfs/")
    metadata_cid = metadata_path[len("/ipfs/") :]

    process = subprocess.run(["ipfs", "dag", "get", metadata_cid], capture_output=True)
    metadata = json.loads(process.stdout)

    return metadata["assets"]["zmetadata"]["href"]["/"]


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

    root = get_dclimate_head(DATASET)
    mapper = ipldstore.get_ipfs_mapper()
    mapper.set_root(root)
    dataset = xarray.open_zarr(mapper)

    production = functools.partial(query_dclimate, dataset, start, end, lat, lon)

    root_file = f".{DATASET}_head"
    root = open(root_file).read().strip()
    resolver = dcdf.new_ipfs_resolver()
    dataset = CpcPrecipDataset(root, resolver)

    start = numpy.datetime64(start)
    end = numpy.datetime64(end)
    experimental = functools.partial(query_dcdf, dataset, start, end, lat, lon)

    # Verify the two sources are giving the same results. Also allow them to prime their
    # respective caches before performing the benchmark
    with time_this("First production"):
        expected = production()
    with time_this("First epxerimental"):
        got = experimental()

    assert numpy.array_equal(expected, got)

    print(f"production: {timeit(production, 100)}")
    print(f"experimental: {timeit(experimental, 100)}")
