import functools
import time

import requests

DCLIMATE_TOKEN = open(".dclimate_token").read().strip()
DATASET = "cpc_precip_global-daily"


def query_dclimate(start, end, lat, lon):
    url = (
        f"https://api.dclimate.net/apiv4/geo_temporal_query/{DATASET}"
        "?output_format=array"
    )
    params = {
        "point_params": {"lat": lat, "lon": lon},
        "time_range": [start, end],
    }
    headers = {"Authorization": DCLIMATE_TOKEN}
    response = requests.post(url, json=params, headers=headers)

    return response.json()


def query_dcdf(start, end, lat, lon):
    url = "http://localhost:8000/"
    params = {
        "point_params": {"lat": lat, "lon": lon},
        "time_range": [start, end],
    }
    response = requests.post(url, json=params)

    return response.json()


def timeit(f, n=5):
    start = time.time()
    for _ in range(n):
        f()
    elapsed = time.time() - start

    return elapsed / n


if __name__ == "__main__":
    import sys

    production = functools.partial(query_dclimate, *sys.argv[1:5])
    experimental = functools.partial(query_dcdf, *sys.argv[1:5])
    assert production() == experimental()

    print(f"production: {timeit(production)}")
    print(f"experimental: {timeit(experimental, 100)}")
