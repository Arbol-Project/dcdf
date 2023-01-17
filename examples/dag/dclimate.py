import json
import subprocess

import requests
import xarray

import ipldstore


DCLIMATE_TOKEN = open(".dclimate_token").read().strip()


def get_heads():
    url = "https://api.dclimate.net/apiv4/get_heads"
    headers = {"Authorization": DCLIMATE_TOKEN}
    response = requests.get(url, headers=headers)
    return response.json()


def get_head(dataset):
    heads = get_heads()
    name = heads[dataset]

    process = subprocess.run(["ipfs", "name", "resolve", name], capture_output=True)
    metadata_path = process.stdout.decode("utf8").strip()
    assert metadata_path.startswith("/ipfs/")
    metadata_cid = metadata_path[len("/ipfs/") :]
    print(f"dclimate metadata: {metadata_cid}")

    process = subprocess.run(["ipfs", "dag", "get", metadata_cid], capture_output=True)
    metadata = json.loads(process.stdout)

    return metadata["assets"]["zmetadata"]["href"]["/"]


class InstrumentedIPLDStore(ipldstore.IPLDStore):

    def __init__(self, castore):
        super().__init__(castore)

    def getitems(self, keys):
        items = super().getitems(keys)
        for key, value in items.items():
            print(f"ipldstore: {key}: {len(value)} bytes")

        return items

    def __getitem__(self, key):
        value = super().__getitem__(key)
        print(f"ipldstore_: {key}: {len(value)} bytes")
        return value


def get_dataset(root, instrument=False):
    MM = InstrumentedIPLDStore if instrument else ipldstore.IPLDStore
    castore = ipldstore.IPFSStore("http://127.0.0.1:5001")
    mapper = MM(castore)
    mapper.set_root(root)
    return xarray.open_zarr(mapper)
