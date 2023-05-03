import code
import os
import sys

import docopt
import numpy as np

import dcdf
import dclimate

"""
Example to illustrate use of DCDF encoding library
"""


class CpcPrecip:
    name = "cpc_precip_global-daily"
    """
    The name of the Dataset in dClimate.
    """

    @staticmethod
    def factory(resolver: dcdf.Resolver) -> dcdf.Dataset:
        """
        Create a new, empty Dataset configured for CPC Precipation data.
        """
        # First we need to set up the coordinates. DCDF deals with 3 dimensional arrays
        # so there will always be 3 coordinates: The first coordinate for time, and two
        # more coordinates for position.

        # CPC Precip data starts in 1979 and has points daily
        t = dcdf.Coordinate.time(
            "time", np.datetime64("1979-01-01"), np.timedelta64(1, "D")
        )

        # Set up the lat/lon coordinates. Dtype defaults to np.float64. We use float32
        # here to match the dClimate dataset
        lat = dcdf.Coordinate.range("latitude", -89.75, 0.5, 360, np.float32)
        lon = dcdf.Coordinate.range("longitude", -179.75, 0.5, 720, np.float32)

        # For shape, we just specify the two dimensional cross-section, which is a
        # function of how many points are in the space coordinates.
        shape = (360, 720)

        dataset = dcdf.Dataset.new([t, lat, lon], shape, resolver)

        # Now that we've created the Dataset, we need to add our precipitation variable.
        # This is where the K-squared raster data is stored. The overarching structure
        # is there are spans which subdivide that dataset along the time axis. Spans can
        # be arbitrarily nested as needed. The bottom layer of spans contains
        # superchunks which subdivide the dataset along the two space axes. These, too,
        # can be arbitrarily nested, with the bottom layer containing K-squared encoded
        # chunks.
        #
        # We can subdivide this data into just about any size chunks we want and at this
        # time there hasn't been enough experimentation to know what the best way to do
        # that is. There is a notion that we should maybe try to keep the subchunk size
        # below 1 megabyte to prevent IPFS from having to chunk up the K-squared chunks.
        # With one IPFS chunk for K-squared chunk, we only have to retrieve the chunks
        # we need to answer our query. This should be regarded as untested hypothesis
        # for now.
        #
        # If we accpet the above premise, though, then we can compute how much
        # uncompressed data will fit in a megabyte and just use that, assuming in
        # practice our chunks will be smaller. A chunk of 64x64x64 32 bit floating point
        # numbers is exactly one megabyte, so let's use that.
        #

        # Chunk size specifies how many time instants to store in a superchunk. We just
        # decided that would be 64 for now.
        chunk_size = 64

        # k2_levels specifies how many quadbox tree levels to use at each layer of data,
        # starting with the topmost superchunks and ending with the leaf-node subchunks.
        # Since each node in a quad tree has four children, then then nth level of a quad
        # tree will have 4^n nodes. Or, put another way, the nth level will be a square
        # with side length of 2^n.
        #
        # In order to represent raster data in a quadtree, the logical size of the
        # raster has to be expanded to be a square with each side being a length that is
        # a power of 2. So this 360x720 dataset will be expanded internally to 1024x1024,
        # and will require 10 quadtree levels in total to store, since 2^10 is 1024. A
        # 64x64 chunk has 4096 points and requires 6 levels. So, if we use 6 level
        # quadtrees for the bottom layer of subchunks, then we still have 4 levels left
        # to represent the entire dataset, so we can use a single layer of 4 level
        # superchunks.
        k2_levels = [4, 6]  # Must add up to 10 for this dataset

        # Span size specifies how many chunks (or subspans) to store per span. Since a
        # span is stored as basically just an array of cids, then the size of a span,
        # aside from a small header, is just a linear function of the number of cids.
        # There is some variability in cid length, but I don't think you're likely to
        # ever see one that is longer than 50 bytes, we'll use that as a pathological
        # worst case. A megabyte is 2^20 bytes, but we'll round down to 1000000 bytes
        # for our case. 1000000/ 50 = 20000, so 20000 seems like a good number that will
        # keep spans under a megabyte.
        span_size = 20000

        # When a new variable is created, it is given one empty span. As data is
        # appended to the variable, superchunks are added to the span until it is full.
        # At that point, a new root span is created, the original span is made the first
        # node of the new root span, a second span is created after the original as a
        # child of the root, and superchunks are added to that new span. The dataset
        # will automatically add spans as needed, nesting to any arbitrary depth.

        # In the case of this dataset, we can store 20000 * 64 = 128000 time instants in
        # single span. CPC Precip data is only up to 16192 so far (2023-05-03), and only
        # adds one point per day, so we're not likely to see this particular dataset
        # outgrow the single root span. Because this dataset is relatively small, to get
        # a data point from it, DCDF only needs to traverse one span, one superchunk,
        # and then the chunk containing the point, using K-squared tree traversal.

        # There are optional arguments to this method for setting the dtype (default:
        # float32) and rounding. The default is to not do any rounding. Because floating
        # point numbers have to be converted to a fixed point representation to be
        # encoded using K-squared, this has some caveats.
        dataset = dataset.add_variable("precip", span_size, chunk_size, k2_levels)

        # Note that mutating the dataset created a new dataset. Fundamentally everything
        # uses immutable data structures.
        return dataset


DATASETS = {"cpc_precip": CpcPrecip}


def initialize_dataset(Dataset):
    """Initialize a new dataset.

    Saves the new CID locally in a file.
    """
    head_file = f".{Dataset.name}_head"
    if os.path.exists(head_file):
        error(f"Dataset already initialized. HEAD is stored at {head_file}")

    # The resolver is the object that stores and retrieves objects from the
    # underlying IPLD store. In the Rust implementation, you can pass the resolver
    # an implementation of the Mapper interface ("trait" in Rust-speak), allowing
    # for the use of arbitrary IPLD-like stores. In this first path of the Python
    # wrapper, I've just hard coded it to use the IPFS Mapper, implemented in the
    # "dcdf-ipfs" Crate in this same repo.
    resolver = dcdf.Resolver()

    # See the factory method of the datasets for more info
    dataset = Dataset.factory(resolver)

    # Commit a dataset to save current state to IPLD and get a cid. Note that most of
    # the dataset will already be written to IPLD. Calling commit here just saves the
    # topmost level. Most everything else has already been written.
    cid = dataset.commit()

    with open(head_file, "w") as out:
        print(cid, file=out)

    print(f"Success. New head saved to {head_file}.")


def copy_data_from_dclimate(Dataset, n_instants=None, commit_every=10):
    """
    Populate our dataset by copying data from dclimate.
    """
    head_file = f".{Dataset.name}_head"
    if not os.path.exists(head_file):
        error(
            f"Dataset doesn't exist. Have you initalized it? HEAD should be"
            f" stored at {head_file}"
        )

    # Get our Dataset
    resolver = dcdf.Resolver()
    head = open(head_file).read().strip()
    dataset = resolver.get_dataset(head)

    # Assume only one variable
    dst = dataset.variables[0]

    # Get the dClimate dataset
    src_dataset = dclimate.get_dataset(dclimate.get_head(Dataset.name))

    # Get the dClimate variable (same name as ours)
    src = getattr(src_dataset, dst.name)

    # See how much is left to copy
    written = dst.shape[0]
    total = src.shape[0]
    remaining = total - written

    if remaining <= 0:
        error("Nothing left to do")

    if n_instants is None:
        n_instants = remaining
    else:
        n_instants = remaining = min(n_instants, remaining)

    commit_count = commit_every

    # Copy one chunk width at a time
    for index in range(written, written + n_instants, dst.chunk_size):
        src_chunk = src[index : index + dst.chunk_size]

        # Again, notice that mutating a dataset creates a new dataset. Data in
        # K-squared, and in IPLD more generally, is immutable.
        print("About to get src data")
        src_data = src_chunk.data
        print("Got src data")
        dataset = dataset.append(dst.name, src_data)
        remaining -= dst.chunk_size

        if commit_count == 1:
            cid = dataset.commit()
            with open(head_file, "w") as out:
                print(cid, file=out)
            print(f"Incremental progress saved. New head saved to {head_file}.")
            commit_count = commit_every

        else:
            commit_count -= 1

        dst = dataset.variables[0]  # Get latest version (data is immutable)
        print(f"Copied {dst.shape[0]}/{src.shape[0]}")

    # Any little nubbin at the end that needs to be tacked on?
    if remaining > 0:
        written = dst.shape[0]
        src_chunk = src[written : written + remaining]

        dataset = dataset.append(dst.name, src_chunk.data)

    # Commit changes. Data is already written, this just formalizes the top level
    # Dataset structure.
    cid = dataset.commit()

    with open(head_file, "w") as out:
        print(cid, file=out)

    print(f"Success. New head saved to {head_file}.")


def shell(Dataset):
    """Open an interactive shell to explore data."""
    head_file = f".{Dataset.name}_head"
    if not os.path.exists(head_file):
        error(
            f"Dataset doesn't exist. Have you initalized it? HEAD should be"
            f" stored at {head_file}"
        )

    # Get our Dataset
    resolver = dcdf.Resolver()
    head = open(head_file).read().strip()
    data = resolver.get_dataset(head)

    # Get the dClimate dataset
    src = dclimate.get_dataset(dclimate.get_head(Dataset.name))

    locals = {
        "data": data,
        "src": src,
        "np": np,
    }

    banner = (
        "Welcome to the DCDF interactive shell.\n\n"
        "Some helpful local variables:\n\n"
        "\tdata: The DCDF dataset.\n"
        "\tsrc: The dClimate version of the same dataset\n"
        "\tnp: NumPY module\n\n"
        "Type 'help(dataset)' for more information."
    )

    code.interact(banner, local=locals)


def main():
    """
    Example of DCDF (K-squared raster)

    Usage:
        example.py <dataset> init
        example.py <dataset> copy [<n_instants>]
        example.py <dataset> shell

    Options:
      -h --help     Show this screen.
    """
    args = docopt.docopt(main.__doc__)
    Dataset = DATASETS.get(args["<dataset>"])
    if Dataset is None:
        error(f"No such dataset: {args['<dataset>']}")

    if args["init"]:
        initialize_dataset(Dataset)

    elif args["copy"]:
        n_instants = args["<n_instants>"]
        if n_instants is not None:
            n_instants = int(n_instants)
        copy_data_from_dclimate(Dataset, n_instants)

    elif args["shell"]:
        shell(Dataset)


def error(message):
    print(message, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
