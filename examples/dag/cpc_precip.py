"""
Encode and retrieve CPC daily precipitation data.

Usage:
    cpc_precip.py init
    cpc_precip.py add <input_file>
    cpc_precip.py query <datetime> <latitude> <longitude>

Options:
  -h --help     Show this screen.
"""
import dataclasses
import docopt
import os
import sys

import numpy
import xarray

import dataset
import dcdf

HEAD = ".cpc_precip_head"


def cli_main():
    resolver = dcdf.new_ipfs_resolver()

    try:
        args = docopt.docopt(__doc__)
        if args["init"]:
            return cli_init(resolver)

        if not os.path.exists(HEAD):
            return cli_error(
                f"Dataset must be initialized. Use `{sys.argv[0]} init` to create  a new "
                f"dataset, or put a CID in the file `{HEAD}` in this directory, if the "
                f"dataset was initialized elsewhere."
            )

        cid = open(HEAD).read().strip()
        data = CpcPrecipDataset(cid, resolver)

        if args["add"]:
            return cli_add(data, args["<input_file>"])

        return cli_error("not implemented")

    except DataError as error:
        return cli_error(error)


def cli_init(resolver):
    if os.path.exists(HEAD):
        return cli_error(f"Dataset already exists. Head is written to {HEAD}")

    root = resolver.init()
    message = "Initial commit."

    cli_new_head(dcdf.commit(message, root, None, resolver))

    return cli_ok(f"New HEAD written to {HEAD}")


def cli_add(dataset, input_file):
    print("Loading...")
    data = xarray.open_dataset(input_file)
    dataset.layout.verify(data)
    dataset.geo.verify(data)
    time = data.time.data[0]

    data = data.precip.data
    shape = data.shape[1:]
    if shape != dataset.shape:
        raise DataError(f"Expected shape: {dataset.shape}, got: {shape}")

    print("Computing encoding requirements...")
    max_value = numpy.nanmax(data)
    suggestion = dcdf.suggest_fraction(data, max_value)

    if suggestion.round:
        print(f"Data must be rounded to use {suggestion.fractional_bits} bit fractions "
              "in order to be able to be encoded.")

    else:
        print("Data can be encoded without loss of precision using "
              f"{suggestion.fractional_bits} bit fractions.")

    # Immutable data means "modifying" the dataset creates a new one
    print("Building...")
    dataset = dataset.add_chunk(time, data, suggestion.fractional_bits, suggestion.round)

    cli_new_head(dataset.cid)

    return cli_ok(f"New head written to {HEAD}")


def cli_new_head(cid):
    with open(HEAD, "w") as f:
        print(cid, file=f)


def cli_ok(message):
    print(message)
    return 0


def cli_error(message):
    print(message, file=sys.stderr)
    return 1


class CpcPrecipDataset(dataset.Dataset):
    LEVELS = 6
    """2^2^5 = 4096 subchunks"""

    LOCAL_THRESHOLD = 0    #1<<12   # 4096
    """Subchunks smaller than this number of bytes will be stored on the superchunk
    rather than as separate objects."""

    def __init__(self, cid, resolver):
        super().__init__(
            cid=cid,
            resolver=resolver,
            layout=DailyLayout(),
            geo=CpcGeoSpace(),
            shape=(360, 720),
        )

    def add_chunk(self, time, data, fraction, round):
        return super().add_chunk(
            time,
            data,
            levels=self.LEVELS,
            fraction=fraction,
            round=round,
            local_threshold=self.LOCAL_THRESHOLD,
        )


class DailyLayout:
    """Layout for data collected daily.

    This layout is two levels, with each top level folder containing a century and each
    century containing superchunks that each contain a year. Each time instant is a day.
    """
    step = numpy.timedelta64(1, "D")

    def verify(self, data):
        """Verify that times in xarray loaded data actually match our assumptions."""
        # Get just the time data
        data = data.time.data

        # Verify shape of time data
        if data.shape not in ((365,), (366,)):
            raise DataError("Expecting 365 or 366 days")

        # Verify that the delta between each point is 1 day
        deltas = set(data[1:] - data[:-1])
        delta = next(iter(deltas))
        if len(deltas) != 1 or delta != self.step:
            raise DataError("Expecting even 1 day spacing between time instants.")

    def locate(self, instant: numpy.datetime64) -> tuple[str, int, numpy.datetime64]:
        # It's confounding that you can't access date components directly in numpy
        # datetimes. Why?!
        year, month, day = str(instant.astype("datetime64[D]")).split("-")

        # Top folder is first two digits of year, then next level is the year
        path = f"{year[:2]}/{year}"

        # Index is the number of full days since Jan 1
        index = (instant - numpy.datetime64(year)) / self.step
        index = int(index)

        # Instant is rounded down to day
        instant = numpy.datetime64(f"{year}-{month}-{day}")

        return path, index, instant

    def time_at(self, path: str, index: int) -> numpy.datetime64:
        # Get timestamp of beginning of year from path
        _, year = path.split("/")
        instant = numpy.datetime64(year)

        # Add index number of days
        instant += index * self.step

        # Done
        return instant


@dataclasses.dataclass
class CpcGeoSpace:
    """GeoSpace with latitude and longitude spaced at regular intervals."""
    lat = numpy.linspace(89.75, -89.75, 360)
    lon = numpy.linspace(0.25, 359.75, 720)

    def verify(self, data):
        """Verify that lat/lon data in the source data matches our assumptions."""
        if not numpy.array_equal(data.lat.data, self.lat):
            raise DataError("Unexpected latitude data.")

        if not numpy.array_equal(data.lon.data, self.lon):
            raise DataError("Unexpected longitude data.")

    def locate(self, lat: float, lon: float
              ) -> tuple[tuple[int, int], tuple[float, float]]:
        row = abs(self.lon - lon).argmin()
        col = abs(self.lat - lat).argmin()

        return (row, col), (self.lon[row], self.lat[col])

    def coord_at(self, row: int, col: int) -> tuple[float, float]:
        return self.lon[row], self.lat[col]


class DataError(Exception):
    """Data does not have expected structure."""


if __name__ == "__main__":
    sys.exit(cli_main())
