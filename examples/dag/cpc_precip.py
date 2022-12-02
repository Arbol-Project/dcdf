"""
Encode and retrieve CPC daily precipitation data.

Usage:
    cpc_precip.py init
    cpc_precip.py shell
    cpc_precip.py serve
    cpc_precip.py add <input_file>
    cpc_precip.py query <datetime> <latitude> <longitude>
    cpc_precip.py query <startdate> <enddate> <latitude> <longitude>
    cpc_precip.py window <startdate> <enddate> <lat1> <lat2> <lon1> <lon2>
    cpc_precip.py search <startdate> <enddate> <lat1> <lat2> <lon1> <lon2> <lower_bound>
        <upper_bound>

Options:
  -h --help     Show this screen.
"""
import code
import dataclasses
import docopt
import os
import sys
import typing

from dateutil.parser import parse as parse_date
import flask
import numpy
import xarray

import dataset
import dcdf

Path = str
HEAD = ".cpc_precip_head"
CHUNK_SIZE = 37  # 10 superchunks/year


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

        elif args["query"]:
            lat = float(args["<latitude>"])
            lon = float(args["<longitude>"])

            if args["<datetime>"]:
                date = parse_date(args["<datetime>"])
                return cli_get(data, date, lat, lon)

            else:
                start = parse_date(args["<startdate>"])
                end = parse_date(args["<enddate>"])
                return cli_cell(data, start, end, lat, lon)

        elif args["window"]:
            start = parse_date(args["<startdate>"])
            end = parse_date(args["<enddate>"])
            lat1 = float(args["<lat1>"])
            lat2 = float(args["<lat2>"])
            lon1 = float(args["<lon1>"])
            lon2 = float(args["<lon2>"])
            return cli_window(data, start, end, lat1, lat2, lon1, lon2)

        elif args["search"]:
            start = parse_date(args["<startdate>"])
            end = parse_date(args["<enddate>"])
            lat1 = float(args["<lat1>"])
            lat2 = float(args["<lat2>"])
            lon1 = float(args["<lon1>"])
            lon2 = float(args["<lon2>"])
            lower = float(args["<lower_bound>"])
            upper = float(args["<upper_bound>"])

            return cli_search(data, start, end, lat1, lat2, lon1, lon2, lower, upper)

        elif args["shell"]:
            banner = (
                "Welcome to the CPC Precipitation Dataset interactive shell.\n"
                "The dataset lives in the 'dataset' variable.\n"
                "Type 'help(dataset)' for more information."
            )
            local = {
                "dataset": data,
                "np": numpy,
            }

            code.interact(banner, local=local)
            return cli_ok()

        elif args["serve"]:
            cli_serve(data)

        return cli_error("not implemented")

    except DataError as error:
        return cli_error(error)


def cli_init(resolver):
    if os.path.exists(HEAD):
        return cli_error(f"Dataset already exists. Head is written to {HEAD}")

    message = "Initial commit."
    cli_new_head(resolver.commit(message, None, None))

    return cli_ok(f"New HEAD written to {HEAD}")


def cli_add(dataset, input_file):
    print("Loading...")
    xdata = xarray.open_dataset(input_file)
    dataset.geo.verify(xdata)
    dataset.layout.verify(xdata)
    shape = xdata.precip.data.shape[1:]
    if shape != dataset.shape:
        raise DataError(f"Expected shape: {dataset.shape}, got: {shape}")

    for start_day in range(0, 366, CHUNK_SIZE):
        time = xdata.time.data[start_day]
        data = xdata.precip.data[start_day : start_day + CHUNK_SIZE]

        print("Computing encoding requirements...")
        max_value = numpy.nanmax(data)
        suggestion = dcdf.suggest_fraction(data, max_value)

        if suggestion.round:
            print(
                f"Data must be rounded to use {suggestion.fractional_bits} bit fractions "
                "in order to be able to be encoded."
            )

        else:
            print(
                "Data can be encoded without loss of precision using "
                f"{suggestion.fractional_bits} bit fractions."
            )

        # Immutable data means "modifying" the dataset creates a new one
        print("Building...")
        dataset = dataset.add_chunk(
            time, data, suggestion.fractional_bits, suggestion.round
        )

    cli_new_head(dataset.cid)

    return cli_ok(f"New head written to {HEAD}")


def cli_new_head(cid):
    with open(HEAD, "w") as f:
        print(cid, file=f)


def cli_get(data, date, lat, lon):
    (date, lat, lon), value = data.get(numpy.datetime64(date), lat, lon)
    message = (
        f"date: {date}\n" f"lat: {lat}\n" f"lon: {lon}\n" f"Precipitation: {value}\n"
    )
    return cli_ok(message)


def cli_cell(data, start, end, lat, lon):
    start = numpy.datetime64(start)
    end = numpy.datetime64(end)
    (start, lat, lon), values = data.cell(start, end, lat, lon)

    print(f"lat: {lat}")
    print(f"lon: {lon}")
    for i, value in enumerate(values):
        time = start + i * data.layout.step
        print(f"{time}: {value}")

    return cli_ok("")


def cli_window(data, start, end, lat1, lat2, lon1, lon2):
    start = numpy.datetime64(start)
    end = numpy.datetime64(end)
    (start, _, _), window = data.window(start, end, lat1, lat2, lon1, lon2)

    for i, page in enumerate(window):
        date = start + i * data.layout.step
        print(date)
        for row in page:
            print(",".join(map(str, row)))
        print("")

    return cli_ok("Done")


def cli_search(data, start, end, lat1, lat2, lon1, lon2, lower, upper):
    start = numpy.datetime64(start)
    end = numpy.datetime64(end)
    for time, lat, lon in data.search(start, end, lat1, lat2, lon1, lon2, lower, upper):
        print(f"{time} {lat} {lon}")

    return cli_ok("Done")


def cli_serve(data):
    """Mimic the current dClimate API for this one query."""
    app = flask.Flask("CPC Precipitation")

    @app.route("/", methods=["POST"])
    def handle():
        request = flask.request.get_json()
        point = request["point_params"]
        lat = float(point["lat"])
        lon = float(point["lon"])
        start, end = request["time_range"]
        start = numpy.datetime64(parse_date(start))
        end = numpy.datetime64(parse_date(end))
        if start > end:
            start, end = end, start

        (start, _, _), values = data.cell(start, end, lat, lon)
        times = [
            numpy.datetime_as_string(start + i * data.layout.step, unit="s")
            for i in range(len(values))
        ]

        return {
            "data": list(map(float, values)),
            "dimensions_order": ["time"],
            "times": times,
            "unit of measurement": "mm",
        }

    app.run(host="0.0.0.0", port=8000)


def cli_ok(message):
    print(message)
    return 0


def cli_error(message):
    print(message, file=sys.stderr)
    return 1


class CpcPrecipDataset(dataset.Dataset):
    LEVELS = 4
    """2^2^4 = 256 subchunks"""

    LOCAL_THRESHOLD = 0  # 1<<12   # 4096
    """Subchunks smaller than this number of bytes will be stored on the superchunk
    rather than as separate objects."""

    def __init__(self, cid, resolver):
        super().__init__(
            cid=cid,
            resolver=resolver,
            layout=DailyLayout(CHUNK_SIZE),
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
    century containing folders for each year with each year containing superchunks that
    each contain a part of a year. Each time instant is a day.
    """

    step = numpy.timedelta64(1, "D")

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

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
        year, chunk, index, instant = self._locate(instant)
        return self._path_for(year, chunk), index, instant

    def _locate(
        self, instant: numpy.datetime64
    ) -> tuple[str, int, int, numpy.datetime64]:
        """For given datetime, return year, chunk, index"""
        # It's confounding that you can't access date components directly in numpy
        # datetimes. Why?!
        year, month, day = str(instant.astype("datetime64[D]")).split("-")

        # Number of full days since Jan 1
        days = (instant - numpy.datetime64(year)) / self.step
        days = int(days)

        # Find superchunk for day
        chunk = days // self.chunk_size
        first_day = chunk * self.chunk_size
        index = days - first_day

        # Instant is rounded down to day
        instant = numpy.datetime64(f"{year}-{month}-{day}")

        return year, chunk, index, instant

    def _path_for(self, year, chunk) -> str:
        # Top folder is first two digits of year, then next level is the year, and the
        # third level is the 0 indexed chunk within that year
        return f"{year[:2]}/{year}/{chunk}"

    def locate_span(
        self, start: numpy.datetime64, end: numpy.datetime64
    ) -> typing.Iterator[tuple[Path, slice]]:
        year, chunk, index, _ = self._locate(start)
        while True:
            next_year, next_chunk, instant = self._locate_next(year, chunk)
            if instant >= end:
                # Last span
                days = (end - numpy.datetime64(year)) // self.step
                first_day = chunk * self.chunk_size
                last_index = days - first_day

                yield self._path_for(year, chunk), slice(index, last_index + 1)
                break

            else:
                yield self._path_for(year, chunk), slice(index, None)
                index = None
                year = next_year
                chunk = next_chunk

    def _locate_next(self, year, chunk) -> tuple[str, int, numpy.datetime64]:
        # Increment chunk
        chunk += 1

        # See if incrementing chunk took us to the next year
        instant = numpy.datetime64(year) + self.step * self.chunk_size * chunk
        next_year = str(instant.astype("datetime64[Y]"))
        if next_year != year:
            # Roll over to the next year
            return next_year, 0, numpy.datetime64(next_year)

        return year, chunk, instant

    def time_at(self, path: str, index: int) -> numpy.datetime64:
        _, year, chunk = path.split("/")
        instant = numpy.datetime64(year)
        instant += (int(chunk) * self.chunk_size + index) * self.step

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

    def locate(
        self, lat: float, lon: float
    ) -> tuple[tuple[int, int], tuple[float, float]]:
        row = abs(self.lat - lat).argmin()
        col = abs(self.lon - lon).argmin()

        return (row, col), (self.lat[row], self.lon[col])

    def coord_at(self, row: int, col: int) -> tuple[float, float]:
        return self.lat[row], self.lon[col]


class DataError(Exception):
    """Data does not have expected structure."""


if __name__ == "__main__":
    sys.exit(cli_main())
