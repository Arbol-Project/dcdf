import dataclasses
import sys
import typing

import numpy

import cli
import dataset

Path = str
CHUNK_SIZE = 37  # 10 superchunks/year
DATASET = "cpc_precip_global-daily"


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
            layout=DailyLayout(CHUNK_SIZE, "precip"),
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

    def __init__(self, chunk_size: int, variable: str):
        self.chunk_size = chunk_size
        self.variable = variable

    def verify(self, data):
        """Verify that times in xarray loaded data actually match our assumptions."""
        # Get just the time data
        data = data.time.data

        # Verify that the delta between each point is 1 day
        deltas = set(data[1:] - data[:-1])
        delta = next(iter(deltas))
        if len(deltas) != 1 or delta != self.step:
            raise dataset.DataError(
                "Expecting even 1 day spacing between time instants."
            )

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

    def chunk(
        self, xdata
    ) -> typing.Iterator[tuple[numpy.datetime64, numpy.typing.NDArray]]:
        data = getattr(xdata, self.variable).data
        for day in range(0, len(data), self.chunk_size):
            yield xdata.time.data[day], data[day : day + self.chunk_size]


@dataclasses.dataclass
class CpcGeoSpace:
    """GeoSpace with latitude and longitude spaced at regular intervals."""

    lat = numpy.linspace(89.75, -89.75, 360)
    lon = numpy.linspace(0.25, 359.75, 720)

    def verify(self, data):
        """Verify that lat/lon data in the source data matches our assumptions."""
        if hasattr(data, "latitude"):
            latitude = getattr(data, "latitude")
        else:
            latitude = getattr(data, "lat")
        if not numpy.array_equal(latitude.data, self.lat):
            raise dataset.DataError("Unexpected latitude data.")

        if hasattr(data, "longitude"):
            longitude = getattr(data, "longitude")
        else:
            longitude = getattr(data, "lon")
        if not numpy.array_equal(longitude.data, self.lon):
            raise dataset.DataError("Unexpected longitude data.")

    def locate(
        self, lat: float, lon: float
    ) -> tuple[tuple[int, int], tuple[float, float]]:
        row = abs(self.lat - lat).argmin()
        col = abs(self.lon - lon).argmin()

        return (row, col), (self.lat[row], self.lon[col])

    def coord_at(self, row: int, col: int) -> tuple[float, float]:
        return self.lat[row], self.lon[col]


if __name__ == "__main__":
    cli = cli.Cli(CpcPrecipDataset, DATASET, "CPC Daily Global Precipitation")
    sys.exit(cli.main())
