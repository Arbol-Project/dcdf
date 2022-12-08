import code
import functools
import os
import sys

from dateutil.parser import parse as parse_date
import docopt
import flask
import numpy
import xarray

from dataset import DataError
import dcdf


class Cli:
    """
    Encode and retrieve {description} data.

    Usage:
        {script} init
        {script} shell
        {script} serve
        {script} add <input_file>
        {script} query <datetime> <latitude> <longitude>
        {script} query <startdate> <enddate> <latitude> <longitude>
        {script} window <startdate> <enddate> <lat1> <lat2> <lon1> <lon2>
        {script} search <startdate> <enddate> <lat1> <lat2> <lon1> <lon2>
            <lower_bound> <upper_bound>

    Options:
      -h --help     Show this screen.
    """

    def __init__(self, Dataset, name, description):
        self.Dataset = Dataset
        self.name = name
        self.description = description
        self.script = sys.argv[0]
        self.resolver = dcdf.new_ipfs_resolver()
        self.head = f".{name}_head"

    def main(self):
        try:
            doc = self.__doc__.format(**self.__dict__)
            args = docopt.docopt(doc)
            if args["init"]:
                return self.init()

            if args["add"]:
                return self.add(args["<input_file>"])

            elif args["query"]:
                lat = float(args["<latitude>"])
                lon = float(args["<longitude>"])

                if args["<datetime>"]:
                    date = parse_date(args["<datetime>"])
                    return self.get(date, lat, lon)

                else:
                    start = parse_date(args["<startdate>"])
                    end = parse_date(args["<enddate>"])
                    return self.cell(start, end, lat, lon)

            elif args["window"]:
                start = parse_date(args["<startdate>"])
                end = parse_date(args["<enddate>"])
                lat1 = float(args["<lat1>"])
                lat2 = float(args["<lat2>"])
                lon1 = float(args["<lon1>"])
                lon2 = float(args["<lon2>"])
                return self.window(start, end, lat1, lat2, lon1, lon2)

            elif args["search"]:
                start = parse_date(args["<startdate>"])
                end = parse_date(args["<enddate>"])
                lat1 = float(args["<lat1>"])
                lat2 = float(args["<lat2>"])
                lon1 = float(args["<lon1>"])
                lon2 = float(args["<lon2>"])
                lower = float(args["<lower_bound>"])
                upper = float(args["<upper_bound>"])

                return self.search(start, end, lat1, lat2, lon1, lon2, lower, upper)

            elif args["shell"]:
                banner = (
                    f"Welcome to the {self.description} interactive shell.\n"
                    "The dataset lives in the 'dataset' variable.\n"
                    "Type 'help(dataset)' for more information."
                )
                local = {
                    "dataset": self.data,
                    "np": numpy,
                }

                code.interact(banner, local=local)
                return self.ok()

            elif args["serve"]:
                self.serve()

            return self.error("not implemented")

        except DataError as error:
            return self.error(error)

    def init(self):
        if os.path.exists(self.head):
            return self.error(f"Dataset already exists. Head is written to {self.head}")

        message = "Initial commit."
        self.new_head(self.resolver.commit(message, None, None))

        return self.ok(f"New HEAD written to {self.head}")

    @property
    @functools.cache
    def data(self):
        if not os.path.exists(self.head):
            return self.error(
                f"Dataset must be initialized. Use `{sys.script} init` to create  a new "
                f"dataset, or put a CID in the file `{self.head}` in this directory, if the "
                f"dataset was initialized elsewhere."
            )

        cid = open(self.head).read().strip()
        return self.Dataset(cid, self.resolver)

    def add(self, input_file):
        print("Loading...")
        dataset = self.data
        xdata = xarray.open_dataset(input_file)
        dataset.geo.verify(xdata)
        dataset.layout.verify(xdata)
        shape = xdata.precip.data.shape[1:]
        if shape != dataset.shape:
            raise DataError(f"Expected shape: {dataset.shape}, got: {shape}")

        for time, data in dataset.layout.chunk(xdata):
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

        self.new_head(dataset.cid)

        return self.ok(f"New head written to {self.head}")

    def new_head(self, cid):
        with open(self.head, "w") as f:
            print(cid, file=f)

    def get(self, date, lat, lon):
        (date, lat, lon), value = self.data.get(numpy.datetime64(date), lat, lon)
        message = (
            f"date: {date}\n"
            f"lat: {lat}\n"
            f"lon: {lon}\n"
            f"Precipitation: {value}\n"
        )
        return self.ok(message)

    def cell(self, start, end, lat, lon):
        start = numpy.datetime64(start)
        end = numpy.datetime64(end)
        (start, lat, lon), values = self.data.cell(start, end, lat, lon)

        print(f"lat: {lat}")
        print(f"lon: {lon}")
        for i, value in enumerate(values):
            time = start + i * self.data.layout.step
            print(f"{time}: {value}")

        return self.ok()

    def window(self, start, end, lat1, lat2, lon1, lon2):
        start = numpy.datetime64(start)
        end = numpy.datetime64(end)
        (start, _, _), window = self.data.window(start, end, lat1, lat2, lon1, lon2)

        for i, page in enumerate(window):
            date = start + i * self.data.layout.step
            print(date)
            for row in page:
                print(",".join(map(str, row)))
            print("")

        return self.ok("Done")

    def search(self, start, end, lat1, lat2, lon1, lon2, lower, upper):
        start = numpy.datetime64(start)
        end = numpy.datetime64(end)
        for time, lat, lon in self.data.search(
            start, end, lat1, lat2, lon1, lon2, lower, upper
        ):
            print(f"{time} {lat} {lon}")

        return self.ok("Done")

    def serve(self):
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

            (start, _, _), values = self.data.cell(start, end, lat, lon)
            times = [
                numpy.datetime_as_string(start + i * self.data.layout.step, unit="s")
                for i in range(len(values))
            ]

            return {
                "data": list(map(float, values)),
                "dimensions_order": ["time"],
                "times": times,
                "unit of measurement": "mm",
            }

        app.run(host="0.0.0.0", port=8000)

    def ok(self, message=None):
        if message:
            print(message)
        return 0

    def error(self, message):
        print(message, file=sys.stderr)
        return 1
