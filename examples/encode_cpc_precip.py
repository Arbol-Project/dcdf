"""
Encode CPC Daily precipitation data.

Usage:
  encode_cpc_precip.py <input_file> <output_file>

Options:
  -h --help     Show this screen.
"""
import dcdf
import numpy
import xarray

from docopt import docopt


def main():
    args = docopt(__doc__)

    print("Loading...")
    data = xarray.open_dataset(args["<input_file>"])
    data = data.precip.data

    print("Computing encoding requirements...")
    max_value = numpy.nanmax(data)
    suggestion = dcdf.suggest_fraction(data, max_value)

    if suggestion.round:
        print(f"Data must be rounded to use {suggestion.fractional_bits} bit fractions "
              "in order to be able to be encoded.")

    else:
        print("Data can be encoded without loss of precision, using "
              f"{suggestion.fractional_bits} bit fractions.")

    print("Building...")
    built = dcdf.build(data, fraction=suggestion.fractional_bits, round=suggestion.round)

    print("Built:")
    print(f"\tCompression: {built.compression * 100:0.2f}%")
    print(f"\tSnapshots: {built.snapshots}")
    print(f"\tLogs: {built.logs}")

    print("Saving...")
    built.save(args["<output_file>"])


if __name__ == "__main__":
    main()
