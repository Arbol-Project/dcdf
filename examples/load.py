"""
Load a DCDF data file

Usage:
  load.py <input_file>

Options:
  -h --help     Show this screen.
"""
import code
import dcdf

from docopt import docopt


def main():
    args = docopt(__doc__)

    print("Loading...")
    with open(args["<input_file>"], "rb") as f:
        data = dcdf.load(f)

    instants, rows, cols = data.shape

    banner = (
        "Welcome to the DCDF interactive console!\n\n"
        f"Time instants: {instants}  Rows: {rows}  Cols: {cols}\n\n"
        "Available queries:\n"
        "\tdata.cell(start, end, row, col)\n"
        "\tdata.window(start, end, top, bottom, left, right)\n"
        "\tdata.search(start, end, top, bottom, left, right, lower, upper)\n"
    )

    code.interact(banner, local={"data": data})


if __name__ == "__main__":
    main()
