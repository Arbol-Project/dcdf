# dClimate Data Format

Software for encoding, publishing, and reading time series raster climate data.

The underlying basis for this work is an implementation of the Heuristic K^2
Raster algorithm as outlined in the paper, ["Space-efficient representations of
raster time series" by Silva Coira, Paramá, de Bernardo, and
Seco](https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf).

This work is unstable and is suitable for experimental use only. **Do not rely
on the data format staying stable.** In the current stage of development,
breaking changes can be made at any time.

Data is stored in a binary format which can be intelligently broken up into hash
addressable chunks for storage in any IPLD based datastore, such as IPFS. The
chunked tree structure allows queries to be performed without having to retrieve
the entire dataset for the time period in question, and the use of compact data
structures allows space efficient files to be queried in place without a
decompression step.

For the time being, the only realy documentation is in the sample code at
`examples/example.py`.

## Quick Start

For development, and until a binary wheel is released, a [Rust
toolchain](https://www.rust-lang.org/tools/install) is required to build and use
this library. Install and activate a Python virtual environment using your
preferred method. Testing has been done using Python 3.9. Other versions may or
may not work at this time.

Then:

    $ pip install -U pip setuptools
    $ pip install "ipldstore @ git+https://github.com/dClimate/ipldstore@v1.0.0"
    $ pip install -e py-dcdf[dev,examples]

From there, you should be able to run anything in the "examples" directory.

When this package sees a release, binary wheels will be available containing
compiled Rust code, so a Rust toolchain will not be needed to use released
versions of the library.
