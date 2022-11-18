# dClimate Data Format

Software for encoding, publishing, and reading time series raster climate data. 

The underlying basis for this work is an implementation of the Heuristic K^2 Raster
algorithm as outlined in the paper, ["Space-efficient representations of raster time
series" by Silva Coira, Paramá, de Bernardo, and Seco](https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf).

This work is unstable and is suitable for experimental use only. **Do not rely on the
data format staying stable.** In the current stage of development, breaking changes can
be made at any time.

Data is stored in a binary format which can be intelligently broken up into hash
addressable chunks for storage in any IPLD based datastore, such as IPFS. The chunked
tree structure allows queries to be performed without having to retrieve the entire
dataset for the time period in question, and the use of compact data structures allows
space efficient files to be queried in place without a decompression step. 

To get a good, quick idea of current capabilities, see `examples/dag/cpc_precip.py` and
`examples/dag/dataset.py` which is a dependency of the former. 

## Quick Start

For development, and until a binary wheel is released, a [Rust
toolchain](https://www.rust-lang.org/tools/install) is required to build and use this
library. Install and activate a Python virtual environment using your preferred method.

Then:

    $ pip install -U pip setuptools
    $ pip install -e py-dcdf[dev,examples]

From there, you should be able to run anything in the "examples" directory.

When this package sees a release, then binary wheels will be available containing
compiled Rust code, so a Rust toolchain will not be needed to use released versions of
the library. 
