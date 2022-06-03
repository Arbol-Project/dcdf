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

What is here currently is just a start. The work in "Low Level Encoding" from the [Road
Map](ROADMAP.md) is largely finished. There is still testing, debugging, documentation,
and possible refactoring to do, but largely this part works.

What this doesn't do yet is any kind of metadata or mapping from logical coordinates to
real coordinates. So, for instance, you'll access a datapoint with the indices of its
time instant, row, and column. There is no facility for mapping these indices to
datetime values, or latitude/longitude coordinates, so that will have to be handled by
the calling code.

This library, currently, can only produce standalone files. Sharding of datasets into
smaller chunks and use of IPLD for storage and retrieval are still in the theoretical
future.

## Quick Start

For development, and until a binary wheel is released, a [Rust
toolchain](https://www.rust-lang.org/tools/install) is required to build and use this
library. Install and activate a Python virtual environment using your preferred method.

Then:

    $ pip install -U pip setuptools
    $ pip install -e py-dcdf[dev,examples]

From there, you should be able to run anything in the "examples" directory. There is an
example, currently, of converting a netcdf file from the cpc daily global precipitation
dataset to a dcdf file that can be loaded with the "load.py" example. The netcdf files
from this dataset contain a year's worth of data.

When this package sees a release, then binary wheels will be available containing
compiled Rust code, so a Rust toolchain will not be needed to use released versions of
the library. 
