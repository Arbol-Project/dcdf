# dClimate Data Format

Software for encoding, publishing, and reading time series raster climate data. 

The underlying basis for this work is an implementation of the Heuristic K^2 Raster
algorithm as outlined in the paper, ["Space-efficient representations of raster time
series" by Silva Coira, Paramá, de Bernardo, and Seco](https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf).

Data is stored in a binary format which can be intelligently broken up into hash
addressable chunks for storage in any IPLD based datastore, such as IPFS. The chunked
tree structure allows queries to be performed without having to retrieve the entire
dataset for the time period in question, and the use of compact data structures allows
space efficient files to be queried in place without a decompression step. 

This is very much a work in progress, and is still at the science fiction stage of
things. 

For more information on the planned implementation, see the [Road Map](ROADMAP.md).

## Quick Start for Development

Install and activate a Python virtual environment using your preferred method.

Then:

$ pip install -U pip setuptools maturin
$ maturin develop -E dev

Rerun that last command any time you need to recompile Rust code:


$ maturin develop -E dev

To run tests:

$ pytest tests
