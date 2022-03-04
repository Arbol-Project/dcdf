# dClimate Data Format Road Map

Development will proceed by way of layers of abstraction, with the lowest levels of
abstraction being built first, and subsequent layers being built atop earlier, lower
layers.

## Low Level Encoding

The low level API will be implemented in Rust with bindings so it can be called from
Python. This is where the Heuristic K^2 Raster algorithm will be implemented. Data will
be passed between the Python and Rust layers using NumPY arrays.

At this level, data is considered only as a numeric 3-dimensional array (or time-series
of 2-dimensional arrays). Mapping of specific real world geographical or time
coordinates to logical coordinates in the array is the responsibility of higher layers.

The general algorithm will be that a particular 3-dimensional array will be decomposed
along the Z-axis (representing time) into a series of 2-dimensional arrays, which we'll
call frames. For each frame a decision will be made to encode it and store it in one of
3 ways, depending on which is more compact:

- **snapshot (K^2 Raster)** A compact representation using K^2 raster that does not
  reference any other frame.

- **log (T - K^2 Raster)** A variation of the K^2 raster which encodes the difference
  between the current frame and the nearest preceding snapshot. (Obviously not an option
  for the first in a series.)

- **uncompressed array** For some datasets with low uniformity, K^2 raster may not do
  better than a plain, uncompressed array. In those cases, just use the uncompressed
  array. 

Each call to the Layer 1 algorithm will produce one binary file which stores the logical
3-dimensional array provided.

## High Level Structure

This may be implemented entirely in Python, or maybe some parts will still be
implemented in Rust.

The high level structure of data will be as follows:

- Stream

  A particular data source, such as CPC rainfall data. Logically, you can think of this
  a long series of 2-dimensional arrays, or a particularly long (along the Z/time axis)
  3-dimensional array.

  - Span

    The stream is divided into regular spans of time. The time interval is arbitrary and
    and could be days, weeks, months, semesters, etc...

    Each span is represented by a metadata file (yaml, json, ...?) which contains
    information about the data source, defines the units used, etc...

    Also contains pointers to:

    - Preceding span
    
    - Map 

      The map is a file which relates the logical coordinates used in the time series
      raster to real world space and time and coordinates. For instance, if there is a
      point in the time series raster with logical (integer) coordinates (x, y, z),
      this provides enough data to (longitude, latitude, date/time). 

      This mapping will need to be the same for all data in this span. If, in the data
      source, this mapping changes over time, then any time when it changes a new span
      will need to be started.

      Because this is stored in an IPLD based datastore, identical maps that are used
      across many spans will only be stored once.

    - Directory

      This is a file which will be encoded similarly to the K^2 Raster used in the low
      level format, but instead of encoding the entire space, it will just contain the
      top few levels of a tree structure whose branches will terminate with pointers
      (hashes) to files which encode their corresponding branch in the low level format.

      - Chunk 1

        File in low level format. Raw data will be meaningless with the higher level
        data to map logical space into real space-time coordinates.

      - Chunk 2 

      - ...

      - Chunk N

## Tree Structure

The fundamental way that the K^2 raster algorithm works is it takes a two dimensional
array and converts it into a tree representation, by dividing the raster into K^2
sub-rasters, and then dividing those rasters into K^2 sub-rasters, and so on to the
lowest level which contains only leaf nodes representing the raw grid cell values in the
raster. In practice, K is likely to almost always be 2, although we can certainly
experiment with other values. In this case, it means a raster is subdivided into 4
quadrants which, in turn, are subdivided into 4 more quadrants, and so on, recursively.

It is this tree structure that allows us to divide a span into some number of chunks to
be stored independently of each other.

## IPLD

The software can use adapters to use any IPLD-like datastore, such as IPFS. The basic
contract needed by the software is just:

    ~~~
    interface IPLD {
        method store_chunk(bytes) -> sha1 {
            // Store a byte array, returning the sha1 hash
        }
    
        method get_chunk(sha1) -> bytes {
            // Retrieve a byte array given the sha1 hash
        }
    }
    ~~~

For development we might use an implementation that just uses the local filesystem, and
then later use an implementation that stores to IPFS or any other compatible system that
might emerge.

## Data Prep

It seems highly likely that the data sources we'd consume for this use data types of a
much higher precision than their actual data. For instance, we might have an array of 64
bit floating point numbers when the data could be represented in 32 bit floats or 32,
24, or 16 bit integer or fixed pointed numbers. There's potentially some utility in
looking at incoming data and seeing if the raw numbers can be stored in a more compact
way, before encoding them using K^2 raster.
