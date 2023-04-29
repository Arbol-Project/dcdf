mod bitmap;
mod block;
mod cache;
mod chunk;
mod dac;
mod dataset;
mod errors;
mod extio;
mod fixed;
mod helpers;
mod links;
mod log;
mod mapper;
mod mmarray;
mod mmbuffer;
mod mmstruct;
mod node;
mod range;
mod resolver;
mod snapshot;
mod span;
mod superchunk;
mod time;

#[cfg(test)]
mod testing;

// Public facing API
pub use dataset::{Coordinate, CoordinateKind, Dataset, Variable};
pub use errors::{Error, Result};
pub use fixed::{from_fixed, to_fixed};
pub use mapper::{Mapper, StoreWrite};
pub mod geom;
pub use mmarray::*;
pub use mmstruct::MMEncoding;
pub use resolver::Resolver;
pub use time::TimeRange;
