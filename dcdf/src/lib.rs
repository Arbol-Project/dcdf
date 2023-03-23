mod cache;

#[allow(dead_code)]
mod codec;

#[allow(dead_code)]
mod dag;

mod errors;
mod extio;
mod fixed;
mod geom;
mod helpers;

pub use cache::Cacheable;

pub use dag::mapper::Mapper;
pub use dag::mapper::StoreWrite;
pub use dag::mmarray::MMArray1;
pub use dag::range::Range;
pub use dag::resolver::LsEntry;
pub use dag::resolver::Resolver;
pub use dag::time::TimeRange;

pub use errors::Error;
pub use errors::Result;

pub use fixed::suggest_fraction;
pub use fixed::Continue;
pub use fixed::Fraction;
pub use fixed::Fraction::{Precise, Round};
pub use fixed::FractionSuggester;

pub use geom::Cube;
pub use geom::Rect;

#[cfg(test)]
mod testing;
