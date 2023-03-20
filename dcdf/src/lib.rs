mod build;
mod cache;
mod codec;
mod dag;
mod errors;
mod extio;
mod fixed;
mod geom;
mod helpers;

pub use build::build_superchunk;
pub use build::SuperchunkBuild;

pub use cache::Cacheable;

pub use dag::mapper::Mapper;
pub use dag::mapper::StoreWrite;
pub use dag::mmarray::MMArray1;
pub use dag::mmarray::MMArray3;
pub use dag::range::Range;
pub use dag::resolver::LsEntry;
pub use dag::resolver::Resolver;
pub use dag::span::Span;
pub use dag::superchunk::Superchunk;
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
