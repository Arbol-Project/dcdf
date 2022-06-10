use ndarray::ArrayView2;
use num_traits::{Num, PrimInt};
use std::fmt::Debug;

use super::*;

impl<I> Snapshot<I>
where
    I: PrimInt + Debug,
{
    /// Wrap Snapshot.build with good default implementation, so we don't have to create a closure
    /// for get in every test.
    pub fn from_array(data: ArrayView2<I>, k: i32) -> Self {
        let get = |row, col| data[[row, col]].to_i64().unwrap();
        let shape = data.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get, [rows, cols], k)
    }
}

impl<I> Log<I>
where
    I: PrimInt + Debug,
{
    /// Wrap Log.build with good default implementation, so we don't have to create closures
    /// for get_s and get_t in every test.
    pub fn from_arrays(snapshot: ArrayView2<I>, log: ArrayView2<I>, k: i32) -> Self {
        let get_s = |row, col| snapshot[[row, col]].to_i64().unwrap();
        let get_t = |row, col| log[[row, col]].to_i64().unwrap();
        let shape = snapshot.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get_s, get_t, [rows, cols], k)
    }
}

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub fn array_search_window<T>(
    data: ArrayView2<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    lower: T,
    upper: T,
) -> Vec<(usize, usize)>
where
    T: Num + Debug + Copy + PartialOrd,
{
    let mut coords: Vec<(usize, usize)> = vec![];
    for row in top..bottom {
        for col in left..right {
            let cell_value = data[[row, col]];
            if lower <= cell_value && cell_value <= upper {
                coords.push((row, col));
            }
        }
    }

    coords
}
