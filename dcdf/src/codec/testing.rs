use ndarray::{Array2, ArrayView2};
use num_traits::{Num, PrimInt};
use std::fmt::Debug;

use super::*;
use crate::geom;

impl<I> Snapshot<I>
where
    I: PrimInt + Debug,
{
    /// Wrap Snapshot.build with function that creates the `get` closure so that it doesn't have to
    /// be repeated in every test of Snapshot.
    pub fn from_array(data: ArrayView2<I>, k: i32) -> Self {
        let get = |row, col| data[[row, col]].to_i64().unwrap();
        let shape = data.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get, [rows, cols], k)
    }

    /// Wrap Snapshot.get_window with function that allocates an array and creates the `set`
    /// enclosure, so it doesn't have to repeated in every test for `get_window`.
    ///
    pub fn get_window(&self, bounds: &geom::Rect) -> Array2<I> {
        let mut window = Array2::zeros([bounds.rows(), bounds.cols()]);
        let set = |row, col, value| window[[row, col]] = I::from(value).unwrap();

        self.fill_window(set, bounds);

        window
    }
}

impl<I> Log<I>
where
    I: PrimInt + Debug,
{
    /// Wrap Log.build with function that creates the `get_s` and `get_t` closures so that they
    /// don't  have to be repeated in every test of Log.
    pub fn from_arrays(snapshot: ArrayView2<I>, log: ArrayView2<I>, k: i32) -> Self {
        let get_s = |row, col| snapshot[[row, col]].to_i64().unwrap();
        let get_t = |row, col| log[[row, col]].to_i64().unwrap();
        let shape = snapshot.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get_s, get_t, [rows, cols], k)
    }

    /// Wrap Log.get_window with function that allocates an array and creates the `set`
    /// enclosure, so it doesn't have to repeated in every test for `get_window`.
    ///
    pub fn get_window(&self, snapshot: &Snapshot<I>, bounds: &geom::Rect) -> Array2<I> {
        let mut window = Array2::zeros([bounds.rows(), bounds.cols()]);
        let set = |row, col, value| window[[row, col]] = I::from(value).unwrap();

        self.fill_window(set, snapshot, bounds);

        window
    }
}

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub fn array_search_window<N>(
    data: ArrayView2<N>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    lower: N,
    upper: N,
) -> Vec<(usize, usize)>
where
    N: Num + Debug + Copy + PartialOrd,
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
