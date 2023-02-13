use std::{
    cmp::min,
    collections::VecDeque,
    fmt::Debug,
    io::{Read, Write},
    marker::PhantomData,
};

use async_trait::async_trait;
use futures::io::{AsyncRead, AsyncWrite};
use num_traits::PrimInt;

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{
        ExtendedAsyncRead, ExtendedAsyncWrite, ExtendedRead, ExtendedWrite, Serialize,
        SerializeAsync,
    },
    geom,
};

use super::bitmap::{BitMap, BitMapBuilder};
use super::dac::Dac;

/// K²-Raster encoded Snapshot
///
/// A Snapshot stores raster data for a particular time instant in a raster time series. Data is
/// stored standalone without reference to any other time instant.
///
pub struct Snapshot<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    _marker: PhantomData<I>,

    /// Bitmap of tree structure, known as T in Silva-Coira
    pub nodemap: BitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira
    pub max: Dac,

    /// Tree node minimum values, known as Lmin in Silva-Coira
    pub min: Dac,

    /// The K in K²-Raster. Each level of the tree structure is divided into k² subtrees.
    /// In practice, this will almost always be 2.
    k: i32,

    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    pub shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl<I> Serialize for Snapshot<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Write a snapshot to a stream
    ///
    fn write_to(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_byte(self.k as u8)?;
        stream.write_u32(self.shape[0] as u32)?;
        stream.write_u32(self.shape[1] as u32)?;
        stream.write_u32(self.sidelen as u32)?;
        self.nodemap.write_to(stream)?;
        self.max.write_to(stream)?;
        self.min.write_to(stream)?;

        Ok(())
    }

    /// Read a snapshot from a stream
    ///
    fn read_from(stream: &mut impl Read) -> Result<Self> {
        let k = stream.read_byte()? as i32;
        let shape = [stream.read_u32()? as usize, stream.read_u32()? as usize];
        let sidelen = stream.read_u32()? as usize;
        let nodemap = BitMap::read_from(stream)?;
        let max = Dac::read_from(stream)?;
        let min = Dac::read_from(stream)?;

        Ok(Self {
            _marker: PhantomData,
            nodemap,
            max,
            min,
            k,
            shape,
            sidelen,
        })
    }
}

#[async_trait]
impl<I> SerializeAsync for Snapshot<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Write a snapshot to a stream
    ///
    async fn write_to_async(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_byte_async(self.k as u8).await?;
        stream.write_u32_async(self.shape[0] as u32).await?;
        stream.write_u32_async(self.shape[1] as u32).await?;
        stream.write_u32_async(self.sidelen as u32).await?;
        self.nodemap.write_to_async(stream).await?;
        self.max.write_to_async(stream).await?;
        self.min.write_to_async(stream).await?;

        Ok(())
    }

    /// Read a snapshot from a stream
    ///
    async fn read_from_async(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let k = stream.read_byte_async().await? as i32;
        let shape = [
            stream.read_u32_async().await? as usize,
            stream.read_u32_async().await? as usize,
        ];
        let sidelen = stream.read_u32_async().await? as usize;
        let nodemap = BitMap::read_from_async(stream).await?;
        let max = Dac::read_from_async(stream).await?;
        let min = Dac::read_from_async(stream).await?;

        Ok(Self {
            _marker: PhantomData,
            nodemap,
            max,
            min,
            k,
            shape,
            sidelen,
        })
    }
}

impl<I> Cacheable for Snapshot<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Return number of bytes in serialized representation
    ///
    fn size(&self) -> u64 {
        1       // k
        + 4 + 4 // shape
        + 4     // sidelen 
        + self.nodemap.size() + self.max.size() + self.min.size()
    }
}

impl<I> Snapshot<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Build a snapshot from a two-dimensional array.
    ///
    /// The notional two-dimensional array is represented by `get`, which is a function that takes
    /// a row and column as arguments and returns an i64. The dimensions of the two-dimensional
    /// array are given by `shape`. `k` is the K from K²-Raster. The recommended value is 2. See
    /// the literature.
    ///
    /// The `get` indirection is used to allow layers further up to inject translation from an
    /// array of arbitrary numeric type to i64 values to store in the snaphsot. Floating point
    /// arrays will use this to convert cell values from floating point to a fixed point
    /// representation.
    ///
    pub fn build<G>(get: G, shape: [usize; 2], k: i32) -> Self
    where
        G: Fn(usize, usize) -> i64,
    {
        let mut nodemap = BitMapBuilder::new();
        let mut max: Vec<i64> = vec![];
        let mut min: Vec<i64> = vec![];

        // Compute the smallest square with sides whose length is a power of K that will contain
        // the passed in data.
        let sidelen = *shape.iter().max().unwrap() as f64;
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let root = K2TreeNode::build(get, shape, k, sidelen);
        let mut to_traverse = VecDeque::new();
        to_traverse.push_back((root.max.unwrap_or(0), root.min.unwrap_or(0), &root));

        // Breadth first traversal
        while let Some((diff_max, diff_min, child)) = to_traverse.pop_front() {
            let child_max = child.max.unwrap_or(0);
            let child_min = child.min.unwrap_or(0);
            max.push(diff_max);

            if !child.children.is_empty() {
                // Non-leaf node
                let elide = child_min == child_max;
                nodemap.push(!elide);
                if !elide {
                    min.push(diff_min);
                    for descendant in &child.children {
                        to_traverse.push_back((
                            child_max - descendant.max.unwrap_or(0),
                            descendant.min.unwrap_or(0) - child_min,
                            &descendant,
                        ));
                    }
                }
            }
        }

        Snapshot {
            _marker: PhantomData,
            nodemap: nodemap.finish(),
            max: Dac::from(max),
            min: Dac::from(min),
            k,
            shape: [shape[0], shape[1]],
            sidelen,
        }
    }

    /// Get a cell value.
    ///
    /// See: Algorithm 2 in Ladra[^note]
    ///
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn get(&self, row: usize, col: usize) -> I {
        if !self.nodemap.get(0) {
            // Special case, single node tree
            return self.max.get(0);
        } else {
            self._get(self.sidelen, row, col, 0, self.max.get(0))
        }
    }

    fn _get(&self, sidelen: usize, row: usize, col: usize, index: usize, max_value: I) -> I {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let index = 1 + self.nodemap.rank(index) * k * k;
        let index = index + row / sidelen * k + col / sidelen;
        let max_value = max_value - self.max.get(index);

        if index >= self.nodemap.length || !self.nodemap.get(index) {
            // Leaf node
            max_value
        } else {
            // Branch
            self._get(sidelen, row % sidelen, col % sidelen, index, max_value)
        }
    }

    /// Get a subarray of Snapshot
    ///
    /// This is based on Algorithm 3 in Ladra[^note], but has been modified to return a submatrix
    /// rather than an unordered sequence of values.
    ///
    /// The passed in `set` function  inserts into a notional two dimensional array that has been
    /// preallocated with the correct dimensions. This indirection allows higher layers to
    /// preallocate a 3 dimensional array at the beginning of a time series query, and provides a
    /// means of injecting data conversion from the underlying stored data to the desired output
    /// numeric type.
    ///
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn fill_window<S>(&self, mut set: S, bounds: &geom::Rect)
    where
        S: FnMut(usize, usize, i64),
    {
        let rows = bounds.rows();
        let cols = bounds.cols();

        if !self.nodemap.get(0) {
            // Special case: single node tree
            let value = self.max.get(0);
            for row in 0..rows {
                for col in 0..cols {
                    set(row, col, value);
                }
            }
        } else {
            self._fill_window(
                &mut set,
                self.sidelen,
                bounds.top,
                bounds.bottom - 1,
                bounds.left,
                bounds.right - 1,
                0,
                self.max.get(0),
                bounds.top,
                bounds.left,
                0,
                0,
            );
        }
    }

    fn _fill_window<S>(
        &self,
        set: &mut S,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index: usize,
        max_value: i64,
        window_top: usize,
        window_left: usize,
        top_offset: usize,
        left_offset: usize,
    ) where
        S: FnMut(usize, usize, i64),
    {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let index = 1 + self.nodemap.rank(index) * k * k;

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_ = index + i * k + j;
                let max_value_ = max_value - self.max.get::<i64>(index_);

                if index_ >= self.nodemap.length || !self.nodemap.get(index_) {
                    // Leaf node
                    for row in top_..=bottom_ {
                        for col in left_..=right_ {
                            set(
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                                max_value_,
                            );
                        }
                    }
                } else {
                    // Branch
                    self._fill_window(
                        set,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_,
                        max_value_,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                    );
                }
            }
        }
    }

    /// Search the window for cells with values in a given range
    ///
    /// See: Algorithm 4 in Ladra[^note]
    ///
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn search_window(&self, bounds: &geom::Rect, lower: I, upper: I) -> Vec<(usize, usize)> {
        let mut cells: Vec<(usize, usize)> = vec![];

        if !self.nodemap.get(0) {
            // Special case: single node tree
            let value: I = self.max.get(0);
            if lower <= value && value <= upper {
                for (row, col) in bounds.iter() {
                    cells.push((row, col));
                }
            }
        } else {
            self._search_window(
                self.sidelen,
                bounds.top,
                bounds.bottom - 1,
                bounds.left,
                bounds.right - 1,
                lower,
                upper,
                0,
                self.min.get(0),
                self.max.get(0),
                &mut cells,
                0,
                0,
            );
        }

        cells
    }

    fn _search_window(
        &self,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: I,
        upper: I,
        index: usize,
        min_value: I,
        max_value: I,
        cells: &mut Vec<(usize, usize)>,
        top_offset: usize,
        left_offset: usize,
    ) {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let index = 1 + self.nodemap.rank(index) * k * k;

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_ = index + i * k + j;
                let max_value_ = max_value - self.max.get(index_);

                if index_ >= self.nodemap.length || !self.nodemap.get(index_) {
                    // Leaf node
                    if lower <= max_value_ && max_value_ <= upper {
                        for row in top_..=bottom_ {
                            for col in left_..=right_ {
                                cells.push((top_offset_ + row, left_offset_ + col));
                            }
                        }
                    }
                } else {
                    // Branch
                    let min_value_ = min_value + self.min.get(self.nodemap.rank(index_));
                    if lower <= min_value && max_value_ <= upper {
                        // All values in branch are within bounds
                        for row in top_..=bottom_ {
                            for col in left_..=right_ {
                                cells.push((top_offset_ + row, left_offset_ + col));
                            }
                        }
                    } else if upper >= min_value_ && lower <= max_value_ {
                        // Some, but not all, values in branch are within bounds.
                        // Recurse into branch
                        self._search_window(
                            sidelen,
                            top_,
                            bottom_,
                            left_,
                            right_,
                            lower,
                            upper,
                            index_,
                            min_value_,
                            max_value_,
                            cells,
                            top_offset_,
                            left_offset_,
                        );
                    }
                }
            }
        }
    }
}

/// Temporary tree structure for building K^2 raster
struct K2TreeNode {
    max: Option<i64>,
    min: Option<i64>,
    children: Vec<K2TreeNode>,
}

impl K2TreeNode {
    fn build<G>(get: G, shape: [usize; 2], k: i32, sidelen: usize) -> Self
    where
        G: Fn(usize, usize) -> i64,
    {
        Self::_build(&get, shape, k as usize, sidelen, 0, 0)
    }

    fn _build<G>(
        get: &G,
        shape: [usize; 2],
        k: usize,
        sidelen: usize,
        row: usize,
        col: usize,
    ) -> Self
    where
        G: Fn(usize, usize) -> i64,
    {
        // Leaf node
        if sidelen == 1 {
            let [rows, cols] = shape;
            let value = if row < rows && col < cols {
                Some(get(row, col))
            } else {
                None
            };
            return K2TreeNode {
                max: value,
                min: value,
                children: vec![],
            };
        }

        // Branch
        let mut children: Vec<K2TreeNode> = vec![];
        let sidelen = sidelen / k;
        for i in 0..k {
            let row_ = row + i * sidelen;
            for j in 0..k {
                let col_ = col + j * sidelen;
                children.push(K2TreeNode::_build(get, shape, k, sidelen, row_, col_));
            }
        }

        let mut max = children[0].max;
        let mut min = children[0].min;
        for child in &children[1..] {
            if let Some(child_max) = child.max {
                if let Some(node_max) = max {
                    if child_max > node_max {
                        max = Some(child_max);
                    }
                } else {
                    max = Some(child_max);
                }
            }
            if let Some(child_min) = child.min {
                if let Some(node_min) = min {
                    if child_min < node_min {
                        min = Some(child_min);
                    }
                } else {
                    min = Some(child_min);
                }
            }
        }

        K2TreeNode { min, max, children }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::array_search_window;
    use super::*;
    use futures::io::Cursor as AsyncCursor;
    use ndarray::{arr2, s, Array2};
    use std::collections::HashSet;
    use std::io::Cursor;

    fn array8() -> Array2<i32> {
        arr2(&[
            [9, 8, 7, 7, 6, 6, 3, 2],
            [7, 7, 7, 7, 6, 6, 3, 3],
            [6, 6, 6, 6, 3, 3, 3, 3],
            [5, 5, 6, 6, 3, 3, 3, 3],
            [4, 5, 5, 5, 4, 4, 4, 4],
            [3, 3, 5, 5, 4, 4, 4, 4],
            [3, 3, 3, 5, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4],
        ])
    }

    fn array9() -> Array2<i32> {
        arr2(&[
            [9, 8, 7, 7, 6, 6, 3, 2, 1],
            [7, 7, 7, 7, 6, 6, 3, 3, 3],
            [6, 6, 6, 6, 3, 3, 3, 3, 3],
            [5, 5, 6, 6, 3, 3, 3, 3, 2],
            [4, 5, 5, 5, 4, 4, 4, 4, 4],
            [3, 3, 5, 5, 4, 4, 4, 4, 4],
            [3, 3, 3, 5, 4, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4, 4],
        ])
    }

    #[test]
    fn build() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        assert_eq!(snapshot.nodemap.length, 17);
        assert_eq!(
            snapshot.nodemap.bitmap,
            vec![0b11110101001001011000000000000000]
        );
        assert_eq!(
            snapshot.max.collect::<i32>(),
            vec![
                9, 0, 3, 4, 5, 0, 2, 3, 3, 0, 3, 3, 3, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 1, 1, 0, 1, 0,
                0, 1, 0, 2, 2, 1, 1, 0, 0, 2, 0, 2, 1,
            ]
        );
        assert_eq!(
            snapshot.min.collect::<i32>(),
            vec![2, 3, 0, 1, 2, 0, 0, 0, 0, 0,]
        );

        assert_eq!(snapshot.shape, [8, 8]);
    }

    #[test]
    fn build_make_sure_fill_values_match_local_nonfill_values_in_same_quadbox() {
        let mut data: Array2<i32> = Array2::zeros([9, 9]) + 5;
        let mut slice = data.slice_mut(s![..8, ..8]);
        slice.assign(&array8());
        let snapshot = Snapshot::from_array(data.view(), 2);

        // Compared to the previous test with an 8x8 array, expecting only 4 new nodes because fill
        // values for expanded 16x16 array will all be 5 because quadboxes except for upper left
        // all contain only 5s or the fill value.
        assert_eq!(snapshot.nodemap.length, 21);
        assert_eq!(snapshot.get(8, 8), 5);
    }

    #[test]
    fn get() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }
    }

    #[test]
    fn get_single_node_tree() {
        let data: Array2<i32> = Array2::zeros([16, 16]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);
        assert_eq!(snapshot.nodemap.bitmap.len(), 1);
        assert_eq!(snapshot.max.levels[0].1.len(), 1);
        assert!(snapshot.min.levels.is_empty());

        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(snapshot.get(row, col), 42);
            }
        }
    }

    #[test]
    fn get_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }
    }

    #[test]
    fn get_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 3);

        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }
    }

    #[test]
    fn get_window() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = snapshot.get_window(&bounds);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = snapshot.get_window(&bounds);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 3);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = snapshot.get_window(&bounds);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_tree() {
        let data: Array2<i32> = Array2::zeros([16, 16]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..16 {
            for bottom in top + 1..=16 {
                for left in 0..16 {
                    for right in left + 1..=16 {
                        let bounds = geom::Rect::new(top, bottom, left, right);
                        let window = snapshot.get_window(&bounds);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = geom::Rect::new(top, bottom, left, right);
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = snapshot.search_window(&window, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = geom::Rect::new(top, bottom, left, right);
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = snapshot.search_window(&window, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 3);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = geom::Rect::new(top, bottom, left, right);
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = snapshot.search_window(&window, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_tree_node_in_range() {
        let data: Array2<i32> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = geom::Rect::new(top, bottom, left, right);
                        let mut expected: HashSet<(usize, usize)> = HashSet::new();
                        for row in top..bottom {
                            for col in left..right {
                                expected.insert((row, col));
                            }
                        }
                        let coords = snapshot.search_window(&window, 41, 43);
                        let coords = HashSet::from_iter(coords.iter().cloned());

                        assert_eq!(coords, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_tree_node_out_of_range() {
        let data: Array2<i32> = Array2::zeros([16, 16]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..16 {
            for bottom in top + 1..=16 {
                for left in 0..16 {
                    for right in left + 1..=16 {
                        let window = geom::Rect::new(top, bottom, left, right);
                        let coords = snapshot.search_window(&window, 0, 41);

                        assert_eq!(coords, vec![]);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_no_values_in_range() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = geom::Rect::new(top, bottom, left, right);
                        let coords = snapshot.search_window(&window, 100, 200);
                        assert_eq!(coords.len(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn serialize_deserialize() -> Result<()> {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        let mut buffer: Vec<u8> = Vec::with_capacity(snapshot.size() as usize);
        snapshot.write_to(&mut buffer)?;
        assert_eq!(buffer.len(), snapshot.size() as usize);

        let mut buffer = Cursor::new(buffer);
        let snapshot: Snapshot<i32> = Snapshot::read_from(&mut buffer)?;

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn serialize_deserialize_async() -> Result<()> {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        let mut buffer: Vec<u8> = Vec::with_capacity(snapshot.size() as usize);
        snapshot.write_to_async(&mut buffer).await?;
        assert_eq!(buffer.len(), snapshot.size() as usize);

        let mut buffer = AsyncCursor::new(buffer);
        let snapshot: Snapshot<i32> = Snapshot::read_from_async(&mut buffer).await?;

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }

        Ok(())
    }
}
