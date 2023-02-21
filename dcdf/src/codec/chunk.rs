use std::{
    cell::RefCell, fmt::Debug, marker::PhantomData, rc::Rc, sync::Arc, vec::IntoIter as VecIntoIter,
};

use async_trait::async_trait;
use futures::io::{AsyncRead, AsyncWrite};
use ndarray::{s, Array2, ArrayBase, DataMut, Ix1, Ix3};
use num_traits::{Float, PrimInt};

use crate::{
    cache::Cacheable,
    dag::resolver::Resolver,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    fixed, geom,
    helpers::rearrange,
};

use super::block::Block;

/// A wrapper for Chunk that adapts it to use floating point data
///
/// Floating point numbers are stored in a fixed point representation and converted back to
/// floating point using `fractional_bits` to determine the number of bits that are used to
/// represent the fractional part of the number.
///
pub struct FChunk<F>
where
    F: Float + Debug + Send + Sync,
{
    _marker: PhantomData<F>,

    /// Number of bits in the fixed point number representation that represent the fractional part
    fractional_bits: usize,

    /// Wrapped Chunk
    chunk: Arc<Chunk<i64>>,
}

impl<F> FChunk<F>
where
    F: Float + Debug + Send + Sync,
{
    /// Create an FChunk that wraps `chunk` using `fractional_bits` for fixed point to floating
    /// point conversion.
    ///
    pub(crate) fn new(chunk: Chunk<i64>, fractional_bits: usize) -> Self {
        Self {
            _marker: PhantomData,
            fractional_bits: fractional_bits,
            chunk: Arc::new(chunk),
        }
    }

    /// Return the dimensions of the 3 dimensional array represented by this chunk.
    ///
    /// Returns [instants, rows, cols]
    ///
    pub(crate) fn shape(&self) -> [usize; 3] {
        self.chunk.shape()
    }

    /// Get a single data point
    ///
    pub(crate) fn get(&self, instant: usize, row: usize, col: usize) -> F {
        fixed::from_fixed(self.chunk.get(instant, row, col), self.fractional_bits)
    }

    /// Fill in a preallocated array with cell's value across time instants.
    ///
    pub(crate) fn fill_cell<S>(
        &self,
        start: usize,
        row: usize,
        col: usize,
        values: &mut ArrayBase<S, Ix1>,
    ) where
        S: DataMut<Elem = F>,
    {
        self.chunk.check_bounds(start + values.len() - 1, row, col);
        for (i, (block, instant)) in self.chunk.iter(start, start + values.len()).enumerate() {
            let block = &self.chunk.blocks[block];
            let value = block.get(instant, row, col);
            values[i] = fixed::from_fixed(value, self.fractional_bits);
        }
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    pub(crate) fn fill_window<S>(
        &self,
        start: usize,
        top: usize,
        left: usize,
        window: &mut ArrayBase<S, Ix3>,
    ) where
        S: DataMut<Elem = F>,
    {
        let shape = window.shape();
        let end = start + shape[0];
        let bottom = top + shape[1];
        let right = left + shape[2];
        self.chunk.check_bounds(end - 1, bottom - 1, right - 1);

        for (i, (block, instant)) in self.chunk.iter(start, end).enumerate() {
            let mut window2d = window.slice_mut(s![i, .., ..]);
            let set = |row, col, value| {
                window2d[[row, col]] = fixed::from_fixed(value, self.fractional_bits)
            };
            let block = &self.chunk.blocks[block];
            block.fill_window(set, instant, &geom::Rect::new(top, bottom, left, right));
        }
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub(crate) fn iter_search<'a>(
        &'a self,
        bounds: &geom::Cube,
        lower: F,
        upper: F,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a {
        self.chunk.iter_search(
            bounds,
            fixed::to_fixed(lower, self.fractional_bits, true),
            fixed::to_fixed(upper, self.fractional_bits, true),
        )
    }
}

#[async_trait]
impl<F> Serialize for FChunk<F>
where
    F: Float + Debug + Send + Sync,
{
    /// Write a chunk to a stream
    ///
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_byte(self.fractional_bits as u8).await?;
        self.chunk.write_to(stream).await?;
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let fractional_bits = stream.read_byte().await? as usize;
        Ok(FChunk::new(
            Chunk::read_from(stream).await?,
            fractional_bits,
        ))
    }
}

impl<F> Cacheable for FChunk<F>
where
    F: Float + Debug + Send + Sync + 'static,
{
    /// Return the number of bytes in the serialized representation
    ///
    fn size(&self) -> u64 {
        Resolver::<F>::HEADER_SIZE
        + 1 // fractional bits
        + self.chunk.size()
    }
}

pub(crate) struct FWindowIter<'a, F>
where
    F: Float + Debug + Send + Sync,
{
    _marker: PhantomData<F>,
    iter: Rc<RefCell<WindowIter<'a, i64>>>,
    fractional_bits: usize,
}

impl<'a, F> Iterator for FWindowIter<'a, F>
where
    F: Float + Debug + Send + Sync,
{
    type Item = Array2<F>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.borrow_mut().next() {
            Some(w) => Some(w.map(|i| fixed::from_fixed(*i, self.fractional_bits))),
            None => None,
        }
    }
}

/// A series of time instants stored in a single file on disk.
///
/// Made up of a series of blocks.
///
pub(crate) struct Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Stored data
    blocks: Vec<Block<I>>,

    /// Index into stored data for finding which block contains a particular time instant
    index: Vec<usize>,
}

impl<I> From<Vec<Block<I>>> for Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Make a new Chunk from a vector of Blocks
    ///
    fn from(blocks: Vec<Block<I>>) -> Self {
        let mut index = Vec::with_capacity(blocks.len());
        let mut count = 0;
        for block in &blocks {
            count += block.logs.len() + 1;
            index.push(count);
        }

        Self { blocks, index }
    }
}

impl<I> Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Return the dimensions of the 3 dimensional array represented by this chunk.
    ///
    /// Returns [instants, rows, cols]
    ///
    pub(crate) fn shape(&self) -> [usize; 3] {
        let [rows, cols] = self.blocks[0].snapshot.shape;
        let instants = self.blocks.iter().map(|i| 1 + i.logs.len()).sum();
        [instants, rows, cols]
    }

    /// Get a single value
    ///
    fn get(&self, instant: usize, row: usize, col: usize) -> I {
        let (block, instant) = self.find_block(instant);
        let block = &self.blocks[block];
        block.get(instant, row, col)
    }

    fn find_block(&self, instant: usize) -> (usize, usize) {
        if instant < self.index[0] {
            // Common special case, first block
            (0, instant)
        } else {
            // Use binary search to locate block
            let mut lower = 0;
            let mut upper = self.blocks.len();
            let mut index = upper / 2;
            loop {
                let here = self.index[index];
                if here == instant {
                    index += 1;
                    break;
                } else if here < instant {
                    lower = index;
                } else if here > instant {
                    if self.index[index - 1] <= instant {
                        break;
                    } else {
                        upper = index;
                    }
                }
                index = (lower + upper) / 2;
            }
            (index, instant - self.index[index - 1])
        }
    }

    /// Iterate over time instants in this chunk.
    ///
    /// Used internally by the other iterators.
    ///
    fn iter<'a>(&'a self, start: usize, end: usize) -> ChunkIter<'a, I> {
        let (block, instant) = self.find_block(start);

        ChunkIter {
            chunk: self,
            block,
            instant,
            remaining: end - start,
        }
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub(crate) fn iter_search(&self, bounds: &geom::Cube, lower: I, upper: I) -> SearchIter<I> {
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut iter = SearchIter {
            iter: Rc::new(RefCell::new(self.iter(bounds.start, bounds.end))),
            bounds: bounds.rect(),
            lower,
            upper,

            instant: bounds.start,
            results: None,
        };
        iter.next_results();

        iter
    }

    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape();
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::Chunk: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

#[async_trait]
impl<I> Serialize for Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Write a chunk to a stream
    ///
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_u32(self.blocks.len() as u32).await?;
        for block in &self.blocks {
            block.write_to(stream).await?;
        }
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let n_blocks = stream.read_u32().await? as usize;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut index = Vec::with_capacity(n_blocks);
        let mut count = 0;
        for _ in 0..n_blocks {
            let block = Block::read_from(stream).await?;
            count += block.logs.len() + 1;
            blocks.push(block);
            index.push(count);
        }
        Ok(Self { blocks, index })
    }
}

impl<I> Cacheable for Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Return the number of bytes in the serialized representation
    ///
    fn size(&self) -> u64 {
        4  // number of blocks
        + self.blocks.iter().map(|b| b.size()).sum::<u64>()
    }
}

/// Iterate over time instants stored in this chunk across several blocks
///
/// Used internally by the other iterators.
///
struct ChunkIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    chunk: &'a Chunk<I>,
    block: usize,
    instant: usize,
    remaining: usize,
}

impl<'a, I> Iterator for ChunkIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.remaining == 0 {
            None
        } else {
            let block_index = self.block;
            let instant = self.instant;

            let block = &self.chunk.blocks[block_index];
            if self.instant == block.logs.len() {
                self.instant = 0;
                self.block += 1;
            } else {
                self.instant += 1;
            }
            self.remaining -= 1;

            Some((block_index, instant))
        }
    }
}

pub(crate) struct CellIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    iter: Rc<RefCell<ChunkIter<'a, I>>>,
    row: usize,
    col: usize,
}

impl<'a, I> Iterator for CellIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        let mut iter = self.iter.borrow_mut();
        match iter.next() {
            Some((block, instant)) => {
                let block = &iter.chunk.blocks[block];
                Some(block.get(instant, self.row, self.col))
            }
            None => None,
        }
    }
}

pub(crate) struct WindowIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    iter: Rc<RefCell<ChunkIter<'a, I>>>,
    bounds: geom::Rect,
}

impl<'a, I> Iterator for WindowIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    type Item = Array2<I>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut iter = self.iter.borrow_mut();
        match iter.next() {
            Some((block, instant)) => {
                let block = &iter.chunk.blocks[block];
                Some(block.get_window(instant, &self.bounds))
            }
            None => None,
        }
    }
}

pub(crate) struct SearchIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    iter: Rc<RefCell<ChunkIter<'a, I>>>,
    bounds: geom::Rect,
    lower: I,
    upper: I,

    instant: usize,
    results: Option<VecIntoIter<(usize, usize)>>,
}

impl<'a, I> SearchIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    fn next_results(&mut self) {
        let mut iter = self.iter.borrow_mut();
        self.results = match iter.next() {
            Some((block, instant)) => {
                let block = &iter.chunk.blocks[block];
                let results = block.search_window(instant, &self.bounds, self.lower, self.upper);
                self.instant += 1;
                Some(results.into_iter())
            }
            None => None,
        }
    }
}

impl<'a, I> Iterator for SearchIter<'a, I>
where
    I: PrimInt + Debug + Send + Sync,
{
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.results {
            Some(results) => {
                let mut result = results.next();
                while result == None {
                    self.next_results();
                    result = match &mut self.results {
                        Some(results) => results.next(),
                        None => break,
                    }
                }
                match result {
                    Some((row, col)) => Some((self.instant - 1, row, col)),
                    None => None,
                }
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::log::Log;
    use super::super::snapshot::Snapshot;
    use super::*;
    use crate::testing::array_search_window;
    use futures::io::Cursor;
    use ndarray::{arr2, Array1, Array3};
    use std::collections::HashSet;

    mod fchunk {
        use super::*;

        impl<F> FChunk<F>
        where
            F: Float + Debug + Send + Sync,
        {
            /// Get a cell's value across time instants.
            ///
            pub(crate) fn get_cell(
                &self,
                start: usize,
                end: usize,
                row: usize,
                col: usize,
            ) -> Array1<F> {
                let mut values = Array1::zeros([end - start]);
                self.fill_cell(start, row, col, &mut values);

                values
            }

            /// Get a subarray of this Chunk.
            ///
            pub(crate) fn get_window(&self, bounds: &geom::Cube) -> Array3<F> {
                let mut window = Array3::zeros([bounds.instants(), bounds.rows(), bounds.cols()]);
                self.fill_window(bounds.start, bounds.top, bounds.left, &mut window);

                window
            }
        }

        fn array() -> Vec<Array2<f32>> {
            let data = vec![
                arr2(&[
                    [9.5, 8.25, 7.75, 7.75, 6.125, 6.125, 3.375, 2.625],
                    [7.75, 7.75, 7.75, 7.75, 6.125, 6.125, 3.375, 3.375],
                    [6.125, 6.125, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
                    [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
                    [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [3.375, 3.375, 3.375, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [4.875, 4.875, 3.375, 4.875, 4.875, 4.875, 4.875, 4.875],
                ]),
                arr2(&[
                    [9.5, 8.25, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
                    [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
                    [6.125, 6.125, 6.125, 6.125, 4.875, 3.375, 3.375, 3.375],
                    [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
                    [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [3.375, 3.375, 4.875, 5.0, 5.0, 4.875, 4.875, 4.875],
                    [4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875],
                ]),
                arr2(&[
                    [9.5, 8.25, 7.75, 7.75, 8.25, 7.75, 5.0, 5.0],
                    [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 5.0, 5.0],
                    [7.75, 7.75, 6.125, 6.125, 4.875, 3.375, 4.875, 4.875],
                    [6.125, 6.125, 6.125, 6.125, 4.875, 4.875, 4.875, 4.875],
                    [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
                    [3.375, 3.375, 4.875, 5.0, 6.125, 4.875, 4.875, 4.875],
                    [4.875, 4.875, 4.875, 4.875, 5.0, 4.875, 4.875, 4.875],
                ]),
            ];

            data.into_iter().cycle().take(100).collect()
        }

        fn chunk(array: Vec<Array2<f32>>) -> Arc<FChunk<f32>> {
            let instants = array.iter();
            let mut current = Vec::with_capacity(4);
            let mut blocks: Vec<Block<i64>> = vec![];

            for instant in instants {
                let instant = instant.map(|n| fixed::to_fixed(*n, 3, false));
                current.push(instant);
                if current.len() == 4 {
                    blocks.push(Block::new(
                        Snapshot::from_array(current[0].view(), 2),
                        vec![
                            Log::from_arrays(current[0].view(), current[1].view(), 2),
                            Log::from_arrays(current[0].view(), current[2].view(), 2),
                            Log::from_arrays(current[0].view(), current[3].view(), 2),
                        ],
                    ));
                    current.clear();
                }
            }

            Arc::new(FChunk::new(Chunk::from(blocks), 3))
        }

        #[test]
        fn test_new() {
            let chunk = chunk(array());
            assert_eq!(chunk.fractional_bits, 3);
            assert_eq!(chunk.shape(), [100, 8, 8]);
        }

        #[test]
        fn get() {
            let data = array();
            let chunk = chunk(data.clone());
            for instant in 0..100 {
                for row in 0..8 {
                    for col in 0..8 {
                        assert_eq!(chunk.get(instant, row, col), data[instant][[row, col]]);
                    }
                }
            }
        }

        #[test]
        fn fill_cell() {
            let data = array();
            let chunk = chunk(data.clone());
            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values = chunk.get_cell(start, end, row, col);
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn fill_cell_end_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values = chunk.get_cell(0, 200, 4, 4);
        }

        #[test]
        #[should_panic]
        fn fill_cell_row_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values = chunk.get_cell(0, 100, 8, 4);
        }

        #[test]
        #[should_panic]
        fn fill_cell_col_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values = chunk.get_cell(0, 100, 4, 8);
        }

        #[test]
        fn fill_window() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window = chunk.get_window(&bounds);
                            assert_eq!(window.shape(), [end - start, bottom - top, right - left]);

                            for i in 0..end - start {
                                assert_eq!(
                                    window.slice(s![i, .., ..]),
                                    data[start + i].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn fill_window_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let bounds = geom::Cube::new(0, 100, 0, 9, 0, 9);
            let _window = chunk.get_window(&bounds);
        }

        #[test]
        fn iter_search() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let lower = (start / 5) as f32;
                            let upper = (end / 10) as f32;

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let results: Vec<(usize, usize, usize)> =
                                chunk.iter_search(&bounds, lower, upper).collect();

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results, expected);
                        }
                    }
                }
            }
        }

        #[test]
        fn iter_search_rearrange() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let lower = (start / 5) as f32;
                            let upper = (end / 10) as f32;

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let results: Vec<(usize, usize, usize)> =
                                chunk.iter_search(&bounds, upper, lower).collect();

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results, expected);
                        }
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn iter_search_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _results: Vec<(usize, usize, usize)> = chunk
                .iter_search(&geom::Cube::new(0, 100, 0, 9, 0, 9), 0.0, 1.0)
                .collect();
        }

        #[tokio::test]
        async fn serialize_deserialize() -> Result<()> {
            let data = array();
            let chunk = chunk(data.clone());
            let mut file: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
            chunk.write_to(&mut file).await?;
            assert_eq!(
                file.len(),
                (chunk.size() - Resolver::<f32>::HEADER_SIZE) as usize
            );

            let mut file = Cursor::new(file);
            let chunk: Arc<FChunk<f32>> = Arc::new(FChunk::read_from(&mut file).await?);

            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values = chunk.get_cell(start, end, row, col);
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }

            Ok(())
        }
    }

    mod chunk {
        use super::*;

        fn array() -> Vec<Array2<i32>> {
            let data = vec![
                arr2(&[
                    [9, 8, 7, 7, 6, 6, 3, 2],
                    [7, 7, 7, 7, 6, 6, 3, 3],
                    [6, 6, 6, 6, 3, 3, 3, 3],
                    [5, 5, 6, 6, 3, 3, 3, 3],
                    [4, 5, 5, 5, 4, 4, 4, 4],
                    [3, 3, 5, 5, 4, 4, 4, 4],
                    [3, 3, 3, 5, 4, 4, 4, 4],
                    [4, 4, 3, 4, 4, 4, 4, 4],
                ]),
                arr2(&[
                    [9, 8, 7, 7, 7, 7, 2, 2],
                    [7, 7, 7, 7, 7, 7, 2, 2],
                    [6, 6, 6, 6, 4, 3, 3, 3],
                    [5, 5, 6, 6, 3, 3, 3, 3],
                    [4, 5, 5, 5, 4, 4, 4, 4],
                    [3, 3, 5, 5, 4, 4, 4, 4],
                    [3, 3, 4, 5, 5, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                ]),
                arr2(&[
                    [9, 8, 7, 7, 8, 7, 5, 5],
                    [7, 7, 7, 7, 7, 7, 5, 5],
                    [7, 7, 6, 6, 4, 3, 4, 4],
                    [6, 6, 6, 6, 4, 4, 4, 4],
                    [4, 5, 5, 5, 4, 4, 4, 4],
                    [3, 3, 5, 5, 4, 4, 4, 4],
                    [3, 3, 4, 5, 6, 4, 4, 4],
                    [4, 4, 4, 4, 5, 4, 4, 4],
                ]),
            ];

            data.into_iter().cycle().take(100).collect()
        }

        fn chunk(array: Vec<Array2<i32>>) -> Arc<Chunk<i32>> {
            let instants = array.iter();
            let mut current = Vec::with_capacity(4);
            let mut blocks: Vec<Block<i32>> = vec![];

            for instant in instants {
                current.push(instant.view());
                if current.len() == 4 {
                    blocks.push(Block::new(
                        Snapshot::from_array(current[0], 2),
                        vec![
                            Log::from_arrays(current[0], current[1], 2),
                            Log::from_arrays(current[0], current[2], 2),
                            Log::from_arrays(current[0], current[3], 2),
                        ],
                    ));
                    current.clear();
                }
            }

            Arc::new(Chunk::from(blocks))
        }

        #[test]
        fn from_blocks() {
            let chunk = chunk(array());
            assert_eq!(chunk.blocks.len(), 25);
            assert_eq!(chunk.shape(), [100, 8, 8]);
        }

        #[test]
        fn get() {
            let data = array();
            let chunk = chunk(data.clone());
            for instant in 0..100 {
                for row in 0..8 {
                    for col in 0..8 {
                        assert_eq!(chunk.get(instant, row, col), data[instant][[row, col]]);
                    }
                }
            }
        }

        #[test]
        fn iter_search() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let lower: i32 = (start / 5).try_into().unwrap();
                            let upper: i32 = (end / 10).try_into().unwrap();

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let results: Vec<(usize, usize, usize)> =
                                chunk.iter_search(&bounds, lower, upper).collect();

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results, expected);
                        }
                    }
                }
            }
        }

        #[test]
        fn iter_search_rearrange() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let lower: i32 = (start / 5).try_into().unwrap();
                            let upper: i32 = (end / 10).try_into().unwrap();

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let results: Vec<(usize, usize, usize)> =
                                chunk.iter_search(&bounds, upper, lower).collect();

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results, expected);
                        }
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn iter_search_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _results: Vec<(usize, usize, usize)> = chunk
                .iter_search(&geom::Cube::new(0, 100, 0, 9, 0, 9), 0, 1)
                .collect();
        }

        #[tokio::test]
        async fn serialize_deserialize() -> Result<()> {
            let data = array();
            let chunk = chunk(data.clone());

            let mut buffer: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
            chunk.write_to(&mut buffer).await?;
            assert_eq!(buffer.len(), chunk.size() as usize);

            let mut buffer = Cursor::new(buffer);
            let chunk: Arc<Chunk<i32>> = Arc::new(Chunk::read_from(&mut buffer).await?);

            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    for instant in start..end {
                        assert_eq!(chunk.get(instant, row, col), data[instant][[row, col]]);
                    }
                }
            }

            Ok(())
        }
    }
}
