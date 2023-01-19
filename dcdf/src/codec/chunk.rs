use std::{
    cell::RefCell,
    fmt::Debug,
    io::{Read, Write},
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
    vec::IntoIter as VecIntoIter,
};

use async_trait::async_trait;
use futures::io::{AsyncRead, AsyncWrite};
use ndarray::{s, Array2, Array3, ArrayBase, DataMut, Ix3};
use num_traits::{Float, PrimInt};

use crate::{
    cache::Cacheable,
    dag::resolver::Resolver,
    errors::Result,
    extio::{
        ExtendedAsyncRead, ExtendedAsyncWrite, ExtendedRead, ExtendedWrite, Serialize,
        SerializeAsync,
    },
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
    pub fn new(chunk: Chunk<i64>, fractional_bits: usize) -> Self {
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
    pub fn shape(&self) -> [usize; 3] {
        self.chunk.shape()
    }

    /// Iterate over a cell's value across time instants.
    ///
    pub fn iter_cell(
        self: &Arc<Self>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> FCellIter<F> {
        FCellIter {
            chunk: Arc::clone(&self),
            iter: Rc::new(RefCell::new(self.chunk.iter_cell(start, end, row, col))),
        }
    }

    /// Get a subarray of this Chunk.
    ///
    pub fn get_window(self: &Arc<Self>, bounds: &geom::Cube) -> Array3<F> {
        self.chunk
            .check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut window = Array3::zeros([bounds.instants(), bounds.rows(), bounds.cols()]);
        self.fill_window(bounds, &mut window);

        window
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    pub(crate) fn fill_window<S>(
        self: &Arc<Self>,
        bounds: &geom::Cube,
        window: &mut ArrayBase<S, Ix3>,
    ) where
        S: DataMut<Elem = F>,
    {
        for (i, (block, instant)) in self.chunk.iter(bounds.start, bounds.end).enumerate() {
            let mut window2d = window.slice_mut(s![i, .., ..]);
            let set = |row, col, value| {
                window2d[[row, col]] = fixed::from_fixed(value, self.fractional_bits)
            };
            let block = &self.chunk.blocks[block];
            block.fill_window(set, instant, &bounds.rect());
        }
    }

    /// Iterate over subarrays of successive time instants.
    ///
    pub fn iter_window(self: &Arc<Self>, bounds: &geom::Cube) -> FWindowIter<F> {
        FWindowIter {
            _marker: PhantomData,
            iter: Rc::new(RefCell::new(self.chunk.iter_window(bounds))),
            fractional_bits: self.fractional_bits,
        }
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub fn iter_search(&self, bounds: &geom::Cube, lower: F, upper: F) -> SearchIter<i64> {
        self.chunk.iter_search(
            bounds,
            fixed::to_fixed(lower, self.fractional_bits, true),
            fixed::to_fixed(upper, self.fractional_bits, true),
        )
    }
}

impl<F> Serialize for FChunk<F>
where
    F: Float + Debug + Send + Sync,
{
    /// Write a chunk to a stream
    ///
    fn write_to(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_byte(self.fractional_bits as u8)?;
        self.chunk.write_to(stream)?;
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    fn read_from(stream: &mut impl Read) -> Result<Self> {
        let fractional_bits = stream.read_byte()? as usize;
        Ok(FChunk::new(Chunk::read_from(stream)?, fractional_bits))
    }
}

#[async_trait]
impl<F> SerializeAsync for FChunk<F>
where
    F: Float + Debug + Send + Sync,
{
    /// Write a chunk to a stream
    ///
    async fn write_to_async(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_byte_async(self.fractional_bits as u8).await?;
        self.chunk.write_to_async(stream).await?;
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    async fn read_from_async(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let fractional_bits = stream.read_byte_async().await? as usize;
        Ok(FChunk::new(
            Chunk::read_from_async(stream).await?,
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

pub struct FCellIter<F>
where
    F: Float + Debug + Send + Sync,
{
    chunk: Arc<FChunk<F>>,
    iter: Rc<RefCell<CellIter<i64>>>,
}

impl<F> Iterator for FCellIter<F>
where
    F: Float + Debug + Send + Sync,
{
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.borrow_mut().next() {
            Some(value) => Some(fixed::from_fixed(value, self.chunk.fractional_bits)),
            None => None,
        }
    }
}

pub struct FWindowIter<F>
where
    F: Float + Debug + Send + Sync,
{
    _marker: PhantomData<F>,
    iter: Rc<RefCell<WindowIter<i64>>>,
    fractional_bits: usize,
}

impl<F> Iterator for FWindowIter<F>
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
pub struct Chunk<I>
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
    pub fn shape(&self) -> [usize; 3] {
        let [rows, cols] = self.blocks[0].snapshot.shape;
        let instants = self.blocks.iter().map(|i| 1 + i.logs.len()).sum();
        [instants, rows, cols]
    }

    /// Iterate over time instants in this chunk.
    ///
    /// Used internally by the other iterators.
    ///
    fn iter(self: &Arc<Self>, start: usize, end: usize) -> ChunkIter<I> {
        let (block, instant) = if start < self.index[0] {
            // Common special case, starting at beginning of chunk
            (0, start)
        } else {
            // Use binary search to locate starting block
            let mut lower = 0;
            let mut upper = self.blocks.len();
            let mut index = upper / 2;
            loop {
                let here = self.index[index];
                if here == start {
                    index += 1;
                    break;
                } else if here < start {
                    lower = index;
                } else if here > start {
                    if self.index[index - 1] <= start {
                        break;
                    } else {
                        upper = index;
                    }
                }
                index = (lower + upper) / 2;
            }
            (index, start - self.index[index - 1])
        };

        ChunkIter {
            chunk: Arc::clone(&self),
            block,
            instant,
            remaining: end - start,
        }
    }

    /// Iterate over a cell's value across time instants.
    ///
    pub fn iter_cell(
        self: &Arc<Self>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> CellIter<I> {
        let (start, end) = rearrange(start, end);
        self.check_bounds(end - 1, row, col);
        CellIter {
            iter: Rc::new(RefCell::new(self.iter(start, end))),
            row,
            col,
        }
    }

    /// Get a subarray of this Chunk.
    ///
    pub fn get_window(self: &Arc<Self>, bounds: &geom::Cube) -> Array3<I> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut windows = Array3::zeros([bounds.instants(), bounds.rows(), bounds.cols()]);

        for (i, (block, instant)) in self.iter(bounds.start, bounds.end).enumerate() {
            let mut window = windows.slice_mut(s![i, .., ..]);
            let set = |row, col, value| window[[row, col]] = I::from(value).unwrap();
            let block = &self.blocks[block];
            block.fill_window(set, instant, &bounds.rect());
        }
        windows
    }

    /// Iterate over subarrays of successive time instants.
    ///
    pub fn iter_window(self: &Arc<Self>, bounds: &geom::Cube) -> WindowIter<I> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        WindowIter {
            iter: Rc::new(RefCell::new(self.iter(bounds.start, bounds.end))),
            bounds: bounds.rect(),
        }
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub fn iter_search(self: &Arc<Self>, bounds: &geom::Cube, lower: I, upper: I) -> SearchIter<I> {
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

impl<I> Serialize for Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Write a chunk to a stream
    ///
    fn write_to(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_u32(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.write_to(stream)?;
        }
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    fn read_from(stream: &mut impl Read) -> Result<Self> {
        let n_blocks = stream.read_u32()? as usize;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut index = Vec::with_capacity(n_blocks);
        let mut count = 0;
        for _ in 0..n_blocks {
            let block = Block::read_from(stream)?;
            count += block.logs.len() + 1;
            blocks.push(block);
            index.push(count);
        }
        Ok(Self { blocks, index })
    }
}

#[async_trait]
impl<I> SerializeAsync for Chunk<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    /// Write a chunk to a stream
    ///
    async fn write_to_async(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_u32_async(self.blocks.len() as u32).await?;
        for block in &self.blocks {
            block.write_to_async(stream).await?;
        }
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    async fn read_from_async(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let n_blocks = stream.read_u32_async().await? as usize;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut index = Vec::with_capacity(n_blocks);
        let mut count = 0;
        for _ in 0..n_blocks {
            let block = Block::read_from_async(stream).await?;
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
struct ChunkIter<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    chunk: Arc<Chunk<I>>,
    block: usize,
    instant: usize,
    remaining: usize,
}

impl<I> Iterator for ChunkIter<I>
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

pub struct CellIter<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    iter: Rc<RefCell<ChunkIter<I>>>,
    row: usize,
    col: usize,
}

impl<I> Iterator for CellIter<I>
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

pub struct WindowIter<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    iter: Rc<RefCell<ChunkIter<I>>>,
    bounds: geom::Rect,
}

impl<I> Iterator for WindowIter<I>
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

pub struct SearchIter<I>
where
    I: PrimInt + Debug + Send + Sync,
{
    iter: Rc<RefCell<ChunkIter<I>>>,
    bounds: geom::Rect,
    lower: I,
    upper: I,

    instant: usize,
    results: Option<VecIntoIter<(usize, usize)>>,
}

impl<I> SearchIter<I>
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

impl<I> Iterator for SearchIter<I>
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
    use super::super::testing::array_search_window;
    use super::*;
    use futures::io::Cursor as AsyncCursor;
    use ndarray::arr2;
    use std::collections::HashSet;
    use std::io::{Cursor, Seek};
    use tempfile::tempfile;

    mod fchunk {
        use super::*;

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
        fn iter_cell() {
            let data = array();
            let chunk = chunk(data.clone());
            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<f32> = chunk.iter_cell(start, end, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }
        }

        #[test]
        fn iter_cell_rearrange() {
            let data = array();
            let chunk = chunk(data.clone());
            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<f32> = chunk.iter_cell(end, start, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn iter_cell_end_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values: Vec<f32> = chunk.iter_cell(0, 200, 4, 4).collect();
        }

        #[test]
        #[should_panic]
        fn iter_cell_row_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values: Vec<f32> = chunk.iter_cell(0, 100, 8, 4).collect();
        }

        #[test]
        #[should_panic]
        fn iter_cell_col_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values: Vec<f32> = chunk.iter_cell(0, 100, 4, 8).collect();
        }

        #[test]
        fn get_window() {
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
        fn get_window_rearrange() {
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
        fn get_window_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let bounds = geom::Cube::new(0, 100, 0, 9, 0, 9);
            let _window = chunk.get_window(&bounds);
        }

        #[test]
        fn iter_window() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let windows: Vec<Array2<f32>> = chunk.iter_window(&bounds).collect();
                            assert_eq!(windows.len(), end - start);

                            for i in 0..windows.len() {
                                assert_eq!(
                                    windows[i],
                                    data[i + start].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn iter_window_rearrange() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let windows: Vec<Array2<f32>> = chunk.iter_window(&bounds).collect();
                            assert_eq!(windows.len(), end - start);

                            for i in 0..windows.len() {
                                assert_eq!(
                                    windows[i],
                                    data[i + start].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn iter_window_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let bounds = geom::Cube::new(0, 100, 0, 9, 0, 9);
            let _window: Vec<Array2<f32>> = chunk.iter_window(&bounds).collect();
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

        #[test]
        fn serialize_deserialize() -> Result<()> {
            let data = array();
            let chunk = chunk(data.clone());
            let mut file: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
            chunk.write_to(&mut file)?;
            assert_eq!(
                file.len(),
                (chunk.size() - Resolver::<f32>::HEADER_SIZE) as usize
            );

            let mut file = Cursor::new(file);
            let chunk: Arc<FChunk<f32>> = Arc::new(FChunk::read_from(&mut file)?);

            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<f32> = chunk.iter_cell(start, end, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }

            Ok(())
        }

        #[tokio::test]
        async fn serialize_deserialize_async() -> Result<()> {
            let data = array();
            let chunk = chunk(data.clone());
            let mut file: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
            chunk.write_to_async(&mut file).await?;
            assert_eq!(
                file.len(),
                (chunk.size() - Resolver::<f32>::HEADER_SIZE) as usize
            );

            let mut file = AsyncCursor::new(file);
            let chunk: Arc<FChunk<f32>> = Arc::new(FChunk::read_from_async(&mut file).await?);

            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<f32> = chunk.iter_cell(start, end, row, col).collect();
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
        fn iter_cell() {
            let data = array();
            let chunk = chunk(data.clone());
            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<i32> = chunk.iter_cell(start, end, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }
        }

        #[test]
        fn iter_cell_rearrange() {
            let data = array();
            let chunk = chunk(data.clone());
            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<i32> = chunk.iter_cell(end, start, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn iter_cell_end_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values: Vec<i32> = chunk.iter_cell(0, 200, 4, 4).collect();
        }

        #[test]
        #[should_panic]
        fn iter_cell_row_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values: Vec<i32> = chunk.iter_cell(0, 100, 8, 4).collect();
        }

        #[test]
        #[should_panic]
        fn iter_cell_col_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let _values: Vec<i32> = chunk.iter_cell(0, 100, 4, 8).collect();
        }

        #[test]
        fn get_window() {
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
        fn get_window_rearrange() {
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
        fn get_window_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let bounds = geom::Cube::new(0, 100, 0, 9, 0, 9);
            let _window = chunk.get_window(&bounds);
        }

        #[test]
        fn iter_window() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let windows: Vec<Array2<i32>> = chunk.iter_window(&bounds).collect();
                            assert_eq!(windows.len(), end - start);

                            for i in 0..windows.len() {
                                assert_eq!(
                                    windows[i],
                                    data[i + start].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn iter_window_rearrange() {
            let data = array();
            let chunk = chunk(data.clone());
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let start = top * left;
                            let end = bottom * right + 36;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let windows: Vec<Array2<i32>> = chunk.iter_window(&bounds).collect();
                            assert_eq!(windows.len(), end - start);

                            for i in 0..windows.len() {
                                assert_eq!(
                                    windows[i],
                                    data[i + start].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        #[should_panic]
        fn iter_window_out_of_bounds() {
            let data = array();
            let chunk = chunk(data.clone());
            let bounds = geom::Cube::new(0, 100, 0, 9, 0, 9);
            let _window: Vec<Array2<i32>> = chunk.iter_window(&bounds).collect();
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

        #[test]
        fn serialize_deserialize() -> Result<()> {
            let data = array();
            let chunk = chunk(data.clone());

            let mut buffer: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
            chunk.write_to(&mut buffer)?;
            assert_eq!(buffer.len(), chunk.size() as usize);

            let mut buffer = Cursor::new(buffer);
            let chunk: Arc<Chunk<i32>> = Arc::new(Chunk::read_from(&mut buffer)?);

            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<i32> = chunk.iter_cell(start, end, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }

            Ok(())
        }

        #[tokio::test]
        async fn serialize_deserialize_async() -> Result<()> {
            let data = array();
            let chunk = chunk(data.clone());

            let mut buffer: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
            chunk.write_to_async(&mut buffer).await?;
            assert_eq!(buffer.len(), chunk.size() as usize);

            let mut buffer = AsyncCursor::new(buffer);
            let chunk: Arc<Chunk<i32>> = Arc::new(Chunk::read_from_async(&mut buffer).await?);

            for row in 0..8 {
                for col in 0..8 {
                    let start = row * col;
                    let end = 100 - col;
                    let values: Vec<i32> = chunk.iter_cell(start, end, row, col).collect();
                    assert_eq!(values.len(), end - start);
                    for i in 0..values.len() {
                        assert_eq!(values[i], data[i + start][[row, col]]);
                    }
                }
            }

            Ok(())
        }
    }
}
