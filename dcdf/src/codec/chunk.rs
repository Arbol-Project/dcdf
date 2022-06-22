use ndarray::{s, Array2, Array3};
use num_traits::{Float, PrimInt};
use std::fmt::Debug;
use std::io;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::vec::IntoIter as VecIntoIter;

use super::block::Block;
use super::helpers::rearrange;
use crate::extio::{ExtendedRead, ExtendedWrite};
use crate::fixed;

/// A wrapper for Chunk that adapts it to use floating point data
///
/// Floating point numbers are stored in a fixed point representation and converted back to
/// floating point using `fractional_bits` to determine the number of bits that are used to
/// represent the fractional part of the number.
///
pub struct FChunk<F>
where
    F: Float + Debug,
{
    _marker: PhantomData<F>,

    /// Number of bits in the fixed point number representation that represent the fractional part
    fractional_bits: usize,

    /// Wrapped Chunk
    chunk: Chunk<i64>,
}

impl<F> FChunk<F>
where
    F: Float + Debug,
{
    /// Create an FChunk that wraps `chunk` using `fractional_bits` for fixed point to floating
    /// point conversion.
    ///
    pub fn new(chunk: Chunk<i64>, fractional_bits: usize) -> Self {
        Self {
            _marker: PhantomData,
            fractional_bits: fractional_bits,
            chunk: chunk,
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
    pub fn iter_cell<'a>(
        &'a self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Box<dyn Iterator<Item = F> + 'a> {
        Box::new(
            self.chunk
                .iter_cell(start, end, row, col)
                .map(|i| fixed::from_fixed(i, self.fractional_bits)),
        )
    }

    /// Get a subarray of this Chunk.
    ///
    pub fn get_window(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array3<F> {
        let (start, end) = rearrange(start, end);
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        self.chunk.check_bounds(end - 1, bottom - 1, right - 1);

        let instants = end - start;
        let rows = bottom - top;
        let cols = right - left;
        let mut windows = Array3::zeros([instants, rows, cols]);

        for (i, (block, instant)) in self.chunk.iter(start, end).enumerate() {
            let mut window = windows.slice_mut(s![i, .., ..]);
            let set = |row, col, value| {
                window[[row, col]] = fixed::from_fixed(value, self.fractional_bits)
            };
            let block = &self.chunk.blocks[block];
            block.fill_window(set, instant, top, bottom, left, right);
        }
        windows
    }

    /// Iterate over subarrays of successive time instants.
    ///
    pub fn iter_window<'a>(
        &'a self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Box<dyn Iterator<Item = Array2<F>> + 'a> {
        Box::new(
            self.chunk
                .iter_window(start, end, top, bottom, left, right)
                .map(|w| w.map(|i| fixed::from_fixed(*i, self.fractional_bits))),
        )
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub fn iter_search(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: F,
        upper: F,
    ) -> SearchIter<i64> {
        self.chunk.iter_search(
            start,
            end,
            top,
            bottom,
            left,
            right,
            fixed::to_fixed(lower, self.fractional_bits, true),
            fixed::to_fixed(upper, self.fractional_bits, true),
        )
    }

    /// Write a chunk to a stream
    ///
    pub fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_byte(self.fractional_bits as u8)?;
        self.chunk.serialize(stream)?;
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    pub fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let fractional_bits = stream.read_byte()? as usize;
        Ok(FChunk::new(Chunk::deserialize(stream)?, fractional_bits))
    }

    /// Return the number of bytes in the serialized representation
    ///
    pub fn size(&self) -> u64 {
        1 // fractional bits
        + self.chunk.size()
    }
}

/// A series of time instants stored in a single file on disk.
///
/// Made up of a series of blocks.
///
pub struct Chunk<I>
where
    I: PrimInt + Debug,
{
    /// Stored data
    blocks: Vec<Block<I>>,

    /// Index into stored data for finding which block contains a particular time instant
    index: Vec<usize>,
}

impl<I> From<Vec<Block<I>>> for Chunk<I>
where
    I: PrimInt + Debug,
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
    I: PrimInt + Debug,
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
    fn iter(&self, start: usize, end: usize) -> ChunkIter<I> {
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
            chunk: self,
            block,
            instant,
            remaining: end - start,
        }
    }

    /// Iterate over a cell's value across time instants.
    ///
    pub fn iter_cell(&self, start: usize, end: usize, row: usize, col: usize) -> CellIter<I> {
        let (start, end) = rearrange(start, end);
        self.check_bounds(end - 1, row, col);
        CellIter {
            iter: self.iter(start, end),
            row,
            col,
        }
    }

    /// Get a subarray of this Chunk.
    ///
    pub fn get_window(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array3<I> {
        let (start, end) = rearrange(start, end);
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        self.check_bounds(end - 1, bottom - 1, right - 1);

        let instants = end - start;
        let rows = bottom - top;
        let cols = right - left;
        let mut windows = Array3::zeros([instants, rows, cols]);

        for (i, (block, instant)) in self.iter(start, end).enumerate() {
            let mut window = windows.slice_mut(s![i, .., ..]);
            let set = |row, col, value| window[[row, col]] = I::from(value).unwrap();
            let block = &self.blocks[block];
            block.fill_window(set, instant, top, bottom, left, right);
        }
        windows
    }

    /// Iterate over subarrays of successive time instants.
    ///
    pub fn iter_window(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> WindowIter<I> {
        let (start, end) = rearrange(start, end);
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        self.check_bounds(end - 1, bottom - 1, right - 1);

        WindowIter {
            iter: self.iter(start, end),
            top,
            bottom,
            left,
            right,
        }
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub fn iter_search(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: I,
        upper: I,
    ) -> SearchIter<I> {
        let (start, end) = rearrange(start, end);
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(end - 1, bottom - 1, right - 1);

        let mut iter = SearchIter {
            iter: self.iter(start, end),
            top,
            bottom,
            left,
            right,
            lower,
            upper,

            instant: start,
            results: None,
        };
        iter.next_results();

        iter
    }

    /// Write a chunk to a stream
    ///
    pub fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_u32(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.serialize(stream)?;
        }
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    pub fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let n_blocks = stream.read_u32()? as usize;
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut index = Vec::with_capacity(n_blocks);
        let mut count = 0;
        for _ in 0..n_blocks {
            let block = Block::deserialize(stream)?;
            count += block.logs.len() + 1;
            blocks.push(block);
            index.push(count);
        }
        Ok(Self { blocks, index })
    }

    /// Return the number of bytes in the serialized representation
    ///
    pub fn size(&self) -> u64 {
        4  // number of blocks
        + self.blocks.iter().map(|b| b.size()).sum::<u64>()
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

/// Iterate over time instants stored in this chunk across several blocks
///
/// Used internally by the other iterators.
///
struct ChunkIter<'a, I>
where
    I: PrimInt + Debug,
{
    chunk: &'a Chunk<I>,
    block: usize,
    instant: usize,
    remaining: usize,
}

impl<'a, I> Iterator for ChunkIter<'a, I>
where
    I: PrimInt + Debug,
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

pub struct CellIter<'a, I>
where
    I: PrimInt + Debug,
{
    iter: ChunkIter<'a, I>,
    row: usize,
    col: usize,
}

impl<'a, I> Iterator for CellIter<'a, I>
where
    I: PrimInt + Debug,
{
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((block, instant)) => {
                let block = &self.iter.chunk.blocks[block];
                Some(block.get(instant, self.row, self.col))
            }
            None => None,
        }
    }
}

pub struct WindowIter<'a, I>
where
    I: PrimInt + Debug,
{
    iter: ChunkIter<'a, I>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
}

impl<'a, I> Iterator for WindowIter<'a, I>
where
    I: PrimInt + Debug,
{
    type Item = Array2<I>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((block, instant)) => {
                let block = &self.iter.chunk.blocks[block];
                Some(block.get_window(instant, self.top, self.bottom, self.left, self.right))
            }
            None => None,
        }
    }
}

pub struct SearchIter<'a, I>
where
    I: PrimInt + Debug,
{
    iter: ChunkIter<'a, I>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    lower: I,
    upper: I,

    instant: usize,
    results: Option<VecIntoIter<(usize, usize)>>,
}

impl<'a, I> SearchIter<'a, I>
where
    I: PrimInt + Debug,
{
    fn next_results(&mut self) {
        self.results = match self.iter.next() {
            Some((block, instant)) => {
                let block = &self.iter.chunk.blocks[block];
                let results = block.search_window(
                    instant,
                    self.top,
                    self.bottom,
                    self.left,
                    self.right,
                    self.lower,
                    self.upper,
                );
                self.instant += 1;
                Some(results.into_iter())
            }
            None => None,
        }
    }
}

impl<'a, I> Iterator for SearchIter<'a, I>
where
    I: PrimInt + Debug,
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
    use ndarray::arr2;
    use std::collections::HashSet;
    use std::io::Seek;
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

        fn chunk(array: Vec<Array2<f32>>) -> FChunk<f32> {
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

            FChunk::new(Chunk::from(blocks), 3)
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
                            let window = chunk.get_window(start, end, top, bottom, left, right);
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
                            let window = chunk.get_window(end, start, bottom, top, right, left);
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
            let _window = chunk.get_window(0, 100, 0, 9, 0, 9);
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
                            let windows: Vec<Array2<f32>> = chunk
                                .iter_window(start, end, top, bottom, left, right)
                                .collect();
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
                            let windows: Vec<Array2<f32>> = chunk
                                .iter_window(end, start, bottom, top, right, left)
                                .collect();
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
            let _window: Vec<Array2<f32>> = chunk.iter_window(0, 100, 0, 9, 0, 9).collect();
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

                            let results: Vec<(usize, usize, usize)> = chunk
                                .iter_search(start, end, top, bottom, left, right, lower, upper)
                                .collect();

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

                            let results: Vec<(usize, usize, usize)> = chunk
                                .iter_search(end, start, bottom, top, right, left, upper, lower)
                                .collect();

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
            let _results: Vec<(usize, usize, usize)> =
                chunk.iter_search(0, 100, 0, 9, 0, 9, 0.0, 1.0).collect();
        }

        #[test]
        fn serialize_deserialize() -> io::Result<()> {
            let data = array();
            let chunk = chunk(data.clone());
            let mut file = tempfile()?;
            chunk.serialize(&mut file)?;
            file.sync_all()?;

            let metadata = file.metadata()?;
            assert_eq!(metadata.len(), chunk.size());

            file.rewind()?;
            let chunk: FChunk<f32> = FChunk::deserialize(&mut file)?;

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

        fn chunk(array: Vec<Array2<i32>>) -> Chunk<i32> {
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

            Chunk::from(blocks)
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
                            let window = chunk.get_window(start, end, top, bottom, left, right);
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
                            let window = chunk.get_window(end, start, bottom, top, right, left);
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
            let _window = chunk.get_window(0, 100, 0, 9, 0, 9);
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
                            let windows: Vec<Array2<i32>> = chunk
                                .iter_window(start, end, top, bottom, left, right)
                                .collect();
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
                            let windows: Vec<Array2<i32>> = chunk
                                .iter_window(end, start, bottom, top, right, left)
                                .collect();
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
            let _window: Vec<Array2<i32>> = chunk.iter_window(0, 100, 0, 9, 0, 9).collect();
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

                            let results: Vec<(usize, usize, usize)> = chunk
                                .iter_search(start, end, top, bottom, left, right, lower, upper)
                                .collect();

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

                            let results: Vec<(usize, usize, usize)> = chunk
                                .iter_search(end, start, bottom, top, right, left, upper, lower)
                                .collect();

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
            let _results: Vec<(usize, usize, usize)> =
                chunk.iter_search(0, 100, 0, 9, 0, 9, 0, 1).collect();
        }
    }
}
