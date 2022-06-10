use ndarray::Array2;
use num_traits::{Float, PrimInt};
use std::fmt::Debug;
use std::io;
use std::io::{Read, Write};
use std::marker::PhantomData;

use super::block::Block;
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
    pub fn new(chunk: Chunk<i64>, fractional_bits: usize) -> Self {
        Self {
            _marker: PhantomData,
            fractional_bits: fractional_bits,
            chunk: chunk,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.chunk.shape()
    }

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

    pub fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_byte(self.fractional_bits as u8)?;
        self.chunk.serialize(stream)?;
        Ok(())
    }

    pub fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let fractional_bits = stream.read_byte()? as usize;
        Ok(FChunk::new(Chunk::deserialize(stream)?, fractional_bits))
    }

    pub fn size(&self) -> u64 {
        1 + self.chunk.size()
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
    pub fn shape(&self) -> [usize; 3] {
        let [rows, cols] = self.blocks[0].snapshot.shape;
        let instants = self.blocks.iter().map(|i| 1 + i.logs.len()).sum();
        [instants, rows, cols]
    }

    // Iterate over time instants in this chunk.
    fn iter(&self, start: usize, end: usize) -> ChunkIter<I> {
        let (block, block_index) = if start < self.index[0] {
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
            block_index,
            remaining: end - start,
        }
    }

    pub fn iter_cell(&self, start: usize, end: usize, row: usize, col: usize) -> CellIter<I> {
        CellIter {
            iter: self.iter(start, end),
            row,
            col,
        }
    }

    pub fn iter_window(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> WindowIter<I> {
        WindowIter {
            iter: self.iter(start, end),
            top,
            bottom,
            left,
            right,
        }
    }

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
        SearchIter {
            iter: self.iter(start, end),
            top,
            bottom,
            left,
            right,
            lower,
            upper,
        }
    }

    pub fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_u32(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.serialize(stream)?;
        }
        Ok(())
    }

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

    pub fn size(&self) -> u64 {
        4 + self.blocks.iter().map(|b| b.size()).sum::<u64>()
    }
}

struct ChunkIter<'a, I>
where
    I: PrimInt + Debug,
{
    chunk: &'a Chunk<I>,
    block: usize,
    block_index: usize,
    remaining: usize,
}

// Unable to use the Iterator trait due to lack of support for Generic Associated Types in Rust
// at this time. (Item cannot contain Block<I>).
//
impl<'a, I> ChunkIter<'a, I>
where
    I: PrimInt + Debug,
{
    fn next(&mut self) -> Option<(usize, &'a Block<I>)> {
        if self.remaining == 0 {
            None
        } else {
            let block = &self.chunk.blocks[self.block];
            let block_index = self.block_index;

            if self.block_index == block.logs.len() {
                self.block_index = 0;
                self.block += 1;
            } else {
                self.block_index += 1;
            }
            self.remaining -= 1;

            Some((block_index, block))
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
            Some((index, block)) => Some(block.get(index, self.row, self.col)),
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
            Some((index, block)) => {
                Some(block.get_window(index, self.top, self.bottom, self.left, self.right))
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
}

impl<'a, I> Iterator for SearchIter<'a, I>
where
    I: PrimInt + Debug,
{
    type Item = Vec<(usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((index, block)) => Some(block.search_window(
                index,
                self.top,
                self.bottom,
                self.left,
                self.right,
                self.lower,
                self.upper,
            )),
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
    use ndarray::{arr2, s, Array2};
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
                            let results: Vec<Vec<(usize, usize)>> = chunk
                                .iter_search(start, end, top, bottom, left, right, lower, upper)
                                .collect();
                            assert_eq!(results.len(), end - start);

                            for i in 0..results.len() {
                                let expected = array_search_window(
                                    data[i + start].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.into_iter());
                                let results: HashSet<(usize, usize)> =
                                    HashSet::from_iter(results[i].clone().into_iter());
                                assert_eq!(results, expected);
                            }
                        }
                    }
                }
            }
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
                            let results: Vec<Vec<(usize, usize)>> = chunk
                                .iter_search(start, end, top, bottom, left, right, lower, upper)
                                .collect();
                            assert_eq!(results.len(), end - start);

                            for i in 0..results.len() {
                                let expected = array_search_window(
                                    data[i + start].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.into_iter());
                                let results: HashSet<(usize, usize)> =
                                    HashSet::from_iter(results[i].clone().into_iter());
                                assert_eq!(results, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}
