//! Encode/Decode Heuristic K²-Raster
//!
//! An implementation of the compact data structure proposed by Silva-Coira, et al.[^bib1],
//! which, in turn, is based on work by Ladra[^bib2] and González[^bib3].
//!
//! The data structures here provide a means of storing raster data compactly while still being
//! able to run queries in-place on the stored data. A separate decompression step is not required
//! in order to read the data.
//!
//! For insight into how this data structure works, please see the literature in footnotes.
//! Reproducing the literature is outside of the scope for this documentation.
//!
//! [^bib1]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient representations
//!     of raster time series, Information Sciences 566 (2021) 300-325.][1]
//!
//! [^bib2]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
//!     structure for raster data, Information Systems 72 (2017) 179-204.
//!
//! [^bib3]: [F. González, S. Grabowski, V. Mäkinen, G. Navarro, Practical implementations of rank
//!     and select queries, in: Poster Proc. of 4th Workshop on Efficient and Experimental
//!     Algorithms (WEA) Greece, 2005, pp. 27-38.][2]
//!
//! [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
//! [2]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9548&rep=rep1&type=pdf

mod bitmap;
mod dac;
mod helpers;
mod log;
mod snapshot;

pub use log::Log;
pub use snapshot::Snapshot;

use ndarray::Array2;
use num_traits::{Float, PrimInt};
use std::fmt::Debug;
use std::io;
use std::io::{Read, Write};
use std::marker::PhantomData;

use super::extio::{ExtendedRead, ExtendedWrite};
use super::fixed;

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

/// A short series of time instants made up of one Snapshot encoding the first time instant and
/// Logs encoding subsequent time instants.
///
pub struct Block<I>
where
    I: PrimInt + Debug,
{
    /// Snapshot of first time instant
    snapshot: Snapshot<I>,

    /// Successive time instants as logs
    logs: Vec<Log<I>>,
}

impl<I> Block<I>
where
    I: PrimInt + Debug,
{
    /// Snapshot of first time instant
    pub fn new(snapshot: Snapshot<I>, logs: Vec<Log<I>>) -> Self {
        Self {
            snapshot: snapshot,
            logs: logs,
        }
    }

    fn get(&self, instant: usize, row: usize, col: usize) -> I
    where
        I: PrimInt + Debug,
    {
        match instant {
            0 => self.snapshot.get(row, col),
            _ => self.logs[instant - 1].get(&self.snapshot, row, col),
        }
    }

    fn get_window(
        &self,
        instant: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array2<I>
    where
        I: PrimInt + Debug,
    {
        match instant {
            0 => self.snapshot.get_window(top, bottom, left, right),
            _ => self.logs[instant - 1].get_window(&self.snapshot, top, bottom, left, right),
        }
    }

    pub fn search_window(
        &self,
        instant: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: I,
        upper: I,
    ) -> Vec<(usize, usize)>
    where
        I: PrimInt + Debug,
    {
        match instant {
            0 => self
                .snapshot
                .search_window(top, bottom, left, right, lower, upper),
            _ => self.logs[instant - 1].search_window(
                &self.snapshot,
                top,
                bottom,
                left,
                right,
                lower,
                upper,
            ),
        }
    }

    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_byte((self.logs.len() + 1) as u8)?;
        self.snapshot.serialize(stream)?;
        for log in &self.logs {
            log.serialize(stream)?;
        }
        Ok(())
    }

    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let n_instants = stream.read_byte()? as usize;
        let snapshot = Snapshot::deserialize(stream)?;
        let mut logs: Vec<Log<I>> = Vec::with_capacity(n_instants - 1);
        for _ in 0..n_instants - 1 {
            let log = Log::deserialize(stream)?;
            logs.push(log);
        }

        Ok(Self { snapshot, logs })
    }

    pub fn size(&self) -> u64 {
        1 + self.snapshot.size() + self.logs.iter().map(|l| l.size()).sum::<u64>()
    }
}

#[cfg(test)]
mod tests;
