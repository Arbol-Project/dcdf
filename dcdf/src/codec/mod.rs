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

use bitmap::{BitMap, BitMapBuilder};
use dac::Dac;

use ndarray::Array2;
use num_traits::{Float, PrimInt};
use std::cmp::min;
use std::collections::VecDeque;
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

/// K²-Raster encoded Snapshot
///
/// A Snapshot stores raster data for a particular time instant in a raster time series. Data is
/// stored standalone without reference to any other time instant.
///
pub struct Snapshot<I>
where
    I: PrimInt + Debug,
{
    _marker: PhantomData<I>,

    /// Bitmap of tree structure, known as T in Silva-Coira
    nodemap: BitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira
    max: Dac,

    /// Tree node minimum values, known as Lmin in Silva-Coira
    min: Dac,

    /// The K in K²-Raster. Each level of the tree structure is divided into k² subtrees.
    /// In practice, this will almost always be 2.
    k: i32,

    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl<I> Snapshot<I>
where
    I: PrimInt + Debug,
{
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_byte(self.k as u8)?;
        stream.write_u32(self.shape[0] as u32)?;
        stream.write_u32(self.shape[1] as u32)?;
        stream.write_u32(self.sidelen as u32)?;
        self.nodemap.serialize(stream)?;
        self.max.serialize(stream)?;
        self.min.serialize(stream)?;

        Ok(())
    }

    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let k = stream.read_byte()? as i32;
        let shape = [stream.read_u32()? as usize, stream.read_u32()? as usize];
        let sidelen = stream.read_u32()? as usize;
        let nodemap = BitMap::deserialize(stream)?;
        let max = Dac::deserialize(stream)?;
        let min = Dac::deserialize(stream)?;

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

    pub fn size(&self) -> u64 {
        1 + 4 + 4 + 4 + self.nodemap.size() + self.max.size() + self.min.size()
    }

    /// Build a snapshot from a two-dimensional array.
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
        to_traverse.push_back((root.max, root.min, &root));

        // Breadth first traversal
        while let Some((diff_max, diff_min, child)) = to_traverse.pop_front() {
            max.push(diff_max);

            if !child.children.is_empty() {
                // Non-leaf node
                let elide = child.min == child.max;
                nodemap.push(!elide);
                if !elide {
                    min.push(diff_min);
                    for descendant in &child.children {
                        to_traverse.push_back((
                            child.max - descendant.max,
                            descendant.min - child.min,
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
        self.check_bounds(row, col);

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
    /// [^note]: S. Ladra, J.R. Paramá, F. Silva-Coira, Scalable and queryable compressed storage
    ///     structure for raster data, Information Systems 72 (2017) 179-204.
    ///
    pub fn get_window(&self, top: usize, bottom: usize, left: usize, right: usize) -> Array2<I> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom - 1, right - 1);

        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);

        if !self.nodemap.get(0) {
            // Special case: single node tree
            let value = self.max.get(0);
            for row in 0..rows {
                for col in 0..cols {
                    window[[row, col]] = value;
                }
            }
        } else {
            self._get_window(
                self.sidelen,
                top,
                bottom - 1,
                left,
                right - 1,
                0,
                self.max.get(0),
                &mut window,
                top,
                left,
                0,
                0,
            );
        }

        window
    }

    fn _get_window(
        &self,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index: usize,
        max_value: I,
        window: &mut Array2<I>,
        window_top: usize,
        window_left: usize,
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
                    for row in top_..=bottom_ {
                        for col in left_..=right_ {
                            window[[
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                            ]] = max_value_;
                        }
                    }
                } else {
                    // Branch
                    self._get_window(
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_,
                        max_value_,
                        window,
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
    pub fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: I,
        upper: I,
    ) -> Vec<(usize, usize)> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(bottom - 1, right - 1);

        let mut cells: Vec<(usize, usize)> = vec![];

        if !self.nodemap.get(0) {
            // Special case: single node tree
            let value: I = self.max.get(0);
            if lower <= value && value <= upper {
                for row in top..bottom {
                    for col in left..right {
                        cells.push((row, col));
                    }
                }
            }
        } else {
            self._search_window(
                self.sidelen,
                top,
                bottom - 1,
                left,
                right - 1,
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

    /// Panics if given point is out of bounds for this snapshot
    fn check_bounds(&self, row: usize, col: usize) {
        if row >= self.shape[0] || col >= self.shape[1] {
            panic!(
                "dcdf::Snapshot: index[{}, {}] is out of bounds for array of shape {:?}",
                row, col, self.shape
            );
        }
    }
}

/// Temporary tree structure for building K^2 raster
struct K2TreeNode {
    max: i64,
    min: i64,
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
            // Fill cells that lay outside of original raster with 0s
            let [rows, cols] = shape;
            let value = if row < rows && col < cols {
                get(row, col)
            } else {
                0
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
            if child.max > max {
                max = child.max;
            }
            if child.min < min {
                min = child.min;
            }
        }

        K2TreeNode { min, max, children }
    }
}

/// K²-Raster encoded Log
///
/// A Log stores raster data for a particular time instant in a raster time series as the
/// difference between this time instant and a reference Snapshot.
///
pub struct Log<I>
where
    I: PrimInt + Debug,
{
    _marker: PhantomData<I>,

    /// Bitmap of tree structure, known as T in Silva-Coira
    nodemap: BitMap,

    /// Bitmap of tree nodes that match referenced snapshot, or have cells that all differ by the
    /// same amount, known as eqB in Silva-Coira paper
    equal: BitMap,

    /// Tree node maximum values, known as Lmax in Silva-Coira
    max: Dac,

    /// Tree node minimum values, known as Lmin in Silva-Coira
    min: Dac,

    /// The K in K²-Raster. Each level of the tree structure is divided into k² subtrees.
    /// In practice, this will almost always be 2.
    k: i32,

    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 2],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,
}

impl<I> Log<I>
where
    I: PrimInt + Debug,
{
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_byte(self.k as u8)?;
        stream.write_u32(self.shape[0] as u32)?;
        stream.write_u32(self.shape[1] as u32)?;
        stream.write_u32(self.sidelen as u32)?;
        self.nodemap.serialize(stream)?;
        self.equal.serialize(stream)?;
        self.max.serialize(stream)?;
        self.min.serialize(stream)?;

        Ok(())
    }

    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let k = stream.read_byte()? as i32;
        let shape = [stream.read_u32()? as usize, stream.read_u32()? as usize];
        let sidelen = stream.read_u32()? as usize;
        let nodemap = BitMap::deserialize(stream)?;
        let equal = BitMap::deserialize(stream)?;
        let max = Dac::deserialize(stream)?;
        let min = Dac::deserialize(stream)?;

        Ok(Self {
            _marker: PhantomData,
            nodemap,
            equal,
            max,
            min,
            k,
            shape,
            sidelen,
        })
    }

    pub fn size(&self) -> u64 {
        1 + 4 + 4 + 4 + self.nodemap.size() + self.equal.size() + self.max.size() + self.min.size()
    }

    /// Build a log from a pair of two-dimensional arrays.
    ///
    pub fn build<GS, GT>(get_s: GS, get_t: GT, shape: [usize; 2], k: i32) -> Self
    where
        GS: Fn(usize, usize) -> i64,
        GT: Fn(usize, usize) -> i64,
    {
        let mut nodemap = BitMapBuilder::new();
        let mut equal = BitMapBuilder::new();
        let mut max: Vec<i64> = vec![];
        let mut min: Vec<i64> = vec![];

        // Compute the smallest square with sides whose length is a power of K that will contain
        // the passed in data.
        let sidelen = *shape.iter().max().unwrap() as f64;
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let root = K2PTreeNode::build(get_s, get_t, shape, k, sidelen);
        let mut to_traverse = VecDeque::new();
        to_traverse.push_back(&root);

        // Breadth first traversal
        while let Some(node) = to_traverse.pop_front() {
            max.push(node.max_t - node.max_s);

            if !node.children.is_empty() {
                // Non-leaf node
                if node.min_t == node.max_t {
                    // Log quadbox is uniform, terminate here
                    nodemap.push(false);
                    equal.push(false);
                } else if node.equal {
                    // Difference of log and snapshot quadboxes is uniform, terminate here
                    nodemap.push(false);
                    equal.push(true);
                } else {
                    // Regular old internal node, keep going
                    nodemap.push(true);
                    min.push(node.min_t - node.min_s);
                    for child in &node.children {
                        to_traverse.push_back(child);
                    }
                }
            }
        }

        Log {
            _marker: PhantomData,
            nodemap: nodemap.finish(),
            equal: equal.finish(),
            max: Dac::from(max),
            min: Dac::from(min),
            k,
            shape: [shape[0], shape[1]],
            sidelen,
        }
    }

    /// Get a cell value
    ///
    /// See: Algorithm 3 in Silva-Coira[^note]
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub fn get(&self, snapshot: &Snapshot<I>, row: usize, col: usize) -> I {
        self.check_bounds(row, col);

        let max_t: I = self.max.get(0);
        let max_s: I = snapshot.max.get(0);
        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);
        if single_t && single_s {
            // Both trees have single node
            max_t + max_s
        } else if single_t && !self.equal.get(0) {
            // Log has single node but it contains a uniform value for all cells
            max_t + max_s
        } else {
            let index_t = if single_t { None } else { Some(0) };
            let index_s = if single_s { None } else { Some(0) };
            let value = self._get(
                snapshot,
                self.sidelen,
                row,
                col,
                index_t,
                index_s,
                max_t.to_i64().unwrap(),
                max_s.to_i64().unwrap(),
            );

            I::from(value).unwrap()
        }
    }

    fn _get(
        &self,
        snapshot: &Snapshot<I>,
        sidelen: usize,
        row: usize,
        col: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: i64,
        max_s: i64,
    ) -> i64 {
        let k = self.k as usize;
        let sidelen = sidelen / k;
        let mut max_s = max_s;
        let mut max_t = max_t;

        let index_s = match index_s {
            Some(index) => {
                let index = 1 + snapshot.nodemap.rank(index) * k * k;
                let index = index + row / sidelen * k + col / sidelen;
                max_s = max_s - snapshot.max.get::<i64>(index);
                Some(index)
            }
            None => None,
        };

        let index_t = match index_t {
            Some(index) => {
                let index = 1 + self.nodemap.rank(index) * k * k;
                let index = index + row / sidelen * k + col / sidelen;
                max_t = self.max.get(index);
                Some(index)
            }
            None => None,
        };

        let leaf_t = match index_t {
            Some(index) => index > self.nodemap.length || !self.nodemap.get(index),
            None => true,
        };

        let leaf_s = match index_s {
            Some(index) => index > snapshot.nodemap.length || !snapshot.nodemap.get(index),
            None => true,
        };

        if leaf_t && leaf_s {
            max_t + max_s
        } else if leaf_s {
            self._get(
                snapshot,
                sidelen,
                row % sidelen,
                col % sidelen,
                index_t,
                None,
                max_t,
                max_s,
            )
        } else if leaf_t {
            if let Some(index_t) = index_t {
                if index_t < self.nodemap.length {
                    let equal = self.equal.get(self.nodemap.rank0(index_t + 1) - 1);
                    if !equal {
                        return max_t + max_s;
                    }
                }
            }
            self._get(
                snapshot,
                sidelen,
                row % sidelen,
                col % sidelen,
                None,
                index_s,
                max_t,
                max_s,
            )
        } else {
            self._get(
                snapshot,
                sidelen,
                row % sidelen,
                col % sidelen,
                index_t,
                index_s,
                max_t,
                max_s,
            )
        }
    }

    /// Get a subarray of log
    ///
    /// This is based on Algorithm 5 in Silva-Coira[^note], but has been modified to return a
    /// submatrix rather than an unordered sequence of values.
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub fn get_window(
        &self,
        snapshot: &Snapshot<I>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> Array2<I> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        self.check_bounds(bottom - 1, right - 1);

        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);

        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);

        if single_t && (single_s || !self.equal.get(0)) {
            // Both trees have single node or log has single node but it contains a uniform value
            // for all cells
            let max_t: I = self.max.get(0);
            let max_s: I = snapshot.max.get(0);
            for row in 0..rows {
                for col in 0..cols {
                    window[[row, col]] = max_t + max_s;
                }
            }
        } else {
            self._get_window(
                snapshot,
                self.sidelen,
                top,
                bottom - 1,
                left,
                right - 1,
                if single_t { None } else { Some(0) },
                if single_s { None } else { Some(0) },
                self.max.get(0),
                snapshot.max.get(0),
                &mut window,
                top,
                left,
                0,
                0,
            );
        }

        window
    }

    fn _get_window(
        &self,
        snapshot: &Snapshot<I>,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        index_t: Option<usize>,
        index_s: Option<usize>,
        max_t: i64,
        max_s: i64,
        window: &mut Array2<I>,
        window_top: usize,
        window_left: usize,
        top_offset: usize,
        left_offset: usize,
    ) {
        let k = self.k as usize;
        let sidelen = sidelen / k;

        let index_t = match index_t {
            Some(index) => Some(1 + self.nodemap.rank(index) * k * k),
            None => None,
        };

        let index_s = match index_s {
            Some(index) => Some(1 + snapshot.nodemap.rank(index) * k * k),
            None => None,
        };

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let index_t_ = match index_t {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let max_t_ = match index_t_ {
                    Some(index) => self.max.get(index),
                    None => max_t,
                };

                let leaf_t = match index_t_ {
                    Some(index) => index > self.nodemap.length || !self.nodemap.get(index),
                    None => true,
                };

                let index_s_ = match index_s {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let max_s_ = match index_s_ {
                    Some(index) => max_s - snapshot.max.get::<i64>(index),
                    None => max_s,
                };

                let leaf_s = match index_s_ {
                    Some(index) => index > snapshot.nodemap.length || !snapshot.nodemap.get(index),
                    None => true,
                };

                if leaf_t && leaf_s {
                    let value = max_t_ + max_s_;
                    for row in top_..=bottom_ {
                        for col in left_..=right_ {
                            window[[
                                top_offset_ + row - window_top,
                                left_offset_ + col - window_left,
                            ]] = I::from(value).unwrap();
                        }
                    }
                } else if leaf_s {
                    self._get_window(
                        snapshot,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_t_,
                        None,
                        max_t_,
                        max_s_,
                        window,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                    );
                } else if leaf_t {
                    if let Some(index) = index_t_ {
                        if !self.nodemap.get(index) {
                            let equal = self.equal.get(self.nodemap.rank0(index + 1) - 1);
                            if !equal {
                                let value = max_t_ + max_s_;
                                for row in top_..=bottom_ {
                                    for col in left_..=right_ {
                                        window[[
                                            top_offset_ + row - window_top,
                                            left_offset_ + col - window_left,
                                        ]] = I::from(value).unwrap();
                                    }
                                }
                                continue;
                            }
                        }
                    }
                    self._get_window(
                        snapshot,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        None,
                        index_s_,
                        max_t_,
                        max_s_,
                        window,
                        window_top,
                        window_left,
                        top_offset_,
                        left_offset_,
                    );
                } else {
                    self._get_window(
                        snapshot,
                        sidelen,
                        top_,
                        bottom_,
                        left_,
                        right_,
                        index_t_,
                        index_s_,
                        max_t_,
                        max_s_,
                        window,
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
    /// See: Algorithm 7 in Silva-Coira[^note]
    ///
    /// [^note]: [F. Silva-Coira, J.R. Paramá, G. de Bernardo, D. Seco, Space-efficient
    ///     representations of raster time series, Information Sciences 566 (2021) 300-325.][1]
    ///
    /// [1]: https://index.ggws.net/downloads/2021-06-18/91/silva-coira2021.pdf
    ///
    pub fn search_window(
        &self,
        snapshot: &Snapshot<I>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: I,
        upper: I,
    ) -> Vec<(usize, usize)> {
        let (left, right) = rearrange(left, right);
        let (top, bottom) = rearrange(top, bottom);
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(bottom - 1, right - 1);

        let mut cells: Vec<(usize, usize)> = vec![];
        let single_t = !self.nodemap.get(0);
        let single_s = !snapshot.nodemap.get(0);

        self._search_window(
            snapshot,
            self.sidelen,
            top,
            bottom - 1,
            left,
            right - 1,
            lower.to_i64().unwrap(),
            upper.to_i64().unwrap(),
            if single_t { None } else { Some(0) },
            if single_s { None } else { Some(0) },
            self.min.get(0),
            snapshot.min.get(0),
            self.max.get(0),
            snapshot.max.get(0),
            &mut cells,
            0,
            0,
        );

        cells
    }

    fn _search_window(
        &self,
        snapshot: &Snapshot<I>,
        sidelen: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i64,
        upper: i64,
        index_t: Option<usize>,
        index_s: Option<usize>,
        min_t: i64,
        min_s: i64,
        max_t: i64,
        max_s: i64,
        cells: &mut Vec<(usize, usize)>,
        top_offset: usize,
        left_offset: usize,
    ) {
        let max_value = max_s + max_t;
        let min_value = min_s + min_t;

        if min_value >= lower && max_value <= upper {
            // Branch meets condition, output all cells
            for row in top..=bottom {
                for col in left..=right {
                    cells.push((top_offset + row, left_offset + col));
                }
            }
            return;
        } else if min_value > upper || max_value < lower {
            // No cells in this branch meet the condition
            return;
        }

        let k = self.k as usize;
        let sidelen = sidelen / k;

        let index_t = match index_t {
            Some(index) => Some(1 + self.nodemap.rank(index) * k * k),
            None => None,
        };

        let index_s = match index_s {
            Some(index) => Some(1 + snapshot.nodemap.rank(index) * k * k),
            None => None,
        };

        for i in top / sidelen..=bottom / sidelen {
            let top_ = top.saturating_sub(i * sidelen);
            let bottom_ = min(sidelen - 1, bottom - i * sidelen);
            let top_offset_ = top_offset + i * sidelen;

            for j in left / sidelen..=right / sidelen {
                let left_ = left.saturating_sub(j * sidelen);
                let right_ = min(sidelen - 1, right - j * sidelen);
                let left_offset_ = left_offset + j * sidelen;

                let mut index_t_ = match index_t {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let mut index_s_ = match index_s {
                    Some(index) => Some(index + i * k + j),
                    None => None,
                };

                let max_t_ = match index_t_ {
                    Some(index) => self.max.get(index),
                    None => max_t,
                };

                let max_s_ = match index_s_ {
                    Some(index) => max_s - snapshot.max.get::<i64>(index),
                    None => max_s,
                };

                let leaf_t = match index_t_ {
                    Some(index) => index >= self.nodemap.length || !self.nodemap.get(index),
                    None => true,
                };

                let leaf_s = match index_s_ {
                    Some(index) => index >= snapshot.nodemap.length || !snapshot.nodemap.get(index),
                    None => true,
                };

                let mut min_t_ = match index_t_ {
                    Some(index) => {
                        if leaf_t {
                            min_t
                        } else {
                            self.min.get(self.nodemap.rank(index))
                        }
                    }
                    None => min_t,
                };

                let mut min_s_ = match index_s_ {
                    Some(index) => {
                        if leaf_s {
                            min_s
                        } else {
                            min_s + snapshot.min.get::<i64>(snapshot.nodemap.rank(index))
                        }
                    }
                    None => min_s,
                };

                if leaf_s {
                    min_s_ = max_s_;
                    index_s_ = None;
                }

                if leaf_t {
                    min_t_ = max_t_;
                    if let Some(index) = index_t_ {
                        if index < self.nodemap.length
                            && !self.equal.get(self.nodemap.rank0(index + 1) - 1)
                        {
                            min_t_ = max_s_ + max_t_ - min_s_;
                        }
                    }
                    index_t_ = None;
                }

                self._search_window(
                    snapshot,
                    sidelen,
                    top_,
                    bottom_,
                    left_,
                    right_,
                    lower,
                    upper,
                    index_t_,
                    index_s_,
                    min_t_,
                    min_s_,
                    max_t_,
                    max_s_,
                    cells,
                    top_offset_,
                    left_offset_,
                );
            }
        }
    }

    /// Panics if given point is out of bounds for this log
    fn check_bounds(&self, row: usize, col: usize) {
        if row >= self.shape[0] || col >= self.shape[1] {
            panic!(
                "dcdf::Log: index[{}, {}] is out of bounds for array of shape {:?}",
                row, col, self.shape
            );
        }
    }
}

// Temporary tree structure for building T - K^2 raster
struct K2PTreeNode {
    max_t: i64,
    min_t: i64,
    max_s: i64,
    min_s: i64,
    diff: i64,
    equal: bool,
    children: Vec<K2PTreeNode>,
}

impl K2PTreeNode {
    fn build<GS, GT>(get_s: GS, get_t: GT, shape: [usize; 2], k: i32, sidelen: usize) -> Self
    where
        GS: Fn(usize, usize) -> i64,
        GT: Fn(usize, usize) -> i64,
    {
        Self::_build(&get_s, &get_t, shape, k as usize, sidelen, 0, 0)
    }

    fn _build<GS, GT>(
        get_s: &GS,
        get_t: &GT,
        shape: [usize; 2],
        k: usize,
        sidelen: usize,
        row: usize,
        col: usize,
    ) -> Self
    where
        GS: Fn(usize, usize) -> i64,
        GT: Fn(usize, usize) -> i64,
    {
        // Leaf node
        if sidelen == 1 {
            // Fill cells that lay outside of original raster with 0s
            let [rows, cols] = shape;
            let value_s = if row < rows && col < cols {
                get_s(row, col)
            } else {
                0
            };
            let value_t = if row < rows && col < cols {
                get_t(row, col)
            } else {
                0
            };
            let diff = value_t - value_s;
            return K2PTreeNode {
                max_t: value_t,
                min_t: value_t,
                max_s: value_s,
                min_s: value_s,
                diff: diff,
                equal: true,
                children: vec![],
            };
        }

        // Branch
        let mut children: Vec<K2PTreeNode> = vec![];
        let sidelen = sidelen / k;
        for i in 0..k {
            let row_ = row + i * sidelen;
            for j in 0..k {
                let col_ = col + j * sidelen;
                children.push(K2PTreeNode::_build(
                    get_s, get_t, shape, k, sidelen, row_, col_,
                ));
            }
        }

        let mut max_t = children[0].max_t;
        let mut min_t = children[0].min_t;
        let mut max_s = children[0].max_s;
        let mut min_s = children[0].min_s;
        let mut equal = children.iter().all(|child| child.equal);
        let diff = children[0].diff;
        for child in &children[1..] {
            if child.max_t > max_t {
                max_t = child.max_t;
            }
            if child.min_t < min_t {
                min_t = child.min_t;
            }
            if child.max_s > max_s {
                max_s = child.max_s;
            }
            if child.min_s < min_s {
                min_s = child.min_s;
            }
            equal = equal && child.diff == diff;
        }

        K2PTreeNode {
            min_t,
            max_t,
            min_s,
            max_s,
            diff,
            equal,
            children,
        }
    }
}

/// Make sure bounds are ordered correctly, eg right is to the right of left, top is above
/// bottom.
fn rearrange<I>(lower: I, upper: I) -> (I, I)
where
    I: PrimInt + Debug,
{
    if lower > upper {
        (upper, lower)
    } else {
        (lower, upper)
    }
}

#[cfg(test)]
mod tests;
