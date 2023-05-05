use std::{cell::RefCell, mem::replace, rc::Rc, vec::IntoIter as VecIntoIter};

use async_trait::async_trait;
use cid::Cid;
use futures::io::{AsyncRead, AsyncWrite};

use crate::{
    block::Block,
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
    helpers::rearrange,
    log::Log,
    mmbuffer::{MMBuffer0, MMBuffer1, MMBuffer3},
    mmstruct::{MMEncoding, MMStruct3, MMStruct3Build},
    snapshot::Snapshot,
};

/// A series of time instants stored in a single file on disk.
///
/// Made up of a series of blocks.
///
pub struct Chunk {
    /// Stored data
    blocks: Vec<Block>,

    /// Index into stored data for finding which block contains a particular time instant
    index: Vec<usize>,

    /// The type of numeric data encoded in this structure.
    ///
    pub encoding: MMEncoding,

    /// Get the number of fractional bits used in encoding this structure.
    /// This is only meaningful for floating point type encodings. Integer encodings will return 0.
    ///
    pub fractional_bits: usize,
}

impl Chunk {
    pub(crate) fn build(buffer: &mut MMBuffer3, shape: [usize; 3], k: i32) -> MMStruct3Build {
        let mut count_snapshots = 0;
        let mut count_logs = 0;

        let [instants, rows, cols] = shape;
        let shape2 = [rows, cols];
        let mut blocks = vec![];

        let first_get = |row, col| buffer.get(0, row, col);
        let mut snapshot = Snapshot::build(first_get, shape2, k);
        let mut snapshot_index = 0;
        let mut logs = vec![];

        for i in 1..instants {
            let get_t = |row, col| buffer.get(i, row, col);
            let new_snapshot = Snapshot::build(get_t, shape2, k);

            let get_s = |row, col| buffer.get(snapshot_index, row, col);
            let new_log = Log::build(get_s, get_t, shape2, k);

            if logs.len() == 254 || new_snapshot.size() <= new_log.size() {
                count_snapshots += 1;
                count_logs += logs.len();

                let block_snapshot = replace(&mut snapshot, new_snapshot);
                let block_logs = replace(&mut logs, vec![]);

                snapshot_index = i;
                blocks.push(Block::new(block_snapshot, block_logs));
            } else {
                logs.push(new_log);
            }
        }

        count_snapshots += 1;
        count_logs += logs.len();
        blocks.push(Block::new(snapshot, logs));

        let chunk = MMStruct3::Subchunk(Chunk::new(
            blocks,
            buffer.encoding(),
            buffer.fractional_bits(),
        ));
        let size = chunk.size();

        MMStruct3Build {
            data: chunk,
            size,
            elided: 0,
            local: 0,
            external: 0,
            logs: count_logs,
            snapshots: count_snapshots,
        }
    }

    /// Make a new Chunk from a vector of Blocks
    ///
    pub(crate) fn new(blocks: Vec<Block>, encoding: MMEncoding, fractional_bits: usize) -> Self {
        let mut index = Vec::with_capacity(blocks.len());
        let mut count = 0;
        for block in &blocks {
            count += block.logs.len() + 1;
            index.push(count);
        }

        Self {
            blocks,
            index,
            encoding,
            fractional_bits,
        }
    }
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
    pub(crate) fn get(&self, instant: usize, row: usize, col: usize, buffer: &mut MMBuffer0) {
        let (block, instant) = self.find_block(instant);
        let block = &self.blocks[block];
        buffer.set(block.get(instant, row, col));
    }

    /// Fill in a preallocated array with cell's value across time instants.
    ///
    pub(crate) fn fill_cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
        buffer: &mut MMBuffer1,
    ) {
        for (i, (block, instant)) in self.iter(start, end).enumerate() {
            let block = &self.blocks[block];
            let value = block.get(instant, row, col);
            buffer.set(i, value);
        }
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    pub(crate) fn fill_window(&self, bounds: geom::Cube, buffer: &mut MMBuffer3) {
        for (i, (block, instant)) in self.iter(bounds.start, bounds.end).enumerate() {
            let set2d = |row, col, value| buffer.set(i, row, col, value);
            let block = &self.blocks[block];
            block.fill_window(set2d, instant, &bounds.rect());
        }
    }

    pub(crate) fn ls(&self) -> Vec<(String, Cid)> {
        vec![]
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
    fn iter<'a>(&'a self, start: usize, end: usize) -> ChunkIter<'a> {
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
    pub(crate) fn iter_search(&self, bounds: &geom::Cube, lower: i64, upper: i64) -> SearchIter {
        let (lower, upper) = rearrange(lower, upper);

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
}

#[async_trait]
impl Serialize for Chunk {
    /// Write a chunk to a stream
    ///
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_byte(self.encoding as u8).await?;
        stream.write_byte(self.fractional_bits as u8).await?;
        stream.write_u32(self.blocks.len() as u32).await?;
        for block in &self.blocks {
            block.write_to(stream).await?;
        }
        Ok(())
    }

    /// Read a chunk from a stream
    ///
    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let encoding = MMEncoding::try_from(stream.read_byte().await?)?;
        let fractional_bits = stream.read_byte().await? as usize;
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
        Ok(Self {
            blocks,
            index,
            encoding,
            fractional_bits,
        })
    }
}

impl Cacheable for Chunk {
    /// Return the number of bytes in the serialized representation
    ///
    fn size(&self) -> u64 {
        1   // encoding
        + 1 // fractional_bits
        + 4 // number of blocks
        + self.blocks.iter().map(|b| b.size()).sum::<u64>()
    }
}

/// Iterate over time instants stored in this chunk across several blocks
///
/// Used internally by the other iterators.
///
struct ChunkIter<'a> {
    chunk: &'a Chunk,
    block: usize,
    instant: usize,
    remaining: usize,
}

impl<'a> Iterator for ChunkIter<'a> {
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

pub(crate) struct CellIter<'a> {
    iter: Rc<RefCell<ChunkIter<'a>>>,
    row: usize,
    col: usize,
}

impl<'a> Iterator for CellIter<'a> {
    type Item = i64;

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

pub(crate) struct SearchIter<'a> {
    iter: Rc<RefCell<ChunkIter<'a>>>,
    bounds: geom::Rect,
    lower: i64,
    upper: i64,

    instant: usize,
    results: Option<VecIntoIter<(usize, usize)>>,
}

impl<'a> SearchIter<'a> {
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

impl<'a> Iterator for SearchIter<'a> {
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
    use super::*;
    use crate::{
        log::Log,
        snapshot::Snapshot,
        testing::{array8, array_search_window2},
    };
    use futures::io::Cursor;
    use ndarray::{s, Array1, Array3};
    use std::collections::HashSet;

    fn chunk(array: Array3<i64>) -> Chunk {
        let mut current = Vec::with_capacity(4);
        let mut blocks: Vec<Block> = vec![];

        for i in 0..array.shape()[0] {
            current.push(array.slice(s![i, .., ..]));
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

        Chunk::new(blocks, MMEncoding::I32, 0)
    }

    #[test]
    fn from_blocks() {
        let chunk = chunk(array8());
        assert_eq!(chunk.blocks.len(), 25);
        assert_eq!(chunk.shape(), [100, 8, 8]);
    }

    #[test]
    fn get() {
        let data = array8();
        let chunk = chunk(data.clone());
        for instant in 0..100 {
            for row in 0..8 {
                for col in 0..8 {
                    let mut buffer = MMBuffer0::I64(0);
                    chunk.get(instant, row, col, &mut buffer);
                    assert_eq!(i64::from(buffer), data[[instant, row, col]]);
                }
            }
        }
    }

    #[test]
    fn fill_cell() {
        let data = array8();
        let chunk = chunk(data.clone());
        for row in 0..8 {
            for col in 0..8 {
                let start = row * 6;
                let end = 100 - col * 6;
                let mut output = Array1::zeros([end - start]);
                let mut buffer = MMBuffer1::new_i64(output.view_mut());
                chunk.fill_cell(start, end, row, col, &mut buffer);

                assert_eq!(output, data.slice(s![start..end, row, col]));
            }
        }
    }

    #[test]
    fn fill_window() {
        let data = array8();
        let chunk = chunk(data.clone());
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let start = top * left;
                        let end = bottom * right + 36;
                        let mut array = Array3::zeros([end - start, bottom - top, right - left]);
                        let mut buffer = MMBuffer3::new_i64(array.view_mut());

                        let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                        chunk.fill_window(bounds, &mut buffer);

                        assert_eq!(array, data.slice(s![start..end, top..bottom, left..right]));
                    }
                }
            }
        }
    }

    #[test]
    fn iter_search() {
        let data = array8();
        let chunk = chunk(data.clone());
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let start = top * left;
                        let end = bottom * right + 36;
                        let lower = (start / 5).try_into().unwrap();
                        let upper = (end / 10).try_into().unwrap();

                        let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                        for i in start..end {
                            let coords = array_search_window2(
                                data.slice(s![i, .., ..]),
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
        let data = array8();
        let chunk = chunk(data.clone());
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let start = top * left;
                        let end = bottom * right + 36;
                        let lower = (start / 5).try_into().unwrap();
                        let upper = (end / 10).try_into().unwrap();

                        let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                        for i in start..end {
                            let coords = array_search_window2(
                                data.slice(s![i, .., ..]),
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

    #[tokio::test]
    async fn serialize_deserialize() -> Result<()> {
        let data = array8();
        let chunk = chunk(data.clone());

        let mut buffer: Vec<u8> = Vec::with_capacity(chunk.size() as usize);
        chunk.write_to(&mut buffer).await?;
        assert_eq!(buffer.len(), chunk.size() as usize);

        let mut buffer = Cursor::new(buffer);
        let chunk = Chunk::read_from(&mut buffer).await?;

        for row in 0..8 {
            for col in 0..8 {
                let start = row * col;
                let end = 100 - col;
                for instant in start..end {
                    let mut buffer = MMBuffer0::I64(0);
                    chunk.get(instant, row, col, &mut buffer);
                    assert_eq!(i64::from(buffer), data[[instant, row, col]]);
                }
            }
        }

        Ok(())
    }
}
