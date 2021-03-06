use ndarray::Array2;
use num_traits::PrimInt;
use std::fmt::Debug;
use std::io;
use std::io::{Read, Write};

use super::log::Log;
use super::snapshot::Snapshot;
use crate::extio::{ExtendedRead, ExtendedWrite};

/// A short series of time instants made up of one Snapshot encoding the first time instant and
/// Logs encoding subsequent time instants.
///
pub struct Block<I>
where
    I: PrimInt + Debug,
{
    /// Snapshot of first time instant
    pub snapshot: Snapshot<I>,

    /// Successive time instants as logs
    pub logs: Vec<Log<I>>,
}

impl<I> Block<I>
where
    I: PrimInt + Debug,
{
    /// Construct a Block from a Snapshot and a series of Logs based on the Snapshot
    ///
    pub fn new(snapshot: Snapshot<I>, logs: Vec<Log<I>>) -> Self {
        if logs.len() > 254 {
            panic!(
                "Too many logs in one block. Maximum is 254. {} passed.",
                logs.len()
            );
        }

        Self {
            snapshot: snapshot,
            logs: logs,
        }
    }

    /// Get the cell value at the given time instant, row, and column
    ///
    pub fn get(&self, instant: usize, row: usize, col: usize) -> I
    where
        I: PrimInt + Debug,
    {
        match instant {
            0 => self.snapshot.get(row, col),
            _ => self.logs[instant - 1].get(&self.snapshot, row, col),
        }
    }

    /// Get a subarray of the given instant
    ///
    /// This will allocate a new `ndarray::Array2` to hold the subarray. This is called by
    /// `chunk::Chunk.iter_window` to iterate over time instants one at a time. To retrieve all
    /// time instants of interest at once, it is more efficient to preallocate an `ndarray::Array3`
    /// and call `block.fill_window` for each instant of interest, which is the strategy used by
    /// `chunk::Chunk.get_window`.
    ///
    pub fn get_window(
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
        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array2::zeros([rows, cols]);
        let set = |row, col, value| window[[row, col]] = I::from(value).unwrap();

        self.fill_window(set, instant, top, bottom, left, right);

        window
    }

    /// Retrieve a subarray of the given instant and write it to a preallocated array.
    ///
    /// The array to write to isn't passed in explicitly. Instead a setter function, `set`, is
    /// passed to write the value. This indirection provides a means of injecting data conversion
    /// from the stored i64 to whatever the destination data type is. This is used, for instance,
    /// to convert from fixed point to floating point representation for floating point datasets.
    ///
    pub fn fill_window<S>(
        &self,
        set: S,
        instant: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) where
        S: FnMut(usize, usize, i64),
    {
        match instant {
            0 => self.snapshot.fill_window(set, top, bottom, left, right),
            _ => self.logs[instant - 1].fill_window(set, &self.snapshot, top, bottom, left, right),
        }
    }

    /// Search within a window for cells with values in a given range.
    ///
    /// Returns a vector of (row, col) pairs.
    ///
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

    /// Write a block to a stream
    ///
    pub fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_byte((self.logs.len() + 1) as u8)?;
        self.snapshot.serialize(stream)?;
        for log in &self.logs {
            log.serialize(stream)?;
        }
        Ok(())
    }

    /// Read a block from a stream
    ///
    pub fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let n_instants = stream.read_byte()? as usize;
        let snapshot = Snapshot::deserialize(stream)?;
        let mut logs: Vec<Log<I>> = Vec::with_capacity(n_instants - 1);
        for _ in 0..n_instants - 1 {
            let log = Log::deserialize(stream)?;
            logs.push(log);
        }

        Ok(Self { snapshot, logs })
    }

    /// Return the number of bytes in the serialized representation of the block
    pub fn size(&self) -> u64 {
        1 // number of instants
        + self.snapshot.size()
        + self.logs.iter().map(|l| l.size()).sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::array_search_window;
    use super::*;
    use ndarray::{arr3, s, Array3};
    use std::collections::HashSet;
    use std::io::Seek;
    use tempfile::tempfile;

    fn array8() -> Array3<i32> {
        arr3(&[
            [
                [9, 8, 7, 7, 6, 6, 3, 2],
                [7, 7, 7, 7, 6, 6, 3, 3],
                [6, 6, 6, 6, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 3, 5, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4],
            ],
            [
                [9, 8, 7, 7, 7, 7, 2, 2],
                [7, 7, 7, 7, 7, 7, 2, 2],
                [6, 6, 6, 6, 4, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 4, 5, 5, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ],
            [
                [9, 8, 7, 7, 8, 7, 5, 5],
                [7, 7, 7, 7, 7, 7, 5, 5],
                [7, 7, 6, 6, 4, 3, 4, 4],
                [6, 6, 6, 6, 4, 4, 4, 4],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 4, 5, 6, 4, 4, 4],
                [4, 4, 4, 4, 5, 4, 4, 4],
            ],
        ])
    }

    fn too_many_logs() -> io::Result<()> {
        let data = array8();
        let snapshot = data.slice(s![0, .., ..]);
        let logs = vec![
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
            data.slice(s![0, .., ..]),
        ];
        let logs = logs.into_iter().cycle().take(300);

        // This statement should panic, as the max number of logs for one block is 254
        let block: Block<i32> = Block::new(
            Snapshot::from_array(snapshot, 2),
            logs.into_iter()
                .map(|log| Log::from_arrays(snapshot, log, 2))
                .collect(),
        );

        let mut file = tempfile()?;
        block.serialize(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        // The number of time instants in the block is written as a single byte, so the maximum
        // number lof logs we can actually store in this block is 254. If we're allowed to actually
        // create the block, then serializing and deserializing will lead to incorrect behavior.
        let block: Block<i32> = Block::deserialize(&mut file)?;
        assert_eq!(block.logs.len(), 300);

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_too_many_logs() {
        too_many_logs().expect("should panic");
    }

    #[test]
    fn get() {
        let data = array8();
        let data = vec![
            data.slice(s![0, .., ..]),
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
        ];

        let block: Block<i32> = Block::new(
            Snapshot::from_array(data[0], 2),
            vec![
                Log::from_arrays(data[0], data[1], 2),
                Log::from_arrays(data[0], data[2], 2),
            ],
        );

        for t in 0..3 {
            for r in 0..8 {
                for c in 0..8 {
                    assert_eq!(block.get(t, r, c), data[t][[r, c]]);
                }
            }
        }
    }

    #[test]
    fn get_window() {
        let data = array8();
        let data = vec![
            data.slice(s![0, .., ..]),
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
        ];

        let block: Block<i32> = Block::new(
            Snapshot::from_array(data[0], 2),
            vec![
                Log::from_arrays(data[0], data[1], 2),
                Log::from_arrays(data[0], data[2], 2),
            ],
        );

        for t in 0..3 {
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            let expected = data[t].slice(s![top..bottom, left..right]);
                            assert_eq!(block.get_window(t, top, bottom, left, right), expected,);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window() {
        let data = array8();
        let data = vec![
            data.slice(s![0, .., ..]),
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
        ];

        let block = Block::new(
            Snapshot::from_array(data[0], 2),
            vec![
                Log::from_arrays(data[0], data[1], 2),
                Log::from_arrays(data[0], data[2], 2),
            ],
        );

        for t in 0..3 {
            for top in 0..8 {
                for bottom in top + 1..=8 {
                    for left in 0..8 {
                        for right in left + 1..=8 {
                            for lower in 0..10 {
                                for upper in lower..10 {
                                    let expected = array_search_window(
                                        data[t], top, bottom, left, right, lower, upper,
                                    );
                                    let expected: HashSet<(usize, usize)> =
                                        HashSet::from_iter(expected.into_iter());
                                    let cells = block
                                        .search_window(t, top, bottom, left, right, lower, upper);
                                    let cells = HashSet::from_iter(cells.into_iter());
                                    assert_eq!(cells, expected);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
