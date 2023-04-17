use async_trait::async_trait;
use futures::io::{AsyncRead, AsyncWrite};
use ndarray::Array2;

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
    {log::Log, snapshot::Snapshot},
};

/// A short series of time instants made up of one Snapshot encoding the first time instant and
/// Logs encoding subsequent time instants.
///
pub(crate) struct Block {
    /// Snapshot of first time instant
    pub(crate) snapshot: Snapshot,

    /// Successive time instants as logs
    pub(crate) logs: Vec<Log>,
}

impl Block {
    /// Construct a Block from a Snapshot and a series of Logs based on the Snapshot
    ///
    pub(crate) fn new(snapshot: Snapshot, logs: Vec<Log>) -> Self {
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
    pub(crate) fn get(&self, instant: usize, row: usize, col: usize) -> i64 {
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
    #[deprecated(note = "should use fill_window at all layers except very top level")]
    pub(crate) fn get_window(&self, instant: usize, bounds: &geom::Rect) -> Array2<i64> {
        let mut window = Array2::zeros([bounds.rows(), bounds.cols()]);
        let set = |row, col, value| window[[row, col]] = value;

        self.fill_window(set, instant, bounds);

        window
    }

    /// Retrieve a subarray of the given instant and write it to a preallocated array.
    ///
    /// The array to write to isn't passed in explicitly. Instead a setter function, `set`, is
    /// passed to write the value. This indirection provides a means of injecting data conversion
    /// from the stored i64 to whatever the destination data type is. This is used, for instance,
    /// to convert from fixed point to floating point representation for floating point datasets.
    ///
    pub(crate) fn fill_window<S>(&self, set: S, instant: usize, bounds: &geom::Rect)
    where
        S: FnMut(usize, usize, i64),
    {
        match instant {
            0 => self.snapshot.fill_window(set, &bounds),
            _ => self.logs[instant - 1].fill_window(set, &self.snapshot, bounds),
        }
    }

    /// Search within a window for cells with values in a given range.
    ///
    /// Returns a vector of (row, col) pairs.
    ///
    pub(crate) fn search_window(
        &self,
        instant: usize,
        bounds: &geom::Rect,
        lower: i64,
        upper: i64,
    ) -> Vec<(usize, usize)> {
        match instant {
            0 => self.snapshot.search_window(bounds, lower, upper),
            _ => self.logs[instant - 1].search_window(&self.snapshot, bounds, lower, upper),
        }
    }
}

#[async_trait]
impl Serialize for Block {
    /// Write a block to a stream
    ///
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        stream.write_byte((self.logs.len() + 1) as u8).await?;
        self.snapshot.write_to(stream).await?;
        for log in &self.logs {
            log.write_to(stream).await?;
        }
        Ok(())
    }

    /// Read a block from a stream
    ///
    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let n_instants = stream.read_byte().await? as usize;
        let snapshot = Snapshot::read_from(stream).await?;
        let mut logs: Vec<Log> = Vec::with_capacity(n_instants - 1);
        for _ in 0..n_instants - 1 {
            let log = Log::read_from(stream).await?;
            logs.push(log);
        }

        Ok(Self { snapshot, logs })
    }
}

impl Cacheable for Block {
    /// Return the number of bytes in the serialized representation of the block
    fn size(&self) -> u64 {
        1 // number of instants
        + self.snapshot.size()
        + self.logs.iter().map(|l| l.size()).sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::array_search_window2;
    use futures::io::Cursor;
    use ndarray::{arr3, s, Array3};
    use std::collections::HashSet;

    fn array8() -> Array3<i64> {
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

    async fn too_many_logs() -> Result<()> {
        let data = array8();
        let snapshot = data.slice(s![0, .., ..]);
        let logs = vec![
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
            data.slice(s![0, .., ..]),
        ];
        let logs = logs.into_iter().cycle().take(300);

        // This statement should panic, as the max number of logs for one block is 254
        let block: Block = Block::new(
            Snapshot::from_array(snapshot, 2),
            logs.into_iter()
                .map(|log| Log::from_arrays(snapshot, log, 2))
                .collect(),
        );

        let mut buffer = Vec::with_capacity(block.size() as usize);
        block.write_to(&mut buffer).await?;

        // The number of time instants in the block is written as a single byte, so the maximum
        // number lof logs we can actually store in this block is 254. If we're allowed to actually
        // create the block, then serializing and deserializing will lead to incorrect behavior.
        let mut file = Cursor::new(buffer);
        let block: Block = Block::read_from(&mut file).await?;
        assert_eq!(block.logs.len(), 300);

        Ok(())
    }

    #[tokio::test]
    #[should_panic]
    async fn test_too_many_logs() {
        too_many_logs().await.expect("should panic");
    }

    #[test]
    fn get() {
        let data = array8();
        let data = vec![
            data.slice(s![0, .., ..]),
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
        ];

        let block: Block = Block::new(
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
    fn fill_window() {
        let data = array8();
        let data = vec![
            data.slice(s![0, .., ..]),
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
        ];

        let block: Block = Block::new(
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
                            let bounds = geom::Rect::new(top, bottom, left, right);

                            let mut window = Array2::zeros([bounds.rows(), bounds.cols()]);
                            let set = |row, col, value| window[[row, col]] = value;

                            block.fill_window(set, t, &bounds);
                            assert_eq!(window, expected);
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
                                    let expected = array_search_window2(
                                        data[t], top, bottom, left, right, lower, upper,
                                    );
                                    let expected: HashSet<(usize, usize)> =
                                        HashSet::from_iter(expected.into_iter());
                                    let window = geom::Rect::new(top, bottom, left, right);
                                    let cells = block.search_window(t, &window, lower, upper);
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

    #[tokio::test]
    async fn serialize_deserialize() -> Result<()> {
        let data = array8();
        let data = vec![
            data.slice(s![0, .., ..]),
            data.slice(s![1, .., ..]),
            data.slice(s![2, .., ..]),
        ];

        let block: Block = Block::new(
            Snapshot::from_array(data[0], 2),
            vec![
                Log::from_arrays(data[0], data[1], 2),
                Log::from_arrays(data[0], data[2], 2),
            ],
        );

        let mut buffer: Vec<u8> = Vec::with_capacity(block.size() as usize);
        block.write_to(&mut buffer).await?;
        assert_eq!(buffer.len(), block.size() as usize);

        let mut buffer = Cursor::new(buffer);
        let block: Block = Block::read_from(&mut buffer).await?;

        for t in 0..3 {
            for r in 0..8 {
                for c in 0..8 {
                    assert_eq!(block.get(t, r, c), data[t][[r, c]]);
                }
            }
        }

        Ok(())
    }
}
