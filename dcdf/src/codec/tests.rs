use ndarray::ArrayView2;
use num_traits::Num;
use std::collections::HashSet;
use std::io;
use std::io::Seek;
use tempfile::tempfile;

use super::*;

impl<I> Snapshot<I>
where
    I: PrimInt + Debug,
{
    /// Wrap Snapshot.build with good default implementation, so we don't have to create a closure
    /// for get in every test.
    pub fn from_array(data: ArrayView2<I>, k: i32) -> Self {
        let get = |row, col| data[[row, col]].to_i64().unwrap();
        let shape = data.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get, [rows, cols], k)
    }
}

impl<I> Log<I>
where
    I: PrimInt + Debug,
{
    /// Wrap Log.build with good default implementation, so we don't have to create closures
    /// for get_s and get_t in every test.
    pub fn from_arrays(snapshot: ArrayView2<I>, log: ArrayView2<I>, k: i32) -> Self {
        let get_s = |row, col| snapshot[[row, col]].to_i64().unwrap();
        let get_t = |row, col| log[[row, col]].to_i64().unwrap();
        let shape = snapshot.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get_s, get_t, [rows, cols], k)
    }
}

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub fn array_search_window<T>(
    data: ArrayView2<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    lower: T,
    upper: T,
) -> Vec<(usize, usize)>
where
    T: Num + Debug + Copy + PartialOrd,
{
    let mut coords: Vec<(usize, usize)> = vec![];
    for row in top..bottom {
        for col in left..right {
            let cell_value = data[[row, col]];
            if lower <= cell_value && cell_value <= upper {
                coords.push((row, col));
            }
        }
    }

    coords
}

mod fchunk {
    use super::*;
    use ndarray::{arr2, s, Array2};

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
    use ndarray::{arr2, s, Array2};

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
