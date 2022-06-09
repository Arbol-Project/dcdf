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
    /// So I don't have to create a closure in every test
    fn from_array(data: ArrayView2<I>, k: i32) -> Self {
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
    /// So I don't have to create closures in every test
    fn from_arrays(snapshot: ArrayView2<I>, log: ArrayView2<I>, k: i32) -> Self {
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
fn array_search_window<T>(
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

mod block {
    use super::*;
    use ndarray::{arr3, s, Array3};

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

mod snapshot {
    use super::*;
    use ndarray::{arr2, s, Array2};

    fn array8() -> Array2<i32> {
        arr2(&[
            [9, 8, 7, 7, 6, 6, 3, 2],
            [7, 7, 7, 7, 6, 6, 3, 3],
            [6, 6, 6, 6, 3, 3, 3, 3],
            [5, 5, 6, 6, 3, 3, 3, 3],
            [4, 5, 5, 5, 4, 4, 4, 4],
            [3, 3, 5, 5, 4, 4, 4, 4],
            [3, 3, 3, 5, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4],
        ])
    }

    fn array9() -> Array2<i32> {
        arr2(&[
            [9, 8, 7, 7, 6, 6, 3, 2, 1],
            [7, 7, 7, 7, 6, 6, 3, 3, 3],
            [6, 6, 6, 6, 3, 3, 3, 3, 3],
            [5, 5, 6, 6, 3, 3, 3, 3, 2],
            [4, 5, 5, 5, 4, 4, 4, 4, 4],
            [3, 3, 5, 5, 4, 4, 4, 4, 4],
            [3, 3, 3, 5, 4, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4, 4],
            [4, 4, 3, 4, 4, 4, 4, 4, 4],
        ])
    }

    #[test]
    fn build() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        assert_eq!(snapshot.nodemap.length, 17);
        assert_eq!(
            snapshot.nodemap.bitmap,
            vec![0b11110101001001011000000000000000]
        );
        assert_eq!(
            snapshot.max.collect::<i32>(),
            vec![
                9, 0, 3, 4, 5, 0, 2, 3, 3, 0, 3, 3, 3, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 1, 1, 0, 1, 0,
                0, 1, 0, 2, 2, 1, 1, 0, 0, 2, 0, 2, 1,
            ]
        );
        assert_eq!(
            snapshot.min.collect::<i32>(),
            vec![2, 3, 0, 1, 2, 0, 0, 0, 0, 0,]
        );

        assert_eq!(snapshot.shape, [8, 8]);
    }

    #[test]
    fn get() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_out_of_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        snapshot.get(0, 9);
    }

    #[test]
    fn get_single_node_tree() {
        let data: Array2<i32> = Array2::zeros([16, 16]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);
        assert_eq!(snapshot.nodemap.bitmap.len(), 1);
        assert_eq!(snapshot.max.levels[0].1.len(), 1);
        assert!(snapshot.min.levels.is_empty());

        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(snapshot.get(row, col), 42);
            }
        }
    }

    #[test]
    fn get_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }
    }

    #[test]
    fn get_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 3);

        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(snapshot.get(row, col), data[[row, col]]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_array9_out_of_bounds() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        snapshot.get(0, 9);
    }

    #[test]
    fn get_window() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = snapshot.get_window(top, bottom, left, right);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_window_lower_right_out_of_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        snapshot.get_window(0, 9, 0, 5);
    }

    #[test]
    fn get_window_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = snapshot.get_window(top, bottom, left, right);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 3);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = snapshot.get_window(top, bottom, left, right);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_rearrange_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = snapshot.get_window(bottom, top, right, left);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_tree() {
        let data: Array2<i32> = Array2::zeros([16, 16]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..16 {
            for bottom in top + 1..=16 {
                for left in 0..16 {
                    for right in left + 1..=16 {
                        let window = snapshot.get_window(top, bottom, left, right);
                        let expected = data.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords =
                                    snapshot.search_window(top, bottom, left, right, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords =
                                    snapshot.search_window(top, bottom, left, right, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.view(), 3);

        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords =
                                    snapshot.search_window(top, bottom, left, right, lower, upper);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_rearrange_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords =
                                    snapshot.search_window(bottom, top, right, left, upper, lower);
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn search_window_out_of_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        snapshot.search_window(0, 9, 0, 5, 4, 6);
    }

    #[test]
    fn search_window_single_tree_node_in_range() {
        let data: Array2<i32> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let mut expected: HashSet<(usize, usize)> = HashSet::new();
                        for row in top..bottom {
                            for col in left..right {
                                expected.insert((row, col));
                            }
                        }
                        let coords = snapshot.search_window(top, bottom, left, right, 41, 43);
                        let coords = HashSet::from_iter(coords.iter().cloned());

                        assert_eq!(coords, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_tree_node_out_of_range() {
        let data: Array2<i32> = Array2::zeros([16, 16]) + 42;
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..16 {
            for bottom in top + 1..=16 {
                for left in 0..16 {
                    for right in left + 1..=16 {
                        let coords = snapshot.search_window(top, bottom, left, right, 0, 41);

                        assert_eq!(coords, vec![]);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_no_values_in_range() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let coords = snapshot.search_window(top, bottom, left, right, 100, 200);
                        assert_eq!(coords.len(), 0);
                    }
                }
            }
        }
    }
}

mod log {
    use super::*;
    use ndarray::{arr3, s, Array3};

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

    fn array8_unsigned() -> Array3<u32> {
        array8().mapv(|x| x as u32)
    }

    fn array9() -> Array3<i32> {
        arr3(&[
            [
                [9, 8, 7, 7, 6, 6, 3, 2, 1],
                [7, 7, 7, 7, 6, 6, 3, 3, 3],
                [6, 6, 6, 6, 3, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3, 2],
                [4, 5, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 3, 5, 4, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4, 4],
            ],
            [
                [9, 8, 7, 7, 7, 7, 2, 2, 2],
                [7, 7, 7, 7, 7, 7, 2, 2, 2],
                [6, 6, 6, 6, 4, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3, 2],
                [4, 5, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 4, 5, 5, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 5],
                [4, 4, 4, 4, 5, 4, 4, 4, 1],
            ],
            [
                [9, 8, 7, 7, 8, 7, 5, 5, 2],
                [7, 7, 7, 7, 7, 7, 5, 5, 2],
                [7, 7, 6, 6, 4, 3, 4, 4, 3],
                [6, 6, 6, 6, 4, 4, 4, 4, 2],
                [4, 5, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4, 4],
                [3, 3, 4, 5, 6, 4, 4, 4, 4],
                [4, 4, 4, 4, 5, 4, 4, 4, 4],
                [5, 4, 4, 4, 5, 5, 5, 5, 10],
            ],
        ])
    }

    #[test]
    fn build() {
        let data = array8();
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        assert_eq!(log.nodemap.length, 17);
        assert_eq!(log.nodemap.bitmap, vec![0b10111001000010010000000000000000]);
        assert_eq!(log.equal.length, 10);
        assert_eq!(log.equal.bitmap, vec![0b10001010000000000000000000000000]);

        assert_eq!(
            log.max.collect::<i32>(),
            vec![
                0, 0, 1, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                0, 0
            ]
        );

        assert_eq!(log.min.collect::<i32>(), vec![0, 0, 0, 0, 0, 1, 0,]);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        assert_eq!(log.nodemap.length, 21);
        assert_eq!(log.nodemap.bitmap, vec![0b11111000010100001001000000000000]);
        assert_eq!(log.equal.length, 12);
        assert_eq!(log.equal.bitmap, vec![0b10100010100000000000000000000000]);

        assert_eq!(
            log.max.collect::<i32>(),
            vec![
                0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 1, 1, 1, 0, 1,
                1, 1, 0, 1, 0, 2, 0, 1, 0,
            ]
        );

        assert_eq!(log.min.collect::<i32>(), vec![1, 1, 1, 0, 0, 1, 0, 1, 0,]);

        assert_eq!(log.shape, [8, 8]);
    }

    #[test]
    fn build_unsigned() {
        let data = array8_unsigned();
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        assert_eq!(log.nodemap.length, 17);
        assert_eq!(log.nodemap.bitmap, vec![0b10111001000010010000000000000000]);
        assert_eq!(log.equal.length, 10);
        assert_eq!(log.equal.bitmap, vec![0b10001010000000000000000000000000]);

        assert_eq!(
            log.max.collect::<i32>(),
            vec![
                0, 0, 1, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                0, 0
            ]
        );

        assert_eq!(log.min.collect::<i32>(), vec![0, 0, 0, 0, 0, 1, 0,]);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        assert_eq!(log.nodemap.length, 21);
        assert_eq!(log.nodemap.bitmap, vec![0b11111000010100001001000000000000]);
        assert_eq!(log.equal.length, 12);
        assert_eq!(log.equal.bitmap, vec![0b10100010100000000000000000000000]);

        assert_eq!(
            log.max.collect::<i32>(),
            vec![
                0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 1, 1, 1, 0, 1,
                1, 1, 0, 1, 0, 2, 0, 1, 0,
            ]
        );

        assert_eq!(log.min.collect::<i32>(), vec![1, 1, 1, 0, 0, 1, 0, 1, 0,]);
    }

    #[test]
    fn get() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    fn get_unsigned() {
        let data = array8_unsigned();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    fn get_single_node_trees() {
        let data_s: Array2<i32> = Array2::zeros([8, 8]) + 20;
        let data_t: Array2<i32> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), 42);
            }
        }
    }

    #[test]
    fn get_single_node_snapshot() {
        let data = array8();
        let data_s: Array2<i32> = Array2::zeros([8, 8]) + 20;
        let data_t = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for row in 0..8 {
            for col in 0..8 {
                assert_eq!(log.get(&snapshot, row, col), data_t[[row, col]]);
            }
        }
    }

    #[test]
    fn get_single_node_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let data_t: Array2<i32> = Array2::zeros([8, 8]) + 20;

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for row in 0..8 {
            for col in 1..8 {
                assert_eq!(log.get(&snapshot, row, col), data_t[[row, col]]);
            }
        }
    }

    #[test]
    fn get_equal_snapshot_and_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_s.view(), 2);

        for row in 0..8 {
            for col in 1..8 {
                assert_eq!(log.get(&snapshot, row, col), data_s[[row, col]]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_out_of_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);

        log.get(&snapshot, 0, 9);
    }

    #[test]
    fn get_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    fn get_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 3);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 3);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[1, row, col]]);
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 3);
        for row in 0..9 {
            for col in 0..9 {
                assert_eq!(log.get(&snapshot, row, col), data[[2, row, col]]);
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_array9_out_of_bounds() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);

        log.get(&snapshot, 0, 9);
    }

    #[test]
    fn get_window() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_unsigned() {
        let data = array8_unsigned();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn get_window_lower_right_out_of_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);
        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);

        log.get_window(&snapshot, 0, 9, 0, 5);
    }

    #[test]
    fn get_window_array9() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_array9_k3() {
        let data = array9();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 3);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_rearrange_bounds() {
        let data = array8();
        let snapshot = Snapshot::from_array(data.slice(s![0, .., ..]), 2);

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![1, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, bottom, top, right, left);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, bottom, top, right, left);
                        let expected = data.slice(s![2, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_trees() {
        let data_s: Array2<i32> = Array2::zeros([8, 8]) + 20;
        let data_t: Array2<i32> = Array2::zeros([8, 8]) + 42;

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data_t.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_snapshot() {
        let data = array8();
        let data_s: Array2<i32> = Array2::zeros([8, 8]) + 20;
        let data_t = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data_t.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_single_node_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let data_t: Array2<i32> = Array2::zeros([8, 8]) + 20;

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data_t.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn get_window_equal_snapshot_and_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);

        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_s.view(), 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data_s.slice(s![top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn search_window() {
        let data = array8();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_unsigned() {
        let data = array8_unsigned();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9() {
        let data = array9();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 2);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_array9_k3() {
        let data = array9();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 3);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![2, .., ..]);
        let log = Log::from_arrays(data0, data2, 3);
        for top in 0..9 {
            for bottom in top + 1..=9 {
                for left in 0..9 {
                    for right in left + 1..=9 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_rearrange_bounds() {
        let data = array8();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data1, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, bottom, top, right, left, upper, lower,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }

        let data2 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data2, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data2, top, bottom, left, right, lower, upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, bottom, top, right, left, upper, lower,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn search_window_out_of_bounds() {
        let data = array8();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);

        log.search_window(&snapshot, 0, 9, 0, 5, 4, 6);
    }

    #[test]
    fn search_window_single_node_trees() {
        let data_s: Array2<i32> = Array2::zeros([8, 8]) + 20;
        let data_t: Array2<i32> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data_t.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_node_snapshot() {
        let data = array8();
        let data_s: Array2<i32> = Array2::zeros([8, 8]) + 42;
        let data_t = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data_t.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_single_node_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let data_t: Array2<i32> = Array2::zeros([8, 8]) + 42;
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_t.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data_t.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_sequal_snapshot_and_log() {
        let data = array8();
        let data_s = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data_s.view(), 2);
        let log = Log::from_arrays(data_s.view(), data_s.view(), 2);

        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        for lower in 4..=9 {
                            for upper in lower..=9 {
                                let expected: Vec<(usize, usize)> = array_search_window(
                                    data_s.view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                let expected: HashSet<(usize, usize)> =
                                    HashSet::from_iter(expected.iter().cloned());

                                let coords = log.search_window(
                                    &snapshot, top, bottom, left, right, lower, upper,
                                );
                                let coords = HashSet::from_iter(coords.iter().cloned());

                                assert_eq!(coords, expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn search_window_no_values_in_range() {
        let data = array8();
        let data0 = data.slice(s![0, .., ..]);
        let snapshot = Snapshot::from_array(data0, 2);

        let data1 = data.slice(s![1, .., ..]);
        let log = Log::from_arrays(data0, data1, 2);
        for top in 0..8 {
            for bottom in top + 1..=8 {
                for left in 0..8 {
                    for right in left + 1..=8 {
                        let coords =
                            log.search_window(&snapshot, top, bottom, left, right, 100, 200);
                        assert_eq!(coords.len(), 0);
                    }
                }
            }
        }
    }
}
