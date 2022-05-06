use std::collections::HashSet;

use super::*;

impl BitMapBuilder {
    /// Count occurences of 1 in BitMap[0...i]
    ///
    /// Naive brute force implementation that is used to double check the indexed implementation in
    /// tests.
    pub fn rank(&self, i: usize) -> usize {
        let mut count = 0;
        for word in self.bitmap.iter().take(i / 8) {
            count += word.count_ones();
        }
        let leftover_bits = i % 8;
        if leftover_bits > 0 {
            let shift = 8 - leftover_bits;
            let word = self.bitmap[i / 8];
            count += (word >> shift).count_ones();
        }
        count.try_into().unwrap()
    }
}

impl Dacs {
    fn len(&self) -> usize {
        self.levels[0].0.length
    }

    fn collect<T>(&self) -> Vec<T>
    where
        T: PrimInt + Debug,
    {
        (0..self.len()).into_iter().map(|i| self.get(i)).collect()
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
    T: PrimInt + Debug,
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
                for bottom in top + 1..8 {
                    for left in 0..8 {
                        for right in left + 1..8 {
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
                for bottom in top + 1..8 {
                    for left in 0..8 {
                        for right in left + 1..8 {
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
    fn from_array() {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..16 {
                for left in 0..16 {
                    for right in left + 1..16 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..16 {
                for left in 0..16 {
                    for right in left + 1..16 {
                        let coords = snapshot.search_window(top, bottom, left, right, 0, 41);

                        assert_eq!(coords, vec![]);
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
    fn from_arrays() {
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
    }

    #[test]
    fn from_arrays_unsigned() {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..9 {
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
                        let window = log.get_window(&snapshot, top, bottom, left, right);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 3);
        for top in 0..9 {
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
                        let window = log.get_window(&snapshot, bottom, top, right, left);
                        let expected = data.slice(s![1, top..bottom, left..right]);
                        assert_eq!(window, expected);
                    }
                }
            }
        }

        let log = Log::from_arrays(data.slice(s![0, .., ..]), data.slice(s![2, .., ..]), 2);
        for top in 0..8 {
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..9 {
                for left in 0..9 {
                    for right in left + 1..9 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
            for bottom in top + 1..8 {
                for left in 0..8 {
                    for right in left + 1..8 {
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
}

mod bitmap {
    use super::*;
    use rand;
    use rand::{Rng, RngCore};
    use std::time;

    #[test]
    fn from_bitmap() {
        let bitmap = BitMapBuilder {
            length: 36,
            bitmap: vec![99, 104, 114, 105, 115],
        };

        let bitmap = BitMap::from(bitmap);
        assert_eq!(bitmap.bitmap, vec![1667789417, 1929379840]);

        let bitmap = BitMapBuilder {
            length: 129,
            bitmap: vec![
                99, 104, 114, 105, 115, 0, 0, 0, 99, 104, 114, 105, 115, 0, 0, 0, 128,
            ],
        };

        let bitmap = BitMap::from(bitmap);
        assert_eq!(
            bitmap.bitmap,
            vec![1667789417, 1929379840, 1667789417, 1929379840, 1 << 31]
        );
        assert_eq!(bitmap.index, vec![40]);
    }

    fn test_rank(bitmap: BitMapBuilder, indexes: &[usize]) {
        // Gather answers using the naive, reference implementation
        println!(
            "Test rank: {} bits, {}/{} lookups",
            bitmap.length,
            indexes.len(),
            indexes.len() * 1000,
        );

        let timer = time::Instant::now();
        let mut answers: Vec<usize> = Vec::with_capacity(indexes.len());
        for index in indexes {
            answers.push(bitmap.rank(*index));
        }
        let reference_impl = timer.elapsed().as_millis();

        // Compare our answers with the reference implementation
        let timer = time::Instant::now();
        let bitmap = BitMap::from(bitmap);
        let make_index = timer.elapsed().as_millis();

        let timer = time::Instant::now();
        for _ in 1..1000 {
            for (index, answer) in indexes.iter().zip(answers.iter()) {
                assert_eq!(*answer, bitmap.rank(*index));
            }
        }
        let our_impl = timer.elapsed().as_millis();

        println!("time to build index: {}", make_index);
        println!("reference impl: {}, our impl: {}", reference_impl, our_impl);
    }

    #[test]
    fn get() {
        let answers = [
            true, false, true, false, true, false, true, false, false, false, true,
        ];
        let mut bitmap = BitMapBuilder::new();

        for answer in answers {
            bitmap.push(answer);
        }

        let bitmap = BitMap::from(bitmap);
        for (index, answer) in answers.iter().enumerate() {
            assert_eq!(bitmap.get(index), *answer);
        }
    }

    #[test]
    fn rank() {
        // 1010101001
        let bitmap = BitMapBuilder {
            length: 10,
            bitmap: vec![170, 64],
        };

        let indexes: Vec<usize> = (0..10).collect();
        test_rank(bitmap, &indexes);
    }

    #[test]
    #[should_panic]
    fn rank_out_of_bounds() {
        // 1010101001
        let bitmap = BitMapBuilder {
            length: 10,
            bitmap: vec![170, 64],
        };

        let bitmap = BitMap::from(bitmap);
        bitmap.rank(11);
    }

    struct RandomRange(usize);

    impl Iterator for RandomRange {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            Some(rand::thread_rng().gen_range(0..self.0))
        }
    }

    fn make_bitmap(n: usize) -> BitMapBuilder {
        let timer = time::Instant::now();
        let mut bytes: Vec<u8> = Vec::with_capacity(n >> 3);
        bytes.resize(n >> 3, 0);
        rand::thread_rng().fill_bytes(&mut bytes);

        let bitmap = BitMapBuilder {
            length: n,
            bitmap: bytes,
        };
        println!("Time to build bitmap: {}", timer.elapsed().as_millis());

        bitmap
    }

    #[test]
    fn rank_megabit() {
        let bitmap = make_bitmap(1 << 20);
        let indexes = RandomRange(1 << 20);
        let indexes: Vec<usize> = indexes.take(100).collect();
        test_rank(bitmap, &indexes);
    }

    #[test]
    #[ignore]
    fn rank_gigabit() {
        let bitmap = make_bitmap(1 << 30);
        let indexes: Vec<usize> = RandomRange(1 << 30).take(1000).collect();
        test_rank(bitmap, &indexes);
    }
}

mod dacs {
    use super::*;

    #[test]
    fn get_i32() {
        let data = vec![0, 2, -3, -2.pow(9), 2.pow(17) + 1, -2.pow(30) - 42];
        let dacs = Dacs::from(data.clone());
        for i in 0..data.len() {
            assert_eq!(dacs.get::<i32>(i), data[i]);
        }
        assert_eq!(dacs.levels[0].0.get(2), false);
    }

    #[test]
    fn this_one() {
        let data: Vec<i32> = vec![-512];
        let dacs = Dacs::from(data.clone());
        assert_eq!(zigzag_decode(zigzag_encode(-512)), -512);
        assert_eq!(dacs.get::<i32>(0), data[0]);
    }
}
