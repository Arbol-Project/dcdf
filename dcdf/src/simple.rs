use super::codec::{Block, Chunk, FChunk, Log, Snapshot};
use super::fixed::to_fixed;

use ndarray::ArrayView2;
use num_traits::{Float, PrimInt};
use std::fmt::Debug;
use std::marker::PhantomData;

fn build<'a, I, T>(instants: T, k: i32) -> Chunk<I> 
where
    I: 'a + PrimInt + Debug,
    T: Iterator<Item = ArrayView2<'a, I>>,
{
    let mut cur_snapshot = None;
    let mut blocks = vec![];
    let mut logs = vec![];

    for instant in instants {
        let shape = instant.shape();
        let rows = shape[0];
        let cols = shape[1];
        match cur_snapshot {
            None => { 
                let get = |row, col| instant[[row, col]].to_i64().unwrap();
                cur_snapshot = Some((instant, Snapshot::build(get, [rows, cols], k))) 
            },
            Some((mut snap_array, mut snapshot)) => {
                let get_t = |row, col| instant[[row, col]].to_i64().unwrap();
                let new_snapshot = Snapshot::build(get_t, [rows, cols], k);

                let get_s = |row, col| snap_array[[row, col]].to_i64().unwrap();
                let new_log = Log::build(get_s, get_t, [rows, cols], k);
                if new_snapshot.size() <= new_log.size() {
                    blocks.push(Block::new(snapshot, logs));
                    logs = vec![];
                    snap_array = instant;
                    snapshot = new_snapshot;
                }
                else {
                    logs.push(new_log);
                }
                cur_snapshot = Some((snap_array, snapshot));
            }
        }
    }

    if let Some((_, snapshot)) = cur_snapshot {
        blocks.push(Block::new(snapshot, logs));
    }

    Chunk::from(blocks)
}

fn buildf<'a, F, T>(instants: T, k: i32, fractional_bits: usize) -> FChunk<F> 
where
    F: 'a + Float + Debug,
    T: Iterator<Item = ArrayView2<'a, F>>,
{
    let mut cur_snapshot = None;
    let mut blocks = vec![];
    let mut logs = vec![];

    for instant in instants {
        let shape = instant.shape();
        let rows = shape[0];
        let cols = shape[1];
        match cur_snapshot {
            None => { 
                let get = |row, col| to_fixed(instant[[row, col]], fractional_bits);
                cur_snapshot = Some((instant, Snapshot::build(get, [rows, cols], k))) 
            },
            Some((mut snap_array, mut snapshot)) => {
                let get_t = |row, col| to_fixed(instant[[row, col]], fractional_bits);
                let new_snapshot = Snapshot::build(get_t, [rows, cols], k);

                let get_s = |row, col| to_fixed(snap_array[[row, col]], fractional_bits);
                let new_log = Log::build(get_s, get_t, [rows, cols], k);
                if new_snapshot.size() <= new_log.size() {
                    blocks.push(Block::new(snapshot, logs));
                    logs = vec![];
                    snap_array = instant;
                    snapshot = new_snapshot;
                }
                else {
                    logs.push(new_log);
                }
                cur_snapshot = Some((snap_array, snapshot));
            }
        }
    }

    if let Some((_, snapshot)) = cur_snapshot {
        blocks.push(Block::new(snapshot, logs));
    }

    let chunk = Chunk::from(blocks);

    FChunk::new(chunk, fractional_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, s, Array2};

    fn array_f32() -> Vec<Array2<f32>> {
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

    fn array_i32() -> Vec<Array2<i32>> 
    {
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

    fn array_u32() -> Vec<Array2<u32>> 
    {
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

    #[test]
    fn build_i32() {
        let data = array_i32();
        let views = data.iter().map(|a| a.view());
        let chunk = build(views, 2);
        assert_eq!(chunk.iter_cell(0, 5, 0, 0).collect::<Vec<i32>>(), vec![9, 9, 9, 9, 9]);
    }

    #[test]
    fn build_u32() {
        let data = array_u32();
        let views = data.iter().map(|a| a.view());
        let chunk = build(views, 2);
        assert_eq!(chunk.iter_cell(0, 5, 0, 0).collect::<Vec<u32>>(), vec![9, 9, 9, 9, 9]);
    }

    #[test]
    fn buildf_f32() {
        let data = array_f32();
        let views = data.iter().map(|a| a.view());
        let chunk = buildf(views, 2, 3);
        assert_eq!(chunk.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(), vec![9.5, 9.5, 9.5, 9.5, 9.5]);
    }
}
