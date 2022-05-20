use super::codec::{Block, Chunk, FChunk, Log, Snapshot};
use super::fixed::to_fixed;

use ndarray::ArrayView2;
use num_traits::{Float, Num, PrimInt};
use std::any::TypeId;
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::mem::size_of;

const MAGIC_NUMBER: u16 = 0xDCDF;
const FORMAT_VERSION: u32 = 0;

struct Build<I>
where
    I: PrimInt + Debug,
{
    data: Chunk<I>,
    logs: usize,
    snapshots: usize,
    compression: f32,
}

fn build<'a, I, T>(instants: T, k: i32) -> Build<I>
where
    I: 'a + PrimInt + Debug,
    T: Iterator<Item = ArrayView2<'a, I>>,
{
    let mut count_snapshots = 0;
    let mut count_logs = 0;
    let mut cur_snapshot = None;
    let mut blocks = vec![];
    let mut logs = vec![];

    let mut instants = instants.peekable();
    let first = instants.peek().expect("No time instants to encode.");
    let shape = first.shape();
    let rows = shape[0];
    let cols = shape[1];

    for instant in instants {
        match cur_snapshot {
            None => {
                let get = |row, col| instant[[row, col]].to_i64().unwrap();
                cur_snapshot = Some((instant, Snapshot::build(get, [rows, cols], k)));
            }
            Some((mut snap_array, mut snapshot)) => {
                let get_t = |row, col| instant[[row, col]].to_i64().unwrap();
                let new_snapshot = Snapshot::build(get_t, [rows, cols], k);

                let get_s = |row, col| snap_array[[row, col]].to_i64().unwrap();
                let new_log = Log::build(get_s, get_t, [rows, cols], k);
                if new_snapshot.size() <= new_log.size() {
                    count_snapshots += 1;
                    count_logs += logs.len();
                    blocks.push(Block::new(snapshot, logs));
                    logs = vec![];
                    snap_array = instant;
                    snapshot = new_snapshot;
                } else {
                    logs.push(new_log);
                }
                cur_snapshot = Some((snap_array, snapshot));
            }
        }
    }

    if let Some((_, snapshot)) = cur_snapshot {
        count_snapshots += 1;
        count_logs += logs.len();
        blocks.push(Block::new(snapshot, logs));
    }

    let chunk = Chunk::from(blocks);
    let count_instants = count_snapshots + count_logs;
    let word_size = size_of::<I>();
    let uncompressed = count_instants * rows * cols * word_size;
    let compression = chunk.size() as f32 / uncompressed as f32;

    Build {
        data: chunk,
        logs: count_logs,
        snapshots: count_snapshots,
        compression: compression,
    }
}

struct FBuild<F>
where
    F: Float + Debug,
{
    data: FChunk<F>,
    logs: usize,
    snapshots: usize,
    compression: f32,
}

fn buildf<'a, F, T>(instants: T, k: i32, fractional_bits: usize) -> FBuild<F>
where
    F: 'a + Float + Debug,
    T: Iterator<Item = ArrayView2<'a, F>>,
{
    let mut count_snapshots = 0;
    let mut count_logs = 0;
    let mut cur_snapshot = None;
    let mut blocks = vec![];
    let mut logs = vec![];

    let mut instants = instants.peekable();
    let first = instants.peek().expect("No time instants to encode.");
    let shape = first.shape();
    let rows = shape[0];
    let cols = shape[1];

    for instant in instants {
        match cur_snapshot {
            None => {
                let get = |row, col| to_fixed(instant[[row, col]], fractional_bits);
                cur_snapshot = Some((instant, Snapshot::build(get, [rows, cols], k)));
            }
            Some((mut snap_array, mut snapshot)) => {
                let get_t = |row, col| to_fixed(instant[[row, col]], fractional_bits);
                let new_snapshot = Snapshot::build(get_t, [rows, cols], k);

                let get_s = |row, col| to_fixed(snap_array[[row, col]], fractional_bits);
                let new_log = Log::build(get_s, get_t, [rows, cols], k);
                if new_snapshot.size() <= new_log.size() {
                    count_snapshots += 1;
                    count_logs += logs.len();
                    blocks.push(Block::new(snapshot, logs));
                    logs = vec![];
                    snap_array = instant;
                    snapshot = new_snapshot;
                } else {
                    logs.push(new_log);
                }
                cur_snapshot = Some((snap_array, snapshot));
            }
        }
    }

    if let Some((_, snapshot)) = cur_snapshot {
        count_snapshots += 1;
        count_logs += logs.len();
        blocks.push(Block::new(snapshot, logs));
    }

    let chunk = FChunk::new(Chunk::from(blocks), fractional_bits);
    let count_instants = count_snapshots + count_logs;
    let word_size = size_of::<F>();
    let uncompressed = count_instants * rows * cols * word_size;
    let compressed = chunk.size() + 2 /* magic number */ + 4 /* version */;
    let compression = chunk.size() as f32 / uncompressed as f32;

    FBuild {
        data: chunk,
        logs: count_logs,
        snapshots: count_snapshots,
        compression: compression,
    }
}

const TYPE_I32: i32 = -4;
const TYPE_U32: i32 = 4;
const TYPE_I64: i32 = -8;
const TYPE_U64: i32 = 8;
const TYPE_F32: i32 = 32;
const TYPE_F64: i32 = 64;

impl<I: 'static> Build<I>
where
    I: PrimInt + Debug,
{
    fn save(&self, stream: &mut File) -> io::Result<()> {
        write_u16(stream, MAGIC_NUMBER)?;
        write_u32(stream, FORMAT_VERSION)?;
        write_i32(stream, self.type_code())?;
        self.data.serialize(stream)?;

        Ok(())
    }
}

impl<I: 'static> Build<I>
where
    I: PrimInt + Debug,
{
    fn type_code(&self) -> i32 {
        if TypeId::of::<I>() == TypeId::of::<i32>() {
            TYPE_I32
        } else if TypeId::of::<I>() == TypeId::of::<u32>() {
            TYPE_U32
        } else if TypeId::of::<I>() == TypeId::of::<i64>() {
            TYPE_I64
        } else if TypeId::of::<I>() == TypeId::of::<u64>() {
            TYPE_U64
        } else {
            panic!("Unsupported type: {:?}", TypeId::of::<I>())
        }
    }
}

impl<F> FBuild<F>
where
    F: Float + Debug,
{
    fn save(&self, stream: &mut File) -> io::Result<()> {
        let type_code = size_of::<F>() as i32 * 8;
        write_u16(stream, MAGIC_NUMBER)?;
        write_u32(stream, FORMAT_VERSION)?;
        write_i32(stream, type_code)?;
        self.data.serialize(stream)?;

        Ok(())
    }
}

enum DataChunk {
    I32(Chunk<i32>),
    U32(Chunk<u32>),
    I64(Chunk<i64>),
    U64(Chunk<u64>),
    F32(FChunk<f32>),
    F64(FChunk<f64>),
}

use DataChunk::F32;
use DataChunk::F64;
use DataChunk::I32;
use DataChunk::I64;
use DataChunk::U32;
use DataChunk::U64;

fn load(stream: &mut File) -> io::Result<DataChunk> {
    let magic_number = read_u16(stream)?;
    if magic_number != MAGIC_NUMBER {
        panic!("File is not a DCDF file.");
    }
    let version = read_u32(stream)?;
    if version != FORMAT_VERSION {
        panic!("Unrecognized file format.");
    }
    let type_code = read_i32(stream)?;
    let chunk = match type_code {
        TYPE_I32 => I32(Chunk::deserialize(stream)?),
        TYPE_U32 => U32(Chunk::deserialize(stream)?),
        TYPE_I64 => I64(Chunk::deserialize(stream)?),
        TYPE_U64 => U64(Chunk::deserialize(stream)?),
        TYPE_F32 => F32(FChunk::deserialize(stream)?),
        TYPE_F64 => F64(FChunk::deserialize(stream)?),
        _ => panic!("Unknown data type"),
    };

    Ok(chunk)
}

fn write_i32(stream: &mut File, word: i32) -> io::Result<()> {
    let buffer = word.to_be_bytes();
    stream.write_all(&buffer);

    Ok(())
}

fn write_u16(stream: &mut File, word: u16) -> io::Result<()> {
    let buffer = word.to_be_bytes();
    stream.write_all(&buffer);

    Ok(())
}

fn write_u32(stream: &mut File, word: u32) -> io::Result<()> {
    let buffer = word.to_be_bytes();
    stream.write_all(&buffer);

    Ok(())
}

fn read_i32(stream: &mut File) -> io::Result<i32> {
    let mut buffer = [0; 4];
    stream.read_exact(&mut buffer)?;

    Ok(i32::from_be_bytes(buffer))
}

fn read_u16(stream: &mut File) -> io::Result<u16> {
    let mut buffer = [0; 2];
    stream.read_exact(&mut buffer)?;

    Ok(u16::from_be_bytes(buffer))
}

fn read_u32(stream: &mut File) -> io::Result<u32> {
    let mut buffer = [0; 4];
    stream.read_exact(&mut buffer)?;

    Ok(u32::from_be_bytes(buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, s, Array2};
    use std::io::Seek;
    use tempfile::tempfile;

    fn array_float() -> Vec<Array2<f32>> {
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

    #[test]
    fn build_i32() {
        let data = array();
        let views = data.iter().map(|a| a.view());
        let built = build(views, 2);
        assert_eq!(
            built.data.iter_cell(0, 5, 0, 0).collect::<Vec<i32>>(),
            vec![9, 9, 9, 9, 9]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.34464845);
    }

    #[test]
    fn save_load_i32() -> io::Result<()> {
        let data = array();
        let views = data.iter().map(|a| a.view());
        let built = build(views, 2);

        let mut file = tempfile()?;
        built.save(&mut file);
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            I32(chunk) => {
                assert_eq!(
                    built.data.iter_cell(0, 5, 0, 0).collect::<Vec<i32>>(),
                    vec![9, 9, 9, 9, 9]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }

    #[test]
    fn save_load_u32() -> io::Result<()> {
        let data = array();
        let data: Vec<Array2<u32>> = data.into_iter().map(|a| a.map(|n| *n as u32)).collect();
        let views = data.iter().map(|a| a.view());
        let built = build(views, 2);

        let mut file = tempfile()?;
        built.save(&mut file);
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            U32(chunk) => {
                assert_eq!(
                    built.data.iter_cell(0, 5, 0, 0).collect::<Vec<u32>>(),
                    vec![9, 9, 9, 9, 9]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }

    #[test]
    fn save_load_i64() -> io::Result<()> {
        let data = array();
        let data: Vec<Array2<i64>> = data.into_iter().map(|a| a.map(|n| *n as i64)).collect();
        let views = data.iter().map(|a| a.view());
        let built = build(views, 2);

        let mut file = tempfile()?;
        built.save(&mut file);
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            I64(chunk) => {
                assert_eq!(
                    built.data.iter_cell(0, 5, 0, 0).collect::<Vec<i64>>(),
                    vec![9, 9, 9, 9, 9]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }

    #[test]
    fn save_load_u64() -> io::Result<()> {
        let data = array();
        let data: Vec<Array2<u64>> = data.into_iter().map(|a| a.map(|n| *n as u64)).collect();
        let views = data.iter().map(|a| a.view());
        let built = build(views, 2);

        let mut file = tempfile()?;
        built.save(&mut file);
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            U64(chunk) => {
                assert_eq!(
                    built.data.iter_cell(0, 5, 0, 0).collect::<Vec<u64>>(),
                    vec![9, 9, 9, 9, 9]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }

    #[test]
    fn buildf_f32() {
        let data = array_float();
        let views = data.iter().map(|a| a.view());
        let built = buildf(views, 2, 3);
        assert_eq!(
            built.data.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(),
            vec![9.5, 9.5, 9.5, 9.5, 9.5]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.3511328);
    }

    #[test]
    fn save_load_f32() -> io::Result<()> {
        let data = array_float();
        let views = data.iter().map(|a| a.view());
        let built = buildf(views, 2, 3);

        let mut file = tempfile()?;
        built.save(&mut file);
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F32(chunk) => {
                assert_eq!(
                    built.data.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(),
                    vec![9.5, 9.5, 9.5, 9.5, 9.5]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }

    #[test]
    fn save_load_f64() -> io::Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let views = data.iter().map(|a| a.view());
        let built = buildf(views, 2, 3);

        let mut file = tempfile()?;
        built.save(&mut file);
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F64(chunk) => {
                assert_eq!(
                    built.data.iter_cell(0, 5, 0, 0).collect::<Vec<f64>>(),
                    vec![9.5, 9.5, 9.5, 9.5, 9.5]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }
}
