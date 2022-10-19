use ndarray::Array2;
use num_traits::{Float, PrimInt};
use std::any::TypeId;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::mem::{replace, size_of};

use super::cache::Cacheable;
use super::codec::{Block, Chunk, FChunk, Log, Snapshot};
use super::errors::Result;
use super::extio::{ExtendedRead, ExtendedWrite, Serialize};
use super::fixed::{to_fixed, Fraction, Precise, Round};

const MAGIC_NUMBER: u16 = 0xDCDF;
const FORMAT_VERSION: u32 = 0;

pub struct Build<I>
where
    I: PrimInt + Debug,
{
    pub data: Chunk<I>,
    pub logs: usize,
    pub snapshots: usize,
    pub compression: f32,
}

pub struct Builder<I>
where
    I: PrimInt + Debug,
{
    count_snapshots: usize,
    count_logs: usize,
    snap_array: Array2<I>,
    snapshot: Snapshot<I>,
    blocks: Vec<Block<I>>,
    logs: Vec<Log<I>>,
    rows: usize,
    cols: usize,
    k: i32,
}

impl<I> Builder<I>
where
    I: PrimInt + Debug,
{
    pub fn new(first: Array2<I>, k: i32) -> Self {
        let shape = first.shape();
        let rows = shape[0];
        let cols = shape[1];

        let get = |row, col| first[[row, col]].to_i64().unwrap();
        let snapshot = Snapshot::build(get, [rows, cols], k);

        Builder {
            count_snapshots: 0,
            count_logs: 0,
            snap_array: first,
            snapshot,
            blocks: vec![],
            logs: vec![],
            rows,
            cols,
            k,
        }
    }

    pub fn push(&mut self, instant: Array2<I>) {
        let get_t = |row, col| instant[[row, col]].to_i64().unwrap();
        let new_snapshot = Snapshot::build(get_t, [self.rows, self.cols], self.k);

        let get_s = |row, col| self.snap_array[[row, col]].to_i64().unwrap();
        let new_log = Log::build(get_s, get_t, [self.rows, self.cols], self.k);

        if self.logs.len() == 254 || new_snapshot.size() <= new_log.size() {
            self.count_snapshots += 1;
            self.count_logs += self.logs.len();

            let snapshot = replace(&mut self.snapshot, new_snapshot);
            let logs = replace(&mut self.logs, vec![]);
            self.snap_array = instant;
            self.blocks.push(Block::new(snapshot, logs));
        } else {
            self.logs.push(new_log);
        }
    }

    pub fn finish(mut self) -> Build<I> {
        self.count_snapshots += 1;
        self.count_logs += self.logs.len();
        self.blocks.push(Block::new(self.snapshot, self.logs));

        let chunk = Chunk::from(self.blocks);
        let count_instants = self.count_snapshots + self.count_logs;
        let word_size = size_of::<I>();

        let compressed = chunk.size() + 2 /* magic number */ + 4 /* version */;
        let uncompressed = count_instants * self.rows * self.cols * word_size;
        let compression = compressed as f32 / uncompressed as f32;

        Build {
            data: chunk,
            logs: self.count_logs,
            snapshots: self.count_snapshots,
            compression: compression,
        }
    }
}

pub fn build<I, T>(mut instants: T, k: i32) -> Build<I>
where
    I: PrimInt + Debug,
    T: Iterator<Item = Array2<I>>,
{
    let first = instants.next().expect("No time instants to encode");
    let mut builder = Builder::new(first, k);
    for instant in instants {
        builder.push(instant);
    }
    builder.finish()
}

pub struct FBuild<F>
where
    F: Float + Debug,
{
    pub data: FChunk<F>,
    pub logs: usize,
    pub snapshots: usize,
    pub compression: f32,
}

pub struct FBuilder<F>
where
    F: Float + Debug,
{
    count_snapshots: usize,
    count_logs: usize,
    snap_array: Array2<F>,
    snapshot: Snapshot<i64>,
    blocks: Vec<Block<i64>>,
    logs: Vec<Log<i64>>,
    rows: usize,
    cols: usize,
    k: i32,
    fraction: Fraction,
}

impl<F> FBuilder<F>
where
    F: Float + Debug,
{
    pub fn new(first: Array2<F>, k: i32, fraction: Fraction) -> Self {
        let shape = first.shape();
        let rows = shape[0];
        let cols = shape[1];

        let get = |row, col| match fraction {
            Precise(bits) => to_fixed(first[[row, col]], bits, false),
            Round(bits) => to_fixed(first[[row, col]], bits, true),
        };
        let snapshot = Snapshot::build(get, [rows, cols], k);

        FBuilder {
            count_snapshots: 0,
            count_logs: 0,
            snap_array: first,
            snapshot,
            blocks: vec![],
            logs: vec![],
            rows,
            cols,
            k,
            fraction,
        }
    }

    pub fn push(&mut self, instant: Array2<F>) {
        let get_t = |row, col| match self.fraction {
            Precise(bits) => to_fixed(instant[[row, col]], bits, false),
            Round(bits) => to_fixed(instant[[row, col]], bits, true),
        };
        let new_snapshot = Snapshot::build(get_t, [self.rows, self.cols], self.k);

        let get_s = |row, col| match self.fraction {
            Precise(bits) => to_fixed(self.snap_array[[row, col]], bits, false),
            Round(bits) => to_fixed(self.snap_array[[row, col]], bits, true),
        };
        let new_log = Log::build(get_s, get_t, [self.rows, self.cols], self.k);

        if self.logs.len() == 254 || new_snapshot.size() <= new_log.size() {
            self.count_snapshots += 1;
            self.count_logs += self.logs.len();

            let snapshot = replace(&mut self.snapshot, new_snapshot);
            let logs = replace(&mut self.logs, vec![]);
            self.snap_array = instant;
            self.blocks.push(Block::new(snapshot, logs));
        } else {
            self.logs.push(new_log);
        }
    }

    pub fn finish(mut self) -> FBuild<F> {
        self.count_snapshots += 1;
        self.count_logs += self.logs.len();
        self.blocks.push(Block::new(self.snapshot, self.logs));

        let fractional_bits = match self.fraction {
            Precise(bits) => bits,
            Round(bits) => bits,
        };
        let chunk = FChunk::new(Chunk::from(self.blocks), fractional_bits);
        let count_instants = self.count_snapshots + self.count_logs;
        let word_size = size_of::<F>();
        let compressed = chunk.size() + 2 /* magic number */ + 4 /* version */;
        let uncompressed = count_instants * self.rows * self.cols * word_size;
        let compression = compressed as f32 / uncompressed as f32;

        FBuild {
            data: chunk,
            logs: self.count_logs,
            snapshots: self.count_snapshots,
            compression: compression,
        }
    }
}

pub fn buildf<F, T>(mut instants: T, k: i32, fraction: Fraction) -> FBuild<F>
where
    F: Float + Debug,
    T: Iterator<Item = Array2<F>>,
{
    let first = instants.next().expect("No time instants to encode");
    let mut builder = FBuilder::new(first, k, fraction);
    for instant in instants {
        builder.push(instant);
    }
    builder.finish()
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
    pub fn save(&self, stream: &mut impl Write) -> Result<()> {
        stream.write_u16(MAGIC_NUMBER)?;
        stream.write_u32(FORMAT_VERSION)?;
        stream.write_i32(self.type_code())?;
        self.data.write_to(stream)?;

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
    pub fn save(&self, stream: &mut impl Write) -> Result<()> {
        let type_code = size_of::<F>() as i32 * 8;
        stream.write_u16(MAGIC_NUMBER)?;
        stream.write_u32(FORMAT_VERSION)?;
        stream.write_i32(type_code)?;
        self.data.write_to(stream)?;

        Ok(())
    }
}

pub enum DataChunk {
    I32(Chunk<i32>),
    U32(Chunk<u32>),
    I64(Chunk<i64>),
    U64(Chunk<u64>),
    F32(FChunk<f32>),
    F64(FChunk<f64>),
}

pub use DataChunk::{F32, F64, I32, I64, U32, U64};

pub fn load(stream: &mut impl Read) -> Result<DataChunk> {
    let magic_number = stream.read_u16()?;
    if magic_number != MAGIC_NUMBER {
        panic!("File is not a DCDF file.");
    }
    let version = stream.read_u32()?;
    if version != FORMAT_VERSION {
        panic!("Unrecognized file format.");
    }
    let type_code = stream.read_i32()?;
    let chunk = match type_code {
        TYPE_I32 => I32(Chunk::read_from(stream)?),
        TYPE_U32 => U32(Chunk::read_from(stream)?),
        TYPE_I64 => I64(Chunk::read_from(stream)?),
        TYPE_U64 => U64(Chunk::read_from(stream)?),
        TYPE_F32 => F32(FChunk::read_from(stream)?),
        TYPE_F64 => F64(FChunk::read_from(stream)?),
        _ => panic!("Unknown data type"),
    };

    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};
    use std::io::Seek;
    use std::sync::Arc;
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
        let built = build(data.into_iter(), 2);
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.iter_cell(0, 5, 0, 0).collect::<Vec<i32>>(),
            vec![9, 9, 9, 9, 9]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.34488282);
    }

    #[test]
    fn save_load_i32() -> Result<()> {
        let data = array();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            I32(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 0, 0).collect::<Vec<i32>>(),
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
    fn save_load_u32() -> Result<()> {
        let data = array();
        let data: Vec<Array2<u32>> = data.into_iter().map(|a| a.map(|n| *n as u32)).collect();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            U32(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 0, 0).collect::<Vec<u32>>(),
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
    fn save_load_i64() -> Result<()> {
        let data = array();
        let data: Vec<Array2<i64>> = data.into_iter().map(|a| a.map(|n| *n as i64)).collect();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            I64(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 0, 0).collect::<Vec<i64>>(),
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
    fn save_load_u64() -> Result<()> {
        let data = array();
        let data: Vec<Array2<u64>> = data.into_iter().map(|a| a.map(|n| *n as u64)).collect();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            U64(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 0, 0).collect::<Vec<u64>>(),
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
        let built = buildf(data.into_iter(), 2, Precise(3));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(),
            vec![9.5, 9.5, 9.5, 9.5, 9.5]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.351875);
    }

    #[test]
    fn save_load_f32() -> Result<()> {
        let data = array_float();
        let built = buildf(data.into_iter(), 2, Precise(3));

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F32(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(),
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
    fn save_load_f64() -> Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = buildf(data.into_iter(), 2, Precise(3));

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F64(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 0, 0).collect::<Vec<f64>>(),
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
    fn save_load_f64_round() -> Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = buildf(data.into_iter(), 2, Round(2));

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F64(chunk) => {
                let chunk = Arc::new(chunk);
                assert_eq!(
                    chunk.iter_cell(0, 5, 2, 4).collect::<Vec<f64>>(),
                    vec![3.5, 5.0, 5.0, 3.5, 5.0]
                );
            }
            _ => {
                assert!(false);
            }
        }

        Ok(())
    }
}
