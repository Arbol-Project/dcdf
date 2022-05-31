use super::codec::{Block, Chunk, FChunk, Log, Snapshot};
use super::fixed::{to_fixed, to_fixed_round};

use ndarray::Array2;
use num_traits::{Float, PrimInt};
use std::any::TypeId;
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::mem::{replace, size_of};

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

        if new_snapshot.size() <= new_log.size() {
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

pub enum Fraction {
    Precise(usize),
    Round(usize),
}

pub use Fraction::{Precise, Round};

fn suggest_fraction<F, T>(instants: T, max_value: F) -> Fraction
where
    F: Float + Debug,
    T: Iterator<Item = Array2<F>>,
{
    // Basic gist is figure out how many bits we need for the whole number part, using the
    // max_value passed in. From there shift each value in the dataset as far to the left as
    // possible given the number of whole number bits needed, then look at the number of trailing
    // zeros on each shifted value to determine how many actual fractional bits we need for that
    // number, and return the maximum number.
    const TOTAL_BITS: usize = 63;
    let whole_bits = 1 + max_value.to_f64().unwrap().log2().floor() as usize;
    let max_fraction_bits = TOTAL_BITS - whole_bits;
    let mut fraction_bits = 0;

    for instant in instants {
        for n in instant {
            let n: f64 = n.to_f64().unwrap();
            let shifted = n * (1_i64 << max_fraction_bits) as f64;

            // If we've left shifted a number as far as it will go and we still have a fractional
            // part, then this dataset will need to be rounded and there will be some loss of
            // precision.
            if shifted.fract() != 0.0 {
                return Round(max_fraction_bits);
            }

            let shifted = shifted as i64;
            if shifted == i64::MAX {
                // Conversion from float to int saturates on overflow, so assume there was an
                // overflow if result is MAX
                panic!("Value {n} is greater than max_value {max_value:?}");
            }

            let these_bits = max_fraction_bits.saturating_sub(shifted.trailing_zeros() as usize);
            if these_bits > fraction_bits {
                fraction_bits = these_bits;
            }
        }
    }
    Precise(fraction_bits)
}

pub struct FBuild<F>
where
    F: Float + Debug,
{
    data: FChunk<F>,
    logs: usize,
    snapshots: usize,
    compression: f32,
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
    fn new(first: Array2<F>, k: i32, fraction: Fraction) -> Self {
        let shape = first.shape();
        let rows = shape[0];
        let cols = shape[1];

        let get = |row, col| match fraction {
            Precise(bits) => to_fixed(first[[row, col]], bits),
            Round(bits) => to_fixed_round(first[[row, col]], bits),
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

    fn push(&mut self, instant: Array2<F>) {
        let get_t = |row, col| match self.fraction {
            Precise(bits) => to_fixed(instant[[row, col]], bits),
            Round(bits) => to_fixed_round(instant[[row, col]], bits),
        };
        let new_snapshot = Snapshot::build(get_t, [self.rows, self.cols], self.k);

        let get_s = |row, col| match self.fraction {
            Precise(bits) => to_fixed(self.snap_array[[row, col]], bits),
            Round(bits) => to_fixed_round(self.snap_array[[row, col]], bits),
        };
        let new_log = Log::build(get_s, get_t, [self.rows, self.cols], self.k);

        if new_snapshot.size() <= new_log.size() {
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

    fn finish(mut self) -> FBuild<F> {
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
    pub fn save(&self, stream: &mut File) -> io::Result<()> {
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
    pub fn save(&self, stream: &mut File) -> io::Result<()> {
        let type_code = size_of::<F>() as i32 * 8;
        write_u16(stream, MAGIC_NUMBER)?;
        write_u32(stream, FORMAT_VERSION)?;
        write_i32(stream, type_code)?;
        self.data.serialize(stream)?;

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

pub fn load(stream: &mut File) -> io::Result<DataChunk> {
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
    stream.write_all(&buffer)?;

    Ok(())
}

fn write_u16(stream: &mut File, word: u16) -> io::Result<()> {
    let buffer = word.to_be_bytes();
    stream.write_all(&buffer)?;

    Ok(())
}

fn write_u32(stream: &mut File, word: u32) -> io::Result<()> {
    let buffer = word.to_be_bytes();
    stream.write_all(&buffer)?;

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
    use ndarray::{arr2, Array2};
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
        let built = build(data.into_iter(), 2);
        assert_eq!(
            built.data.iter_cell(0, 5, 0, 0).collect::<Vec<i32>>(),
            vec![9, 9, 9, 9, 9]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.34488282);
    }

    #[test]
    fn save_load_i32() -> io::Result<()> {
        let data = array();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            I32(chunk) => {
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
    fn save_load_u32() -> io::Result<()> {
        let data = array();
        let data: Vec<Array2<u32>> = data.into_iter().map(|a| a.map(|n| *n as u32)).collect();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            U32(chunk) => {
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
    fn save_load_i64() -> io::Result<()> {
        let data = array();
        let data: Vec<Array2<i64>> = data.into_iter().map(|a| a.map(|n| *n as i64)).collect();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            I64(chunk) => {
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
    fn save_load_u64() -> io::Result<()> {
        let data = array();
        let data: Vec<Array2<u64>> = data.into_iter().map(|a| a.map(|n| *n as u64)).collect();
        let built = build(data.into_iter(), 2);

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            U64(chunk) => {
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
    fn suggest_fraction_3bits() {
        let data = array_float();
        let fraction = suggest_fraction(data.into_iter(), 15.0);
        match fraction {
            Precise(bits) => {
                assert_eq!(bits, 3);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn suggest_fraction_4bits() {
        let data = vec![arr2(&[[16.0, 1.0 / 16.0]])];
        let fraction = suggest_fraction(data.into_iter(), 16.0);
        match fraction {
            Precise(bits) => {
                assert_eq!(bits, 4);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn suggest_fraction_all_the_bits() {
        let data = vec![
            // 1/10 is infinite repeating digits in binary, like 1/3 in decimal. We still don't end
            // up rounding, though, because we can represent all the bits that are in the f64
            // representation. This doesn't introduce any more rounding error than is already
            // there.
            arr2(&[[16.0, 0.1]]),
        ];
        let fraction = suggest_fraction(data.into_iter(), 16.0);
        match fraction {
            Precise(bits) => {
                assert_eq!(bits, 55);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn suggest_fraction_loss_of_precision() {
        let data = vec![
            // 1/10 is infinite repeating digits in binary, like 1/3 in decimal. Unlike the test
            // just before this one, we do have a loss of precision as the 9 bits needed for the
            // whole number part of 316 will push some bits off the right hand side of the fixed
            // point representation for 0.1.
            arr2(&[[316.0, 0.1]]),
        ];
        let fraction = suggest_fraction(data.into_iter(), 316.0);
        match fraction {
            Round(bits) => {
                assert_eq!(bits, 54);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn buildf_f32() {
        let data = array_float();
        let built = buildf(data.into_iter(), 2, Precise(3));
        assert_eq!(
            built.data.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(),
            vec![9.5, 9.5, 9.5, 9.5, 9.5]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.35136718);
    }

    #[test]
    fn save_load_f32() -> io::Result<()> {
        let data = array_float();
        let built = buildf(data.into_iter(), 2, Precise(3));

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F32(chunk) => {
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
    fn save_load_f64() -> io::Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = buildf(data.into_iter(), 2, Precise(3));

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F64(chunk) => {
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
    fn save_load_f64_round() -> io::Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = buildf(data.into_iter(), 2, Round(2));

        let mut file = tempfile()?;
        built.save(&mut file)?;
        file.sync_all()?;
        file.rewind()?;

        match load(&mut file)? {
            F64(chunk) => {
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
