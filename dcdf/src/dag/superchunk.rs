use cid::Cid;
use ndarray::{s, Array2, Array3, ArrayBase, DataMut, Ix3};
use num_traits::Float;
use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::mem;
use std::rc::Rc;

use super::mapper::Mapper;
use super::resolver::Resolver;
use crate::codec::Dac;
use crate::codec::FChunk;
use crate::fixed::{from_fixed, to_fixed, Fraction, Precise, Round};
use crate::helpers::rearrange;
use crate::simple::FBuilder;

pub struct Superchunk<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    /// The K in K²-Raster. Each level of the tree structure is divided into k² subtrees.
    /// In practice, this will almost always be 2.
    k: i32,

    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    pub shape: [usize; 3],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,

    /// Number of tree levels stored in Superchunk. Lower levels are stored in subchunks.
    /// The idea of tree levels here is notional, as only the superchunk's leaf nodes are
    /// represented in any meaningful way. Number of subchunks equals k.pow(2).pow(levels)
    levels: usize,

    /// References to subchunks
    references: Vec<Reference>,

    /// Max values for each subchunk at each instant
    max: Dac,

    /// Min values for each subchunk at each instant
    min: Dac,

    /// Locally stored subchunks
    local: Vec<Rc<FChunk<N>>>,

    /// Resolver for retrieving subchunks
    resolver: Rc<RefCell<Resolver<M, N>>>,

    /// Number of fractional bits stored in fixed point number representation
    fractional_bits: usize,

    /// The side length of the subchunks stored in this superchunk
    chunks_sidelen: usize,

    /// The number of subchunks per side in the logical grid represented by this superchunk
    subsidelen: usize,
}

impl<M, N> Superchunk<M, N>
where
    M: Mapper + 'static,
    N: Float + Debug + 'static,
{
    /// Iterate over a cell's value across time instants.
    ///
    pub fn iter_cell(
        self: &Rc<Self>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> io::Result<CellIter<N>> {
        let (start, end) = rearrange(start, end);
        self.check_bounds(end - 1, row, col);

        // Theoretically compiler will optimize to single DIVREM instructions
        // https://stackoverflow.com/questions/69051429
        //      /what-is-the-function-to-get-the-quotient-and-remainder-divmod-for-rust
        let chunk_row = row / self.chunks_sidelen;
        let local_row = row % self.chunks_sidelen;
        let chunk_col = col / self.chunks_sidelen;
        let local_col = col % self.chunks_sidelen;

        let chunk = chunk_row * self.subsidelen + chunk_col;
        match self.references[chunk] {
            Reference::Elided => Ok(CellIter {
                inner: Rc::new(RefCell::new(SuperCellIter::new(self, start, end, chunk))),
            }),
            Reference::Local(index) => {
                let chunk = &self.local[index];
                Ok(CellIter {
                    inner: Rc::new(RefCell::new(
                        chunk.iter_cell(start, end, local_row, local_col),
                    )),
                })
            }
            Reference::External(cid) => {
                let chunk = self.resolver.borrow_mut().get_chunk(cid)?;

                Ok(CellIter {
                    inner: Rc::new(RefCell::new(
                        chunk.iter_cell(start, end, local_row, local_col),
                    )),
                })
            }
        }
    }

    /// Get a subarray of this Chunk.
    ///
    pub fn get_window(
        self: &Rc<Self>,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> io::Result<Array3<N>> {
        let (start, end) = rearrange(start, end);
        let (top, bottom) = rearrange(top, bottom);
        let (left, right) = rearrange(left, right);
        self.check_bounds(end - 1, bottom - 1, right - 1);

        let instants = end - start;
        let rows = bottom - top;
        let cols = right - left;
        let mut window = Array3::zeros([instants, rows, cols]);

        self.fill_window(start, top, left, &mut window)?;

        Ok(window)
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    pub(crate) fn fill_window<S>(
        self: &Rc<Self>,
        start: usize,
        top: usize,
        left: usize,
        window: &mut ArrayBase<S, Ix3>,
    ) -> io::Result<()> 
    where
        S: DataMut<Elem = N>,
    {
        let shape = window.shape();
        let instants = shape[0];
        let rows = shape[1];
        let cols = shape[2];

        let bottom = top + rows;
        let right = left + cols;

        let chunks_top = top / self.chunks_sidelen;
        let chunks_bottom = (bottom - 1) / self.chunks_sidelen;
        let chunks_left = left / self.chunks_sidelen;
        let chunks_right = (right - 1) / self.chunks_sidelen;

        for chunk_row in chunks_top..=chunks_bottom {
            let chunk_top = chunk_row * self.chunks_sidelen;
            let window_top = cmp::max(chunk_top, top);
            let local_top = window_top - chunk_top;
            let slice_top = window_top - top;

            let chunk_bottom = chunk_top + self.chunks_sidelen;
            let window_bottom = cmp::min(chunk_bottom, bottom);
            let slice_bottom = window_bottom - top;

            for chunk_col in chunks_left..=chunks_right {
                let chunk_left = chunk_col * self.chunks_sidelen;
                let window_left = cmp::max(chunk_left, left);
                let local_left = window_left - chunk_left;
                let slice_left = window_left - left;

                let chunk_right = chunk_left + self.chunks_sidelen;
                let window_right = cmp::min(chunk_right, right);
                let slice_right = window_right - left;

                let mut subwindow = window.slice_mut(s![.., slice_top..slice_bottom, slice_left..slice_right]);

                let chunk = chunk_row * self.subsidelen + chunk_col;
                match self.references[chunk] {
                    Reference::Elided => {
                        let stride = self.subsidelen * self.subsidelen;
                        let mut index = chunk + start * stride;
                        for i in 0..instants {
                            let value = self.max.get(index);
                            let value = from_fixed(value, self.fractional_bits);
                            let mut slice = subwindow.slice_mut(s![i, .., ..]);
                            slice.fill(value);
                            index += stride;
                        }
                    },
                    Reference::Local(index) => {
                        let chunk = &self.local[index];
                        chunk.fill_window(start, local_top, local_left, &mut subwindow);
                    },
                    Reference::External(cid) => {
                        let chunk = self.resolver.borrow_mut().get_chunk(cid)?;
                        chunk.fill_window(start, local_top, local_left, &mut subwindow);
                    },
                }
            }
        }

        Ok(())
    }
 
    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape;
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::Superchunk: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

pub struct CellIter<N>
where
    N: Float + Debug,
{
    inner: Rc<RefCell<dyn Iterator<Item = N>>>,
}

impl<N> Iterator for CellIter<N>
where
    N: Float + Debug,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.borrow_mut().next()
    }
}

struct SuperCellIter<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    superchunk: Rc<Superchunk<M, N>>,
    index: usize,
    stride: usize,
    remaining: usize,
}

impl<M, N> SuperCellIter<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    fn new(
        superchunk: &Rc<Superchunk<M, N>>,
        start: usize,
        end: usize,
        chunk_index: usize,
    ) -> Self {
        let stride = superchunk.subsidelen * superchunk.subsidelen;
        let index = chunk_index + start * stride;
        let remaining = end - start;

        Self {
            superchunk: Rc::clone(superchunk),
            index,
            stride,
            remaining,
        }
    }
}

impl<M, N> Iterator for SuperCellIter<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        match self.remaining {
            0 => None,
            _ => {
                let value = self.superchunk.max.get(self.index);
                self.index += self.stride;
                self.remaining -= 1;

                Some(from_fixed(value, self.superchunk.fractional_bits))
            }
        }
    }
}

enum Reference {
    Elided,
    Local(usize),
    External(Cid),
}

fn build_superchunk<I, M, N>(
    mut instants: I,
    mapper: M,
    levels: usize,
    k: i32,
    fraction: Fraction,
    local_threshold: u64,
) -> io::Result<Superchunk<M, N>>
where
    I: Iterator<Item = Array2<N>>,
    M: Mapper,
    N: Float + Debug,
{
    let first = instants.next().expect("No time instants to encode");
    let mut builder = SuperchunkBuilder::new(first, k, fraction, levels, mapper, local_threshold);
    for instant in instants {
        builder.push(instant);
    }
    builder.finish()
}

struct SuperchunkBuilder<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    builders: Vec<FBuilder<N>>,
    min: Vec<N>,
    max: Vec<N>,
    rows: usize,
    cols: usize,
    mapper: M,
    levels: usize,
    fraction: Fraction,
    local_threshold: u64,
    k: i32,
    sidelen: usize,
    subsidelen: usize,
    chunks_sidelen: usize,
}

impl<M, N> SuperchunkBuilder<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    fn new(
        first: Array2<N>,
        k: i32,
        fraction: Fraction,
        levels: usize,
        mapper: M,
        local_threshold: u64,
    ) -> Self {
        let shape = first.shape();
        let rows = shape[0];
        let cols = shape[1];

        let sidelen = *shape.iter().max().unwrap() as f64;
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let subsidelen = k.pow(levels as u32) as usize;
        let chunks_sidelen = sidelen / subsidelen;
        let subchunks = subsidelen.pow(2);
        let mut builders = Vec::with_capacity(subchunks);
        let mut max = Vec::new();
        let mut min = Vec::new();

        for (subarray, min_value, max_value) in iter_subarrays(first, subsidelen, chunks_sidelen) {
            builders.push(FBuilder::new(subarray, k, fraction));
            min.push(min_value);
            max.push(max_value);
        }

        Self {
            builders,
            min,
            max,
            rows,
            cols,
            mapper,
            levels,
            fraction,
            local_threshold,
            k,
            sidelen,
            subsidelen,
            chunks_sidelen,
        }
    }

    fn push(&mut self, a: Array2<N>) {
        for ((subarray, min_value, max_value), builder) in
            iter_subarrays(a, self.subsidelen, self.chunks_sidelen).zip(&mut self.builders)
        {
            builder.push(subarray);
            self.min.push(min_value);
            self.max.push(max_value);
        }
    }

    /// Return whether a chunk, referred to be index, should be elided.
    ///
    /// A chunk can be elided if the minimum value equals the maximum value for every time instant
    /// in that chunk.
    ///
    fn elide(&self, i: usize) -> bool {
        let mut i = i;
        let stride = self.subsidelen * self.subsidelen;
        let end = self.max.len();
        while i < end {
            if self.max[i] != self.min[i] {
                return false;
            }
            i += stride;
        }

        true
    }

    fn finish(mut self) -> io::Result<Superchunk<M, N>> {
        // Swap builders out of self before moving so that self can be borrowed by methods later.
        let builders = mem::replace(&mut self.builders, vec![]);
        let chunks: Vec<FChunk<N>> = builders
            .into_iter()
            .map(|builder| builder.finish().data)
            .collect();
        let mut local_references: HashMap<Cid, usize> = HashMap::new();
        let mut local = Vec::new();
        let mut references = Vec::new();
        let instants = chunks[0].shape()[0];
        for (i, chunk) in chunks.into_iter().enumerate() {
            if self.elide(i) {
                references.push(Reference::Elided);
            } else if chunk.size() < self.local_threshold {
                let mut hasher = self.mapper.hash();
                chunk.serialize(&mut hasher)?;

                let cid = hasher.finish();
                let index = match local_references.get(&cid) {
                    Some(index) => *index,
                    None => {
                        let index = local.len();
                        local.push(Rc::new(chunk));
                        local_references.insert(cid, index);

                        index
                    }
                };
                references.push(Reference::Local(index))
            } else {
                let mut writer = self.mapper.store();
                chunk.serialize(&mut writer)?;

                let cid = writer.finish();
                references.push(Reference::External(cid));
            }
        }

        let (round, bits) = match self.fraction {
            Round(bits) => (true, bits),
            Precise(bits) => (false, bits),
        };

        let max: Vec<i64> = self
            .max
            .into_iter()
            .map(|n| to_fixed(n, bits, round))
            .collect();
        let min: Vec<i64> = self
            .min
            .into_iter()
            .map(|n| to_fixed(n, bits, round))
            .collect();

        Ok(Superchunk {
            k: self.k,
            shape: [instants, self.rows, self.cols],
            sidelen: self.sidelen,
            levels: self.levels,
            references,
            max: Dac::from(max),
            min: Dac::from(min),
            local,
            resolver: Rc::new(RefCell::new(Resolver::new(self.mapper))),
            fractional_bits: bits,
            chunks_sidelen: self.chunks_sidelen,
            subsidelen: self.subsidelen,
        })
    }
}

/// Iterate over subarrays of array. 
///
/// Used to build individual chunks that comprise the superchunk.
///
fn iter_subarrays<N>(a: Array2<N>, subsidelen: usize, chunks_sidelen: usize) -> SubarrayIterator<N>
where
    N: Float + Debug,
{
    SubarrayIterator {
        a,
        subsidelen,
        chunks_sidelen,
        row: 0,
        col: 0,
    }
}

struct SubarrayIterator<N>
where
    N: Float + Debug,
{
    a: Array2<N>,
    subsidelen: usize,
    chunks_sidelen: usize,
    row: usize,
    col: usize,
}

impl<N> Iterator for SubarrayIterator<N>
where
    N: Float + Debug,
{
    type Item = (Array2<N>, N, N);

    fn next(&mut self) -> Option<Self::Item> {
        if self.row == self.subsidelen {
            return None;
        }

        let shape = self.a.shape();
        let rows = shape[0];
        let cols = shape[1];

        let top = self.row * self.chunks_sidelen;
        let left = self.col * self.chunks_sidelen;

        let col = self.col + 1;
        if col == self.subsidelen {
            self.col = 0;
            self.row += 1;
        } else {
            self.col = col;
        }

        if top >= rows || left >= cols {
            // This subarray is entirely outside the bounds of the actual array. This can happen
            // becaue the logical array is expanded to have a square shape with the side lengths a
            // power of k.
            return Some((Array2::zeros([0, 0]), N::zero(), N::zero()));
        }

        let bottom = cmp::min(top + self.chunks_sidelen, rows);
        let right = cmp::min(left + self.chunks_sidelen, cols);
        let subarray = self.a.slice(s![top..bottom, left..right]).to_owned();

        let value = subarray[[0, 0]];
        let (min_value, max_value) =
            subarray
                .iter()
                .fold((value, value), |(min_value, max_value), value| {
                    let min_value = if *value < min_value {
                        *value
                    } else {
                        min_value
                    };
                    let max_value = if *value > max_value {
                        *value
                    } else {
                        max_value
                    };
                    (min_value, max_value)
                });

        Some((subarray, min_value, max_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dag::mapper::{Sha2_256Write, StoreWrite};

    use cid::Cid;
    use ndarray::arr2;
    use paste::paste;
    use std::collections::HashSet;
    use std::io::{Read, Write};

    /// A test implementation of Mapper that stores objects in RAM
    ///
    struct MemoryMapper {
        objects: HashMap<Cid, Vec<u8>>,
    }

    impl MemoryMapper {
        fn new() -> Self {
            Self {
                objects: HashMap::new(),
            }
        }
    }

    impl Mapper for MemoryMapper {
        fn store(&mut self) -> Box<dyn StoreWrite + '_> {
            Box::new(MemoryMapperStoreWrite::new(self, false))
        }

        fn hash(&mut self) -> Box<dyn StoreWrite + '_> {
            Box::new(MemoryMapperStoreWrite::new(self, true))
        }

        fn load(&mut self, cid: Cid) -> Option<Box<dyn Read + '_>> {
            let object = self.objects.get(&cid)?;
            Some(Box::new(object.as_slice()))
        }
    }

    struct MemoryMapperStoreWrite<'a> {
        mapper: &'a mut MemoryMapper,
        writer: Sha2_256Write<Vec<u8>>,
        hash_only: bool,
    }

    impl<'a> MemoryMapperStoreWrite<'a> {
        fn new(mapper: &'a mut MemoryMapper, hash_only: bool) -> Self {
            let writer = Sha2_256Write::wrap(Vec::new());
            Self {
                mapper,
                writer,
                hash_only,
            }
        }
    }

    impl<'a> Write for MemoryMapperStoreWrite<'a> {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writer.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.writer.flush()
        }
    }

    impl<'a> StoreWrite for MemoryMapperStoreWrite<'a> {
        fn finish(&mut self) -> Cid {
            let cid = self.writer.finish();
            if !self.hash_only {
                let object = mem::replace(&mut self.writer.inner, vec![]);
                self.mapper.objects.insert(cid, object);
            }

            cid
        }
    }

    fn mapper() -> MemoryMapper {
        MemoryMapper::new()
    }

    fn array8() -> Vec<Array2<f32>> {
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

    fn array(sidelen: usize) -> Vec<Array2<f32>> {
        let data = array8();

        data.into_iter()
            .map(|a| Array2::from_shape_fn((sidelen, sidelen), |(row, col)| a[[row % 8, col % 8]]))
            .collect()
    }

    macro_rules! test_all_the_things {
        ($name:ident) => { paste! {
            #[test]
            fn [<$name _test_iter_cell>]() -> io::Result<()> {
                let (data, chunk) = $name()?;
                let [instants, rows, cols] = chunk.shape;
                for row in 0..rows {
                    for col in 0..cols {
                        let start = row + col;
                        let end = instants - start;
                        let values: Vec<f32> = chunk.iter_cell(start, end, row, col)?.collect();
                        assert_eq!(values.len(), end - start);
                        for i in 0..values.len() {
                            assert_eq!(values[i], data[i + start][[row, col]]);
                        }
                    }
                }

                Ok(())
            }

            #[test]
            fn [<$name _test_iter_cell_rearrange>]() -> io::Result<()> {
                let (data, chunk) = $name()?;
                let [instants, rows, cols] = chunk.shape;
                for row in 0..rows {
                    for col in 0..cols {
                        let start = row + col;
                        let end = instants - start;
                        let values: Vec<f32> = chunk.iter_cell(end, start, row, col)?.collect();
                        assert_eq!(values.len(), end - start);
                        for i in 0..values.len() {
                            assert_eq!(values[i], data[i + start][[row, col]]);
                        }
                    }
                }

                Ok(())
            }

            #[test]
            #[should_panic]
            fn [<$name _test_iter_cell_time_out_of_bounds>]() {
                let (_, chunk) = $name().expect("This should work");
                let [instants, rows, cols] = chunk.shape;

                let values: Vec<f32> = chunk.iter_cell(0, instants + 1, rows, cols)
                    .expect("This isn't what causes the panic").collect();
                assert_eq!(values.len(), instants + 1);
            }

            #[test]
            #[should_panic]
            fn [<$name _test_iter_cell_row_out_of_bounds>]() {
                let (_, chunk) = $name().expect("This should work");
                let [instants, rows, cols] = chunk.shape;

                let values: Vec<f32> = chunk.iter_cell(0, instants, rows + 1, cols)
                    .expect("This isn't what causes the panic").collect();
                assert_eq!(values.len(), instants + 1);
            }

            #[test]
            #[should_panic]
            fn [<$name _test_iter_cell_col_out_of_bounds>]() {
                let (_, chunk) = $name().expect("This should work");
                let [instants, rows, cols] = chunk.shape;

                let values: Vec<f32> = chunk.iter_cell(0, instants, rows, cols + 1)
                    .expect("This isn't what causes the panic").collect();
                assert_eq!(values.len(), instants + 1);
            }

            #[test]
            fn [<$name _test_get_window>]() -> io::Result<()> {
                let (data, chunk) = $name()?;
                let [instants, rows, cols] = chunk.shape;
                for top in 0..rows / 2 {
                    let bottom = top + rows / 2;
                    for left in 0..cols / 2 {
                        let right = left + cols / 2;
                        let start = top + bottom;
                        let end = instants - start;
                        let window = chunk.get_window(start, end, top, bottom, left, right)?;

                        assert_eq!(window.shape(), 
                                   [end - start, bottom - top, right - left]);
                        
                        for i in 0..end - start {
                            assert_eq!(
                                window.slice(s![i, .., ..]),
                                data[start + i].slice(s![top..bottom, left..right])
                            );
                        }
                    }
                }

                Ok(())
            }
        }}
    }

    fn no_subchunks() -> io::Result<(Vec<Array2<f32>>, Rc<Superchunk<MemoryMapper, f32>>)> {
        let data = array8();
        let chunk = build_superchunk(data.clone().into_iter(), mapper(), 3, 2, Precise(3), 0)?;
        let chunk = Rc::new(chunk);
        assert_eq!(chunk.references.len(), 64);

        Ok((data, chunk))
    }

    test_all_the_things!(no_subchunks);

    fn local_subchunks() -> io::Result<(Vec<Array2<f32>>, Rc<Superchunk<MemoryMapper, f32>>)> {
        let data = array(16);
        let chunk = build_superchunk(
            data.clone().into_iter(),
            mapper(),
            2,
            2,
            Precise(3),
            1 << 14,
        )?;
        let chunk = Rc::new(chunk);
        assert_eq!(chunk.references.len(), 16);
        assert_eq!(chunk.local.len(), 4);

        Ok((data, chunk))
    }

    test_all_the_things!(local_subchunks);

    fn external_subchunks() -> io::Result<(Vec<Array2<f32>>, Rc<Superchunk<MemoryMapper, f32>>)> {
        let data = array(16);
        let chunk = build_superchunk(data.clone().into_iter(), mapper(), 2, 2, Precise(3), 0)?;
        let chunk = Rc::new(chunk);
        assert_eq!(chunk.references.len(), 16);

        let cids = chunk.references.iter().map(|r| match r {
            Reference::External(cid) => *cid,
            _ => {
                panic!("Expecting extnernal references only");
            }
        });

        let unique_cids: HashSet<Cid> = HashSet::from_iter(cids);
        assert_eq!(unique_cids.len(), 4);

        Ok((data, chunk))
    }

    test_all_the_things!(external_subchunks);

    fn mixed_subchunks() -> io::Result<(Vec<Array2<f32>>, Rc<Superchunk<MemoryMapper, f32>>)> {
        let data = array(17);
        let chunk = build_superchunk(data.clone().into_iter(), mapper(), 2, 2, Precise(3), 8000)?;
        let chunk = Rc::new(chunk);
        assert_eq!(chunk.references.len(), 16);

        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in chunk.references.iter() {
            match r {
                Reference::External(_) => {
                    external_count += 1;
                }
                Reference::Local(_) => {
                    local_count += 1;
                }
                Reference::Elided => {
                    elided_count += 1;
                }
            }
        }
        assert_eq!(external_count, 4);
        assert_eq!(local_count, 4);
        assert_eq!(elided_count, 8);

        Ok((data, chunk))
    }

    test_all_the_things!(mixed_subchunks);
}
