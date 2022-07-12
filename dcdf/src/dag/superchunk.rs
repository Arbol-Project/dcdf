use cid::Cid;
use ndarray::{s, Array2};
use num_traits::Float;
use std::fmt::Debug;
use std::io;

use super::mapper::Mapper;
use super::resolver::Resolver;
use crate::codec::Dac;
use crate::codec::FChunk;
use crate::fixed::{to_fixed, Fraction, Precise, Round};
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
    local: Vec<FChunk<N>>,

    /// Resolver for retrieving subchunks
    resolver: Resolver<M, N>,

    stride: usize,
    subsidelen: usize,
}

impl<M, N> Superchunk<M, N>
where
    M: Mapper,
    N: Float + Debug,
{
    /// Iterate over a cell's value across time instants.
    ///
    pub fn iter_cell<'a>(
        &'a mut self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> io::Result<Box<dyn Iterator<Item = N> + 'a>> {
        // Theoretically compiler will optimize to single DIVREM instructions
        // https://stackoverflow.com/questions/69051429
        //      /what-is-the-function-to-get-the-quotient-and-remainder-divmod-for-rust
        let chunk_row = row / self.stride;
        let local_row = row % self.stride;
        let chunk_col = col / self.stride;
        let local_col = col % self.stride;

        let chunk = chunk_row * self.subsidelen + chunk_col;
        match self.references[chunk] {
            Reference::Local(index) => {
                let chunk = &self.local[index];
                Ok(chunk.iter_cell(start, end, local_row, local_col))
            }
            Reference::External(cid) => {
                let chunk = self.resolver.get_chunk(cid)?;

                Ok(chunk.iter_cell(start, end, local_row, local_col))
            }
        }
    }
}

enum Reference {
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
    stride: usize,
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
        let stride = sidelen / subsidelen;
        let subchunks = subsidelen.pow(2);
        let mut builders = Vec::with_capacity(subchunks);
        let mut max = Vec::new();
        let mut min = Vec::new();

        for (subarray, min_value, max_value) in iter_subarrays(first, subsidelen, stride) {
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
            stride,
        }
    }

    fn push(&mut self, a: Array2<N>) {
        for ((subarray, min_value, max_value), builder) in
            iter_subarrays(a, self.subsidelen, self.stride).zip(&mut self.builders)
        {
            builder.push(subarray);
            self.min.push(min_value);
            self.max.push(max_value);
        }
    }

    fn finish(mut self) -> io::Result<Superchunk<M, N>> {
        let chunks: Vec<FChunk<N>> = self
            .builders
            .into_iter()
            .map(|builder| builder.finish().data)
            .collect();
        let mut local = Vec::new();
        let mut references = Vec::new();
        let instants = chunks[0].shape()[0];
        for chunk in chunks {
            if chunk.size() < self.local_threshold {
                let index = local.len();
                local.push(chunk);
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
            resolver: Resolver::new(self.mapper),
            stride: self.stride,
            subsidelen: self.subsidelen,
        })
    }
}

fn iter_subarrays<N>(a: Array2<N>, subsidelen: usize, stride: usize) -> SubarrayIterator<N>
where
    N: Float + Debug,
{
    SubarrayIterator {
        a,
        subsidelen,
        stride,
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
    stride: usize,
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

        let top = self.row * self.stride;
        let bottom = top + self.stride;
        let left = self.col * self.stride;
        let right = left + self.stride;
        let subarray = self.a.slice(s![top..bottom, left..right]).to_owned();

        let col = self.col + 1;
        if col == self.subsidelen {
            self.col = 0;
            self.row += 1;
        } else {
            self.col = col;
        }

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

    use super::super::mapper::{Sha2_256Write, StoreWrite};
    use cid::Cid;
    use ndarray::arr2;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::mem::replace;

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
            Box::new(MemoryMapperStoreWrite::new(self))
        }

        fn load(&mut self, cid: Cid) -> Option<Box<dyn Read + '_>> {
            let object = self.objects.get(&cid)?;
            Some(Box::new(object.as_slice()))
        }
    }

    struct MemoryMapperStoreWrite<'a> {
        mapper: &'a mut MemoryMapper,
        writer: Sha2_256Write<Vec<u8>>,
    }

    impl<'a> MemoryMapperStoreWrite<'a> {
        fn new(mapper: &'a mut MemoryMapper) -> Self {
            let writer = Sha2_256Write::wrap(Vec::new());
            Self { mapper, writer }
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
            let object = replace(&mut self.writer.inner, vec![]);
            self.mapper.objects.insert(cid, object);

            cid
        }
    }

    fn mapper() -> MemoryMapper {
        MemoryMapper::new()
    }

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

    fn test_all_the_things(
        data: Vec<Array2<f32>>,
        chunk: Superchunk<MemoryMapper, f32>,
    ) -> io::Result<()> {
        test_iter_cell(data, chunk)?;

        Ok(())
    }

    fn test_iter_cell(
        data: Vec<Array2<f32>>,
        mut chunk: Superchunk<MemoryMapper, f32>,
    ) -> io::Result<()> {
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
    fn build_no_subchunks() -> io::Result<()> {
        let data = array();
        let chunk = build_superchunk(data.clone().into_iter(), mapper(), 3, 2, Precise(3), 0)?;
        assert_eq!(chunk.references.len(), 64);
        test_all_the_things(data, chunk)?;

        Ok(())
    }
}
