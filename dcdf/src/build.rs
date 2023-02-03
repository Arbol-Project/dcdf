use std::{
    cmp,
    collections::HashMap,
    fmt::Debug,
    mem::{replace, size_of},
    sync::Arc,
};

use cid::Cid;
use ndarray::{s, Array2};
use num_traits::Float;
use parking_lot::Mutex;

use super::{
    cache::Cacheable,
    codec::{Block, Chunk, Dac, FChunk, Log, Snapshot},
    dag::{
        links::Links,
        resolver::Resolver,
        superchunk::{Reference, Superchunk},
    },
    errors::Result,
    fixed::{to_fixed, Fraction, Precise, Round},
};

pub struct SubchunkBuild<N>
where
    N: Float + Debug + Send + Sync,
{
    pub data: FChunk<N>,
    pub logs: usize,
    pub snapshots: usize,
    pub compression: f32,
}

pub struct SubchunkBuilder<N>
where
    N: Float + Debug + Send + Sync,
{
    count_snapshots: usize,
    count_logs: usize,
    snap_array: Array2<N>,
    snapshot: Snapshot<i64>,
    blocks: Vec<Block<i64>>,
    logs: Vec<Log<i64>>,
    rows: usize,
    cols: usize,
    k: i32,
    fraction: Fraction,
}

impl<N> SubchunkBuilder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub fn new(first: Array2<N>, k: i32, fraction: Fraction) -> Self {
        let shape = first.shape();
        let rows = shape[0];
        let cols = shape[1];

        let get = |row, col| match fraction {
            Precise(bits) => to_fixed(first[[row, col]], bits, false),
            Round(bits) => to_fixed(first[[row, col]], bits, true),
        };
        let snapshot = Snapshot::build(get, [rows, cols], k);

        SubchunkBuilder {
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

    pub fn push(&mut self, instant: Array2<N>) {
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

    pub fn finish(mut self) -> SubchunkBuild<N> {
        self.count_snapshots += 1;
        self.count_logs += self.logs.len();
        self.blocks.push(Block::new(self.snapshot, self.logs));

        let fractional_bits = match self.fraction {
            Precise(bits) => bits,
            Round(bits) => bits,
        };
        let chunk = FChunk::new(Chunk::from(self.blocks), fractional_bits);
        let count_instants = self.count_snapshots + self.count_logs;
        let word_size = size_of::<N>();
        let compressed = chunk.size() + 2 /* magic number */ + 4 /* version */;
        let uncompressed = count_instants * self.rows * self.cols * word_size;
        let compression = compressed as f32 / uncompressed as f32;

        SubchunkBuild {
            data: chunk,
            logs: self.count_logs,
            snapshots: self.count_snapshots,
            compression: compression,
        }
    }
}

pub struct SuperchunkBuild<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub data: Superchunk<N>,
    pub size: u64,
    pub size_external: u64,
    pub sizes: Vec<u64>,
    pub compression: f32,
    pub elided: usize,
    pub local: usize,
    pub external: usize,
}

pub async fn build_superchunk<I, N>(
    mut instants: I,
    resolver: Arc<Resolver<N>>,
    levels: usize,
    k: i32,
    fraction: Fraction,
    local_threshold: u64,
) -> Result<SuperchunkBuild<N>>
where
    I: Iterator<Item = Array2<N>>,
    N: Float + Debug + Send + Sync,
{
    let first = instants.next().expect("No time instants to encode");
    let mut builder = SuperchunkBuilder::new(first, k, fraction, levels, resolver, local_threshold);
    for instant in instants {
        builder.push(instant).await;
    }
    builder.finish().await
}

pub struct SuperchunkBuilder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    builders: Vec<SubchunkBuilder<N>>,
    min: Vec<N>,
    max: Vec<N>,
    rows: usize,
    cols: usize,
    resolver: Arc<Resolver<N>>,
    levels: usize,
    fraction: Fraction,
    local_threshold: u64,
    sidelen: usize,
    subsidelen: usize,
    chunks_sidelen: usize,
}

// TODO Parallelize building of subchunks

impl<N> SuperchunkBuilder<N>
where
    N: Float + Debug + Send + Sync,
{
    pub fn new(
        first: Array2<N>,
        k: i32,
        fraction: Fraction,
        levels: usize,
        resolver: Arc<Resolver<N>>,
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

        println!("Building superchunk with {subsidelen}x{subsidelen} ({subchunks}) subchunks");
        println!("\tsubchunk size: {chunks_sidelen}x{chunks_sidelen}");
        for (subarray, min_value, max_value) in iter_subarrays(first, subsidelen, chunks_sidelen) {
            builders.push(SubchunkBuilder::new(subarray, k, fraction));
            min.push(min_value);
            max.push(max_value);
        }

        Self {
            builders,
            min,
            max,
            rows,
            cols,
            resolver,
            levels,
            fraction,
            local_threshold,
            sidelen,
            subsidelen,
            chunks_sidelen,
        }
    }

    pub async fn push(&mut self, a: Array2<N>) {
        for ((subarray, min_value, max_value), builder) in
            iter_subarrays(a, self.subsidelen, self.chunks_sidelen).zip(&mut self.builders)
        {
            builder.push(subarray);
            self.min.push(min_value);
            self.max.push(max_value);
        }
    }

    /// Return whether a chunk, referred to by index, should be elided.
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

    pub async fn finish(mut self) -> Result<SuperchunkBuild<N>> {
        // Swap builders out of self before moving so that self can be borrowed by methods later.
        let builders = replace(&mut self.builders, vec![]);
        let builds: Vec<SubchunkBuild<N>> = builders
            .into_iter()
            .map(|builder| builder.finish())
            .collect();

        let mut local_references: HashMap<Cid, usize> = HashMap::new();
        let mut local = Vec::new();
        let mut external_references: HashMap<Cid, usize> = HashMap::new();
        let mut external = Links::new();
        let mut references = Vec::new();
        let mut sizes = Vec::new();
        let mut elided = 0;
        let instants = builds[0].data.shape()[0];
        for (i, build) in builds.into_iter().enumerate() {
            if self.elide(i) {
                elided += 1;
                references.push(Reference::Elided);
            } else if build.data.size() < self.local_threshold {
                let cid = self.resolver.hash_subchunk(&build.data)?;
                let index = match local_references.get(&cid) {
                    Some(index) => *index,
                    None => {
                        let index = local.len();
                        local.push(Arc::new(build.data));
                        local_references.insert(cid, index);

                        index
                    }
                };
                references.push(Reference::Local(index))
            } else {
                sizes.push(build.data.size());
                let cid = self.resolver.save_async(build.data).await?;
                let index = match external_references.get(&cid) {
                    Some(index) => *index,
                    None => {
                        let index = external.len();
                        external.push(cid);
                        external_references.insert(cid, index);

                        index
                    }
                };
                references.push(Reference::External(index));
            }
        }

        let size_external = external.size();
        let local_len = local.len();
        let external_len = external.len();
        let external_cid = self.resolver.save_async(external).await?;

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

        let data = Superchunk::new(
            [instants, self.rows, self.cols],
            self.sidelen,
            self.levels,
            references,
            Dac::from(max),
            Dac::from(min),
            local,
            external_cid,
            Mutex::new(None),
            Arc::clone(&self.resolver),
            bits,
            self.chunks_sidelen,
            self.subsidelen,
        );

        let size = data.size();
        let compressed = size + size_external + sizes.iter().sum::<u64>();
        let word_size = size_of::<N>();
        let uncompressed = instants * self.rows * self.cols * word_size;
        let compression = compressed as f32 / uncompressed as f32;

        Ok(SuperchunkBuild {
            data,
            size,
            size_external,
            sizes,
            compression,
            elided,
            local: local_len,
            external: external_len,
        })
    }
}

/// Iterate over subarrays of array.
///
/// Used to build individual chunks that comprise the superchunk.
///
fn iter_subarrays<N>(a: Array2<N>, subsidelen: usize, chunks_sidelen: usize) -> SubarrayIterator<N>
where
    N: Float + Debug + Send + Sync,
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
    N: Float + Debug + Send + Sync,
{
    a: Array2<N>,
    subsidelen: usize,
    chunks_sidelen: usize,
    row: usize,
    col: usize,
}

impl<N> Iterator for SubarrayIterator<N>
where
    N: Float + Debug + Send + Sync,
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
    use ndarray::{arr2, Array2};
    use std::sync::Arc;

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

    fn build_subchunk<N, T>(mut instants: T, k: i32, fraction: Fraction) -> SubchunkBuild<N>
    where
        N: Float + Debug + Send + Sync + 'static,
        T: Iterator<Item = Array2<N>>,
    {
        let first = instants.next().expect("No time instants to encode");
        let mut builder = SubchunkBuilder::new(first, k, fraction);
        for instant in instants {
            builder.push(instant);
        }
        builder.finish()
    }

    #[test]
    fn build_subchunk_f32() {
        let data = array_float();
        let built = build_subchunk(data.into_iter(), 2, Precise(3));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.iter_cell(0, 5, 0, 0).collect::<Vec<f32>>(),
            vec![9.5, 9.5, 9.5, 9.5, 9.5]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.3521875);
    }

    #[test]
    fn build_subchunk_f64() {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = build_subchunk(data.into_iter(), 2, Precise(3));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.iter_cell(0, 5, 0, 0).collect::<Vec<f64>>(),
            vec![9.5, 9.5, 9.5, 9.5, 9.5]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.17609376);
    }

    #[test]
    fn build_subchunk_f64_round() {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = build_subchunk(data.into_iter(), 2, Round(2));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.iter_cell(0, 5, 2, 4).collect::<Vec<f64>>(),
            vec![3.5, 5.0, 5.0, 3.5, 5.0]
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);
        assert_eq!(built.compression, 0.16294922);
    }
}
