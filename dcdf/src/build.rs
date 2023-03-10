use std::{cmp, collections::HashMap, fmt::Debug, mem::replace, sync::Arc};

use async_recursion::async_recursion;
use cid::Cid;
use futures::stream::{FuturesOrdered, StreamExt};
use ndarray::{s, Array2};
use num_traits::Float;

use super::{
    cache::Cacheable,
    codec::{Block, Chunk, Dac, FChunk, Log, Snapshot},
    dag::{
        links::Links,
        mmarray::MMArray3,
        resolver::Resolver,
        superchunk::{Reference, Superchunk},
    },
    errors::Result,
    fixed::{to_fixed, Fraction, Precise, Round},
};

pub struct MMArray3Build<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub data: MMArray3<N>,
    pub size: u64,

    pub elided: usize,
    pub local: usize,
    pub external: usize,

    pub snapshots: usize,
    pub logs: usize,
}

pub(crate) enum MMArray3Builder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    Superchunk(SuperchunkBuilder<N>),
    Subchunk(SubchunkBuilder<N>),
    Elide,
}

impl<N> MMArray3Builder<N>
where
    N: Float + Debug + Send + Sync,
{
    #[async_recursion]
    pub(crate) async fn push(&mut self, a: Array2<N>) {
        match self {
            MMArray3Builder::Superchunk(builder) => builder.push(a).await,
            MMArray3Builder::Subchunk(builder) => builder.push(a),
            MMArray3Builder::Elide => {}
        };
    }

    #[async_recursion]
    pub(crate) async fn finish(self) -> Result<Option<MMArray3Build<N>>> {
        let build = match self {
            MMArray3Builder::Superchunk(builder) => Some(builder.finish().await?),
            MMArray3Builder::Subchunk(builder) => Some(builder.finish()?),
            MMArray3Builder::Elide => None,
        };

        Ok(build)
    }
}

pub(crate) struct SubchunkBuilder<N>
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
    pub(crate) fn new(first: Array2<N>, k: i32, fraction: Fraction) -> Self {
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

    pub(crate) fn push(&mut self, instant: Array2<N>) {
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

    pub(crate) fn finish(mut self) -> Result<MMArray3Build<N>> {
        self.count_snapshots += 1;
        self.count_logs += self.logs.len();
        self.blocks.push(Block::new(self.snapshot, self.logs));

        let fractional_bits = match self.fraction {
            Precise(bits) => bits,
            Round(bits) => bits,
        };
        let chunk = MMArray3::Subchunk(FChunk::new(Chunk::from(self.blocks), fractional_bits));
        let size = chunk.size();

        Ok(MMArray3Build {
            data: chunk,
            size,
            elided: 0,
            local: 0,
            external: 0,
            logs: self.count_logs,
            snapshots: self.count_snapshots,
        })
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
    levels: &[u32],
    k: i32,
    fraction: Fraction,
    local_threshold: u64,
) -> Result<MMArray3Build<N>>
where
    I: Iterator<Item = Array2<N>>,
    N: Float + Debug + Send + Sync,
{
    let first = instants.next().expect("No time instants to encode");

    // Find longest side
    let sidelen = *first.shape().iter().max().unwrap() as f64;

    // Find total number of tree levels needed to represent this data
    let total_levels = sidelen.log(k as f64).ceil() as u32;

    // Make sure levels passed in by user match up to levels needed to encode array
    let user_levels = levels.iter().sum::<u32>();
    if user_levels != total_levels {
        panic!(
            "Need {total_levels} tree levels to encode array, but {user_levels} levels \
            passed in."
        );
    }

    let mut builder = SuperchunkBuilder::new(first, k, fraction, levels, resolver, local_threshold);
    for instant in instants {
        builder.push(instant).await;
    }
    builder.finish().await
}

pub(crate) struct SuperchunkBuilder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    builders: Vec<MMArray3Builder<N>>,
    instants: usize,
    min: Vec<N>,
    max: Vec<N>,
    rows: usize,
    cols: usize,
    resolver: Arc<Resolver<N>>,
    levels: u32,
    fraction: Fraction,
    local_threshold: u64,
    sidelen: usize,
    subsidelen: usize,
    chunks_sidelen: usize,
}

impl<N> SuperchunkBuilder<N>
where
    N: Float + Debug + Send + Sync,
{
    pub(crate) fn new(
        first: Array2<N>,
        k: i32,
        fraction: Fraction,
        levels: &[u32],
        resolver: Arc<Resolver<N>>,
        local_threshold: u64,
    ) -> Self {
        let shape = first.shape();
        let rows = shape[0];
        let cols = shape[1];

        // Find longest side
        let sidelen = *shape.iter().max().unwrap() as f64;

        // Adjust sidelen to be lowest power of k that is equal to or greater than the longest side
        let sidelen = k.pow(sidelen.log(k as f64).ceil() as u32) as usize;

        let sublevels = &levels[1..];
        let at_bottom = sublevels.len() == 1;
        let levels = levels[0];

        let subsidelen = k.pow(levels as u32) as usize;
        let chunks_sidelen = sidelen / subsidelen;
        let subchunks = subsidelen.pow(2);
        let mut builders = Vec::with_capacity(subchunks);
        let mut max = Vec::new();
        let mut min = Vec::new();

        for subarray in iter_subarrays(first, subsidelen, chunks_sidelen) {
            match subarray {
                Some((subarray, min_value, max_value)) => {
                    let builder = if at_bottom {
                        MMArray3Builder::Subchunk(SubchunkBuilder::new(subarray, k, fraction))
                    } else {
                        // Find out how many tree levels are needed to represent this subarray. In
                        // cases where the array sides are greatly expanded to find a power of K, we
                        // may wind up with a subarray (in the lower right quadrant) that is
                        // is significantly smaller than other subarrays at the same level and
                        // should be encoded as a superchunk instead of as a subchunk.
                        let sidelen = *subarray.shape().iter().max().unwrap() as f64;
                        let needed_levels = sidelen.log(k as f64).ceil() as u32;
                        if needed_levels <= sublevels[0] {
                            MMArray3Builder::Subchunk(SubchunkBuilder::new(subarray, k, fraction))
                        } else {
                            MMArray3Builder::Superchunk(SuperchunkBuilder::new(
                                subarray,
                                k,
                                fraction,
                                sublevels,
                                Arc::clone(&resolver),
                                local_threshold,
                            ))
                        }
                    };
                    builders.push(builder);
                    min.push(min_value);
                    max.push(max_value);
                }
                None => {
                    builders.push(MMArray3Builder::Elide);
                    min.push(N::zero());
                    max.push(N::zero());
                }
            }
        }

        Self {
            builders,
            instants: 1,
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

    pub(crate) async fn push(&mut self, a: Array2<N>) {
        for (subarray, builder) in
            iter_subarrays(a, self.subsidelen, self.chunks_sidelen).zip(&mut self.builders)
        {
            match subarray {
                Some((subarray, min_value, max_value)) => {
                    builder.push(subarray).await;
                    self.min.push(min_value);
                    self.max.push(max_value);
                }
                None => {
                    self.min.push(N::zero());
                    self.max.push(N::zero());
                }
            }
        }
        self.instants += 1;
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

    pub(crate) async fn finish(mut self) -> Result<MMArray3Build<N>> {
        // Swap builders out of self before moving so that self can be borrowed by methods later.
        let builders = replace(&mut self.builders, vec![]);

        // Asynchronously finish each builder and collect the results
        let builds = builders
            .into_iter()
            .map(|builder| builder.finish())
            .collect::<FuturesOrdered<_>>()
            .collect::<Vec<Result<Option<MMArray3Build<N>>>>>()
            .await
            // Map Vec<Result<_>> to Result<Vec<_>>
            .into_iter()
            .collect::<Result<Vec<Option<MMArray3Build<N>>>>>()?;

        let mut local_references: HashMap<Cid, usize> = HashMap::new();
        let mut local = Vec::new();
        let mut external_references: HashMap<Cid, usize> = HashMap::new();
        let mut external = Links::new();
        let mut references = Vec::new();
        let mut sizes = Vec::new();
        let mut elided = 0;
        let mut snapshots = 0;
        let mut logs = 0;
        for (i, build) in builds.into_iter().enumerate() {
            if self.elide(i) {
                elided += 1;
                references.push(Reference::Elided);
            } else {
                let build = build.unwrap();
                if build.data.size() < self.local_threshold {
                    let cid = self.resolver.hash(&build.data).await?;
                    let index = match local_references.get(&cid) {
                        Some(index) => *index,
                        None => {
                            let index = local.len();
                            local.push(Arc::new(build.data));
                            local_references.insert(cid, index);

                            index
                        }
                    };
                    references.push(Reference::Local(index));
                    snapshots += build.snapshots;
                    logs += build.logs;
                } else {
                    sizes.push(build.data.size());
                    let cid = self.resolver.save(build.data).await?;
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
                    snapshots += build.snapshots;
                    logs += build.logs;
                }
            }
        }

        let size_external = external.size();
        let local_len = local.len();
        let external_len = external.len();
        let external_cid = self.resolver.save(external).await?;

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
            [self.instants, self.rows, self.cols],
            self.sidelen,
            self.levels,
            references,
            Dac::from(max),
            Dac::from(min),
            local,
            external_cid,
            Arc::clone(&self.resolver),
            bits,
            self.chunks_sidelen,
            self.subsidelen,
        );

        let size = data.size();

        Ok(MMArray3Build {
            data: MMArray3::Superchunk(data),
            size: size + size_external + sizes.iter().sum::<u64>(),
            elided,
            local: local_len,
            external: external_len,
            snapshots,
            logs,
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
    type Item = Option<(Array2<N>, N, N)>;

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
            return Some(None);
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

        Some(Some((subarray, min_value, max_value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::build_subchunk;
    use ndarray::{arr1, arr2, Array2};
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

    #[tokio::test]
    async fn build_subchunk_f32() -> Result<()> {
        let data = array_float();
        let built = build_subchunk(data.into_iter(), 2, Precise(3));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.get_cell(0, 5, 0, 0).await?,
            arr1(&[9.5, 9.5, 9.5, 9.5, 9.5]),
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);

        Ok(())
    }

    #[tokio::test]
    async fn build_subchunk_f64() -> Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = build_subchunk(data.into_iter(), 2, Precise(3));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.get_cell(0, 5, 0, 0).await?,
            arr1(&[9.5, 9.5, 9.5, 9.5, 9.5]),
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);

        Ok(())
    }

    #[tokio::test]
    async fn build_subchunk_f64_round() -> Result<()> {
        let data = array_float();
        let data: Vec<Array2<f64>> = data.into_iter().map(|a| a.map(|n| *n as f64)).collect();
        let built = build_subchunk(data.into_iter(), 2, Round(2));
        let chunk = Arc::new(built.data);
        assert_eq!(
            chunk.get_cell(0, 5, 2, 4).await?,
            arr1(&[3.5, 5.0, 5.0, 3.5, 5.0]),
        );
        assert_eq!(built.snapshots, 1);
        assert_eq!(built.logs, 99);

        Ok(())
    }
}
