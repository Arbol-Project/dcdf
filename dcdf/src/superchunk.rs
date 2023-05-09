use std::{
    cmp,
    collections::HashMap,
    fmt::Debug,
    pin::Pin,
    sync::{Arc, Weak},
};

use async_recursion::async_recursion;
use async_trait::async_trait;
use cid::Cid;
use futures::{
    future::{join_all, ready},
    io::{AsyncRead, AsyncWrite},
    lock::Mutex as AsyncMutex,
    stream::{self, once, FuturesOrdered, FuturesUnordered, Stream, StreamExt},
};

use crate::{
    cache::Cacheable,
    errors::{Error, Result},
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
    helpers::rearrange,
    links::Links,
    mmbuffer::{MMBuffer0, MMBuffer1, MMBuffer3},
    mmstruct::{MMEncoding, MMStruct3, MMStruct3Build},
    node::{Node, NODE_SUPERCHUNK},
    resolver::Resolver,
    {chunk::Chunk, dac::Dac},
};

/// A time series raster subdivided into a number of K²-Raster encoded chunks.
///
/// The encoded subchunks are stored in some IPLD-like data store (represented by a concrete
/// implementation of ``Mapper``.)
///
pub(crate) struct Superchunk {
    /// Shape of the encoded raster. Since K² matrix is grown to a square with sides whose length
    /// are a power of K, we need to keep track of the dimensions of the original raster so we can
    /// perform range checking.
    shape: [usize; 3],

    /// Length of one side of logical matrix, ie number of rows, number of columns, which are equal
    /// since it is a square
    sidelen: usize,

    /// Number of tree levels stored in Superchunk. Lower levels are stored in subchunks.
    /// The idea of tree levels here is notional, as only the superchunk's leaf nodes are
    /// represented in any meaningful way. Number of subchunks equals k.pow(2).pow(levels)
    levels: u32,

    /// References to subchunks
    references: Vec<Reference>,

    /// Max values for each subchunk at each instant
    max: Dac,

    /// Min values for each subchunk at each instant
    min: Dac,

    /// Locally stored subchunks
    local: Vec<Arc<MMStruct3>>,

    /// Hashes of externally stored subchunks
    external_cid: Cid,
    external: AsyncMutex<Option<Weak<Links>>>,

    /// Resolver for retrieving subchunks
    pub(crate) resolver: Arc<Resolver>,

    /// Number of fractional bits stored in fixed point number representation
    pub fractional_bits: usize,

    /// The type of numeric data encoded in this structure.
    ///
    pub encoding: MMEncoding,

    /// The side length of the subchunks stored in this superchunk
    chunks_sidelen: usize,

    /// The number of subchunks per side in the logical grid represented by this superchunk
    subsidelen: usize,
}

impl Superchunk {
    #[async_recursion]
    pub async fn build(
        resolver: Arc<Resolver>,
        buffer: &mut MMBuffer3<'_>,
        shape: [usize; 3],
        levels: &[u32],
        k: i32,
    ) -> Result<MMStruct3Build> {
        let [instants, rows, cols] = shape;

        // Find longest side
        let sidelen = *shape[1..].iter().max().unwrap() as f64;

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

        // Adjust sidelen to be lowest power of k that is equal to or greater than the longest side
        let sidelen = k.pow(total_levels) as usize;

        let sublevels = &levels[1..];
        let at_bottom = sublevels.len() == 1;
        let levels = levels[0];

        let subsidelen = k.pow(levels as u32) as usize;
        let chunks_sidelen = sidelen / subsidelen;

        // Compute subchunks in parallel, by iterating over subarrays
        let mut futures = FuturesOrdered::new();
        let mut elided = vec![]; // bitmap would be more compact but also more overhead. worth it?
        let mut min_max = vec![];

        for row in 0..subsidelen {
            let top = row * chunks_sidelen;
            let bottom = cmp::min(top + chunks_sidelen, rows);
            for col in 0..subsidelen {
                let left = col * chunks_sidelen;
                let right = cmp::min(left + chunks_sidelen, cols);

                if top >= rows || left >= cols {
                    // This subarray is entirely outside the bounds of the actual array. This can
                    // happen becaue the logical array is expanded to have a square shape with the
                    // side lengths a power of k.
                    elided.push(true);
                    min_max.push(vec![(0, 0); instants]);
                } else {
                    let mut sub_buffer = buffer.slice(0, instants, top, bottom, left, right);
                    let shape = [instants, bottom - top, right - left];

                    let subchunk_min_max = sub_buffer.min_max();
                    let can_elide = subchunk_min_max
                        .iter()
                        .all(|(min_value, max_value)| min_value == max_value);
                    min_max.push(subchunk_min_max);

                    if can_elide {
                        elided.push(true);
                    } else {
                        let build_subchunk = at_bottom || {
                            // Find out how many tree levels are needed to represent this subarray.
                            // In cases where the array sides are greatly expanded to find a power
                            // of K, we may wind up with a subarray (in the lower right quadrant)
                            // that is is significantly smaller than other subarrays at the same
                            // level and should be encoded as a subchunk instead of as a superchunk.
                            let sidelen = *shape[1..].iter().max().unwrap() as f64;
                            let needed_levels = sidelen.log(k as f64).ceil() as u32;

                            needed_levels <= sublevels[0]
                        };

                        let resolver = Arc::clone(&resolver);
                        let future = async move {
                            sub_buffer.compute_fractional_bits();
                            if build_subchunk {
                                Ok(Chunk::build(&mut sub_buffer, shape, k))
                            } else {
                                Superchunk::build(resolver, &mut sub_buffer, shape, sublevels, k)
                                    .await
                            }
                        };

                        futures.push_back(future);
                        elided.push(false);
                    }
                }
            }
        }

        let mut subchunks = futures
            .collect::<Vec<Result<_>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .into_iter();

        let mut min = Vec::with_capacity(instants);
        let mut max = Vec::with_capacity(instants);
        for i in 0..instants {
            for subchunk in &min_max {
                let (subchunk_min, subchunk_max) = subchunk[i];
                min.push(subchunk_min);
                max.push(subchunk_max);
            }
        }
        let mut external_references: HashMap<Cid, usize> = HashMap::new();
        let mut external = Links::new();
        let mut references = Vec::new();
        let mut sizes = Vec::new();
        let mut n_elided = 0;
        let mut n_snapshots = 0;
        let mut n_logs = 0;
        let n_subchunks = subsidelen * subsidelen;
        for i in 0..n_subchunks {
            if elided[i] {
                n_elided += 1;
                references.push(Reference::Elided);
            } else {
                let build = subchunks.next().unwrap();
                let can_elide = (i..n_subchunks * instants)
                    .step_by(n_subchunks)
                    .all(|n| max[n] == min[n]);
                if can_elide {
                    n_elided += 1;
                    references.push(Reference::Elided);
                } else {
                    sizes.push(build.data.size());
                    let cid = resolver.save(&build.data).await?;
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
                    n_snapshots += build.snapshots;
                    n_logs += build.logs;
                }
            }
        }

        let size_external = external.size();
        let external_len = external.len();
        let external_cid = resolver.save(&external).await?;

        let data = Superchunk::new(
            shape,
            sidelen,
            levels,
            references,
            Dac::from(max),
            Dac::from(min),
            vec![],
            external_cid,
            Arc::clone(&resolver),
            buffer.fractional_bits(),
            buffer.encoding(),
            chunks_sidelen,
            subsidelen,
        );

        let size = data.size();

        Ok(MMStruct3Build {
            data: MMStruct3::Superchunk(data),
            size: size + size_external + sizes.iter().sum::<u64>(),
            elided: n_elided,
            local: 0,
            external: external_len,
            snapshots: n_snapshots,
            logs: n_logs,
        })
    }

    pub(crate) fn new(
        shape: [usize; 3],
        sidelen: usize,
        levels: u32,
        references: Vec<Reference>,
        max: Dac,
        min: Dac,
        local: Vec<Arc<MMStruct3>>,
        external_cid: Cid,
        resolver: Arc<Resolver>,
        fractional_bits: usize,
        encoding: MMEncoding,
        chunks_sidelen: usize,
        subsidelen: usize,
    ) -> Self {
        Self {
            shape,
            sidelen,
            levels,
            references,
            max,
            min,
            local,
            external_cid,
            external: AsyncMutex::new(None),
            resolver,
            fractional_bits,
            encoding,
            chunks_sidelen,
            subsidelen,
        }
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Get a cell's value at a particular time instant.
    ///
    pub async fn get(
        &self,
        instant: usize,
        row: usize,
        col: usize,
        buffer: &mut MMBuffer0,
    ) -> Result<()> {
        // Theoretically compiler will optimize to single DIVREM instructions
        // https://stackoverflow.com/questions/69051429
        //      /what-is-the-function-to-get-the-quotient-and-remainder-divmod-for-rust
        let chunk_row = row / self.chunks_sidelen;
        let local_row = row % self.chunks_sidelen;
        let chunk_col = col / self.chunks_sidelen;
        let local_col = col % self.chunks_sidelen;

        let chunk = chunk_row * self.subsidelen + chunk_col;
        match self.references[chunk] {
            Reference::Elided => {
                let stride = self.subsidelen * self.subsidelen;
                let index = chunk + instant * stride;

                buffer.set(self.max.get(index));
            }
            Reference::Local(index) => {
                let chunk = &self.local[index];
                buffer.set_fractional_bits(chunk.fractional_bits());
                chunk.get(instant, local_row, local_col, buffer).await?;
            }
            Reference::External(index) => {
                let external = &self.external().await?;
                let cid = &external[index];
                let chunk = self.resolver.get_mmstruct3(cid).await?;

                buffer.set_fractional_bits(chunk.fractional_bits());
                chunk.get(instant, local_row, local_col, buffer).await?;
            }
        }
        Ok(())
    }

    /// Fill in a preallocated array with a cell's value across time instants.
    ///
    #[async_recursion]
    pub async fn fill_cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
        buffer: &mut MMBuffer1,
    ) -> Result<()> {
        // Theoretically compiler will optimize to single DIVREM instructions
        // https://stackoverflow.com/questions/69051429
        //      /what-is-the-function-to-get-the-quotient-and-remainder-divmod-for-rust
        let chunk_row = row / self.chunks_sidelen;
        let local_row = row % self.chunks_sidelen;
        let chunk_col = col / self.chunks_sidelen;
        let local_col = col % self.chunks_sidelen;

        let chunk = chunk_row * self.subsidelen + chunk_col;
        match self.references[chunk] {
            Reference::Elided => {
                for (i, value) in SuperCellIter::new(self, start, end, chunk).enumerate() {
                    buffer.set(i, value);
                }
            }
            Reference::Local(index) => {
                let chunk = &self.local[index];
                buffer.set_fractional_bits(chunk.fractional_bits());
                chunk
                    .fill_cell(start, end, local_row, local_col, buffer)
                    .await?;
            }
            Reference::External(index) => {
                let external = &self.external().await?;
                let cid = &external[index];
                let chunk = self.resolver.get_mmstruct3(cid).await?;
                buffer.set_fractional_bits(chunk.fractional_bits());
                chunk
                    .fill_cell(start, end, local_row, local_col, buffer)
                    .await?;
            }
        }

        Ok(())
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    #[async_recursion]
    pub async fn fill_window(&self, window: geom::Cube, buffer: &mut MMBuffer3<'_>) -> Result<()> {
        let mut futures = vec![];
        let subchunks = self.subchunks_for(&window.rect());
        for subchunk in subchunks {
            let mut buffer = buffer.slice(
                0,
                window.end - window.start,
                subchunk.slice.top,
                subchunk.slice.bottom,
                subchunk.slice.left,
                subchunk.slice.right,
            );

            let future = async move {
                let bounds = geom::Cube::new(
                    window.start,
                    window.end,
                    subchunk.local.top,
                    subchunk.local.bottom,
                    subchunk.local.left,
                    subchunk.local.right,
                );
                match self.references[subchunk.index] {
                    Reference::Elided => {
                        let stride = self.subsidelen * self.subsidelen;
                        let mut index = subchunk.index + bounds.start * stride;
                        for i in 0..bounds.instants() {
                            buffer.fill_instant(i, self.max.get(index));
                            index += stride;
                        }
                    }
                    Reference::Local(index) => {
                        let chunk = &self.local[index];
                        buffer.set_fractional_bits(chunk.fractional_bits());
                        chunk.fill_window(bounds, &mut buffer).await?;
                    }
                    Reference::External(index) => {
                        let external = &self.external().await?;
                        let cid = &external[index];
                        let chunk = self.resolver.get_mmstruct3(cid).await?;
                        buffer.set_fractional_bits(chunk.fractional_bits());
                        chunk.fill_window(bounds, &mut buffer).await?;
                    }
                }

                Ok::<(), Error>(())
            };

            futures.push(future);
        }

        join_all(futures).await;

        Ok(())
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        myself: &Arc<MMStruct3>,
        bounds: geom::Cube,
        lower: i64,
        upper: i64,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        let myself = Arc::clone(myself);
        let chunk = match &*myself {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk");
            }
        };
        let (lower, upper) = rearrange(lower, upper);

        // Use local min and max to figure out which subchunks have cells in range
        let has_cells = |subchunk: &WindowSubchunk| {
            let stride = chunk.subsidelen * chunk.subsidelen;
            let mut index = subchunk.index + bounds.start * stride;
            for _ in bounds.start..bounds.end {
                let min = chunk.min.get(index);
                let max = chunk.max.get(index);
                if upper >= min && lower <= max {
                    return true;
                }
                index += stride;
            }

            false
        };

        let subchunks = chunk
            .subchunks_for(&bounds.rect())
            .into_iter()
            .filter(has_cells);

        let stream = FuturesUnordered::new();
        for subchunk in subchunks {
            let me = Arc::clone(&myself);
            let future = async move {
                match Self::search_subchunk(me, bounds, lower, upper, subchunk).await {
                    Ok(stream) => stream,
                    Err(err) => once(ready(Err(err))).boxed(),
                }
            };

            stream.push(future);
        }

        stream.flatten_unordered(None).boxed()
    }

    async fn search_subchunk(
        myself: Arc<MMStruct3>,
        bounds: geom::Cube,
        lower: i64,
        upper: i64,
        subchunk: WindowSubchunk,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>>> {
        let chunk = match &*myself {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("Not a superchunk");
            }
        };

        let top = subchunk.chunk.top;
        let left = subchunk.chunk.left;
        let bounds = geom::Cube::new(
            bounds.start,
            bounds.end,
            subchunk.local.top,
            subchunk.local.bottom,
            subchunk.local.left,
            subchunk.local.right,
        );
        match chunk.references[subchunk.index] {
            Reference::Elided => {
                let mut cells = Vec::new();
                let n_subchunks = chunk.subsidelen * chunk.subsidelen;
                let start_index = subchunk.index + n_subchunks * bounds.start;
                let end_index = start_index + (bounds.end - bounds.start) * n_subchunks;
                for (instant, index) in (start_index..end_index).step_by(n_subchunks).enumerate() {
                    let value = chunk.max.get(index);
                    if lower <= value && value <= upper {
                        let instant = instant + bounds.start;
                        for row in bounds.top..bounds.bottom {
                            for col in bounds.left..bounds.right {
                                cells.push(Ok((instant, row + top, col + left)));
                            }
                        }
                    }
                }

                Ok(stream::iter(cells).boxed())
            }
            Reference::Local(index) => {
                let subchunk = &chunk.local[index];
                let iter = subchunk
                    .search(bounds, lower, upper)
                    .map(move |r| {
                        r.and_then(|(instant, row, col)| Ok((instant, row + top, col + left)))
                    })
                    .boxed();

                Ok(iter)
            }
            Reference::External(index) => {
                let external = chunk.external().await?;
                let cid = &external[index];
                let subchunk = chunk.resolver.get_mmstruct3(&cid).await?;
                let iter = subchunk
                    .search(bounds, lower, upper)
                    .map(move |r| {
                        r.and_then(|(instant, row, col)| Ok((instant, row + top, col + left)))
                    })
                    .boxed();

                Ok(iter)
            }
        }
    }

    /// Get the subchunks that overlap a given window
    ///
    fn subchunks_for(&self, window: &geom::Rect) -> Vec<WindowSubchunk> {
        let mut subchunks = vec![];

        let chunks = geom::Rect::new(
            window.top / self.chunks_sidelen,
            (window.bottom - 1) / self.chunks_sidelen,
            window.left / self.chunks_sidelen,
            (window.right - 1) / self.chunks_sidelen,
        );

        for row in chunks.top..=chunks.bottom {
            let chunk_top = row * self.chunks_sidelen;
            let window_top = cmp::max(chunk_top, window.top);
            let local_top = window_top - chunk_top;
            let slice_top = window_top - window.top;

            let chunk_bottom = chunk_top + self.chunks_sidelen;
            let window_bottom = cmp::min(chunk_bottom, window.bottom);
            let local_bottom = window_bottom - chunk_top;
            let slice_bottom = window_bottom - window.top;

            for col in chunks.left..=chunks.right {
                let chunk_left = col * self.chunks_sidelen;
                let window_left = cmp::max(chunk_left, window.left);
                let local_left = window_left - chunk_left;
                let slice_left = window_left - window.left;

                let chunk_right = chunk_left + self.chunks_sidelen;
                let window_right = cmp::min(chunk_right, window.right);
                let local_right = window_right - chunk_left;
                let slice_right = window_right - window.left;

                let index = row * self.subsidelen + col;

                subchunks.push(WindowSubchunk {
                    index,
                    chunk: geom::Rect::new(chunk_top, chunk_bottom, chunk_left, chunk_right),
                    local: geom::Rect::new(local_top, local_bottom, local_left, local_right),
                    slice: geom::Rect::new(slice_top, slice_bottom, slice_left, slice_right),
                });
            }
        }

        subchunks
    }

    async fn external(&self) -> Result<Arc<Links>> {
        // Try to use cached version. Because we only store a weak reference locally, the LRU cache
        // is still in charge of whether an object is still available.
        let mut external = self.external.lock().await;
        if let Some(links) = &*external {
            if let Some(links) = links.upgrade() {
                return Ok(links);
            }
        }

        let links = self.resolver.get_links(&self.external_cid).await?;
        *external = Some(Arc::downgrade(&links));

        Ok(links)
    }
}

impl Cacheable for Superchunk {
    /// Return size of serialized superchunk in bytes
    fn size(&self) -> u64 {
        Resolver::HEADER_SIZE +
        4 * 3  // shape
        + 4    // sidelen
        + 1    // levels
        + 4    // chunks_sidelen
        + 4    // subsidelen
        + 1    // fractional_bits
        + 4    // n_references
        + self.references.iter().map(|r| r.size()).sum::<u64>()
        + self.external_cid.encoded_len() as u64
        + 4    // n_local
        + self.local.iter().map(|l| l.size()).sum::<u64>()
        + self.max.size()
        + self.min.size()
    }
}

#[async_trait]
impl Node for Superchunk {
    const NODE_TYPE: u8 = NODE_SUPERCHUNK;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_u32(self.shape[0] as u32).await?;
        stream.write_u32(self.shape[1] as u32).await?;
        stream.write_u32(self.shape[2] as u32).await?;

        stream.write_u32(self.sidelen as u32).await?;
        stream.write_byte(self.levels as u8).await?;
        stream.write_u32(self.chunks_sidelen as u32).await?;
        stream.write_u32(self.subsidelen as u32).await?;
        stream.write_byte(self.fractional_bits as u8).await?;
        stream.write_byte(self.encoding as u8).await?;

        stream.write_u32(self.references.len() as u32).await?;
        for reference in &self.references {
            reference.write_to(stream).await?;
        }

        stream.write_cid(&self.external_cid).await?;

        stream.write_u32(self.local.len() as u32).await?;
        for chunk in &self.local {
            chunk.save_to(resolver, stream).await?;
        }

        self.max.write_to(stream).await?;
        self.min.write_to(stream).await?;

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let instants = stream.read_u32().await? as usize;
        let rows = stream.read_u32().await? as usize;
        let cols = stream.read_u32().await? as usize;
        let shape = [instants, rows, cols];

        let sidelen = stream.read_u32().await? as usize;
        let levels = stream.read_byte().await? as u32;
        let chunks_sidelen = stream.read_u32().await? as usize;
        let subsidelen = stream.read_u32().await? as usize;
        let fractional_bits = stream.read_byte().await? as usize;
        let encoding = MMEncoding::try_from(stream.read_byte().await?)?;

        let n_references = stream.read_u32().await? as usize;
        let mut references = Vec::with_capacity(n_references);
        for _ in 0..n_references {
            let reference = Reference::read_from(stream).await?;
            references.push(reference);
        }

        let external_cid = stream.read_cid().await?;

        let n_local = stream.read_u32().await? as usize;
        let mut local = Vec::with_capacity(n_local);
        for _ in 0..n_local {
            let chunk = MMStruct3::load_from(resolver, stream).await?;
            local.push(Arc::new(chunk));
        }

        let max = Dac::read_from(stream).await?;
        let min = Dac::read_from(stream).await?;

        Ok(Self {
            shape,
            sidelen,
            levels,
            chunks_sidelen,
            subsidelen,
            fractional_bits,
            encoding,
            references,
            local,
            external_cid,
            external: AsyncMutex::new(None),
            max,
            min,
            resolver: Arc::clone(resolver),
        })
    }
    fn ls(&self) -> Vec<(String, Cid)> {
        vec![(String::from("subchunks"), self.external_cid)]
    }
}

/// Represents the part of a window that has some overlap with a subchunk of a superchunk.
///
#[derive(Debug)]
struct WindowSubchunk {
    /// Index of subchunk in references
    pub(crate) index: usize,

    /// Coordinates of subchunk in superchunk
    pub(crate) chunk: geom::Rect,

    /// Coordinates of window slice in chunk
    pub(crate) local: geom::Rect,

    /// Coordinates of window slice in window
    pub(crate) slice: geom::Rect,
}

struct SuperCellIter<'a> {
    superchunk: &'a Superchunk,
    index: usize,
    stride: usize,
    remaining: usize,
}

impl<'a> SuperCellIter<'a> {
    fn new(superchunk: &'a Superchunk, start: usize, end: usize, chunk_index: usize) -> Self {
        let stride = superchunk.subsidelen * superchunk.subsidelen;
        let index = chunk_index + start * stride;
        let remaining = end - start;

        Self {
            superchunk,
            index,
            stride,
            remaining,
        }
    }
}

impl<'a> Iterator for SuperCellIter<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        match self.remaining {
            0 => None,
            _ => {
                let value = self.superchunk.max.get(self.index);
                self.index += self.stride;
                self.remaining -= 1;

                Some(value)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum Reference {
    Elided,
    Local(usize),
    External(usize),
}

const REFERENCE_ELIDED: u8 = 0;
const REFERENCE_LOCAL: u8 = 1;
const REFERENCE_EXTERNAL: u8 = 2;

#[async_trait]
impl Serialize for Reference {
    /// Write self to a stream
    async fn write_to(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        match self {
            Reference::Elided => {
                stream.write_byte(REFERENCE_ELIDED).await?;
            }
            Reference::Local(index) => {
                stream.write_byte(REFERENCE_LOCAL).await?;
                stream.write_u32(*index as u32).await?;
            }
            Reference::External(index) => {
                stream.write_byte(REFERENCE_EXTERNAL).await?;
                stream.write_u32(*index as u32).await?;
            }
        }

        Ok(())
    }

    /// Read Self from a stream
    async fn read_from(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let node = match stream.read_byte().await? {
            REFERENCE_ELIDED => Reference::Elided,
            REFERENCE_LOCAL => Reference::Local(stream.read_u32().await? as usize),
            REFERENCE_EXTERNAL => Reference::External(stream.read_u32().await? as usize),
            _ => panic!("Unrecognized reference type"),
        };

        Ok(node)
    }
}

impl Cacheable for Reference {
    /// Return the number of bytes in the serialized representation
    fn size(&self) -> u64 {
        match self {
            Reference::Elided => 1,
            Reference::Local(_) => 1 + 4,
            Reference::External(_) => 1 + 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing;

    use std::collections::HashSet;

    use ndarray::{s, Array1, Array3};
    use paste::paste;

    macro_rules! test_all_the_things {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_get>]() -> Result<()> {
                    let (data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for instant in 0..instants {
                        for row in 0..rows {
                            for col in 0..cols {
                                let mut buffer = MMBuffer0::I64(0);
                                chunk.get(instant, row, col, &mut buffer).await?;
                                assert_eq!(i64::from(buffer), data[[instant, row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_fill_cell>]() -> Result<()> {
                    let (data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for row in 0..rows {
                        for col in 0..cols {
                            let start = row + col;
                            let end = instants - start;
                            let mut array = Array1::zeros([end - start]);
                            let mut buffer = MMBuffer1::new_i64(array.view_mut());
                            chunk.fill_cell(start, end, row, col, &mut buffer).await?;

                            assert_eq!(array, data.slice(s![start..end, row, col]));
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_fill_window>]() -> Result<()> {
                    let (data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;

                            let mut array = Array3::zeros([end - start, bottom - top, right - left]);
                            let mut buffer = MMBuffer3::new_i64(array.view_mut());
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            chunk.fill_window(bounds, &mut buffer).await?;

                            assert_eq!(array, data.slice(s![start..end, top..bottom, left..right]));
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_search>]() -> Result<()>{
                    let (data, chunk) = $name().await?;
                    let chunk = Arc::new(MMStruct3::Superchunk(chunk));
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let lower = (start / 5) as i64;
                            let upper = (end / 10) as i64;

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let expected = testing::array_search_window3(
                                data.view(),
                                bounds,
                                lower,
                                upper,
                            ).into_iter().collect::<HashSet<_>>();

                            let results = chunk
                                .search(bounds, lower, upper)
                                .map(|r| r.unwrap())
                                .collect::<HashSet<_>>().await;

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_ls>]() -> Result<()> {
                    let (_, chunk) = $name().await?;
                    let ls = chunk.ls();
                    assert_eq!(ls.len(), 1);
                    assert_eq!(ls[0], (String::from("subchunks"), chunk.external_cid));

                    Ok(())
                }
            }
        };
    }

    type DataChunk = Result<(Array3<i64>, Superchunk)>;

    async fn no_subchunks() -> DataChunk {
        let mut data = testing::array8();
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build =
            Superchunk::build(testing::resolver(), &mut buffer, [100, 8, 8], &[3, 0], 2).await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 64);

        Ok((data, superchunk))
    }

    test_all_the_things!(no_subchunks);

    async fn no_subchunks_four_levels() -> DataChunk {
        let mut data = testing::array(16);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build =
            Superchunk::build(testing::resolver(), &mut buffer, [100, 16, 16], &[4, 0], 2).await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 256);

        Ok((data, superchunk))
    }

    test_all_the_things!(no_subchunks_four_levels);

    async fn no_subchunks_coarse() -> DataChunk {
        let data = testing::array8();
        let mut data = Array3::from_shape_fn((100, 16, 16), |(instant, row, col)| {
            data[[instant, row / 2, col / 2]]
        });
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build =
            Superchunk::build(testing::resolver(), &mut buffer, [100, 16, 16], &[3, 1], 2).await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 64);
        for reference in &superchunk.references {
            match reference {
                Reference::Elided => continue,
                _ => panic!("not elided"),
            }
        }

        Ok((data, superchunk))
    }

    test_all_the_things!(no_subchunks_coarse);

    async fn external_subchunks() -> DataChunk {
        let mut data = testing::array(16);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build =
            Superchunk::build(testing::resolver(), &mut buffer, [100, 16, 16], &[2, 2], 2).await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 16);

        for reference in &superchunk.references {
            match reference {
                Reference::External(_) => {}
                _ => {
                    panic!("Expecting external references only");
                }
            }
        }

        assert_eq!(superchunk.external().await?.len(), 4);

        Ok((data, superchunk))
    }

    test_all_the_things!(external_subchunks);

    async fn mixed_subchunks() -> DataChunk {
        let mut data = testing::array(17);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build =
            Superchunk::build(testing::resolver(), &mut buffer, [100, 17, 17], &[2, 3], 2).await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 16);

        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in superchunk.references.iter() {
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
        assert_eq!(external_count, 8);
        assert_eq!(local_count, 0);
        assert_eq!(elided_count, 8);

        Ok((data, superchunk))
    }

    test_all_the_things!(mixed_subchunks);

    async fn elide_everything() -> DataChunk {
        let length = 100;
        let mut data = Array3::zeros([100, 16, 16]);

        for i in 0..length {
            data.slice_mut(s![i as usize, .., ..]).fill(i);
        }

        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build =
            Superchunk::build(testing::resolver(), &mut buffer, [100, 16, 16], &[2, 2], 2).await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in superchunk.references.iter() {
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
        assert_eq!(external_count, 0);
        assert_eq!(local_count, 0);
        assert_eq!(elided_count, 16);

        Ok((data, superchunk))
    }

    test_all_the_things!(elide_everything);

    async fn nested_superchunks() -> DataChunk {
        let mut data = testing::array(17);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build = Superchunk::build(
            testing::resolver(),
            &mut buffer,
            [100, 17, 17],
            &[1, 2, 2],
            2,
        )
        .await?;
        let superchunk = match build.data {
            MMStruct3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };

        Ok((data, superchunk))
    }

    test_all_the_things!(nested_superchunks);
}
