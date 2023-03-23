use std::{
    cmp,
    fmt::Debug,
    pin::Pin,
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use cid::Cid;
use futures::{
    future::{join_all, ready},
    io::{AsyncRead, AsyncWrite},
    lock::Mutex as AsyncMutex,
    stream::{self, once, FuturesUnordered, Stream, StreamExt},
};
use ndarray::{s, ArrayBase, DataMut, Ix1, Ix3};
use num_traits::Float;

use crate::{
    cache::Cacheable,
    codec::Dac,
    errors::{Error, Result},
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    fixed::{from_fixed, to_fixed},
    geom,
    helpers::rearrange,
};

use super::{
    links::Links,
    mmarray::MMDep3,
    node::{Node, NODE_SUPERCHUNK},
    resolver::Resolver,
};

/// A time series raster subdivided into a number of K²-Raster encoded chunks.
///
/// The encoded subchunks are stored in some IPLD-like data store (represented by a concrete
/// implementation of ``Mapper``.)
///
pub struct OldSuperchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
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
    local: Vec<Arc<MMDep3<N>>>,

    /// Hashes of externally stored subchunks
    external_cid: Cid,
    external: AsyncMutex<Option<Weak<Links>>>,

    /// Resolver for retrieving subchunks
    pub(crate) resolver: Arc<Resolver<N>>,

    /// Number of fractional bits stored in fixed point number representation
    fractional_bits: usize,

    /// The side length of the subchunks stored in this superchunk
    chunks_sidelen: usize,

    /// The number of subchunks per side in the logical grid represented by this superchunk
    subsidelen: usize,
}

impl<N> OldSuperchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub(crate) fn new(
        shape: [usize; 3],
        sidelen: usize,
        levels: u32,
        references: Vec<Reference>,
        max: Dac,
        min: Dac,
        local: Vec<Arc<MMDep3<N>>>,
        external_cid: Cid,
        resolver: Arc<Resolver<N>>,
        fractional_bits: usize,
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
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<N> {
        self.check_bounds(instant, row, col);

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
                let value = self.max.get(index);

                Ok(from_fixed(value, self.fractional_bits))
            }
            Reference::Local(index) => {
                let chunk = &self.local[index];
                Ok(chunk.get(instant, local_row, local_col).await?)
            }
            Reference::External(index) => {
                let external = &self.external().await?;
                let cid = &external[index];
                let chunk = self.resolver.get_mmdep3(cid).await?;

                Ok(chunk.get(instant, local_row, local_col).await?)
            }
        }
    }

    /// Fill in a preallocated array with a cell's value across time instants.
    ///
    pub async fn fill_cell<S>(
        &self,
        start: usize,
        row: usize,
        col: usize,
        values: &mut ArrayBase<S, Ix1>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N> + Send,
    {
        self.check_bounds(start + values.len() - 1, row, col);

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
                for (i, value) in
                    SuperCellIter::new(self, start, start + values.len(), chunk).enumerate()
                {
                    values[i] = value;
                }
            }
            Reference::Local(index) => {
                let chunk = &self.local[index];
                chunk.fill_cell(start, local_row, local_col, values).await?;
            }
            Reference::External(index) => {
                let external = &self.external().await?;
                let cid = &external[index];
                let chunk = self.resolver.get_mmdep3(cid).await?;
                chunk.fill_cell(start, local_row, local_col, values).await?;
            }
        }

        Ok(())
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    pub async fn fill_window<S>(
        &self,
        start: usize,
        top: usize,
        left: usize,
        window: &mut ArrayBase<S, Ix3>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N> + Send,
    {
        let mut futures = vec![];
        let shape = window.shape();
        let end = start + shape[0];
        let bottom = top + shape[1];
        let right = left + shape[2];
        self.check_bounds(end - 1, bottom - 1, right - 1);

        let subchunks = self.subchunks_for(&geom::Rect::new(top, bottom, left, right));
        for subchunk in subchunks {
            let mut subwindow = unsafe {
                window
                    .slice_mut(s![
                        ..,
                        subchunk.slice.top..subchunk.slice.bottom,
                        subchunk.slice.left..subchunk.slice.right
                    ])
                    .raw_view_mut()
                    .deref_into_view_mut()
            };

            let future = async move {
                let bounds = geom::Cube::new(
                    start,
                    end,
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
                            let value = self.max.get(index);
                            let value = from_fixed(value, self.fractional_bits);
                            let mut slice = subwindow.slice_mut(s![i, .., ..]);
                            slice.fill(value);
                            index += stride;
                        }
                    }
                    Reference::Local(index) => {
                        let chunk = &self.local[index];
                        chunk
                            .fill_window(bounds.start, bounds.top, bounds.left, &mut subwindow)
                            .await?;
                    }
                    Reference::External(index) => {
                        let external = &self.external().await?;
                        let cid = &external[index];
                        let chunk = self.resolver.get_mmdep3(cid).await?;
                        chunk
                            .fill_window(bounds.start, bounds.top, bounds.left, &mut subwindow)
                            .await?;
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
        myself: &Arc<MMDep3<N>>,
        bounds: geom::Cube,
        lower: N,
        upper: N,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        let myself = Arc::clone(myself);
        let chunk = match &*myself {
            MMDep3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk");
            }
        };
        let (lower, upper) = rearrange(lower, upper);
        chunk.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        // Use local min and max to figure out which subchunks have cells in range
        let lower_fixed = to_fixed(lower, chunk.fractional_bits, true);
        let upper_fixed = to_fixed(upper, chunk.fractional_bits, true);
        let has_cells = |subchunk: &WindowSubchunk| {
            let stride = chunk.subsidelen * chunk.subsidelen;
            let mut index = subchunk.index + bounds.start * stride;
            for _ in bounds.start..bounds.end {
                let min = chunk.min.get(index);
                let max = chunk.max.get(index);
                if upper_fixed > min && lower_fixed < max {
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
        myself: Arc<MMDep3<N>>,
        bounds: geom::Cube,
        lower: N,
        upper: N,
        subchunk: WindowSubchunk,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>>> {
        let chunk = match &*myself {
            MMDep3::Superchunk(chunk) => chunk,
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
                let subiter = SuperSearchIter::new(
                    chunk,
                    subchunk.index,
                    lower,
                    upper,
                    bounds.start,
                    bounds.end,
                    subchunk.local,
                );
                let cells: Vec<Result<(usize, usize, usize)>> = subiter
                    .map(|(instant, row, col)| Ok((instant, row + top, col + left)))
                    .collect();

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
                let subchunk = chunk.resolver.get_mmdep3(&cid).await?;
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

impl<N> Cacheable for OldSuperchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Return size of serialized superchunk in bytes
    fn size(&self) -> u64 {
        Resolver::<N>::HEADER_SIZE +
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
impl<N> Node<N> for OldSuperchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_SUPERCHUNK;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver<N>>,
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
        resolver: &Arc<Resolver<N>>,
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
            let chunk = MMDep3::load_from(resolver, stream).await?;
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

struct SuperCellIter<'a, N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    superchunk: &'a OldSuperchunk<N>,
    index: usize,
    stride: usize,
    remaining: usize,
}

impl<'a, N> SuperCellIter<'a, N>
where
    N: Float + Debug + Send + Sync,
{
    fn new(superchunk: &'a OldSuperchunk<N>, start: usize, end: usize, chunk_index: usize) -> Self {
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

impl<'a, N> Iterator for SuperCellIter<'a, N>
where
    N: Float + Debug + Send + Sync,
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

struct SuperSearchIter<'a, N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    chunk: &'a OldSuperchunk<N>,
    index: usize,
    lower: i64,
    upper: i64,

    end: usize,
    bounds: geom::Rect,

    instant: usize,
    row: usize,
    col: usize,
}

impl<'a, N> SuperSearchIter<'a, N>
where
    N: Float + Debug + Send + Sync,
{
    fn new(
        chunk: &'a OldSuperchunk<N>,
        index: usize,
        lower: N,
        upper: N,
        start: usize,
        end: usize,
        bounds: geom::Rect,
    ) -> Self {
        let row = bounds.top;
        let col = bounds.left;

        let mut iter = Self {
            chunk,
            index,
            lower: to_fixed(lower, chunk.fractional_bits, true),
            upper: to_fixed(upper, chunk.fractional_bits, true),
            end,
            bounds,

            instant: start,
            row,
            col,
        };

        iter.next_instant();

        iter
    }

    fn advance(&mut self) {
        self.col += 1;
        if self.col == self.bounds.right {
            self.col = self.bounds.left;
            self.row += 1;
            if self.row == self.bounds.bottom {
                self.row = self.bounds.top; // Da capo
                self.instant += 1;
                self.next_instant();
            }
        }
    }

    fn next_instant(&mut self) {
        // Search for next instant with value in bounds
        let stride = self.chunk.subsidelen * self.chunk.subsidelen;
        while self.instant < self.end {
            let value: i64 = self.chunk.max.get(self.index + self.instant * stride);
            if self.lower <= value && value <= self.upper {
                break;
            }
            self.instant += 1;
        }
    }
}

impl<'a, N> Iterator for SuperSearchIter<'a, N>
where
    N: Float + Debug + Send + Sync,
{
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.instant < self.end {
            let point = (self.instant, self.row, self.col);

            self.advance();

            Some(point)
        } else {
            None
        }
    }
}

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

    use crate::{build::build_superchunk, fixed::Fraction::Precise, testing};

    use std::collections::HashSet;
    use std::iter::zip;

    use ndarray::Array2;
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
                                let value = chunk.get(instant, row, col).await?;
                                assert_eq!(value, data[instant][[row, col]]);
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
                            let values = chunk.get_cell(start, end, row, col).await?;
                            assert_eq!(values.len(), end - start);
                            for i in 0..values.len() {
                                assert_eq!(values[i], data[i + start][[row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_cell_time_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants + 1, rows, cols).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_cell_row_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants, rows + 1, cols).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_cell_col_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants, rows, cols + 1).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
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
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window = chunk.get_window(&bounds).await?;

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

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_window_time_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_window_row_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_window_col_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[tokio::test]
                async fn [<$name _test_search>]() -> Result<()>{
                    let (data, chunk) = $name().await?;
                    let chunk = Arc::new(MMDep3::Superchunk(chunk));
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let lower = (start / 5) as f32;
                            let upper = (end / 10) as f32;

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = testing::float_array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let results: Vec<(usize, usize, usize)> = OldSuperchunk::
                                search(&chunk, bounds, lower, upper)
                                .map(|r| r.unwrap())
                                .collect().await;

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_search_rearrange>]() -> Result<()>{
                    let (data, chunk) = $name().await?;
                    let chunk = Arc::new(MMDep3::Superchunk(chunk));
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let lower = (start / 5) as f32;
                            let upper = (end / 10) as f32;

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = testing::float_array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(end, start, bottom, top, right, left);
                            let results: Vec<(usize, usize, usize)> = OldSuperchunk::
                                search(&chunk, bounds, upper, lower)
                                .map(|r| r.unwrap())
                                .collect().await;

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                #[allow(unused_must_use)]
                async fn [<$name _test_search_time_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    let chunk = Arc::new(MMDep3::Superchunk(chunk));
                    OldSuperchunk::search(&chunk, bounds, 0.0, 100.0);
                }

                #[tokio::test]
                #[should_panic]
                #[allow(unused_must_use)]
                async fn [<$name _test_search_row_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    let chunk = Arc::new(MMDep3::Superchunk(chunk));
                    OldSuperchunk::search(&chunk, bounds, 0.0, 100.0);
                }

                #[tokio::test]
                #[should_panic]
                #[allow(unused_must_use)]
                async fn [<$name _test_search_col_out_of_bounds>]() {
                    let (_, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    let chunk = Arc::new(MMDep3::Superchunk(chunk));
                    OldSuperchunk::search(&chunk, bounds, 0.0, 100.0);
                }

                #[tokio::test]
                async fn [<$name _test_save_load>]() -> Result<()> {
                    let (data, chunk) = $name().await?;
                    let resolver = Arc::clone(&chunk.resolver);
                    let cid = resolver.save(chunk).await?;
                    let chunk = resolver.get_superchunk(&cid).await?;

                    let [instants, rows, cols] = chunk.shape();
                    for row in 0..rows {
                        for col in 0..cols {
                            let start = row + col;
                            let end = instants - start;
                            let values = chunk.get_cell(start, end, row, col).await?;
                            assert_eq!(values.len(), end - start);
                            for i in 0..values.len() {
                                assert_eq!(values[i], data[i + start][[row, col]]);
                            }
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

                #[tokio::test]
                async fn [<$name _test_high_level_ls>]() -> Result<()> {
                    let (_, chunk) = $name().await?;
                    let resolver = Arc::clone(&chunk.resolver);
                    let chunk_cid = resolver.save(chunk).await?;
                    let chunk = resolver.get_superchunk(&chunk_cid).await?;
                    let external = chunk.external().await?;
                    let ls = resolver.ls(&chunk_cid).await?.expect("Can't ls superchunk");
                    assert_eq!(ls.len(), 1);
                    assert_eq!(ls[0].name, String::from("subchunks"));
                    assert_eq!(ls[0].cid, chunk.external_cid);
                    assert_eq!(ls[0].node_type.unwrap(), "Links");
                    assert_eq!(ls[0].size.unwrap(), external.size());

                    let ls = resolver.ls(&ls[0].cid).await?.expect("Can't ls links");
                    assert_eq!(ls.len(), external.len());
                    for (expected, entry) in zip(external.iter().enumerate(), ls) {
                        let (i, cid) = expected;
                        let subchunk = resolver.get_mmdep3(&cid).await?;
                        assert_eq!(entry.name, i.to_string());
                        assert_eq!(entry.cid, cid.clone());
                        assert_eq!(entry.size.unwrap(), subchunk.size());
                    }

                    Ok(())
                }
            }
        };
    }

    type DataChunk = Result<(Vec<Array2<f32>>, OldSuperchunk<f32>)>;

    async fn no_subchunks() -> DataChunk {
        let data = testing::oldfarray8();
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[3, 0],
            2,
            Precise(3),
            0,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 64);

        Ok((data, superchunk))
    }

    test_all_the_things!(no_subchunks);

    async fn no_subchunks_coarse() -> DataChunk {
        let data = testing::oldfarray8();
        let data: Vec<Array2<f32>> = data
            .into_iter()
            .map(|a| Array2::from_shape_fn((16, 16), |(row, col)| a[[row / 2, col / 2]]))
            .collect();

        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[3, 1],
            2,
            Precise(3),
            0,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
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

    async fn local_subchunks() -> DataChunk {
        let data = testing::oldfarray(16);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[2, 2],
            2,
            Precise(3),
            1 << 14,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };
        assert_eq!(superchunk.references.len(), 16);
        assert_eq!(superchunk.local.len(), 4);

        Ok((data, superchunk))
    }

    test_all_the_things!(local_subchunks);

    async fn external_subchunks() -> DataChunk {
        let data = testing::oldfarray(16);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[2, 2],
            2,
            Precise(3),
            0,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
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
        let data = testing::oldfarray(17);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[2, 3],
            2,
            Precise(3),
            8000,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
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
        assert_eq!(external_count, 4);
        assert_eq!(local_count, 4);
        assert_eq!(elided_count, 8);

        Ok((data, superchunk))
    }

    test_all_the_things!(mixed_subchunks);

    async fn elide_everything() -> DataChunk {
        let length = 100;
        let shape = (16, 16);
        let mut data = vec![];

        for i in 0..length {
            let slice = Array2::from_elem(shape, i as f32);
            data.push(slice);
        }

        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[2, 2],
            2,
            Precise(3),
            0,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
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
        let data = testing::oldfarray(17);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            &[1, 2, 2],
            2,
            Precise(3),
            0,
        )
        .await?;
        let superchunk = match build.data {
            MMDep3::Superchunk(chunk) => chunk,
            _ => {
                panic!("not a superchunk")
            }
        };

        Ok((data, superchunk))
    }

    test_all_the_things!(nested_superchunks);
}
