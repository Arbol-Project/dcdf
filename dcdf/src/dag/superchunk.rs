use std::{
    cell::RefCell,
    cmp,
    collections::HashMap,
    fmt::Debug,
    io, mem,
    rc::Rc,
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use cid::Cid;
use futures::{
    future::join_all,
    io::{AsyncRead, AsyncWrite},
};
use ndarray::{s, Array2, Array3, ArrayBase, DataMut, Ix3};
use num_traits::Float;
use parking_lot::Mutex;

use crate::{
    cache::Cacheable,
    codec::Dac,
    codec::FChunk,
    errors::{Error, Result},
    extio::{
        ExtendedAsyncRead, ExtendedAsyncWrite, ExtendedRead, ExtendedWrite, Serialize,
        SerializeAsync,
    },
    fixed::{from_fixed, to_fixed, Fraction, Precise, Round},
    geom,
    helpers::rearrange,
    simple::{FBuild, FBuilder},
};

use super::{
    links::Links,
    node::{AsyncNode, Node, NODE_SUPERCHUNK},
    resolver::Resolver,
};

/// A time series raster subdivided into a number of K²-Raster encoded chunks.
///
/// The encoded subchunks are stored in some IPLD-like data store (represented by a concrete
/// implementation of ``Mapper``.)
///
pub struct Superchunk<N>
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
    levels: usize,

    /// References to subchunks
    references: Vec<Reference>,

    /// Max values for each subchunk at each instant
    max: Dac,

    /// Min values for each subchunk at each instant
    min: Dac,

    /// Locally stored subchunks
    local: Vec<Arc<FChunk<N>>>,

    /// Hashes of externally stored subchunks
    external_cid: Cid,
    external: Mutex<Option<Weak<Links>>>,

    /// Resolver for retrieving subchunks
    resolver: Arc<Resolver<N>>,

    /// Number of fractional bits stored in fixed point number representation
    fractional_bits: usize,

    /// The side length of the subchunks stored in this superchunk
    chunks_sidelen: usize,

    /// The number of subchunks per side in the logical grid represented by this superchunk
    subsidelen: usize,
}

impl<N> Superchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub(crate) fn new(
        shape: [usize; 3],
        sidelen: usize,
        levels: usize,
        references: Vec<Reference>,
        max: Dac,
        min: Dac,
        local: Vec<Arc<FChunk<N>>>,
        external_cid: Cid,
        external: Mutex<Option<Weak<Links>>>,
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
            external,
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
    pub fn get(self: &Arc<Self>, instant: usize, row: usize, col: usize) -> Result<N> {
        let mut iter = self.iter_cell(instant, instant + 1, row, col)?;

        Ok(iter.next().unwrap())
    }

    /// Get a cell's value at a particular time instant.
    ///
    pub async fn get_async(self: &Arc<Self>, instant: usize, row: usize, col: usize) -> Result<N> {
        let mut iter = self.iter_cell_async(instant, instant + 1, row, col).await?;

        Ok(iter.next().unwrap())
    }

    /// Iterate over a cell's value across time instants.
    ///
    pub fn iter_cell(
        self: &Arc<Self>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<CellIter<N>> {
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
            Reference::External(index) => {
                let external = &self.external()?;
                let cid = &external[index];
                let chunk = self.resolver.get_subchunk(cid)?;

                Ok(CellIter {
                    inner: Rc::new(RefCell::new(
                        chunk.iter_cell(start, end, local_row, local_col),
                    )),
                })
            }
        }
    }

    /// Iterate over a cell's value across time instants.
    ///
    pub async fn iter_cell_async(
        self: &Arc<Self>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<CellIter<N>> {
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
            Reference::External(index) => {
                let external = &self.external_async().await?;
                let cid = &external[index];
                let chunk = self.resolver.get_subchunk_async(cid).await?;

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
    pub fn get_window(self: &Arc<Self>, bounds: &geom::Cube) -> Result<Array3<N>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut window = Array3::zeros([bounds.instants(), bounds.rows(), bounds.cols()]);
        self.fill_window(bounds, &mut window)?;

        Ok(window)
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    fn fill_window<S>(
        self: &Arc<Self>,
        bounds: &geom::Cube,
        window: &mut ArrayBase<S, Ix3>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N>,
    {
        for subchunk in self.subchunks_for(&bounds.rect()) {
            let mut subwindow = window.slice_mut(s![
                ..,
                subchunk.slice.top..subchunk.slice.bottom,
                subchunk.slice.left..subchunk.slice.right
            ]);

            let bounds = geom::Cube::new(
                bounds.start,
                bounds.end,
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
                    chunk.fill_window(&bounds, &mut subwindow);
                }
                Reference::External(index) => {
                    let external = &self.external()?;
                    let cid = &external[index];
                    let chunk = self.resolver.get_subchunk(cid)?;
                    chunk.fill_window(&bounds, &mut subwindow);
                }
            }
        }

        Ok(())
    }

    /// Iterate over subarrays of successive time instants.
    ///
    pub fn iter_window(self: &Arc<Self>, bounds: &geom::Cube) -> Result<WindowIter<N>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut subiters: Vec<SubwindowIter<N>> = vec![];
        for subchunk in self.subchunks_for(&bounds.rect()) {
            let bounds = geom::Cube::new(
                bounds.start,
                bounds.end,
                subchunk.local.top,
                subchunk.local.bottom,
                subchunk.local.left,
                subchunk.local.right,
            );

            match self.references[subchunk.index] {
                Reference::Elided => {
                    let subiter = SuperWindowIter::new(
                        self,
                        bounds.start,
                        bounds.end,
                        subchunk.index,
                        subchunk.slice.bottom - subchunk.slice.top,
                        subchunk.slice.right - subchunk.slice.left,
                    );
                    subiters.push(SubwindowIter {
                        subiter: Rc::new(RefCell::new(subiter)),
                        top: subchunk.slice.top,
                        left: subchunk.slice.left,
                    });
                }
                Reference::Local(index) => {
                    let chunk = &self.local[index];
                    let subiter = chunk.iter_window(&bounds);
                    subiters.push(SubwindowIter {
                        subiter: Rc::new(RefCell::new(subiter)),
                        top: subchunk.slice.top,
                        left: subchunk.slice.left,
                    });
                }
                Reference::External(index) => {
                    let external = self.external()?;
                    let cid = external[index];
                    let chunk = self.resolver.get_subchunk(&cid)?;
                    let subiter = chunk.iter_window(&bounds);
                    subiters.push(SubwindowIter {
                        subiter: Rc::new(RefCell::new(subiter)),
                        top: subchunk.slice.top,
                        left: subchunk.slice.left,
                    });
                }
            }
        }

        Ok(WindowIter {
            subiters,
            rows: bounds.rows(),
            cols: bounds.cols(),
            remaining: bounds.instants(),
        })
    }

    /// Get a subarray of this Chunk.
    ///
    pub async fn get_window_async(self: &Arc<Self>, bounds: &geom::Cube) -> Result<Array3<N>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut window = Array3::zeros([bounds.instants(), bounds.rows(), bounds.cols()]);
        self.fill_window_async(bounds, &mut window).await?;

        Ok(window)
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    async fn fill_window_async<S>(
        self: &Arc<Self>,
        bounds: &geom::Cube,
        window: &mut ArrayBase<S, Ix3>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N>,
    {
        let mut futures = vec![];

        let subchunks = self.subchunks_for(&bounds.rect());
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
                    bounds.start,
                    bounds.end,
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
                        chunk.fill_window(&bounds, &mut subwindow);
                    }
                    Reference::External(index) => {
                        let external = &self.external_async().await?;
                        let cid = &external[index];
                        let chunk = self.resolver.get_subchunk_async(cid).await?;
                        chunk.fill_window(&bounds, &mut subwindow);
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
    /// Returns an iterator that produces coordinate triplets [instant, row, col] of matching
    /// cells.
    ///
    pub fn iter_search(
        self: &Arc<Self>,
        bounds: &geom::Cube,
        lower: N,
        upper: N,
    ) -> Result<SearchIter<N>> {
        let (lower, upper) = rearrange(lower, upper);
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        Ok(SearchIter::new(self, bounds, lower, upper)?)
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

    fn external(&self) -> Result<Arc<Links>> {
        // Try to use cached version. Because we only store a weak reference locally, the LRU cache
        // is still in charge of whether an object is still available.
        let mut external = self.external.lock();
        if let Some(links) = &*external {
            if let Some(links) = links.upgrade() {
                return Ok(links);
            }
        }

        let links = self.resolver.get_links(&self.external_cid)?;
        *external = Some(Arc::downgrade(&links));

        Ok(links)
    }

    async fn external_async(&self) -> Result<Arc<Links>> {
        // Try to use cached version. Because we only store a weak reference locally, the LRU cache
        // is still in charge of whether an object is still available.
        let mut external = self.external.lock();
        if let Some(links) = &*external {
            if let Some(links) = links.upgrade() {
                return Ok(links);
            }
        }

        let links = self.resolver.get_links_async(&self.external_cid).await?;
        *external = Some(Arc::downgrade(&links));

        Ok(links)
    }
}

impl<N> Cacheable for Superchunk<N>
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

impl<N> Node<N> for Superchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_SUPERCHUNK;

    /// Load superchunk from a stream
    fn load_from(resolver: &Arc<Resolver<N>>, stream: &mut impl io::Read) -> Result<Self> {
        let instants = stream.read_u32()? as usize;
        let rows = stream.read_u32()? as usize;
        let cols = stream.read_u32()? as usize;
        let shape = [instants, rows, cols];

        let sidelen = stream.read_u32()? as usize;
        let levels = stream.read_byte()? as usize;
        let chunks_sidelen = stream.read_u32()? as usize;
        let subsidelen = stream.read_u32()? as usize;
        let fractional_bits = stream.read_byte()? as usize;

        let n_references = stream.read_u32()? as usize;
        let mut references = Vec::with_capacity(n_references);
        for _ in 0..n_references {
            let reference = Reference::read_from(stream)?;
            references.push(reference);
        }

        let external_cid = Cid::read_bytes(&mut *stream)?;

        let n_local = stream.read_u32()? as usize;
        let mut local = Vec::with_capacity(n_local);
        for _ in 0..n_local {
            let chunk = FChunk::read_from(stream)?;
            local.push(Arc::new(chunk));
        }

        let max = Dac::read_from(stream)?;
        let min = Dac::read_from(stream)?;

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
            external: Mutex::new(None),
            max,
            min,
            resolver: Arc::clone(resolver),
        })
    }

    /// Save superchunk to a stream
    fn save_to(self, _resolver: &Arc<Resolver<N>>, stream: &mut impl io::Write) -> Result<()> {
        stream.write_u32(self.shape[0] as u32)?;
        stream.write_u32(self.shape[1] as u32)?;
        stream.write_u32(self.shape[2] as u32)?;

        stream.write_u32(self.sidelen as u32)?;
        stream.write_byte(self.levels as u8)?;
        stream.write_u32(self.chunks_sidelen as u32)?;
        stream.write_u32(self.subsidelen as u32)?;
        stream.write_byte(self.fractional_bits as u8)?;

        stream.write_u32(self.references.len() as u32)?;
        for reference in &self.references {
            reference.write_to(stream)?;
        }

        self.external_cid.write_bytes(&mut *stream)?;

        stream.write_u32(self.local.len() as u32)?;
        for chunk in &self.local {
            chunk.write_to(stream)?;
        }

        self.max.write_to(stream)?;
        self.min.write_to(stream)?;

        Ok(())
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        vec![(String::from("subchunks"), self.external_cid)]
    }
}

#[async_trait]
impl<N> AsyncNode<N> for Superchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Save an object into the DAG
    ///
    async fn save_to_async(
        self,
        _resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_u32_async(self.shape[0] as u32).await?;
        stream.write_u32_async(self.shape[1] as u32).await?;
        stream.write_u32_async(self.shape[2] as u32).await?;

        stream.write_u32_async(self.sidelen as u32).await?;
        stream.write_byte_async(self.levels as u8).await?;
        stream.write_u32_async(self.chunks_sidelen as u32).await?;
        stream.write_u32_async(self.subsidelen as u32).await?;
        stream.write_byte_async(self.fractional_bits as u8).await?;

        stream.write_u32_async(self.references.len() as u32).await?;
        for reference in &self.references {
            reference.write_to_async(stream).await?;
        }

        stream.write_cid(&self.external_cid).await?;

        stream.write_u32_async(self.local.len() as u32).await?;
        for chunk in &self.local {
            chunk.write_to_async(stream).await?;
        }

        self.max.write_to_async(stream).await?;
        self.min.write_to_async(stream).await?;

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from_async(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let instants = stream.read_u32_async().await? as usize;
        let rows = stream.read_u32_async().await? as usize;
        let cols = stream.read_u32_async().await? as usize;
        let shape = [instants, rows, cols];

        let sidelen = stream.read_u32_async().await? as usize;
        let levels = stream.read_byte_async().await? as usize;
        let chunks_sidelen = stream.read_u32_async().await? as usize;
        let subsidelen = stream.read_u32_async().await? as usize;
        let fractional_bits = stream.read_byte_async().await? as usize;

        let n_references = stream.read_u32_async().await? as usize;
        let mut references = Vec::with_capacity(n_references);
        for _ in 0..n_references {
            let reference = Reference::read_from_async(stream).await?;
            references.push(reference);
        }

        let external_cid = stream.read_cid().await?;

        let n_local = stream.read_u32_async().await? as usize;
        let mut local = Vec::with_capacity(n_local);
        for _ in 0..n_local {
            let chunk = FChunk::read_from_async(stream).await?;
            local.push(Arc::new(chunk));
        }

        let max = Dac::read_from_async(stream).await?;
        let min = Dac::read_from_async(stream).await?;

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
            external: Mutex::new(None),
            max,
            min,
            resolver: Arc::clone(resolver),
        })
    }
}

/// Represents the part of a window that has some overlap with a subchunk of a superchunk.
///
#[derive(Debug)]
struct WindowSubchunk {
    /// Index of subchunk in references
    pub index: usize,

    /// Coordinates of subchunk in superchunk
    pub chunk: geom::Rect,

    /// Coordinates of window slice in chunk
    pub local: geom::Rect,

    /// Coordinates of window slice in window
    pub slice: geom::Rect,
}

pub struct CellIter<N>
where
    N: Float + Debug + Send + Sync,
{
    inner: Rc<RefCell<dyn Iterator<Item = N>>>,
}

impl<N> Iterator for CellIter<N>
where
    N: Float + Debug + Send + Sync,
{
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.borrow_mut().next()
    }
}

struct SuperCellIter<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    superchunk: Arc<Superchunk<N>>,
    index: usize,
    stride: usize,
    remaining: usize,
}

impl<N> SuperCellIter<N>
where
    N: Float + Debug + Send + Sync,
{
    fn new(superchunk: &Arc<Superchunk<N>>, start: usize, end: usize, chunk_index: usize) -> Self {
        let stride = superchunk.subsidelen * superchunk.subsidelen;
        let index = chunk_index + start * stride;
        let remaining = end - start;

        Self {
            superchunk: Arc::clone(superchunk),
            index,
            stride,
            remaining,
        }
    }
}

impl<N> Iterator for SuperCellIter<N>
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

pub struct WindowIter<N>
where
    N: Float + Debug + Send + Sync,
{
    subiters: Vec<SubwindowIter<N>>,
    rows: usize,
    cols: usize,
    remaining: usize,
}

struct SubwindowIter<N>
where
    N: Float + Debug + Send + Sync,
{
    subiter: Rc<RefCell<dyn Iterator<Item = Array2<N>>>>,
    top: usize,
    left: usize,
}

impl<N> Iterator for WindowIter<N>
where
    N: Float + Debug + Send + Sync,
{
    type Item = Array2<N>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.remaining {
            0 => None,
            _ => {
                let mut window = Array2::zeros([self.rows, self.cols]);
                for subiter in &self.subiters {
                    let top = subiter.top;
                    let left = subiter.left;
                    let mut subiter = subiter.subiter.borrow_mut();
                    let subwindow = subiter.next().expect("There should still be subwindows.");
                    let shape = subwindow.shape();
                    let rows = shape[0];
                    let cols = shape[1];

                    window
                        .slice_mut(s![top..top + rows, left..left + cols])
                        .assign(&subwindow);
                }

                self.remaining -= 1;

                Some(window)
            }
        }
    }
}

struct SuperWindowIter<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    superchunk: Arc<Superchunk<N>>,
    index: usize,
    stride: usize,
    remaining: usize,
    rows: usize,
    cols: usize,
}

impl<N> SuperWindowIter<N>
where
    N: Float + Debug + Send + Sync,
{
    fn new(
        superchunk: &Arc<Superchunk<N>>,
        start: usize,
        end: usize,
        chunk_index: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        let stride = superchunk.subsidelen * superchunk.subsidelen;
        let index = chunk_index + start * stride;
        let remaining = end - start;

        Self {
            superchunk: Arc::clone(superchunk),
            index,
            stride,
            remaining,
            rows,
            cols,
        }
    }
}

impl<N> Iterator for SuperWindowIter<N>
where
    N: Float + Debug + Send + Sync,
{
    type Item = Array2<N>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.remaining {
            0 => None,
            _ => {
                let value = self.superchunk.max.get(self.index);
                let value = from_fixed(value, self.superchunk.fractional_bits);
                self.index += self.stride;
                self.remaining -= 1;

                Some(Array2::from_elem((self.rows, self.cols), value))
            }
        }
    }
}

pub struct SearchIter<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    chunk: Arc<Superchunk<N>>,
    start: usize,
    end: usize,
    lower: N,
    upper: N,

    top: usize,
    left: usize,
    subchunks: Vec<WindowSubchunk>,
    iter: Option<Rc<RefCell<dyn Iterator<Item = (usize, usize, usize)>>>>,
}

impl<N> SearchIter<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn new(chunk: &Arc<Superchunk<N>>, bounds: &geom::Cube, lower: N, upper: N) -> Result<Self> {
        let has_cells = |subchunk: &WindowSubchunk| {
            let lower = to_fixed(lower, chunk.fractional_bits, true);
            let upper = to_fixed(upper, chunk.fractional_bits, true);
            let stride = chunk.subsidelen * chunk.subsidelen;
            let mut index = subchunk.index + bounds.start * stride;
            for _ in bounds.start..bounds.end {
                let min = chunk.min.get(index);
                let max = chunk.max.get(index);
                if upper > min && lower < max {
                    return true;
                }
                index += stride;
            }

            false
        };

        let subchunks = chunk
            .subchunks_for(&bounds.rect())
            .into_iter()
            .filter(has_cells)
            .collect();

        let mut iter = Self {
            chunk: chunk.clone(),
            start: bounds.start,
            end: bounds.end,
            lower,
            upper,
            top: 0,
            left: 0,
            subchunks,
            iter: None,
        };

        iter.next_subchunk()?;

        Ok(iter)
    }

    fn next_subchunk(&mut self) -> Result<()> {
        self.iter = match self.subchunks.pop() {
            Some(subchunk) => {
                self.top = subchunk.chunk.top;
                self.left = subchunk.chunk.left;

                let bounds = geom::Cube::new(
                    self.start,
                    self.end,
                    subchunk.local.top,
                    subchunk.local.bottom,
                    subchunk.local.left,
                    subchunk.local.right,
                );
                match self.chunk.references[subchunk.index] {
                    Reference::Elided => {
                        let subiter = SuperSearchIter::new(
                            &self.chunk,
                            subchunk.index,
                            self.lower,
                            self.upper,
                            self.start,
                            self.end,
                            subchunk.local,
                        );

                        Some(Rc::new(RefCell::new(subiter)))
                    }
                    Reference::Local(index) => {
                        let chunk = &self.chunk.local[index];
                        let subiter = chunk.iter_search(&bounds, self.lower, self.upper);

                        Some(Rc::new(RefCell::new(subiter)))
                    }
                    Reference::External(index) => {
                        let external = self.chunk.external()?;
                        let cid = &external[index];
                        let chunk = self.chunk.resolver.get_subchunk(&cid)?;
                        let subiter = chunk.iter_search(&bounds, self.lower, self.upper);

                        Some(Rc::new(RefCell::new(subiter)))
                    }
                }
            }
            None => None,
        };

        Ok(())
    }
}

impl<N> Iterator for SearchIter<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    type Item = Result<(usize, usize, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.iter {
            Some(iter) => {
                let next = iter.borrow_mut().next();
                match next {
                    Some((instant, row, col)) => {
                        Some(Ok((instant, row + self.top, col + self.left)))
                    }
                    None => match self.next_subchunk() {
                        Ok(_) => self.next(),
                        Err(err) => Some(Err(err)),
                    },
                }
            }
            None => None,
        }
    }
}

struct SuperSearchIter<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    chunk: Arc<Superchunk<N>>,
    index: usize,
    lower: i64,
    upper: i64,

    end: usize,
    bounds: geom::Rect,

    instant: usize,
    row: usize,
    col: usize,
}

impl<N> SuperSearchIter<N>
where
    N: Float + Debug + Send + Sync,
{
    fn new(
        chunk: &Arc<Superchunk<N>>,
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
            chunk: chunk.clone(),
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

impl<N> Iterator for SuperSearchIter<N>
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

impl Serialize for Reference {
    /// Write self to a stream
    fn write_to(&self, stream: &mut impl io::Write) -> Result<()> {
        match self {
            Reference::Elided => {
                stream.write_byte(REFERENCE_ELIDED)?;
            }
            Reference::Local(index) => {
                stream.write_byte(REFERENCE_LOCAL)?;
                stream.write_u32(*index as u32)?;
            }
            Reference::External(index) => {
                stream.write_byte(REFERENCE_EXTERNAL)?;
                stream.write_u32(*index as u32)?;
            }
        }

        Ok(())
    }

    /// Read Self from a stream
    fn read_from(stream: &mut impl io::Read) -> Result<Self> {
        let node = match stream.read_byte()? {
            REFERENCE_ELIDED => Reference::Elided,
            REFERENCE_LOCAL => Reference::Local(stream.read_u32()? as usize),
            REFERENCE_EXTERNAL => Reference::External(stream.read_u32()? as usize),
            _ => panic!("Unrecognized reference type"),
        };

        Ok(node)
    }
}

#[async_trait]
impl SerializeAsync for Reference {
    /// Write self to a stream
    async fn write_to_async(&self, stream: &mut (impl AsyncWrite + Unpin + Send)) -> Result<()> {
        match self {
            Reference::Elided => {
                stream.write_byte_async(REFERENCE_ELIDED).await?;
            }
            Reference::Local(index) => {
                stream.write_byte_async(REFERENCE_LOCAL).await?;
                stream.write_u32_async(*index as u32).await?;
            }
            Reference::External(index) => {
                stream.write_byte_async(REFERENCE_EXTERNAL).await?;
                stream.write_u32_async(*index as u32).await?;
            }
        }

        Ok(())
    }

    /// Read Self from a stream
    async fn read_from_async(stream: &mut (impl AsyncRead + Unpin + Send)) -> Result<Self> {
        let node = match stream.read_byte_async().await? {
            REFERENCE_ELIDED => Reference::Elided,
            REFERENCE_LOCAL => Reference::Local(stream.read_u32_async().await? as usize),
            REFERENCE_EXTERNAL => Reference::External(stream.read_u32_async().await? as usize),
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

pub fn build_superchunk<I, N>(
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
        builder.push(instant);
    }
    builder.finish()
}

pub struct SuperchunkBuilder<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    builders: Vec<FBuilder<N>>,
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
            resolver,
            levels,
            fraction,
            local_threshold,
            sidelen,
            subsidelen,
            chunks_sidelen,
        }
    }

    pub fn push(&mut self, a: Array2<N>) {
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

    pub fn finish(mut self) -> Result<SuperchunkBuild<N>> {
        // Swap builders out of self before moving so that self can be borrowed by methods later.
        let builders = mem::replace(&mut self.builders, vec![]);
        let builds: Vec<FBuild<N>> = builders
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
                let cid = self.resolver.save(build.data)?;
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
        let external_cid = self.resolver.save(external)?;

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

        let data = Superchunk {
            shape: [instants, self.rows, self.cols],
            sidelen: self.sidelen,
            levels: self.levels,
            references,
            max: Dac::from(max),
            min: Dac::from(min),
            local,
            external_cid,
            external: Mutex::new(None),
            resolver: Arc::clone(&self.resolver),
            fractional_bits: bits,
            chunks_sidelen: self.chunks_sidelen,
            subsidelen: self.subsidelen,
        };

        let size = data.size();
        let compressed = size + size_external + sizes.iter().sum::<u64>();
        let word_size = mem::size_of::<N>();
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

    use super::super::testing;
    use crate::build::build_superchunk as build_superchunk_async;

    use std::collections::HashSet;
    use std::iter::zip;

    use paste::paste;

    macro_rules! test_all_the_things {
        ($name:ident) => {
            paste! {
                #[test]
                fn [<$name _test_get>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    for instant in 0..instants {
                        for row in 0..rows {
                            for col in 0..cols {
                                assert_eq!(chunk.get(instant, row, col)?, data[instant][[row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_get_async>]() -> Result<()> {
                    let (data, chunk) = [<$name _async>]().await?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    for instant in 0..instants {
                        for row in 0..rows {
                            for col in 0..cols {
                                let value = chunk.get_async(instant, row, col).await?;
                                assert_eq!(value, data[instant][[row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[test]
                fn [<$name _test_iter_cell>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
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
                fn [<$name _test_iter_cell_rearrange>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
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
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;

                    let values: Vec<f32> = chunk.iter_cell(0, instants + 1, rows, cols)
                        .expect("This isn't what causes the panic").collect();
                    assert_eq!(values.len(), instants + 1);
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_cell_row_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("This should work");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;

                    let values: Vec<f32> = chunk.iter_cell(0, instants, rows + 1, cols)
                        .expect("This isn't what causes the panic").collect();
                    assert_eq!(values.len(), instants + 1);
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_cell_col_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("This should work");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;

                    let values: Vec<f32> = chunk.iter_cell(0, instants, rows, cols + 1)
                        .expect("This isn't what causes the panic").collect();
                    assert_eq!(values.len(), instants + 1);
                }

                #[test]
                fn [<$name _test_get_window>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window = chunk.get_window(&bounds)?;

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
                async fn [<$name _test_get_window_async>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window = chunk.get_window_async(&bounds).await?;

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

                #[test]
                #[should_panic]
                fn [<$name _test_get_window_time_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    chunk.get_window(&bounds).expect("Unexpected error.");
                }

                #[test]
                #[should_panic]
                fn [<$name _test_get_window_row_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    chunk.get_window(&bounds).expect("Unexpected error.");
                }

                #[test]
                #[should_panic]
                fn [<$name _test_get_window_col_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    chunk.get_window(&bounds).expect("Unexpected error.");
                }

                #[test]
                fn [<$name _test_iter_window>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window: Vec<Array2<f32>> = chunk.iter_window(&bounds)?.collect();

                            assert_eq!(window.len(), end - start);
                            for i in 0..end - start {
                                assert_eq!(window[i].shape(), [bottom - top, right - left]);
                                assert_eq!(
                                    window[i],
                                    data[start + i].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }

                    Ok(())
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_window_time_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    chunk.iter_window(&bounds).expect("Unexpected error.");
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_window_row_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    chunk.iter_window(&bounds).expect("Unexpected error.");
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_window_col_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    chunk.iter_window(&bounds).expect("Unexpected error.");
                }

                #[test]
                fn [<$name _test_iter_search>]() -> Result<()>{
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
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
                                let coords = testing::array_search_window(
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
                            let results: Vec<(usize, usize, usize)> = chunk
                                .iter_search(&bounds, lower, upper)?
                                .map(|r| r.expect("I/O error"))
                                .collect();

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[test]
                fn [<$name _test_iter_search_rearrange>]() -> Result<()>{
                    let (data, chunk) = $name()?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
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
                                let coords = testing::array_search_window(
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
                            let results: Vec<(usize, usize, usize)> = chunk
                                .iter_search(&bounds, upper, lower)?
                                .map(|r| r.expect("I/O error"))
                                .collect();

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_search_time_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    chunk.iter_search(&bounds, 0.0, 100.0).expect("Unexpected error.");
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_search_row_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    chunk.iter_search(&bounds, 0.0, 100.0).expect("Unexpected error.");
                }

                #[test]
                #[should_panic]
                fn [<$name _test_iter_search_col_out_of_bounds>]() {
                    let (_, chunk) = $name().expect("Unexpected error.");
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape;
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    chunk.iter_search(&bounds, 0.0, 100.0).expect("Unexpected error.");
                }

                #[test]
                fn [<$name _test_save_load>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let resolver = Arc::clone(&chunk.resolver);
                    let cid = resolver.save(chunk)?;
                    let chunk = resolver.get_superchunk(&cid)?;

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

                #[tokio::test]
                async fn [<$name _test_save_load_async>]() -> Result<()> {
                    let (data, chunk) = $name()?;
                    let resolver = Arc::clone(&chunk.resolver);
                    let cid = resolver.save_async(chunk).await?;
                    let chunk = resolver.get_superchunk_async(&cid).await?;

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
                fn [<$name _test_ls>]() -> Result<()> {
                    let (_, chunk) = $name()?;
                    let ls = chunk.ls();
                    assert_eq!(ls.len(), 1);
                    assert_eq!(ls[0], (String::from("subchunks"), chunk.external_cid));

                    Ok(())
                }

                #[test]
                fn [<$name _test_high_level_ls>]() -> Result<()> {
                    let (_, chunk) = $name()?;
                    let resolver = Arc::clone(&chunk.resolver);
                    let chunk_cid = resolver.save(chunk)?;
                    let chunk = resolver.get_superchunk(&chunk_cid)?;
                    let external = chunk.external()?;
                    let ls = resolver.ls(&chunk_cid)?.expect("Can't ls superchunk");
                    assert_eq!(ls.len(), 1);
                    assert_eq!(ls[0].name, String::from("subchunks"));
                    assert_eq!(ls[0].cid, chunk.external_cid);
                    assert_eq!(ls[0].node_type.unwrap(), "Links");
                    assert_eq!(ls[0].size.unwrap(), external.size());

                    let ls = resolver.ls(&ls[0].cid)?.expect("Can't ls links");
                    assert_eq!(ls.len(), external.len());
                    for (expected, entry) in zip(external.iter().enumerate(), ls) {
                        let (i, cid) = expected;
                        let subchunk = resolver.get_subchunk(&cid)?;
                        assert_eq!(entry.name, i.to_string());
                        assert_eq!(entry.cid, cid.clone());
                        assert_eq!(entry.node_type.unwrap(), "Subchunk");
                        assert_eq!(entry.size.unwrap(), subchunk.size());
                    }

                    Ok(())
                }
            }
        };
    }

    type DataChunk = Result<(Vec<Array2<f32>>, Superchunk<f32>)>;

    fn no_subchunks() -> DataChunk {
        let data = testing::array8();
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            3,
            2,
            Precise(3),
            0,
        )?;
        assert_eq!(build.data.references.len(), 64);

        Ok((data, build.data))
    }

    async fn no_subchunks_async() -> DataChunk {
        let data = testing::array8();
        let build = build_superchunk_async(
            data.clone().into_iter(),
            testing::resolver(),
            3,
            2,
            Precise(3),
            0,
        )
        .await?;
        assert_eq!(build.data.references.len(), 64);

        Ok((data, build.data))
    }

    test_all_the_things!(no_subchunks);

    fn no_subchunks_coarse() -> DataChunk {
        let data = testing::array8();
        let data: Vec<Array2<f32>> = data
            .into_iter()
            .map(|a| Array2::from_shape_fn((16, 16), |(row, col)| a[[row / 2, col / 2]]))
            .collect();

        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            3,
            2,
            Precise(3),
            0,
        )?;
        assert_eq!(build.data.references.len(), 64);
        for reference in &build.data.references {
            match reference {
                Reference::Elided => continue,
                _ => panic!("not elided"),
            }
        }

        Ok((data, build.data))
    }

    async fn no_subchunks_coarse_async() -> DataChunk {
        let data = testing::array8();
        let data: Vec<Array2<f32>> = data
            .into_iter()
            .map(|a| Array2::from_shape_fn((16, 16), |(row, col)| a[[row / 2, col / 2]]))
            .collect();

        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            3,
            2,
            Precise(3),
            0,
        )?;
        assert_eq!(build.data.references.len(), 64);
        for reference in &build.data.references {
            match reference {
                Reference::Elided => continue,
                _ => panic!("not elided"),
            }
        }

        Ok((data, build.data))
    }

    test_all_the_things!(no_subchunks_coarse);

    fn local_subchunks() -> DataChunk {
        let data = testing::array(16);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            1 << 14,
        )?;
        assert_eq!(build.data.references.len(), 16);
        assert_eq!(build.data.local.len(), 4);

        Ok((data, build.data))
    }

    async fn local_subchunks_async() -> DataChunk {
        let data = testing::array(16);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            1 << 14,
        )?;
        assert_eq!(build.data.references.len(), 16);
        assert_eq!(build.data.local.len(), 4);

        Ok((data, build.data))
    }

    test_all_the_things!(local_subchunks);

    fn external_subchunks() -> DataChunk {
        let data = testing::array(16);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            0,
        )?;
        assert_eq!(build.data.references.len(), 16);

        for reference in &build.data.references {
            match reference {
                Reference::External(_) => {}
                _ => {
                    panic!("Expecting external references only");
                }
            }
        }

        assert_eq!(build.data.external()?.len(), 4);

        Ok((data, build.data))
    }

    async fn external_subchunks_async() -> DataChunk {
        let data = testing::array(16);
        let build = build_superchunk_async(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            0,
        )
        .await?;
        assert_eq!(build.data.references.len(), 16);

        for reference in &build.data.references {
            match reference {
                Reference::External(_) => {}
                _ => {
                    panic!("Expecting external references only");
                }
            }
        }

        assert_eq!(build.data.external()?.len(), 4);

        Ok((data, build.data))
    }

    test_all_the_things!(external_subchunks);

    fn mixed_subchunks() -> DataChunk {
        let data = testing::array(17);
        let build = build_superchunk(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            8000,
        )?;
        assert_eq!(build.data.references.len(), 16);

        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in build.data.references.iter() {
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

        Ok((data, build.data))
    }

    async fn mixed_subchunks_async() -> DataChunk {
        let data = testing::array(17);
        let build = build_superchunk_async(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            8000,
        )
        .await?;
        assert_eq!(build.data.references.len(), 16);

        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in build.data.references.iter() {
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

        Ok((data, build.data))
    }

    test_all_the_things!(mixed_subchunks);

    fn elide_everything() -> DataChunk {
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
            2,
            2,
            Precise(3),
            0,
        )?;

        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in build.data.references.iter() {
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

        Ok((data, build.data))
    }

    async fn elide_everything_async() -> DataChunk {
        let length = 100;
        let shape = (16, 16);
        let mut data = vec![];

        for i in 0..length {
            let slice = Array2::from_elem(shape, i as f32);
            data.push(slice);
        }

        let build = build_superchunk_async(
            data.clone().into_iter(),
            testing::resolver(),
            2,
            2,
            Precise(3),
            0,
        )
        .await?;

        let mut local_count = 0;
        let mut external_count = 0;
        let mut elided_count = 0;
        for r in build.data.references.iter() {
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

        Ok((data, build.data))
    }

    test_all_the_things!(elide_everything);
}
