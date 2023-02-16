use std::{
    collections::HashMap,
    fmt::Debug,
    io::{self, Write},
    mem,
    pin::Pin,
    result,
    sync::Arc,
};

use async_trait::async_trait;
use cid::{multihash::MultihashGeneric, Cid};
use futures::{
    io::{AsyncRead, AsyncWrite, Cursor, Error as AioError},
    task::{Context, Poll},
};
use multihash::{Hasher, Sha2_256};
use ndarray::{Array1, arr2, Array2, ArrayView2, Array3};
use num_traits::{Float, Num};
use parking_lot::Mutex;

use crate::{
    errors::Result,
    geom,
};

use super::{
    mapper::{Mapper, StoreWrite},
    resolver::Resolver,
    superchunk::Superchunk,
};

pub(crate) type AioResult<T> = result::Result<T, AioError>;

/// The SHA_256 multicodec code
const SHA2_256: u64 = 0x12;

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub(crate) fn array_search_window<N>(
    data: ArrayView2<N>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    lower: N,
    upper: N,
) -> Vec<(usize, usize)>
where
    N: Num + Debug + Copy + PartialOrd,
{
    let mut coords: Vec<(usize, usize)> = vec![];
    for row in top..bottom {
        for col in left..right {
            let cell_value = data[[row, col]];
            if lower <= cell_value && cell_value <= upper {
                coords.push((row, col));
            }
        }
    }

    coords
}

/// A test implementation of Mapper that stores objects in RAM
///
struct MemoryMapper {
    objects: Mutex<HashMap<Cid, Vec<u8>>>,
}

impl MemoryMapper {
    fn new() -> Self {
        Self {
            objects: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Mapper for MemoryMapper {
    async fn store(&self) -> Box<dyn StoreWrite + '_> {
        Box::new(MemoryMapperStoreWrite::new(self, false))
    }

    async fn hash(&self) -> Box<dyn StoreWrite + '_> {
        Box::new(MemoryMapperStoreWrite::new(self, true))
    }

    async fn load(&self, cid: &Cid) -> Option<Box<dyn AsyncRead + Unpin + Send + '_>> {
        let objects = self.objects.lock();
        let object = objects.get(cid)?;
        Some(Box::new(Cursor::new(object.clone())))
    }

    async fn size_of(&self, cid: &Cid) -> io::Result<Option<u64>> {
        let objects = self.objects.lock();
        Ok(objects
            .get(cid)
            .and_then(|object| Some(object.len() as u64)))
    }
}

struct MemoryMapperStoreWrite<'a> {
    mapper: &'a MemoryMapper,
    buffer: Vec<u8>,
    hash: Sha2_256,
    hash_only: bool,
}

impl<'a> MemoryMapperStoreWrite<'a> {
    fn new(mapper: &'a MemoryMapper, hash_only: bool) -> Self {
        Self {
            mapper,
            buffer: Vec::new(),
            hash: Sha2_256::default(),
            hash_only,
        }
    }
}

impl<'a> AsyncWrite for MemoryMapperStoreWrite<'a> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<AioResult<usize>> {
        let result = self.buffer.write(buf);
        if let Ok(len) = result {
            self.hash.update(&buf[..len]);
        }

        Poll::Ready(result)
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<AioResult<()>> {
        self.buffer.flush().expect("This should be a noop, anyway.");

        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<AioResult<()>> {
        Poll::Ready(Ok(()))
    }
}

#[async_trait]
impl<'a> StoreWrite for MemoryMapperStoreWrite<'a> {
    async fn finish(mut self: Box<Self>) -> Cid {
        let object = mem::replace(&mut self.buffer, vec![]);
        let digest = self.hash.finalize();
        let hash = MultihashGeneric::wrap(SHA2_256, &digest).expect("Not really sure.");
        let cid = Cid::new_v1(SHA2_256, hash);

        if !self.hash_only {
            self.mapper.objects.lock().insert(cid, object);
        }

        cid
    }
}

pub(crate) fn cid_for(data: &str) -> Cid {
    let mut hash = Sha2_256::default();
    hash.update(&data.as_bytes());

    let digest = hash.finalize();
    let hash = MultihashGeneric::wrap(SHA2_256, &digest).expect("Not really sure.");

    Cid::new_v1(SHA2_256, hash)
}

pub(crate) fn resolver<N>() -> Arc<Resolver<N>>
where
    N: Float + Debug + Send + Sync,
{
    Arc::new(Resolver::new(Box::new(MemoryMapper::new()), 0))
}

pub(crate) fn array8() -> Vec<Array2<f32>> {
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

pub(crate) fn array(sidelen: usize) -> Vec<Array2<f32>> {
    let data = array8();

    data.into_iter()
        .map(|a| Array2::from_shape_fn((sidelen, sidelen), |(row, col)| a[[row % 8, col % 8]]))
        .collect()
}

impl<N> Superchunk<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Get a cell's value across time instants. Convenience for calling fill_cell in tests.
    ///
    pub(crate) async fn get_cell(
        self: &Arc<Self>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<Array1<N>> {
        let mut values = Array1::zeros([end - start]);
        self.fill_cell(start, row, col, &mut values).await?;

        Ok(values)
    }

    /// Get a subarray of this Chunk.
    ///
    pub(crate) async fn get_window(self: &Arc<Self>, bounds: &geom::Cube) -> Result<Array3<N>> {
        let mut window = Array3::zeros([bounds.instants(), bounds.rows(), bounds.cols()]);
        self.fill_window(bounds.start, bounds.top, bounds.left, &mut window).await?;

        Ok(window)
    }
}
