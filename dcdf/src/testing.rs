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
use ndarray::{arr2, s, Array2, Array3, ArrayView2, ArrayView3};
use num_traits::Num;
use parking_lot::Mutex;

use crate::{
    geom,
    {log::Log, snapshot::Snapshot},
    {
        mapper::{Mapper, StoreWrite},
        resolver::Resolver,
    },
};

pub(crate) type AioResult<T> = result::Result<T, AioError>;

/// The SHA_256 multicodec code
const SHA2_256: u64 = 0x12;

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub(crate) fn array_search_window2<N>(
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

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub(crate) fn array_search_window3<N>(
    data: ArrayView3<N>,
    bounds: geom::Cube,
    lower: N,
    upper: N,
) -> Vec<(usize, usize, usize)>
where
    N: Num + Debug + Copy + PartialOrd,
{
    let mut coords: Vec<(usize, usize, usize)> = vec![];
    for instant in bounds.start..bounds.end {
        for row in bounds.top..bounds.bottom {
            for col in bounds.left..bounds.right {
                let cell_value = data[[instant, row, col]];
                if lower <= cell_value && cell_value <= upper {
                    coords.push((instant, row, col));
                }
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

pub(crate) fn resolver() -> Arc<Resolver> {
    Arc::new(Resolver::new(Box::new(MemoryMapper::new()), 0))
}

pub(crate) fn array8() -> Array3<i64> {
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

    let mut array = Array3::zeros([100, 8, 8]);
    for (i, a) in data.into_iter().cycle().take(100).enumerate() {
        array.slice_mut(s![i, .., ..]).assign(&a);
    }

    array
}

pub(crate) fn array(sidelen: usize) -> Array3<i64> {
    let data = array8();

    Array3::from_shape_fn(
        (data.shape()[0], sidelen, sidelen),
        |(instant, row, col)| data[[instant, row % 8, col % 8]],
    )
}

pub(crate) fn farray8() -> Array3<f32> {
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
            [3.375, 3.375, 4.875, 5.0, 4.875, 4.875, 4.875, 4.875],
            [4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875],
        ]),
        arr2(&[
            [9.5, 8.25, 7.75, 7.75, 8.25, 7.75, 5.0, 5.0],
            [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 5.0, 5.0],
            [7.75, 7.75, 6.125, 6.125, 4.875, 3.375, 4.875, 4.875],
            [6.125, 6.125, 6.125, 6.125, 4.875, 4.875, 4.875, 4.875],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 4.875, 5.0, 4.875, 4.875, 4.875, 4.875],
            [4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875],
        ]),
        arr2(&[
            [f32::NAN, 8.25, 7.75, 7.75, 6.125, 6.125, 3.375, 2.625],
            [7.75, 7.75, 7.75, 7.75, 6.125, 6.125, 3.375, 3.375],
            [6.125, 6.125, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
            [5.0, 5.0, 6.125, 6.125, 3.375, 3.375, 3.375, 3.375],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [f32::NAN, f32::NAN, 3.375, 5.0, 4.875, 4.875, 4.875, 4.875],
            [f32::NAN, f32::NAN, 3.375, 4.875, 4.875, 4.875, 4.875, 4.875],
        ]),
        arr2(&[
            [9.5, f32::NAN, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
            [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 2.625, 2.625],
            [f32::NAN, 6.125, 6.125, 6.125, 4.875, 3.375, 3.375, 3.375],
            [5.0, 5.0, f32::NAN, 6.125, 3.375, 3.375, 3.375, 3.375],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 4.875, 5.0, 4.875, 4.875, 4.875, 4.875],
            [4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875, 4.875],
        ]),
        arr2(&[
            [9.5, 8.25, 7.75, 7.75, 8.25, 7.75, 5.0, 5.0],
            [7.75, 7.75, 7.75, 7.75, 7.75, 7.75, 5.0, 5.0],
            [7.75, 7.75, 6.125, 6.125, 4.875, 3.375, 4.875, 4.875],
            [6.125, 6.125, 6.125, 6.125, 4.875, 4.875, 4.875, 4.875],
            [4.875, 5.0, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 5.0, 5.0, 4.875, 4.875, 4.875, 4.875],
            [3.375, 3.375, 4.875, 5.0, 4.875, 4.875, f32::NAN, f32::NAN],
            [
                4.875,
                4.875,
                4.875,
                4.875,
                4.875,
                f32::NAN,
                f32::NAN,
                f32::NAN,
            ],
        ]),
    ];

    let mut array = Array3::zeros([100, 8, 8]);
    for (i, a) in data.into_iter().cycle().take(100).enumerate() {
        array.slice_mut(s![i, .., ..]).assign(&a);
    }

    array
}

pub(crate) fn farray(sidelen: usize) -> Array3<f32> {
    let data = farray8();

    Array3::from_shape_fn(
        (data.shape()[0], sidelen, sidelen),
        |(instant, row, col)| data[[instant, row % 8, col % 8]],
    )
}

impl Snapshot {
    /// Wrap Snapshot.build with function that creates the `get` closure so that it doesn't have to
    /// be repeated in every test of Snapshot.
    pub(crate) fn from_array(data: ArrayView2<i64>, k: i32) -> Self {
        let get = |row, col| data[[row, col]];
        let shape = data.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get, [rows, cols], k)
    }

    /// Wrap Snapshot.get_window with function that allocates an array and creates the `set`
    /// enclosure, so it doesn't have to repeated in every test for `get_window`.
    ///
    pub(crate) fn get_window(&self, bounds: &geom::Rect) -> Array2<i64> {
        let mut window = Array2::zeros([bounds.rows(), bounds.cols()]);
        let set = |row, col, value| window[[row, col]] = value;

        self.fill_window(set, bounds);

        window
    }
}

impl Log {
    /// Wrap Log.build with function that creates the `get_s` and `get_t` closures so that they
    /// don't  have to be repeated in every test of Log.
    pub(crate) fn from_arrays(snapshot: ArrayView2<i64>, log: ArrayView2<i64>, k: i32) -> Self {
        let get_s = |row, col| snapshot[[row, col]];
        let get_t = |row, col| log[[row, col]];
        let shape = snapshot.shape();
        let rows = shape[0];
        let cols = shape[1];
        Self::build(get_s, get_t, [rows, cols], k)
    }

    /// Wrap Log.get_window with function that allocates an array and creates the `set`
    /// enclosure, so it doesn't have to repeated in every test for `get_window`.
    ///
    pub(crate) fn get_window(&self, snapshot: &Snapshot, bounds: &geom::Rect) -> Array2<i64> {
        let mut window = Array2::zeros([bounds.rows(), bounds.cols()]);
        let set = |row, col, value| window[[row, col]] = value;

        self.fill_window(set, snapshot, bounds);

        window
    }
}
