use std::collections::HashMap;
use std::fmt::Debug;
use std::io;
use std::io::{Cursor, Read, Write};
use std::mem;
use std::sync::Arc;

use cid::Cid;
use multihash::{Hasher, MultihashGeneric, Sha2_256};
use ndarray::{arr2, Array2, ArrayView2};
use num_traits::{Float, Num};
use parking_lot::Mutex;

use crate::errors::Result;
use crate::fixed::Precise;

use super::mapper::{Mapper, StoreWrite};
use super::resolver::Resolver;
use super::superchunk::{build_superchunk, Superchunk};

/// The SHA_256 multicodec code
const SHA2_256: u64 = 0x12;

/// Reference implementation for search_window that works on an ndarray::Array2, for comparison
/// to the K^2 raster implementations.
pub fn array_search_window<N>(
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

impl Mapper for MemoryMapper {
    fn store(&self) -> Box<dyn StoreWrite + '_> {
        Box::new(MemoryMapperStoreWrite::new(self, false))
    }

    fn hash(&self) -> Box<dyn StoreWrite + '_> {
        Box::new(MemoryMapperStoreWrite::new(self, true))
    }

    fn load(&self, cid: &Cid) -> Option<Box<dyn Read + '_>> {
        let objects = self.objects.lock();
        let object = objects.get(cid)?;
        Some(Box::new(Cursor::new(object.clone())))
    }
}

struct MemoryMapperStoreWrite<'a> {
    mapper: &'a MemoryMapper,
    writer: Box<Sha2_256Write<Vec<u8>>>,
    hash_only: bool,
}

impl<'a> MemoryMapperStoreWrite<'a> {
    fn new(mapper: &'a MemoryMapper, hash_only: bool) -> Self {
        let writer = Box::new(Sha2_256Write::wrap(Vec::new()));
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
    fn finish(mut self: Box<Self>) -> Cid {
        let object = mem::replace(&mut self.writer.inner, vec![]);
        let cid = self.writer.finish();
        if !self.hash_only {
            self.mapper.objects.lock().insert(cid, object);
        }

        cid
    }
}

/// An implementor of `StoreWrite` that computes CIDs using Sha2 256.
///
pub struct Sha2_256Write<W: Write> {
    pub inner: W,
    hash: Sha2_256,
}

impl<W> Sha2_256Write<W>
where
    W: Write,
{
    /// Wrap an existing output stream
    ///
    pub fn wrap(inner: W) -> Self {
        Self {
            inner,
            hash: Sha2_256::default(),
        }
    }
}

impl<W> Write for Sha2_256Write<W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let result = self.inner.write(buf);
        if let Ok(len) = result {
            self.hash.update(&buf[..len]);
        }
        result
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W> StoreWrite for Sha2_256Write<W>
where
    W: Write,
{
    fn finish(mut self: Box<Self>) -> Cid {
        let digest = self.hash.finalize();
        let hash = MultihashGeneric::wrap(SHA2_256, &digest).expect("Not really sure.");

        Cid::new_v1(SHA2_256, hash)
    }
}

pub fn resolver<N>() -> Arc<Resolver<N>>
where
    N: Float + Debug,
{
    Arc::new(Resolver::new(Box::new(MemoryMapper::new()), 0))
}

pub fn array8() -> Vec<Array2<f32>> {
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

pub fn array(sidelen: usize) -> Vec<Array2<f32>> {
    let data = array8();

    data.into_iter()
        .map(|a| Array2::from_shape_fn((sidelen, sidelen), |(row, col)| a[[row % 8, col % 8]]))
        .collect()
}

pub fn superchunk(
    data: &Vec<Array2<f32>>,
    resolver: &Arc<Resolver<f32>>,
) -> Result<Superchunk<f32>> {
    let chunk = build_superchunk(
        data.clone().into_iter(),
        Arc::clone(resolver),
        3,
        2,
        Precise(3),
        0,
    )?;

    Ok(chunk)
}
