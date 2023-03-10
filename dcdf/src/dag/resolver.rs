use std::{any::TypeId, fmt::Debug, io, sync::Arc};

use cid::Cid;
use futures::{io::AsyncRead, FutureExt};
use num_traits::Float;

use crate::{
    cache::{Cache, Cacheable},
    codec::FChunk,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite},
};

use super::{
    links::Links,
    mapper::Mapper,
    mmarray::MMArray3,
    node::{self, Node},
    superchunk::Superchunk,
};

const MAGIC_NUMBER: u16 = 0xDCDF + 1;
const FORMAT_VERSION: u32 = 0;

const TYPE_F32: u8 = 32;
const TYPE_F64: u8 = 64;

/// The `Resolver` manages storage and retrieval of objects from an IPLD datastore
///
/// To store and load objects, a Resolver must be provided with a concrete `Mapper` implementation.
/// Loaded objects are stored in RAM in an LRU cache up to a specified size limit, for fast
/// re-retrieval of recently used objects.
///
pub struct Resolver<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    mapper: Box<dyn Mapper>,
    async_cache: Cache<Cid, CacheItem<N>>,
}

enum CacheItem<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    Links(Arc<Links>),
    MMArray3(Arc<MMArray3<N>>),
    Subchunk(Arc<FChunk<N>>),
    Superchunk(Arc<Superchunk<N>>),
}

impl<N> CacheItem<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn ls(&self) -> Vec<(String, Cid)> {
        match self {
            CacheItem::Links(links) => <Links as Node<N>>::ls(links),
            CacheItem::Subchunk(chunk) => chunk.ls(),
            CacheItem::Superchunk(chunk) => chunk.ls(),
            CacheItem::MMArray3(chunk) => chunk.ls(),
        }
    }
}

impl<N> Cacheable for CacheItem<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        match self {
            CacheItem::Links(links) => links.size(),
            CacheItem::Subchunk(chunk) => chunk.size(),
            CacheItem::Superchunk(chunk) => chunk.size(),
            CacheItem::MMArray3(chunk) => chunk.size(),
        }
    }
}

impl<N> Resolver<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub(crate) const HEADER_SIZE: u64 = 2 + 4 + 1 + 1;

    /// Create a new `Resolver`
    ///
    /// # Arguments
    ///
    /// * `mapper` - A boxed implementation of `Mapper`, which handles writing to and reading from
    ///   the underlying data store.
    /// * `cache_bytes` - The size limit, in bytes, for the LRU cache used by the resolver to hold
    ///   recently used objects in RAM. All objects stored in the cache implement `Cacheable` which
    ///   allows them to self-report the number of bytes they take up in RAM, usually approximated
    ///   by reporting the number of bytes in the serialized representation (close enough).
    ///
    pub fn new(mapper: Box<dyn Mapper>, cache_bytes: u64) -> Self {
        let async_cache = Cache::new(cache_bytes);
        Self {
            mapper,
            async_cache,
        }
    }

    /// Get a `Superchunk` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the superchunk to retreive.
    ///
    pub async fn get_superchunk(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Superchunk<N>>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Superchunk(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting superchunk."),
        }
    }

    /// Get an `MMArray3` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the superchunk to retreive.
    ///
    pub async fn get_mmarray3(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<MMArray3<N>>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::MMArray3(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting 3 dimensional MM array."),
        }
    }

    /// Get a `Links` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the links to retreive.
    ///
    pub(crate) async fn get_links(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Links>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Links(links) => Ok(Arc::clone(&links)),
            _ => panic!("Expecting links."),
        }
    }

    async fn check_cache(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<CacheItem<N>>> {
        let resolver = Arc::clone(self);
        let load = |cid: Cid| async move { resolver.retrieve(cid.clone()).await }.boxed();
        self.async_cache.get(cid, load).await
    }

    /// Compute the hash for a subchunk.
    ///
    pub(crate) async fn hash<O>(self: &Arc<Resolver<N>>, object: &O) -> Result<Cid>
    where
        O: Node<N>,
    {
        let mut hasher = self.mapper.hash().await;
        object.save_to(self, &mut hasher).await?;

        Ok(hasher.finish().await)
    }

    /// Store a node
    ///
    pub(crate) async fn save<O>(self: &Arc<Resolver<N>>, node: O) -> Result<Cid>
    where
        O: Node<N>,
    {
        let mut stream = self.mapper.store().await;
        stream.write_u16(MAGIC_NUMBER).await?;
        stream.write_u32(FORMAT_VERSION).await?;
        stream.write_byte(Self::type_code()).await?;
        stream.write_byte(O::NODE_TYPE).await?;

        node.save_to(&self, &mut stream).await?;

        Ok(stream.finish().await)
    }

    /// Retrieve a node
    ///
    async fn retrieve(self: &Arc<Resolver<N>>, cid: Cid) -> Result<Option<CacheItem<N>>> {
        match self.mapper.load(&cid).await {
            None => Ok(None),
            Some(mut stream) => {
                let node_type = self.read_header(&mut stream).await?;
                let item = match node_type {
                    node::NODE_LINKS => {
                        CacheItem::Links(Arc::new(Links::load_from(self, &mut stream).await?))
                    }
                    node::NODE_SUBCHUNK => {
                        CacheItem::Subchunk(Arc::new(FChunk::load_from(self, &mut stream).await?))
                    }
                    node::NODE_SUPERCHUNK => CacheItem::Superchunk(Arc::new(
                        Superchunk::load_from(self, &mut stream).await?,
                    )),
                    node::NODE_MMARRAY3 => {
                        CacheItem::MMArray3(Arc::new(MMArray3::load_from(self, &mut stream).await?))
                    }
                    _ => panic!("Unrecognized node type: {node_type}"),
                };

                Ok(Some(item))
            }
        }
    }

    async fn read_header(&self, stream: &mut (impl AsyncRead + Unpin + Send)) -> io::Result<u8> {
        let magic_number = stream.read_u16().await?;
        if magic_number != MAGIC_NUMBER {
            panic!("File is not a DCDF graph node file.");
        }

        let version = stream.read_u32().await?;
        if version != FORMAT_VERSION {
            panic!("Unrecognized file format.");
        }

        if Self::type_code() != stream.read_byte().await? {
            panic!("Numeric type doesn't match.");
        }

        stream.read_byte().await
    }

    pub async fn ls(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Option<Vec<LsEntry>>> {
        match self.retrieve(cid.clone()).await? {
            None => Ok(None),
            Some(object) => {
                let mut ls = Vec::new();
                for (name, cid) in object.ls() {
                    let node_type = self.node_type_of(&cid).await?;
                    let size = self.mapper.size_of(&cid).await?;
                    let entry = LsEntry {
                        cid,
                        name,
                        node_type,
                        size,
                    };
                    ls.push(entry)
                }

                Ok(Some(ls))
            }
        }
    }

    async fn node_type_of(&self, cid: &Cid) -> Result<Option<&'static str>> {
        match self.mapper.load(cid).await {
            None => Ok(None),
            Some(mut stream) => {
                let mut code = self.read_header(&mut stream).await?;
                if code == node::NODE_MMARRAY3 {
                    code = stream.read_byte().await?;
                }
                let node_type = match code {
                    node::NODE_LINKS => "Links",
                    node::NODE_SUBCHUNK => "Subchunk",
                    node::NODE_SUPERCHUNK => "Superchunk",
                    _ => panic!("Unrecognized node type: {code}"),
                };

                Ok(Some(node_type))
            }
        }
    }

    fn type_code() -> u8 {
        if TypeId::of::<N>() == TypeId::of::<f32>() {
            TYPE_F32
        } else if TypeId::of::<N>() == TypeId::of::<f64>() {
            TYPE_F64
        } else {
            panic!("Unsupported type: {:?}", TypeId::of::<N>())
        }
    }
}

pub struct LsEntry {
    pub cid: Cid,
    pub name: String,
    pub node_type: Option<&'static str>,
    pub size: Option<u64>,
}
