use std::{
    any::TypeId,
    fmt::Debug,
    io::{self, Read},
    sync::Arc,
};

use cid::Cid;
use futures::{io::AsyncRead, FutureExt};
use num_traits::Float;

use crate::{
    cache::{Cache, Cacheable},
    cache_async::Cache as AsyncCache,
    codec::FChunk,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, ExtendedRead, ExtendedWrite, Serialize},
};

use super::{
    commit::Commit,
    folder::Folder,
    links::Links,
    mapper::{Mapper, StoreWrite},
    node::{self, AsyncNode, Node},
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
    cache: Cache<Cid, CacheItem<N>>,
    async_cache: AsyncCache<Cid, CacheItem<N>>,
}

enum CacheItem<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    Commit(Arc<Commit<N>>),
    Folder(Arc<Folder<N>>),
    Links(Arc<Links>),
    Subchunk(Arc<FChunk<N>>),
    Superchunk(Arc<Superchunk<N>>),
}

impl<N> CacheItem<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn ls(&self) -> Vec<(String, Cid)> {
        match self {
            CacheItem::Commit(commit) => commit.ls(),
            CacheItem::Folder(folder) => folder.ls(),
            CacheItem::Links(links) => <Links as Node<N>>::ls(links),
            CacheItem::Subchunk(chunk) => chunk.ls(),
            CacheItem::Superchunk(chunk) => chunk.ls(),
        }
    }
}

impl<N> Cacheable for CacheItem<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        match self {
            CacheItem::Commit(commit) => commit.size(),
            CacheItem::Folder(folder) => folder.size(),
            CacheItem::Links(links) => links.size(),
            CacheItem::Subchunk(chunk) => chunk.size(),
            CacheItem::Superchunk(chunk) => chunk.size(),
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
        let cache = Cache::new(cache_bytes);
        let async_cache = AsyncCache::new(cache_bytes);
        Self {
            mapper,
            cache,
            async_cache,
        }
    }

    /// Get a `Folder` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the folder to retreive.
    ///
    pub fn get_folder(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Folder<N>>> {
        let item = self.cache.get(cid, |cid| self.retrieve(&cid))?;

        match &*item {
            CacheItem::Folder(folder) => Ok(Arc::clone(&folder)),
            _ => panic!("Expecting folder."),
        }
    }

    pub async fn get_folder_async(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Folder<N>>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Folder(folder) => Ok(Arc::clone(&folder)),
            _ => panic!("Expecting folder."),
        }
    }

    /// Get a `Commit` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the commit to retreive.
    ///
    pub fn get_commit(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Commit<N>>> {
        let item = self.cache.get(cid, |cid| self.retrieve(&cid))?;

        match &*item {
            CacheItem::Commit(commit) => Ok(Arc::clone(&commit)),
            _ => panic!("Expecting commit."),
        }
    }

    pub async fn get_commit_async(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Commit<N>>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Commit(commit) => Ok(Arc::clone(&commit)),
            _ => panic!("Expecting commit."),
        }
    }

    /// Get a `Superchunk` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the superchunk to retreive.
    ///
    pub fn get_superchunk(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Superchunk<N>>> {
        let item = self.cache.get(cid, |cid| self.retrieve(&cid))?;

        match &*item {
            CacheItem::Superchunk(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting superchunk."),
        }
    }

    /// Get a `Superchunk` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the superchunk to retreive.
    ///
    pub async fn get_superchunk_async(
        self: &Arc<Resolver<N>>,
        cid: &Cid,
    ) -> Result<Arc<Superchunk<N>>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Superchunk(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting superchunk."),
        }
    }

    /// Get an `Fchunk` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the chunk to retreive.
    ///
    pub(crate) fn get_subchunk(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<FChunk<N>>> {
        let load = |cid| self.retrieve(&cid);
        let item = self.cache.get(cid, load)?;

        match &*item {
            CacheItem::Subchunk(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting subchunk."),
        }
    }

    /// Get an `Fchunk` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the chunk to retreive.
    ///
    pub(crate) async fn get_subchunk_async(
        self: &Arc<Resolver<N>>,
        cid: &Cid,
    ) -> Result<Arc<FChunk<N>>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Subchunk(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting subchunk."),
        }
    }

    /// Get a `Links` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the links to retreive.
    ///
    pub(crate) fn get_links(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Links>> {
        let item = self.cache.get(cid, |cid| self.retrieve(&cid))?;

        match &*item {
            CacheItem::Links(links) => Ok(Arc::clone(&links)),
            _ => panic!("Expecting links."),
        }
    }

    /// Get a `Links` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the links to retreive.
    ///
    pub(crate) async fn get_links_async(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Links>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Links(links) => Ok(Arc::clone(&links)),
            _ => panic!("Expecting links."),
        }
    }

    async fn check_cache(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<CacheItem<N>>> {
        let resolver = Arc::clone(self);
        let load = |cid: Cid| async move { resolver.retrieve_async(cid.clone()).await }.boxed();
        self.async_cache.get(cid, load).await
    }

    /// Compute the hash for a subchunk.
    ///
    pub(crate) fn hash_subchunk(self: &Arc<Resolver<N>>, object: &FChunk<N>) -> Result<Cid> {
        let mut hasher = self.mapper.hash();
        object.write_to(&mut hasher)?;

        Ok(hasher.finish())
    }

    /// Store a node
    ///
    pub fn save<O>(self: &Arc<Resolver<N>>, node: O) -> Result<Cid>
    where
        O: Node<N>,
    {
        let mut stream = self.mapper.store();
        stream.write_u16(MAGIC_NUMBER)?;
        stream.write_u32(FORMAT_VERSION)?;
        stream.write_byte(Self::type_code())?;
        stream.write_byte(O::NODE_TYPE)?;

        node.save_to(&self, &mut stream)?;

        Ok(stream.finish())
    }

    /// Store a node
    ///
    pub async fn save_async<O>(self: &Arc<Resolver<N>>, node: O) -> Result<Cid>
    where
        O: AsyncNode<N>,
    {
        let mut stream = self.mapper.store_async().await;
        stream.write_u16_async(MAGIC_NUMBER).await?;
        stream.write_u32_async(FORMAT_VERSION).await?;
        stream.write_byte_async(Self::type_code()).await?;
        stream.write_byte_async(O::NODE_TYPE).await?;

        node.save_to_async(&self, &mut stream).await?;

        Ok(stream.finish_async().await)
    }

    /// Retrieve a node
    ///
    fn retrieve(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Option<CacheItem<N>>> {
        match self.mapper.load(cid) {
            None => Ok(None),
            Some(mut stream) => {
                let node_type = self.read_header(&mut stream)?;
                let item = match node_type {
                    node::NODE_COMMIT => {
                        CacheItem::Commit(Arc::new(Commit::load_from(self, &mut stream)?))
                    }
                    node::NODE_LINKS => {
                        CacheItem::Links(Arc::new(Links::load_from(self, &mut stream)?))
                    }
                    node::NODE_FOLDER => {
                        CacheItem::Folder(Arc::new(Folder::load_from(self, &mut stream)?))
                    }
                    node::NODE_SUBCHUNK => {
                        CacheItem::Subchunk(Arc::new(FChunk::load_from(self, &mut stream)?))
                    }
                    node::NODE_SUPERCHUNK => {
                        CacheItem::Superchunk(Arc::new(Superchunk::load_from(self, &mut stream)?))
                    }
                    _ => panic!("Unrecognized node type: {node_type}"),
                };

                Ok(Some(item))
            }
        }
    }

    /// Retrieve a node
    ///
    async fn retrieve_async(self: &Arc<Resolver<N>>, cid: Cid) -> Result<Option<CacheItem<N>>> {
        match self.mapper.load_async(&cid).await {
            None => Ok(None),
            Some(mut stream) => {
                let node_type = self.read_header_async(&mut stream).await?;
                let item = match node_type {
                    node::NODE_COMMIT => CacheItem::Commit(Arc::new(
                        Commit::load_from_async(self, &mut stream).await?,
                    )),
                    node::NODE_LINKS => {
                        CacheItem::Links(Arc::new(Links::load_from_async(self, &mut stream).await?))
                    }
                    node::NODE_FOLDER => CacheItem::Folder(Arc::new(
                        Folder::load_from_async(self, &mut stream).await?,
                    )),
                    node::NODE_SUBCHUNK => CacheItem::Subchunk(Arc::new(
                        FChunk::load_from_async(self, &mut stream).await?,
                    )),
                    node::NODE_SUPERCHUNK => CacheItem::Superchunk(Arc::new(
                        Superchunk::load_from_async(self, &mut stream).await?,
                    )),
                    _ => panic!("Unrecognized node type: {node_type}"),
                };

                Ok(Some(item))
            }
        }
    }

    fn read_header(&self, stream: &mut impl Read) -> io::Result<u8> {
        let magic_number = stream.read_u16()?;
        if magic_number != MAGIC_NUMBER {
            panic!("File is not a DCDF graph node file.");
        }

        let version = stream.read_u32()?;
        if version != FORMAT_VERSION {
            panic!("Unrecognized file format.");
        }

        if Self::type_code() != stream.read_byte()? {
            panic!("Numeric type doesn't match.");
        }

        stream.read_byte()
    }

    async fn read_header_async(
        &self,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> io::Result<u8> {
        let magic_number = stream.read_u16_async().await?;
        if magic_number != MAGIC_NUMBER {
            panic!("File is not a DCDF graph node file.");
        }

        let version = stream.read_u32_async().await?;
        if version != FORMAT_VERSION {
            panic!("Unrecognized file format.");
        }

        if Self::type_code() != stream.read_byte_async().await? {
            panic!("Numeric type doesn't match.");
        }

        stream.read_byte_async().await
    }

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Returns `Option::None` if given `cid` isn't in the store.
    ///
    pub fn load(&self, cid: &Cid) -> Option<Box<dyn Read + '_>> {
        self.mapper.load(cid)
    }

    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    pub fn store(&self) -> Box<dyn StoreWrite + '_> {
        self.mapper.store()
    }

    pub fn ls(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Option<Vec<LsEntry>>> {
        match self.retrieve(cid)? {
            None => Ok(None),
            Some(object) => {
                let mut ls = Vec::new();
                for (name, cid) in object.ls() {
                    let node_type = self.node_type_of(&cid)?;
                    let size = self.mapper.size_of(&cid)?;
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

    fn node_type_of(&self, cid: &Cid) -> Result<Option<&'static str>> {
        match self.mapper.load(cid) {
            None => Ok(None),
            Some(mut stream) => {
                let code = self.read_header(&mut stream)?;
                let node_type = match code {
                    node::NODE_COMMIT => "Commit",
                    node::NODE_LINKS => "Links",
                    node::NODE_FOLDER => "Folder",
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
