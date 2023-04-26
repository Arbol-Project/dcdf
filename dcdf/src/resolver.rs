use std::{io, sync::Arc};

use cid::Cid;
use futures::{io::AsyncRead, FutureExt};

use crate::{
    cache::{Cache, Cacheable},
    dataset::Dataset,
    errors::{Error, Result},
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite},
    links::Links,
    mapper::Mapper,
    mmstruct::MMStruct3,
    node::{self, Node},
};

const MAGIC_NUMBER: u16 = 0xDCDF + 1;
const FORMAT_VERSION: u32 = 0;

/// The `Resolver` manages storage and retrieval of objects from an IPLD datastore
///
/// To store and load objects, a Resolver must be provided with a concrete `Mapper` implementation.
/// Loaded objects are stored in RAM in an LRU cache up to a specified size limit, for fast
/// re-retrieval of recently used objects.
///
pub struct Resolver {
    mapper: Box<dyn Mapper>,
    cache: Cache<Cid, CacheItem>,
}

enum CacheItem {
    Dataset(Arc<Dataset>),
    Links(Arc<Links>),
    MMStruct3(Arc<MMStruct3>),
}

impl CacheItem {
    fn ls(&self) -> Vec<(String, Cid)> {
        match self {
            CacheItem::Dataset(_dataset) => {
                todo!();
            }
            CacheItem::Links(links) => links.ls(),
            CacheItem::MMStruct3(node) => node.ls(),
        }
    }
}

impl Cacheable for CacheItem {
    fn size(&self) -> u64 {
        match self {
            CacheItem::Dataset(dataset) => dataset.size(),
            CacheItem::Links(links) => links.size(),
            CacheItem::MMStruct3(chunk) => chunk.size(),
        }
    }
}

impl Resolver {
    pub(crate) const HEADER_SIZE: u64 = 2 + 4 + 1;

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
        Self { mapper, cache }
    }

    /// Get a `Dataset` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the dataset to retreive.
    ///
    pub async fn get_dataset(self: &Arc<Resolver>, cid: &Cid) -> Result<Arc<Dataset>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Dataset(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting 3 dimensional MM struct."),
        }
    }

    /// Get an `MMStruct3` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the array to retreive.
    ///
    pub(crate) async fn get_mmstruct3(self: &Arc<Resolver>, cid: &Cid) -> Result<Arc<MMStruct3>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::MMStruct3(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting 3 dimensional MM struct."),
        }
    }

    /// Get a `Links` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the links to retreive.
    ///
    pub(crate) async fn get_links(self: &Arc<Resolver>, cid: &Cid) -> Result<Arc<Links>> {
        let item = self.check_cache(cid).await?;
        match &*item {
            CacheItem::Links(links) => Ok(Arc::clone(&links)),
            _ => panic!("Expecting links."),
        }
    }

    async fn check_cache(self: &Arc<Resolver>, cid: &Cid) -> Result<Arc<CacheItem>> {
        let resolver = Arc::clone(self);
        let load = |cid: Cid| async move { resolver.retrieve(cid.clone()).await }.boxed();
        self.cache.get(cid, load).await
    }

    /// Store a node
    ///
    pub(crate) async fn save<O>(self: &Arc<Resolver>, node: &O) -> Result<Cid>
    where
        O: Node,
    {
        let mut stream = self.mapper.store().await;
        stream.write_u16(MAGIC_NUMBER).await?;
        stream.write_u32(FORMAT_VERSION).await?;
        stream.write_byte(O::NODE_TYPE).await?;

        node.save_to(&self, &mut stream).await?;

        Ok(stream.finish().await)
    }

    /// Retrieve a node
    ///
    async fn retrieve(self: &Arc<Resolver>, cid: Cid) -> Result<Option<CacheItem>> {
        match self.mapper.load(&cid).await {
            None => Err(Error::NotFound(cid)), // Maybe a NotFound error here, instead?
            Some(mut stream) => {
                let node_type = self.read_header(&mut stream).await?;
                let item = match node_type {
                    node::NODE_DATASET => {
                        let mut dataset = Dataset::load_from(self, &mut stream).await?;
                        dataset.cid = Some(cid);
                        CacheItem::Dataset(Arc::new(dataset))
                    }
                    node::NODE_LINKS => {
                        CacheItem::Links(Arc::new(Links::load_from(self, &mut stream).await?))
                    }
                    node::NODE_MMSTRUCT3 => CacheItem::MMStruct3(Arc::new(
                        MMStruct3::load_from(self, &mut stream).await?,
                    )),
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

        stream.read_byte().await
    }

    pub async fn ls(self: &Arc<Resolver>, cid: &Cid) -> Result<Vec<LsEntry>> {
        match self.retrieve(cid.clone()).await? {
            None => Err(Error::NotFound(cid.clone())),
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

                Ok(ls)
            }
        }
    }

    async fn node_type_of(&self, cid: &Cid) -> Result<Option<&'static str>> {
        match self.mapper.load(cid).await {
            None => Ok(None),
            Some(mut stream) => {
                let mut code = self.read_header(&mut stream).await?;
                if code == node::NODE_MMSTRUCT3 {
                    code = stream.read_byte().await?;
                }
                let node_type = match code {
                    node::NODE_LINKS => "Links",
                    node::NODE_SUBCHUNK => "Subchunk",
                    node::NODE_SUPERCHUNK => "Superchunk",
                    node::NODE_SPAN => "Span",
                    _ => panic!("Unrecognized node type: {code}"),
                };

                Ok(Some(node_type))
            }
        }
    }
}

pub struct LsEntry {
    pub cid: Cid,
    pub name: String,
    pub node_type: Option<&'static str>,
    pub size: Option<u64>,
}
