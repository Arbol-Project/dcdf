use cid::Cid;
use num_traits::Float;
use std::fmt::Debug;
use std::io::Read;
use std::sync::Arc;

use crate::cache::{Cache, Cacheable};
use crate::codec::FChunk;
use crate::errors::Result;
use crate::extio::Serialize;

use super::commit::Commit;
use super::folder::Folder;
use super::mapper::{Mapper, StoreWrite};
use super::node::Node;
use super::superchunk::Superchunk;

/// The `Resolver` manages storage and retrieval of objects from an IPLD datastore
///
/// To store and load objects, a Resolver must be provided with a concrete `Mapper` implementation.
/// Loaded objects are stored in RAM in an LRU cache up to a specified size limit, for fast
/// re-retrieval of recently used objects.
///
pub struct Resolver<N>
where
    N: Float + Debug + 'static,
{
    mapper: Box<dyn Mapper>,
    cache: Cache<Cid, CacheItem<N>>,
}

enum CacheItem<N>
where
    N: Float + Debug + 'static,
{
    Commit(Arc<Commit<N>>),
    Subchunk(Arc<FChunk<N>>),
    Superchunk(Arc<Superchunk<N>>),
}

impl<N> Cacheable for CacheItem<N>
where
    N: Float + Debug + 'static,
{
    fn size(&self) -> u64 {
        match self {
            CacheItem::Commit(_) => 1, // not important
            CacheItem::Subchunk(chunk) => chunk.size(),
            CacheItem::Superchunk(chunk) => chunk.size(),
        }
    }
}

impl<N> Resolver<N>
where
    N: Float + Debug + 'static,
{
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

    /// Get a `Folder` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the folder to retreive.
    ///
    pub fn get_folder(self: &Arc<Resolver<N>>, cid: &Cid) -> Arc<Folder<N>> {
        Folder::new(self, *cid, self.mapper.ls(cid))
    }

    /// Get a `Commit` from the data store.
    ///
    /// # Arguments
    ///
    /// * `cid` - The CID of the commit to retreive.
    ///
    pub fn get_commit(self: &Arc<Resolver<N>>, cid: &Cid) -> Result<Arc<Commit<N>>> {
        let item = self.cache.get(cid, |cid| {
            let commit = Commit::retrieve(self, &cid)?;
            match commit {
                Some(commit) => Ok(Some(CacheItem::Commit(Arc::new(commit)))),
                None => Ok(None),
            }
        })?;

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
        let item = self.cache.get(cid, |cid| {
            let chunk = Superchunk::retrieve(self, &cid)?;
            match chunk {
                Some(chunk) => Ok(Some(CacheItem::Superchunk(Arc::new(chunk)))),
                None => Ok(None),
            }
        })?;

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
        let item = self.cache.get(cid, |cid| {
            let chunk = FChunk::retrieve(self, &cid)?;
            match chunk {
                Some(chunk) => Ok(Some(CacheItem::Subchunk(Arc::new(chunk)))),
                None => Ok(None),
            }
        })?;

        match &*item {
            CacheItem::Subchunk(chunk) => Ok(Arc::clone(&chunk)),
            _ => panic!("Expecting subchunk."),
        }
    }

    /// Compute the hash for a subchunk.
    ///
    pub(crate) fn hash_subchunk(self: &Arc<Resolver<N>>, object: &FChunk<N>) -> Result<Cid> {
        let mut hasher = self.mapper.hash();
        object.write_to(&mut hasher)?;

        Ok(hasher.finish())
    }

    /// Save a node, possibly as a folder
    ///
    pub fn save<O>(self: &Arc<Resolver<N>>, node: O) -> Result<Cid>
    where
        O: Node<N>,
    {
        Ok(node.store(self)?)
    }

    /// Save a leaf node in the underlying data store.
    ///
    pub(crate) fn save_leaf<O>(self: &Arc<Resolver<N>>, node: O) -> Result<Cid>
    where
        O: Node<N>,
    {
        let mut writer = self.mapper.store();
        node.save_to(&mut writer)?;

        Ok(writer.finish())
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

    /// Initialize a new DAG
    ///
    pub fn init(self: &Arc<Resolver<N>>) -> Arc<Folder<N>> {
        let cid = self.mapper.init();

        Folder::new(self, cid, vec![])
    }

    /// Place an object in the DAG filesystem tree and return the new root CID
    ///
    pub fn insert(&self, root: &Cid, path: &str, object: &Cid) -> Cid {
        self.mapper.insert(root, path, object)
    }
}
