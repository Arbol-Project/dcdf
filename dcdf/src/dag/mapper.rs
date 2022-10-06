use cid::Cid;
use std::io::{Read, Write};

/// A trait for storing and loading data from an arbitrary IPLD store.
///
pub trait Mapper: Send + Sync {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    fn store(&self) -> Box<dyn StoreWrite + '_>;

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    fn hash(&self) -> Box<dyn StoreWrite + '_>;

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    fn load(&self, cid: &Cid) -> Option<Box<dyn Read + '_>>;

    /// Return a CID for an empty filesystem folder
    ///
    fn init(&self) -> Cid;

    /// Place an object in the DAG filesystem tree and return the new root CID
    ///
    fn insert(&self, root: &Cid, path: &str, object: &Cid) -> Cid;

    /// Get a listing of the contents of a folder in the DAG
    ///
    fn ls(&self, cid: &Cid) -> Vec<Link>;
}

pub struct Link {
    pub name: String,
    pub cid: Cid,
    pub size: u64,
}

/// An extension to Write that computes a hash for the written data.
///
pub trait StoreWrite: Write {
    /// Close the output stream and return the `cid` for the newly written object.
    ///
    fn finish(self: Box<Self>) -> Cid;
}
