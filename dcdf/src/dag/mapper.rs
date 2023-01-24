use std::io::{self, Read, Write};

use async_trait::async_trait;
use cid::Cid;
use futures::{AsyncRead, AsyncWrite};

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

    /// Get the size, in bytes, of object identified by `cid`
    ///
    fn size_of(&self, cid: &Cid) -> io::Result<Option<u64>>;
}

/// A trait for storing and loading data from an arbitrary IPLD store.
///
#[async_trait]
pub trait AsyncMapper: Send + Sync {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    async fn store_async(&self) -> Box<dyn StoreAsyncWrite + '_>;

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    async fn hash_async(&self) -> Box<dyn StoreAsyncWrite + '_>;

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    async fn load_async(&self, cid: &Cid) -> Option<Box<dyn AsyncRead + '_>>;

    /// Get the size, in bytes, of object identified by `cid`
    ///
    async fn size_of_async(&self, cid: &Cid) -> io::Result<Option<u64>>;
}

/// An extension to Write that computes a hash for the written data.
///
pub trait StoreWrite: Write {
    /// Close the output stream and return the `cid` for the newly written object.
    ///
    fn finish(self: Box<Self>) -> Cid;
}

#[async_trait]
pub trait StoreAsyncWrite: AsyncWrite {
    /// Close the output stream and return the `cid` for the newly written object.
    ///
    async fn finish_async(self: Box<Self>) -> Cid;
}
