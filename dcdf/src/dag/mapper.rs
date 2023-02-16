use std::io;

use async_trait::async_trait;
use cid::Cid;
use futures::{AsyncRead, AsyncWrite};

/// A trait for storing and loading data from an arbitrary IPLD store.
///
#[async_trait]
pub trait Mapper: Send + Sync {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    async fn store(&self) -> Box<dyn StoreWrite + '_>;

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    async fn hash(&self) -> Box<dyn StoreWrite + '_>;

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    async fn load(&self, cid: &Cid) -> Option<Box<dyn AsyncRead + Unpin + Send + '_>>;

    /// Get the size, in bytes, of object identified by `cid`
    ///
    async fn size_of(&self, cid: &Cid) -> io::Result<Option<u64>>;
}

#[async_trait]
pub trait StoreWrite: AsyncWrite + Unpin + Send {
    /// Close the output stream and return the `cid` for the newly written object.
    ///
    async fn finish(self: Box<Self>) -> Cid;
}
