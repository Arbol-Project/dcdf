use cid::Cid;
use multihash::{Hasher, MultihashGeneric, Sha2_256};
use std::io;
use std::io::{Read, Write};

/// The SHA_256 multicodec code
const SHA2_256: u64 = 0x12;

/// A trait for storing and loading data from an arbitrary IPLD store.
///
pub trait Mapper {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    fn store(&mut self) -> Box<dyn StoreWrite + '_>;

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    fn hash(&mut self) -> Box<dyn StoreWrite + '_>;

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    fn load(&mut self, cid: Cid) -> Option<Box<dyn Read + '_>>;
}

/// An extension to Write that computes a hash for the written data.
///
pub trait StoreWrite: Write {
    /// Close the output stream and return the `cid` for the newly written object.
    ///
    fn finish(&mut self) -> Cid;
}

/// An implmentor of `StoreWrite` that computes CIDs using Sha2 256.
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
    fn finish(&mut self) -> Cid {
        let digest = self.hash.finalize();
        let hash = MultihashGeneric::wrap(SHA2_256, &digest).expect("Not really sure.");

        Cid::new_v1(SHA2_256, hash)
    }
}
