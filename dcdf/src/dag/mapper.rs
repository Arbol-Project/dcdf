use cid::Cid;
use multihash::{Hasher, MultihashGeneric, Sha2_256};
use std::io;
use std::io::{Read, Write};

/// The SHA_256 multicodec code
const SHA2_256: u64 = 0x12;

pub trait Mapper {
    fn store(&mut self) -> Box<dyn StoreWrite + '_>;

    fn load(&mut self, cid: Cid) -> Option<Box<dyn Read + '_>>;
}

pub trait StoreWrite: Write {
    fn finish(&mut self) -> Cid;
}

pub struct Sha2_256Write<W: Write> {
    pub inner: W,
    hash: Sha2_256,
}

impl<W> Sha2_256Write<W>
where
    W: Write,
{
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
