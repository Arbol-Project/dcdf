//! A concrete implementation of the `dcdf::Mapper` interface for IPFS.
//!
//! This allows a DCDF DAG to be stored in IPFS.
//!
use std::cmp;
use std::io::{self, Cursor, Read, Write};
use std::str::FromStr;

use bytes::Bytes;
use cid::Cid;
use futures::{Stream, StreamExt};
use ipfs_api_backend_hyper::{request, Error as IpfsError, IpfsApi, IpfsClient};
use tokio::runtime::Runtime;

use dcdf;

type IpfsStream = Box<dyn Stream<Item = Result<Bytes, IpfsError>> + Unpin>;

pub struct IpfsMapper {
    client: IpfsClient,
    runtime: Runtime,
}

impl IpfsMapper {
    pub fn new() -> Self {
        Self {
            client: IpfsClient::default(),
            runtime: Runtime::new().expect("Failed to create tokio runtime"),
        }
    }
}

impl dcdf::Mapper for IpfsMapper {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    /// This impelementation writes all data to memory and then uploads to IPFS when `finish` is
    /// called.
    ///
    fn store(&self) -> Box<dyn dcdf::StoreWrite + '_> {
        Box::new(IpfsStoreWrite::new(self, false))
    }

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    fn hash(&self) -> Box<dyn dcdf::StoreWrite + '_> {
        Box::new(IpfsStoreWrite::new(self, true))
    }

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    fn load(&self, cid: &Cid) -> Option<Box<dyn Read + '_>> {
        let stream = self.client.cat(&cid.to_string());
        let reader = IpfsReader::new(&self.runtime, stream);

        Some(Box::new(reader))
    }
}

/// A writer for writing an object to IPFS
///
/// All writes will, in fact, be written to RAM until `IpfsStoreWrite::finish` is called, at which
/// time, the object will be uploaded to IPFS and a CID will be obtained.
///
struct IpfsStoreWrite<'a> {
    mapper: &'a IpfsMapper,
    buffer: Vec<u8>,
    only_hash: bool,
}

impl<'a> IpfsStoreWrite<'a> {
    fn new(mapper: &'a IpfsMapper, only_hash: bool) -> Self {
        Self {
            mapper,
            buffer: Vec::new(),
            only_hash,
        }
    }
}

impl<'a> Write for IpfsStoreWrite<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buffer.flush()
    }
}

impl<'a> dcdf::StoreWrite for IpfsStoreWrite<'a> {
    fn finish(self: Box<Self>) -> Cid {
        let data = Cursor::new(self.buffer);
        let req = request::Add {
            only_hash: Some(self.only_hash),
            ..Default::default()
        };
        let response = self
            .mapper
            .runtime
            .block_on(self.mapper.client.add_with_options(data, req));
        match response {
            Ok(response) => Cid::from_str(&response.hash).expect("invalid hash"),
            Err(e) => {
                panic!("error adding file: {}", e);
            }
        }
    }
}

/// Reader for reading an object from IPFS
///
struct IpfsReader<'a> {
    runtime: &'a Runtime,
    stream: IpfsStream,
    block: Option<Bytes>,
    index: usize,
}

impl<'a> IpfsReader<'a> {
    fn new(runtime: &'a Runtime, stream: IpfsStream) -> Self {
        let mut reader = Self {
            runtime,
            stream,
            block: None,
            index: 0,
        };

        reader.next_block();

        reader
    }

    fn next_block(&mut self) {
        let block = self.runtime.block_on(self.stream.next());
        let block = match block {
            Some(response) => Some(response.expect("Error reading block")),
            None => None,
        };

        self.block = block;
        self.index = 0;
    }
}

impl<'a> Read for IpfsReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &self.block {
            None => Ok(0),
            Some(block) => {
                let available = block.len() - self.index;
                let read = cmp::min(buf.len(), available);
                let end = self.index + read;
                buf[0..read].copy_from_slice(&block[self.index..end]);
                self.index += read;
                if self.index == block.len() {
                    self.next_block();
                }

                Ok(read)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ndarray::{arr2, Array2};

    use dcdf::Folder;

    fn resolver() -> Arc<dcdf::Resolver<f32>> {
        Arc::new(dcdf::Resolver::new(Box::new(IpfsMapper::new()), 0))
    }

    fn array8() -> Vec<Array2<f32>> {
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

    fn array(sidelen: usize) -> Vec<Array2<f32>> {
        let data = array8();

        data.into_iter()
            .map(|a| Array2::from_shape_fn((sidelen, sidelen), |(row, col)| a[[row % 8, col % 8]]))
            .collect()
    }

    fn superchunk(
        data: &Vec<Array2<f32>>,
        resolver: &Arc<dcdf::Resolver<f32>>,
    ) -> dcdf::Result<dcdf::Superchunk<f32>> {
        let build = dcdf::build_superchunk(
            data.clone().into_iter(),
            Arc::clone(resolver),
            3,
            2,
            dcdf::Precise(3),
            0,
        )?;

        Ok(build.data)
    }

    #[test]
    fn make_a_couple_of_commits() -> dcdf::Result<()> {
        // Store DAG structure
        let resolver = resolver();
        let data1 = array(16);
        let superchunk1 = superchunk(&data1, &resolver)?;

        let a = Folder::new(&resolver);
        let a = a.insert("data", superchunk1)?;

        let c = Folder::new(&resolver);
        let c = c.insert("a", a)?;
        let c_cid = resolver.save(c)?;

        let commit1 = dcdf::Commit::new("First commit", c_cid, None, &resolver);
        let commit1_cid = resolver.save(commit1)?;

        let data2 = array(15);
        let superchunk2 = superchunk(&data2, &resolver)?;

        let b = Folder::new(&resolver);
        let b = b.insert("data", superchunk2)?;

        let c = resolver.get_folder(&c_cid)?;
        let c = c.insert("b", b)?;
        let c_cid = resolver.save(c)?;

        let commit2 = dcdf::Commit::new("Second commit", c_cid, Some(commit1_cid), &resolver);

        let cid = resolver.save(commit2)?;
        println!("HEAD: {:?}", cid);

        // Read DAG structure
        let commit = resolver.get_commit(&cid)?;
        assert_eq!(commit.message(), "Second commit");

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a)?;
        let b = c.get("b").expect("no value for b");
        let b = resolver.get_folder(&b)?;

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        let superchunk = resolver.get_superchunk(&b.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let commit = commit.prev()?.expect("Expected previous commit");
        assert_eq!(commit.message(), "First commit");

        let c = commit.root();
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder(&a)?;

        let superchunk = resolver.get_superchunk(&a.get("data").expect("no value for data"))?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        assert!(c.get("b").is_none());
        assert!(commit.prev()?.is_none());

        Ok(())
    }
}
