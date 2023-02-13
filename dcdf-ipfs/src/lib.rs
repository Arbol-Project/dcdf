//! A concrete implementation of the `dcdf::Mapper` interface for IPFS.
//!
//! This allows a DCDF DAG to be stored in IPFS.
//!
use std::{
    cmp,
    io::{self, Read, Write},
    pin::Pin,
    result,
    str::FromStr,
};

use async_trait::async_trait;
use bytes::Bytes;
use cid::Cid;
use futures::{
    io::{AsyncRead, AsyncWrite, Error as AioError},
    task::{Context, Poll},
    Stream, StreamExt, TryStreamExt,
};
use reqwest::{
    multipart::{Form, Part},
    Client, Error as ReqwestError,
};
use serde::Deserialize;
use tokio::runtime::Runtime;

use dcdf;

type AioResult<T> = result::Result<T, AioError>;
type IpfsStream = Box<dyn Stream<Item = AioResult<Bytes>> + Unpin>;

struct IpfsClient {
    api_uri: String,
    client: Client,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct AddResponse {
    hash: String,
    //name: String,
    //size: String,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "PascalCase")]
struct StatResponse {
    size: u64,
    //hash: String,
    //cumulative_size: u64,
    //blocks: u64,

    //#[serde(rename = "Type")]
    //typ: String,

    //#[serde(default)]
    //size_local: Option<u64>,

    //#[serde(default)]
    //local: Option<bool>,
}

fn reqwest_error(error: ReqwestError) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("{error}"))
}

impl IpfsClient {
    fn new() -> Self {
        Self {
            api_uri: String::from("http://localhost:5001/api/v0/"),
            client: Client::new(),
        }
    }

    async fn add(&self, data: Vec<u8>, only_hash: bool) -> AioResult<Cid> {
        let only_hash = if only_hash { "true" } else { "false" };
        let uri = format!(
            "{}{}?chunker=size-1048576&only-hash={}",
            self.api_uri, "add", only_hash
        );
        let form = Form::new().part("path", Part::bytes(data));
        let response = self
            .client
            .post(uri)
            .multipart(form)
            .send()
            .await
            .map_err(reqwest_error)?;
        let response = response
            .json::<AddResponse>()
            .await
            .map_err(reqwest_error)?;

        Ok(Cid::from_str(&response.hash).expect("invalid hash"))
    }

    async fn cat(&self, cid: &Cid) -> AioResult<Box<dyn AsyncRead + Unpin + Send>> {
        Ok(Box::new(self._cat(cid).await?.into_async_read()))
    }

    async fn cat_stream(&self, cid: &Cid) -> AioResult<IpfsStream> {
        Ok(Box::new(self._cat(cid).await?))
    }

    async fn _cat(&self, cid: &Cid) -> AioResult<impl Stream<Item = AioResult<Bytes>> + Unpin> {
        let uri = format!("{}{}?arg={}", self.api_uri, "cat", cid.to_string());
        let response = self.client.post(uri).send().await.map_err(reqwest_error)?;

        Ok(response
            .bytes_stream()
            .map(|result| result.map_err(reqwest_error)))
    }

    async fn stat(&self, cid: &Cid) -> AioResult<StatResponse> {
        let uri = format!(
            "{}{}?arg=/ipfs/{}",
            self.api_uri,
            "files/stat",
            cid.to_string()
        );
        let response = self.client.post(uri).send().await.map_err(reqwest_error)?;
        let response = response
            .json::<StatResponse>()
            .await
            .map_err(reqwest_error)?;

        Ok(response)
    }
}

pub struct IpfsMapper {
    client: IpfsClient,
    runtime: Option<Runtime>,
}

impl IpfsMapper {
    pub fn new(blocking: bool) -> Self {
        Self {
            client: IpfsClient::new(),
            runtime: if blocking {
                Some(Runtime::new().expect("Failed to create tokio runtime"))
            } else {
                None
            },
        }
    }
}

impl dcdf::Mapper for IpfsMapper {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    /// This implementation writes all data to memory and then uploads to IPFS when `finish` is
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
        let runtime = self.runtime.as_ref().unwrap();
        let stream = runtime.block_on(self.client.cat_stream(&cid)).ok()?;
        let reader = IpfsReader::new(runtime, stream);

        Some(Box::new(reader))
    }

    /// Get the size, in bytes, of object identified by `cid`
    ///
    fn size_of(&self, cid: &Cid) -> io::Result<Option<u64>> {
        let response = self
            .runtime
            .as_ref()
            .unwrap()
            .block_on(self.client.stat(&cid))
            .expect("Unable to stat file");

        Ok(Some(response.size))
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
        let response = self
            .mapper
            .runtime
            .as_ref()
            .unwrap()
            .block_on(self.mapper.client.add(self.buffer, self.only_hash));

        response.expect("error adding file")
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

#[async_trait]
impl dcdf::AsyncMapper for IpfsMapper {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    async fn store_async(&self) -> Box<dyn dcdf::StoreAsyncWrite + '_> {
        Box::new(IpfsStoreWrite::new(self, false))
    }

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    async fn hash_async(&self) -> Box<dyn dcdf::StoreAsyncWrite + '_> {
        Box::new(IpfsStoreWrite::new(self, true))
    }

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    async fn load_async(&self, cid: &Cid) -> Option<Box<dyn AsyncRead + Unpin + Send + '_>> {
        let stream = self
            .client
            .cat(cid)
            .await
            .expect("This should return a result, probably.");

        Some(stream)
    }

    /// Get the size, in bytes, of object identified by `cid`
    ///
    async fn size_of_async(&self, cid: &Cid) -> AioResult<Option<u64>> {
        let response = self.client.stat(cid).await.expect("Unable to stat file");

        Ok(Some(response.size))
    }
}

impl<'a> AsyncWrite for IpfsStoreWrite<'a> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<AioResult<usize>> {
        Poll::Ready(self.buffer.write(buf))
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<AioResult<()>> {
        Poll::Ready(self.buffer.flush())
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<AioResult<()>> {
        Poll::Ready(Ok(()))
    }
}

#[async_trait]
impl<'a> dcdf::StoreAsyncWrite for IpfsStoreWrite<'a> {
    async fn finish_async(self: Box<Self>) -> Cid {
        let cid = self.mapper.client.add(self.buffer, self.only_hash).await;
        cid.expect("error adding file")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ndarray::{arr2, Array2};

    use dcdf::Cacheable;
    use dcdf::Folder;

    fn resolver(blocking: bool) -> Arc<dcdf::Resolver<f32>> {
        Arc::new(dcdf::Resolver::new(Box::new(IpfsMapper::new(blocking)), 0))
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

    async fn superchunk_async(
        data: &Vec<Array2<f32>>,
        resolver: &Arc<dcdf::Resolver<f32>>,
    ) -> dcdf::Result<dcdf::Superchunk<f32>> {
        let build = dcdf::build_superchunk_async(
            data.clone().into_iter(),
            Arc::clone(resolver),
            3,
            2,
            dcdf::Precise(3),
            0,
        )
        .await?;

        Ok(build.data)
    }

    #[test]
    fn make_a_couple_of_commits() -> dcdf::Result<()> {
        // Store DAG structure
        let resolver = resolver(true);
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

        let ls = resolver.ls(&cid)?.expect("Couldn't find commit");
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0].name, String::from("root"));
        assert_eq!(ls[0].cid, commit.root);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), commit.root().size());
        assert_eq!(ls[1].name, String::from("prev"));
        assert_eq!(ls[1].cid, commit.prev.unwrap());
        assert_eq!(ls[1].node_type.unwrap(), "Commit");
        assert_eq!(ls[1].size.unwrap(), commit.prev().unwrap().unwrap().size());

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

    #[tokio::test]
    async fn make_a_couple_of_commits_async() -> dcdf::Result<()> {
        // Store DAG structure
        let resolver = resolver(false);
        let data1 = array(16);
        let superchunk1 = superchunk_async(&data1, &resolver).await?;

        let a = Folder::new(&resolver);
        let a = a.insert_async("data", superchunk1).await?;

        let c = Folder::new(&resolver);
        let c = c.insert_async("a", a).await?;
        let c_cid = resolver.save_async(c).await?;

        let commit1 = dcdf::Commit::new("First commit", c_cid, None, &resolver);
        let commit1_cid = resolver.save_async(commit1).await?;

        let data2 = array(15);
        let superchunk2 = superchunk_async(&data2, &resolver).await?;

        let b = Folder::new(&resolver);
        let b = b.insert_async("data", superchunk2).await?;

        let c = resolver.get_folder_async(&c_cid).await?;
        let c = c.insert_async("b", b).await?;
        let c_cid = resolver.save_async(c).await?;

        let commit2 = dcdf::Commit::new("Second commit", c_cid, Some(commit1_cid), &resolver);

        let cid = resolver.save_async(commit2).await?;
        println!("HEAD: {:?}", cid);

        // Read DAG structure
        let commit = resolver.get_commit_async(&cid).await?;
        assert_eq!(commit.message(), "Second commit");

        let ls = resolver
            .ls_async(&cid)
            .await?
            .expect("Couldn't find commit");
        assert_eq!(ls.len(), 2);
        assert_eq!(ls[0].name, String::from("root"));
        assert_eq!(ls[0].cid, commit.root);
        assert_eq!(ls[0].node_type.unwrap(), "Folder");
        assert_eq!(ls[0].size.unwrap(), commit.root_async().await.size());
        assert_eq!(ls[1].name, String::from("prev"));
        assert_eq!(ls[1].cid, commit.prev.unwrap());
        assert_eq!(ls[1].node_type.unwrap(), "Commit");
        assert_eq!(
            ls[1].size.unwrap(),
            commit.prev_async().await.unwrap().unwrap().size()
        );

        let c = commit.root_async().await;
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder_async(&a).await?;
        let b = c.get("b").expect("no value for b");
        let b = resolver.get_folder_async(&b).await?;

        let superchunk = resolver
            .get_superchunk_async(&a.get("data").expect("no value for data"))
            .await?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        let superchunk = resolver
            .get_superchunk_async(&b.get("data").expect("no value for data"))
            .await?;
        assert_eq!(superchunk.shape(), [100, 15, 15]);

        let commit = commit
            .prev_async()
            .await?
            .expect("Expected previous commit");
        assert_eq!(commit.message(), "First commit");

        let c = commit.root_async().await;
        let a = c.get("a").expect("no value for a");
        let a = resolver.get_folder_async(&a).await?;

        let superchunk = resolver
            .get_superchunk_async(&a.get("data").expect("no value for data"))
            .await?;
        assert_eq!(superchunk.shape(), [100, 16, 16]);

        assert!(c.get("b").is_none());
        assert!(commit.prev_async().await?.is_none());

        Ok(())
    }
}
