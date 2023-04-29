//! A concrete implementation of the `dcdf::Mapper` interface for IPFS.
//!
//! This allows a DCDF DAG to be stored in IPFS.
//!
use std::{
    io::{self, Write},
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

use dcdf;

type AioResult<T> = result::Result<T, AioError>;

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
}

impl IpfsMapper {
    pub fn new() -> Self {
        Self {
            client: IpfsClient::new(),
        }
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

#[async_trait]
impl dcdf::Mapper for IpfsMapper {
    /// Obtain an output stream for writing an object to the store.
    ///
    /// The CID for the object can be obtained from the `finish` method of the returned
    /// `StoreWrite` object.
    ///
    async fn store(&self) -> Box<dyn dcdf::StoreWrite + '_> {
        Box::new(IpfsStoreWrite::new(self, false))
    }

    /// Same as `store` but doesn't actually store the object, just computes its hash.
    ///
    async fn hash(&self) -> Box<dyn dcdf::StoreWrite + '_> {
        Box::new(IpfsStoreWrite::new(self, true))
    }

    /// Obtain an input stream for reading an object from the store.
    ///
    /// Should return `Option::None` if given `cid` isn't in the store.
    ///
    async fn load(&self, cid: &Cid) -> Option<Box<dyn AsyncRead + Unpin + Send + '_>> {
        let stream = self
            .client
            .cat(cid)
            .await
            .expect("This should return a result, probably.");

        Some(stream)
    }

    /// Get the size, in bytes, of object identified by `cid`
    ///
    async fn size_of(&self, cid: &Cid) -> AioResult<Option<u64>> {
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
impl<'a> dcdf::StoreWrite for IpfsStoreWrite<'a> {
    async fn finish(self: Box<Self>) -> Cid {
        let cid = self.mapper.client.add(self.buffer, self.only_hash).await;
        cid.expect("error adding file")
    }
}

#[cfg(test)]
mod ipfs_tests {
    use super::*;

    use std::sync::Arc;

    use dcdf::{
        from_fixed, geom, to_fixed, Coordinate, Dataset, MMArray3F32, MMArray3F64, MMArray3I32,
        MMArray3I64, MMEncoding, Resolver, Result,
    };
    use ndarray::{arr2, s, Array3};
    use num_traits::{cast, Float, PrimInt};

    fn resolver() -> Arc<dcdf::Resolver> {
        Arc::new(dcdf::Resolver::new(Box::new(IpfsMapper::new()), 0))
    }

    pub(crate) fn array8() -> Array3<i64> {
        let data = vec![
            arr2(&[
                [9, 8, 7, 7, 6, 6, 3, 2],
                [7, 7, 7, 7, 6, 6, 3, 3],
                [6, 6, 6, 6, 3, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 3, 5, 4, 4, 4, 4],
                [4, 4, 3, 4, 4, 4, 4, 4],
            ]),
            arr2(&[
                [9, 8, 7, 7, 7, 7, 2, 2],
                [7, 7, 7, 7, 7, 7, 2, 2],
                [6, 6, 6, 6, 4, 3, 3, 3],
                [5, 5, 6, 6, 3, 3, 3, 3],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 4, 5, 5, 4, 4, 4],
                [4, 4, 4, 4, 4, 4, 4, 4],
            ]),
            arr2(&[
                [9, 8, 7, 7, 8, 7, 5, 5],
                [7, 7, 7, 7, 7, 7, 5, 5],
                [7, 7, 6, 6, 4, 3, 4, 4],
                [6, 6, 6, 6, 4, 4, 4, 4],
                [4, 5, 5, 5, 4, 4, 4, 4],
                [3, 3, 5, 5, 4, 4, 4, 4],
                [3, 3, 4, 5, 6, 4, 4, 4],
                [4, 4, 4, 4, 5, 4, 4, 4],
            ]),
        ];

        let mut array = Array3::zeros([100, 8, 8]);
        for (i, a) in data.into_iter().cycle().take(100).enumerate() {
            array.slice_mut(s![i, .., ..]).assign(&a);
        }

        array
    }

    pub(crate) fn farray8() -> Array3<f32> {
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

        let mut array = Array3::zeros([100, 8, 8]);
        for (i, a) in data.into_iter().cycle().take(100).enumerate() {
            array.slice_mut(s![i, .., ..]).assign(&a);
        }

        array
    }

    fn make_int_data<N: PrimInt>(instants: usize) -> Array3<N> {
        let data = array8();
        let data = Array3::from_shape_fn([instants, 16, 16], |(t, y, x)| {
            cast(data[[t % 3, y % 8, x % 8]]).unwrap()
        });

        data
    }

    fn make_float_data<N: Float>(instants: usize) -> Array3<N> {
        let data = farray8();
        let data = Array3::from_shape_fn([instants, 16, 16], |(t, y, x)| {
            cast(data[[t % 3, y % 8, x % 8]]).unwrap()
        });

        data
    }

    fn make_one(resolver: Arc<Resolver>) -> Dataset {
        let t = Coordinate::time("t", 0, 100);
        let y = Coordinate::range_f64("y", -160.0, 20.0, 16);
        let x = Coordinate::range_f64("x", -200.0, 25.0, 16);
        Dataset::new([t, y, x], [16, 16], resolver)
    }

    async fn populate(
        resolver: &Arc<Resolver>,
        dataset: Dataset,
    ) -> Result<(
        Array3<f32>,
        Array3<f64>,
        Array3<i32>,
        Array3<i64>,
        Array3<f32>,
        Array3<f64>,
        Dataset,
    )> {
        assert!(dataset.cid.is_none());

        let dataset = dataset
            .add_variable("apples", None, 10, 20, vec![2, 2], MMEncoding::F32)
            .await?;
        let mut apple_data = make_float_data::<f32>(360);
        let dataset = dataset
            .append_f32("apples", apple_data.slice_mut(s![..99_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_f32("apples", apple_data.slice_mut(s![99..200_usize, .., ..]))
            .await?;
        let dataset = dataset
            .append_f32("apples", apple_data.slice_mut(s![200_usize.., .., ..]))
            .await?;

        let dataset = dataset
            .add_variable("pears", None, 10, 20, vec![2, 2], MMEncoding::F64)
            .await?;
        let mut pear_data = make_float_data::<f64>(500);
        let dataset = dataset.append_f64("pears", pear_data.view_mut()).await?;

        let dataset = dataset
            .add_variable("bananas", None, 10, 20, vec![2, 2], MMEncoding::I32)
            .await?;
        let mut banana_data = make_int_data::<i32>(511);
        let dataset = dataset
            .append_i32("bananas", banana_data.view_mut())
            .await?;

        let dataset = dataset
            .add_variable("grapes", None, 10, 20, vec![2, 2], MMEncoding::I64)
            .await?;
        let mut grape_data = make_int_data::<i64>(365);
        let dataset = dataset.append_i64("grapes", grape_data.view_mut()).await?;

        assert!(dataset.prev.is_none());
        let cid = dataset.commit().await?;
        let dataset = resolver.get_dataset(&cid).await?;
        assert_eq!(dataset.cid, Some(cid));

        let dataset = dataset
            .add_variable("dates", Some(2), 10, 20, vec![2, 2], MMEncoding::F32)
            .await?;

        let mut date_data = make_float_data::<f32>(489);
        let dataset = dataset.append_f32("dates", date_data.view_mut()).await?;
        date_data.mapv_inplace(|v| from_fixed(to_fixed(v, 2, true), 2));

        assert!(dataset.cid.is_none());
        assert_eq!(dataset.prev, Some(cid));

        let dataset = dataset
            .add_variable("melons", Some(2), 10, 20, vec![2, 2], MMEncoding::F64)
            .await?;
        let mut melon_data = make_float_data::<f64>(275);
        let dataset = dataset.append_f64("melons", melon_data.view_mut()).await?;
        melon_data.mapv_inplace(|v| from_fixed(to_fixed(v, 2, true), 2));

        assert!(dataset.cid.is_none());
        assert_eq!(dataset.prev, Some(cid));

        Ok((
            apple_data,
            pear_data,
            banana_data,
            grape_data,
            date_data,
            melon_data,
            dataset,
        ))
    }

    async fn verify_i32(array: Array3<i32>, mmarray: MMArray3I32) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    async fn verify_i64(array: Array3<i64>, mmarray: MMArray3I64) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    async fn verify_f32(array: Array3<f32>, mmarray: MMArray3F32) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    async fn verify_f64(array: Array3<f64>, mmarray: MMArray3F64) -> Result<()> {
        let [instants, rows, cols] = mmarray.shape();
        let extracted = mmarray
            .window(geom::Cube::new(0, instants, 0, rows, 0, cols))
            .await?;
        assert_eq!(array, extracted);

        Ok(())
    }

    #[tokio::test]
    async fn test_populate_variables_and_save_load() -> Result<()> {
        let resolver = resolver();
        let dataset = make_one(Arc::clone(&resolver));
        let (apple_data, pear_data, banana_data, grape_data, date_data, melon_data, dataset) =
            populate(&resolver, dataset).await?;
        assert_eq!(dataset.variables.len(), 6);

        let cid = dataset.commit().await?;
        let dataset = resolver.get_dataset(&cid).await?;

        let apples = dataset.get_variable("apples").unwrap();
        assert_eq!(apples.name, "apples");
        assert!(apples.round.is_none());
        assert_eq!(apples.span_size, 10);
        assert_eq!(apples.chunk_size, 20);
        assert_eq!(apples.k2_levels, &[2, 2]);
        assert_eq!(apples.encoding, MMEncoding::F32);

        let apples_mmstruct = apples.data_f32().await?;
        verify_f32(apple_data, apples_mmstruct).await?;

        let pears = dataset.get_variable("pears").unwrap();
        assert_eq!(pears.name, "pears");
        assert!(pears.round.is_none());
        assert_eq!(pears.span_size, 10);
        assert_eq!(pears.chunk_size, 20);
        assert_eq!(pears.k2_levels, &[2, 2]);
        assert_eq!(pears.encoding, MMEncoding::F64);

        let pears_mmstruct = pears.data_f64().await?;
        verify_f64(pear_data, pears_mmstruct).await?;

        let bananas = dataset.get_variable("bananas").unwrap();
        assert_eq!(bananas.name, "bananas");
        assert!(bananas.round.is_none());
        assert_eq!(bananas.span_size, 10);
        assert_eq!(bananas.chunk_size, 20);
        assert_eq!(bananas.k2_levels, &[2, 2]);
        assert_eq!(bananas.encoding, MMEncoding::I32);

        let bananas_mmstruct = bananas.data_i32().await?;
        verify_i32(banana_data, bananas_mmstruct).await?;

        let grapes = dataset.get_variable("grapes").unwrap();
        assert_eq!(grapes.name, "grapes");
        assert!(grapes.round.is_none());
        assert_eq!(grapes.span_size, 10);
        assert_eq!(grapes.chunk_size, 20);
        assert_eq!(grapes.k2_levels, &[2, 2]);
        assert_eq!(grapes.encoding, MMEncoding::I64);

        let grapes_mmstruct = grapes.data_i64().await?;
        verify_i64(grape_data, grapes_mmstruct).await?;

        let dates = dataset.get_variable("dates").unwrap();
        assert_eq!(dates.name, "dates");
        assert_eq!(dates.round, Some(2));
        assert_eq!(dates.span_size, 10);
        assert_eq!(dates.chunk_size, 20);
        assert_eq!(dates.k2_levels, &[2, 2]);
        assert_eq!(dates.encoding, MMEncoding::F32);

        let dates_mmstruct = dates.data_f32().await?;
        verify_f32(date_data, dates_mmstruct).await?;

        let melons = dataset.get_variable("melons").unwrap();
        assert_eq!(melons.name, "melons");
        assert_eq!(melons.round, Some(2));
        assert_eq!(melons.span_size, 10);
        assert_eq!(melons.chunk_size, 20);
        assert_eq!(melons.k2_levels, &[2, 2]);
        assert_eq!(melons.encoding, MMEncoding::F64);

        let melons_mmstruct = melons.data_f64().await?;
        verify_f64(melon_data, melons_mmstruct).await?;

        Ok(())
    }
}
