use std::{fmt::Debug, pin::Pin, sync::Arc};

use async_recursion::async_recursion;
use async_trait::async_trait;
use cid::Cid;
use futures::{
    stream::{self, Stream, StreamExt},
    AsyncRead, AsyncWrite,
};

use crate::{
    cache::Cacheable,
    codec::chunk::Chunk,
    errors::{Error, Result},
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
};

use super::{
    mmbuffer::{MMBuffer0, MMBuffer1, MMBuffer3},
    node::{Node, NODE_MMSTRUCT3, NODE_SPAN, NODE_SUBCHUNK, NODE_SUPERCHUNK},
    resolver::Resolver,
    span::Span,
    superchunk::Superchunk,
};

pub struct MMStruct3Build {
    pub(crate) data: MMStruct3,
    pub size: u64,

    pub elided: usize,
    pub local: usize,
    pub external: usize,

    pub snapshots: usize,
    pub logs: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MMEncoding {
    I32 = 4,
    I64 = 8,
    F32 = 32,
    F64 = 64,
}

impl TryFrom<u8> for MMEncoding {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            4 => Ok(MMEncoding::I32),
            8 => Ok(MMEncoding::I64),
            32 => Ok(MMEncoding::F32),
            64 => Ok(MMEncoding::F64),
            _ => {
                panic!("Bad Value {value}");
            } //Err(Error::BadValue),
        }
    }
}
pub(crate) enum MMStruct3 {
    Span(Span),
    Subchunk(Chunk),
    Superchunk(Superchunk),
}

impl MMStruct3 {
    /// Get the type of data encoded in this structure.
    ///
    pub fn encoding(&self) -> MMEncoding {
        match self {
            MMStruct3::Span(chunk) => chunk.encoding,
            MMStruct3::Subchunk(chunk) => chunk.encoding,
            MMStruct3::Superchunk(chunk) => chunk.encoding,
        }
    }

    /// Get the number of fractional bits used in encoding this structure.
    /// This is only meaningful for floating point type encodings. Integer encodings will return 0.
    ///
    pub fn fractional_bits(&self) -> usize {
        match self {
            MMStruct3::Span(_) => 0,
            MMStruct3::Subchunk(chunk) => chunk.fractional_bits,
            MMStruct3::Superchunk(chunk) => chunk.fractional_bits,
        }
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        match self {
            MMStruct3::Span(chunk) => chunk.shape(),
            MMStruct3::Subchunk(chunk) => chunk.shape(),
            MMStruct3::Superchunk(chunk) => chunk.shape(),
        }
    }

    /// Get a cell's value at a particular time instant.
    ///
    #[async_recursion]
    pub async fn get(
        &self,
        instant: usize,
        row: usize,
        col: usize,
        buffer: &mut MMBuffer0,
    ) -> Result<()> {
        match self {
            MMStruct3::Span(chunk) => {
                chunk.get(instant, row, col, buffer).await?;
            }
            MMStruct3::Subchunk(chunk) => {
                chunk.get(instant, row, col, buffer);
            }
            MMStruct3::Superchunk(chunk) => {
                chunk.get(instant, row, col, buffer).await?;
            }
        }

        Ok(())
    }

    /// Fill in a preallocated array with a cell's value across time instants.
    ///
    #[async_recursion]
    pub(crate) async fn fill_cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
        buffer: &mut MMBuffer1,
    ) -> Result<()> {
        match self {
            MMStruct3::Span(chunk) => chunk.fill_cell(start, end, row, col, buffer).await?,
            MMStruct3::Subchunk(chunk) => chunk.fill_cell(start, end, row, col, buffer),
            MMStruct3::Superchunk(chunk) => chunk.fill_cell(start, end, row, col, buffer).await?,
        }

        Ok(())
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    #[async_recursion]
    pub(crate) async fn fill_window(
        &self,
        bounds: geom::Cube,
        buffer: &mut MMBuffer3,
    ) -> Result<()> {
        match self {
            MMStruct3::Span(chunk) => chunk.fill_window(bounds, buffer).await?,
            MMStruct3::Subchunk(chunk) => chunk.fill_window(bounds, buffer),
            MMStruct3::Superchunk(chunk) => chunk.fill_window(bounds, buffer).await?,
        }

        Ok(())
    }

    /// Search a subarray for cells that fall in a given mmarray.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        self: &Arc<Self>,
        bounds: geom::Cube,
        lower: i64,
        upper: i64,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        match &**self {
            MMStruct3::Span(chunk) => chunk.search(bounds, lower, upper),
            MMStruct3::Subchunk(chunk) => stream::iter(
                chunk
                    .iter_search(&bounds, lower, upper)
                    .map(|r| Ok(r))
                    .collect::<Vec<Result<(usize, usize, usize)>>>(),
            )
            .boxed(),
            MMStruct3::Superchunk(_) => Superchunk::search(self, bounds, lower, upper),
        }
    }
}

impl Cacheable for MMStruct3 {
    fn size(&self) -> u64 {
        let size = match self {
            MMStruct3::Span(chunk) => chunk.size(),
            MMStruct3::Subchunk(chunk) => chunk.size(),
            MMStruct3::Superchunk(chunk) => chunk.size(),
        };

        size + 1
    }
}

#[async_trait]
impl Node for MMStruct3 {
    const NODE_TYPE: u8 = NODE_MMSTRUCT3;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        match self {
            MMStruct3::Span(chunk) => {
                stream.write_byte(NODE_SPAN).await?;
                chunk.save_to(resolver, stream).await?;
            }
            MMStruct3::Subchunk(chunk) => {
                stream.write_byte(NODE_SUBCHUNK).await?;
                chunk.write_to(stream).await?;
            }
            MMStruct3::Superchunk(chunk) => {
                stream.write_byte(NODE_SUPERCHUNK).await?;
                chunk.save_to(resolver, stream).await?;
            }
        }

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let node_type = stream.read_byte().await?;
        let chunk = match node_type {
            NODE_SUBCHUNK => Self::Subchunk(Chunk::read_from(stream).await?),
            NODE_SPAN => Self::Span(Span::load_from(resolver, stream).await?),
            NODE_SUPERCHUNK => Self::Superchunk(Superchunk::load_from(resolver, stream).await?),
            _ => {
                panic!("Unkonwn MMStruct3 type: {node_type}");
            }
        };

        Ok(chunk)
    }

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)> {
        match self {
            Self::Span(chunk) => chunk.ls(),
            Self::Subchunk(chunk) => chunk.ls(),
            Self::Superchunk(chunk) => chunk.ls(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::resolver::Resolver;
    use crate::{
        dag::{mmbuffer::MMBuffer3, node::Node},
        testing,
    };

    use std::collections::HashSet;

    use futures::StreamExt;
    use ndarray::{s, Array1, Array3};
    use paste::paste;

    macro_rules! mmstruct3_tests {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_get>]() -> Result<()> {
                    let (_resolver, data, mmstruct) = $name().await?;
                    let [instants, rows, cols] = mmstruct.shape();
                    for instant in 0..instants {
                        for row in 0..rows {
                            for col in 0..cols {
                                let mut buffer = MMBuffer0::I64(0);
                                mmstruct.get(instant, row, col, &mut buffer).await?;
                                assert_eq!(i64::from(buffer), data[[instant, row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_fill_cell>]() -> Result<()> {
                    let (_, data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for row in 0..rows {
                        for col in 0..cols {
                            let start = row + col;
                            let end = instants - start;
                            let mut array = Array1::zeros([end - start]);
                            let mut buffer = MMBuffer1::new_i64(array.view_mut());
                            chunk.fill_cell(start, end, row, col, &mut buffer).await?;

                            assert_eq!(array, data.slice(s![start..end, row, col]));
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_fill_window>]() -> Result<()> {
                    let (_, data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;

                            let mut array = Array3::zeros([end - start, bottom - top, right - left]);
                            let mut buffer = MMBuffer3::new_i64(array.view_mut());
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            chunk.fill_window(bounds, &mut buffer).await?;

                            assert_eq!(array, data.slice(s![start..end, top..bottom, left..right]));
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_search>]() -> Result<()>{
                    let (_, data, chunk) = $name().await?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let lower = (start / 5) as i64;
                            let upper = (end / 10) as i64;

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let expected = testing::array_search_window3(
                                data.view(),
                                bounds,
                                lower,
                                upper,
                            ).into_iter().collect::<HashSet<_>>();

                            let results = chunk
                                .search(bounds, lower, upper)
                                .map(|r| r.unwrap())
                                .collect::<HashSet<_>>().await;

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_save_load>]() -> Result<()> {
                    let (resolver, data, chunk) = $name().await?;
                    assert_eq!(chunk.encoding(), MMEncoding::I64);
                    assert_eq!(chunk.fractional_bits(), 0);

                    let cid = resolver.save(chunk).await?;
                    let chunk = resolver.get_mmstruct3(&cid).await?;
                    assert_eq!(chunk.encoding(), MMEncoding::I64);
                    assert_eq!(chunk.fractional_bits(), 0);

                    let [instants, rows, cols] = chunk.shape();
                    for instant in (0..instants).step_by(5) {
                        for row in (0..rows).step_by(2) {
                            for col in (0..cols).step_by(2) {
                                let mut buffer = MMBuffer0::I64(0);
                                chunk.get(instant, row, col, &mut buffer).await?;
                                assert_eq!(i64::from(buffer), data[[instant, row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_ls>]() -> Result<()> {
                    let (_, _, chunk) = $name().await?;
                    let ls = chunk.ls();
                    match chunk {
                        MMStruct3::Subchunk(_) => {
                            assert_eq!(ls.len(), 0);
                        }
                        MMStruct3::Superchunk(_) => {
                            assert_eq!(ls.len(), 1);
                            assert_eq!(ls[0].0, "subchunks");
                        }
                        MMStruct3::Span(_) => {
                            assert_eq!(ls.len(), 5);
                            assert_eq!(ls[0].0, "0");
                            assert_eq!(ls[4].0, "4");
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_high_level_ls>]() -> Result<()> {
                    let (resolver, _, chunk) = $name().await?;
                    match chunk {
                        MMStruct3::Subchunk(_) => {
                            let cid = resolver.save(chunk).await?;
                            let ls = resolver.ls(&cid).await?;
                            assert_eq!(ls.len(), 0);
                        }
                        MMStruct3::Superchunk(_) => {
                            let cid = resolver.save(chunk).await?;
                            let ls = resolver.ls(&cid).await?;
                            assert_eq!(ls.len(), 1);
                            assert_eq!(ls[0].name, "subchunks");
                            assert_eq!(ls[0].node_type.unwrap(), "Links");

                            let ls = resolver.ls(&ls[0].cid).await?;
                            assert_eq!(ls.len(), 3);
                            assert_eq!(ls[0].name, "0");
                            assert_eq!(ls[0].node_type.unwrap(), "Superchunk");

                        }
                        MMStruct3::Span(_) => {
                            let cid = resolver.save(chunk).await?;
                            let ls = resolver.ls(&cid).await?;
                            assert_eq!(ls.len(), 5);
                            assert_eq!(ls[0].name, "0");
                            assert_eq!(ls[0].node_type.unwrap(), "Superchunk");
                        }
                    }


                    Ok(())
                }
            }
        };
    }

    type DataStruct3 = Result<(Arc<Resolver>, Array3<i64>, MMStruct3)>;

    async fn chunk() -> DataStruct3 {
        let resolver = testing::resolver();
        let mut data = testing::array(16);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;

        Ok((resolver, data, chunk))
    }

    mmstruct3_tests!(chunk);

    async fn superchunk() -> DataStruct3 {
        let resolver = testing::resolver();
        let mut data = testing::array(17);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build = Superchunk::build(
            Arc::clone(&resolver),
            &mut buffer,
            [100, 17, 17],
            &[1, 2, 2],
            2,
        )
        .await?;

        Ok((resolver, data, build.data))
    }

    mmstruct3_tests!(superchunk);

    async fn span() -> DataStruct3 {
        let resolver = testing::resolver();
        let mut data = testing::array(17);
        let mut span = Span::new([17, 17], 20, Arc::clone(&resolver), MMEncoding::I64);
        for i in (0_usize..100_usize).step_by(20) {
            let mut buffer = MMBuffer3::new_i64(data.slice_mut(s![i..i + 20, .., ..]));
            let build = Superchunk::build(
                Arc::clone(&resolver),
                &mut buffer,
                [20, 17, 17],
                &[1, 2, 2],
                2,
            )
            .await?;

            span = span.append(build.data).await?;
        }

        Ok((resolver, data, MMStruct3::Span(span)))
    }

    mmstruct3_tests!(span);
}
