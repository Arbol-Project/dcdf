use std::{fmt::Debug, pin::Pin, sync::Arc};

use async_recursion::async_recursion;
use async_trait::async_trait;
use cid::Cid;
use futures::{
    stream::{self, Stream, StreamExt},
    AsyncRead, AsyncWrite,
};
use ndarray::{Array1, ArrayBase, DataMut, Ix1, Ix3};
use num_traits::Float;

use crate::{
    cache::Cacheable,
    codec::FChunk,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite, Serialize},
    geom,
};

use super::{
    node::{
        Node, NODE_MMARRAY1, NODE_MMARRAY3, NODE_RANGE, NODE_SPAN, NODE_SUBCHUNK, NODE_SUPERCHUNK,
    },
    range::Range,
    resolver::Resolver,
    span::Span,
    superchunk::Superchunk,
};

pub enum MMArray3<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    Span(Span<N>),
    Subchunk(FChunk<N>),
    Superchunk(Superchunk<N>),
}

impl<N> MMArray3<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        match self {
            MMArray3::Span(span) => span.shape(),
            MMArray3::Subchunk(chunk) => chunk.shape(),
            MMArray3::Superchunk(chunk) => chunk.shape(),
        }
    }

    /// Get a cell's value at a particular time instant.
    ///
    #[async_recursion]
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<N> {
        match self {
            MMArray3::Span(chunk) => chunk.get(instant, row, col).await,
            MMArray3::Subchunk(chunk) => Ok(chunk.get(instant, row, col)),
            MMArray3::Superchunk(chunk) => chunk.get(instant, row, col).await,
        }
    }

    /// Fill in a preallocated array with a cell's value across time instants.
    ///
    #[async_recursion]
    pub async fn fill_cell<S>(
        &self,
        start: usize,
        row: usize,
        col: usize,
        values: &mut ArrayBase<S, Ix1>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N> + Send,
    {
        let mut values = unsafe { values.raw_view_mut().deref_into_view_mut() };
        match self {
            MMArray3::Span(chunk) => chunk.fill_cell(start, row, col, &mut values).await,
            MMArray3::Subchunk(chunk) => Ok(chunk.fill_cell(start, row, col, &mut values)),
            MMArray3::Superchunk(chunk) => chunk.fill_cell(start, row, col, &mut values).await,
        }
    }

    /// Fill in a preallocated array with subarray from this chunk
    ///
    #[async_recursion]
    pub async fn fill_window<S>(
        &self,
        start: usize,
        top: usize,
        left: usize,
        window: &mut ArrayBase<S, Ix3>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N> + Send,
    {
        let mut window = unsafe { window.raw_view_mut().deref_into_view_mut() };
        match self {
            MMArray3::Span(chunk) => chunk.fill_window(start, top, left, &mut window).await,
            MMArray3::Subchunk(chunk) => Ok(chunk.fill_window(start, top, left, &mut window)),
            MMArray3::Superchunk(chunk) => chunk.fill_window(start, top, left, &mut window).await,
        }
    }

    /// Search a subarray for cells that fall in a given mmarray.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        self: &Arc<Self>,
        bounds: geom::Cube,
        lower: N,
        upper: N,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        match &**self {
            MMArray3::Span(chunk) => chunk.search(bounds, lower, upper),
            MMArray3::Subchunk(chunk) => stream::iter(
                chunk
                    .iter_search(&bounds, lower, upper)
                    .map(|r| Ok(r))
                    .collect::<Vec<Result<(usize, usize, usize)>>>(),
            )
            .boxed(),
            MMArray3::Superchunk(_) => Superchunk::search(self, bounds, lower, upper),
        }
    }
}

impl<N> Cacheable for MMArray3<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        let size = match self {
            MMArray3::Span(chunk) => chunk.size(),
            MMArray3::Subchunk(chunk) => chunk.size(),
            MMArray3::Superchunk(chunk) => chunk.size(),
        };

        size + 1
    }
}

#[async_trait]
impl<N> Node<N> for MMArray3<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_MMARRAY3;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        match self {
            MMArray3::Span(chunk) => {
                stream.write_byte(NODE_SPAN).await?;
                chunk.save_to(resolver, stream).await?;
            }
            MMArray3::Subchunk(chunk) => {
                stream.write_byte(NODE_SUBCHUNK).await?;
                chunk.write_to(stream).await?;
            }
            MMArray3::Superchunk(chunk) => {
                stream.write_byte(NODE_SUPERCHUNK).await?;
                chunk.save_to(resolver, stream).await?;
            }
        }

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let node_type = stream.read_byte().await?;
        let chunk = match node_type {
            NODE_SPAN => Self::Span(Span::load_from(resolver, stream).await?),
            NODE_SUBCHUNK => Self::Subchunk(FChunk::read_from(stream).await?),
            NODE_SUPERCHUNK => Self::Superchunk(Superchunk::load_from(resolver, stream).await?),
            _ => {
                panic!("Unkonwn MMArray3 type: {node_type}");
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

pub enum MMArray1<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    Range(Range<N>),
}

impl<N> MMArray1<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub fn get(&self, index: usize) -> N {
        match self {
            MMArray1::Range(mmarray) => mmarray.get(index),
        }
    }
    pub fn slice(&self, start: usize, stop: usize) -> Array1<N> {
        match self {
            MMArray1::Range(mmarray) => mmarray.slice(start, stop),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MMArray1::Range(mmarray) => mmarray.len(),
        }
    }

    pub fn shape(&self) -> [usize; 1] {
        match self {
            MMArray1::Range(mmarray) => mmarray.shape(),
        }
    }
}

impl<N> Cacheable for MMArray1<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        let size = match self {
            Self::Range(range) => range.size(),
        };

        size + 1
    }
}

#[async_trait]
impl<N> Node<N> for MMArray1<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_MMARRAY1;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        match self {
            Self::Range(range) => {
                stream.write_byte(NODE_RANGE).await?;
                range.save_to(resolver, stream).await?;
            }
        }

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let node_type = stream.read_byte().await?;
        let chunk = match node_type {
            NODE_RANGE => Self::Range(Range::load_from(resolver, stream).await?),
            _ => {
                panic!("Unkonwn MMArray1 type: {node_type}");
            }
        };

        Ok(chunk)
    }

    /// List other nodes contained by this node
    fn ls(&self) -> Vec<(String, Cid)> {
        match self {
            Self::Range(range) => range.ls(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::resolver::Resolver;
    use crate::{build::build_superchunk, fixed::Fraction::Precise, testing};

    use std::collections::HashSet;

    use futures::StreamExt;
    use ndarray::{array, s, Array2};
    use paste::paste;

    macro_rules! mmarray3_tests {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_get>]() -> Result<()> {
                    let (_, data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for instant in 0..instants {
                        for row in 0..rows {
                            for col in 0..cols {
                                let value = chunk.get(instant, row, col).await?;
                                assert_eq!(value, data[instant][[row, col]]);
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
                            let values = chunk.get_cell(start, end, row, col).await?;
                            assert_eq!(values.len(), end - start);
                            for i in 0..values.len() {
                                assert_eq!(values[i], data[i + start][[row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_cell_time_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants + 1, rows, cols).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_cell_row_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants, rows + 1, cols).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_cell_col_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants, rows, cols + 1).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
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
                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window = chunk.get_window(&bounds).await?;

                            assert_eq!(window.shape(),
                                       [end - start, bottom - top, right - left]);

                            for i in 0..end - start {
                                assert_eq!(
                                    window.slice(s![i, .., ..]),
                                    data[start + i].slice(s![top..bottom, left..right])
                                );
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_window_time_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_window_row_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_fill_window_col_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
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
                            let lower = (start / 5) as f32;
                            let upper = (end / 10) as f32;

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = testing::array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let results: Vec<(usize, usize, usize)> = chunk
                                .search(bounds, lower, upper)
                                .map(|r| r.unwrap())
                                .collect().await;

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }


                #[tokio::test]
                async fn [<$name _test_search_rearrange>]() -> Result<()>{
                    let (_, data, chunk) = $name().await?;
                    let chunk = Arc::new(chunk);
                    let [instants, rows, cols] = chunk.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let lower = (start / 5) as f32;
                            let upper = (end / 10) as f32;

                            let mut expected: HashSet<(usize, usize, usize)> = HashSet::new();
                            for i in start..end {
                                let coords = testing::array_search_window(
                                    data[i].view(),
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    lower,
                                    upper,
                                );
                                for (row, col) in coords {
                                    expected.insert((i, row, col));
                                }
                            }

                            let bounds = geom::Cube::new(end, start, bottom, top, right, left);
                            let results: Vec<(usize, usize, usize)> = chunk
                                .search(bounds, upper, lower)
                                .map(|r| r.unwrap())
                                .collect().await;

                            let results: HashSet<(usize, usize, usize)> =
                                HashSet::from_iter(results.clone().into_iter());

                            assert_eq!(results.len(), expected.len());
                            assert_eq!(results, expected);
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                #[allow(unused_must_use)]
                async fn [<$name _test_search_time_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    Arc::new(chunk).search(bounds, 0.0, 100.0);
                }

                #[tokio::test]
                #[should_panic]
                #[allow(unused_must_use)]
                async fn [<$name _test_search_row_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    Arc::new(chunk).search(bounds, 0.0, 100.0);
                }

                #[tokio::test]
                #[should_panic]
                #[allow(unused_must_use)]
                async fn [<$name _test_search_col_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    Arc::new(chunk).search(bounds, 0.0, 100.0);
                }

                #[tokio::test]
                async fn [<$name _test_save_load>]() -> Result<()> {
                    let (resolver, data, chunk) = $name().await?;
                    let cid = resolver.save(chunk).await?;
                    let chunk = resolver.get_mmarray3(&cid).await?;

                    let [instants, rows, cols] = chunk.shape();
                    for row in 0..rows {
                        for col in 0..cols {
                            let start = row + col;
                            let end = instants - start;
                            let values = chunk.get_cell(start, end, row, col).await?;
                            assert_eq!(values.len(), end - start);
                            for i in 0..values.len() {
                                assert_eq!(values[i], data[i + start][[row, col]]);
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
                        MMArray3::Span(_) => { unimplemented!(); },
                        MMArray3::Subchunk(_) => {
                            assert_eq!(ls.len(), 0);
                        }
                        MMArray3::Superchunk(_) => {
                            assert_eq!(ls.len(), 1);
                            assert_eq!(ls[0].0, String::from("subchunks"));
                        }
                    }

                    Ok(())
                }
            }
        };
    }

    type DataArray3 = Result<(Arc<Resolver<f32>>, Vec<Array2<f32>>, MMArray3<f32>)>;

    async fn subchunk() -> DataArray3 {
        let data = testing::array8();
        let resolver = testing::resolver();
        let build = testing::build_subchunk(data.clone().into_iter(), 2, Precise(3));

        Ok((resolver, data, build.data))
    }

    mmarray3_tests!(subchunk);

    async fn superchunk() -> DataArray3 {
        let data = testing::array(17);
        let resolver = testing::resolver();
        let build = build_superchunk(
            data.clone().into_iter(),
            Arc::clone(&resolver),
            &[2, 3],
            2,
            Precise(3),
            8000,
        )
        .await?;

        Ok((resolver, data, build.data))
    }

    mmarray3_tests!(superchunk);

    macro_rules! mmarray1_tests {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_get>]() -> Result<()> {
                    let (_resolver, data, mmarray) = $name().await?;
                    for i in 0..mmarray.len() {
                        assert_eq!(mmarray.get(i), data[i]);
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_get_out_bounds>]() {
                    let (_resolver, _data, mmarray) = $name().await.unwrap();
                    assert_eq!(mmarray.get(mmarray.len()), 130.0); // Out of bounds
                }

                #[tokio::test]
                async fn [<$name _test_slice>]() -> Result<()> {
                    let (_resolver, data, mmarray) = $name().await?;
                    for i in 0..mmarray.len() / 2 {
                        let (start, end) = (i, mmarray.len() - i);
                        assert_eq!(mmarray.slice(start, end), data.slice(s![start..end]));
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_slice_out_of_bounds>]() {
                    let (_resolver, _data, mmarray) = $name().await.unwrap();
                    let start = mmarray.len() - 1;
                    let end = start + 2;
                    assert_eq!(mmarray.slice(start, end), array![125.0, 130.0]);
                }

                #[tokio::test]
                async fn [<$name _test_save_load>]() -> Result<()> {
                    let (resolver, data, mmarray) = $name().await?;
                    let cid = resolver.save(mmarray).await?;
                    let mmarray = resolver.get_mmarray1(&cid).await?;

                    for i in 0..mmarray.len() {
                        assert_eq!(mmarray.get(i), data[i]);
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_ls>]() -> Result<()> {
                    let (_, _, mmarray) = $name().await?;
                    assert_eq!(mmarray.ls(), vec![]);

                    Ok(())
                }
            }
        };
    }

    type DataArray1 = Result<(Arc<Resolver<f32>>, Array1<f32>, MMArray1<f32>)>;

    async fn range() -> DataArray1 {
        let resolver = testing::resolver();
        let data = Array1::range(-20.0, 130.0, 5.0);
        let range = MMArray1::Range(Range::new(-20.0, 5.0, 30));

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        Ok((resolver, data, range))
    }

    mmarray1_tests!(range);
}
