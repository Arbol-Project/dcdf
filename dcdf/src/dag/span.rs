use std::{cmp, fmt::Debug, pin::Pin, sync::Arc};

use async_recursion::async_recursion;
use async_trait::async_trait;
use cid::Cid;
use futures::{
    future::ready,
    stream::{once, FuturesUnordered, Stream, StreamExt, TryStreamExt},
    AsyncRead, AsyncWrite,
};
use ndarray::{s, ArrayBase, DataMut, Ix1, Ix3};
use num_traits::Float;

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite},
    geom,
};

use super::{
    mmarray::MMArray3,
    node::{Node, NODE_SPAN},
    resolver::Resolver,
};

/// A span of time
///
pub struct Span<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    shape: [usize; 3],
    stride: usize,
    spans: Vec<Cid>,
    resolver: Arc<Resolver<N>>,
}

impl<N> Span<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    pub fn new(shape: [usize; 2], stride: usize, resolver: Arc<Resolver<N>>) -> Self {
        Self {
            shape: [0, shape[0], shape[1]],
            stride,
            spans: vec![],
            resolver,
        }
    }

    pub async fn append(self, span: MMArray3<N>) -> Result<Self> {
        // Can only append to this span if the last subspan is full
        if self.spans.len() > 0 {
            let last_span = self
                .resolver
                .get_mmarray3(&self.spans[self.spans.len() - 1])
                .await?;

            if last_span.shape()[0] != self.stride {
                panic!("Can't append to span when last subspan is not full");
            }
        }

        // Make sure dimensions match up
        let span_shape = span.shape();
        if span_shape[1] != self.shape[1] || span_shape[2] != self.shape[2] {
            panic!(
                "Shape of subpsan ({}, {}) doesn't match shape of span ({}, {})",
                span_shape[1], span_shape[2], self.shape[1], self.shape[2]
            );
        }

        // Make sure subspan fits into one slot
        if span_shape[0] > self.stride {
            panic!(
                "Attempt to add subspan with length ({}) greater than stride ({})",
                span_shape[0], self.stride
            )
        }

        let shape = [self.shape[0] + span_shape[0], span_shape[1], span_shape[2]];
        let mut spans = self.spans;
        spans.push(self.resolver.save(span).await?);

        Ok({
            Self {
                shape,
                stride: self.stride,
                spans,
                resolver: self.resolver,
            }
        })
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// Get a cell's value at a particular time instant.
    ///
    #[async_recursion]
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<N> {
        self.check_bounds(instant, row, col);

        let (span, instant) = self.find_span(instant);
        let cid = &self.spans[span];
        let chunk = self.resolver.get_mmarray3(cid).await?;

        chunk.get(instant, row, col).await
    }

    /// Fill in a preallocated array with a cell's value across time instants.
    ///
    #[async_recursion]
    pub async fn fill_cell<S>(
        &self,
        start: usize,
        row: usize,
        col: usize,
        window: &mut ArrayBase<S, Ix1>,
    ) -> Result<()>
    where
        S: DataMut<Elem = N> + Send,
    {
        self.check_bounds(start + window.len() - 1, row, col);

        let mut futures = FuturesUnordered::new();
        let (mut span, mut instant) = self.find_span(start);
        let mut start = 0;
        let instants = window.shape()[0];
        while start < instants {
            let span_len = cmp::min(self.stride - instant, instants - start);
            let mut slice = window.slice_mut(s![start..start + span_len]);
            let mut slice = unsafe { slice.raw_view_mut().deref_into_view_mut() };
            let cid = &self.spans[span];
            let resolver = Arc::clone(&self.resolver);
            let future = async move {
                let chunk = resolver.get_mmarray3(cid).await?;
                chunk.fill_cell(instant, row, col, &mut slice).await
            };
            futures.push(future);

            instant = 0;
            span += 1;
            start += span_len;
        }

        while let Some(_) = futures.try_next().await? {
            // Just trying to flush out any errors that occurred in any of the futures
            // TODO: There must be a more elegant way to do this?
            continue;
        }

        Ok(())
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
        let [instants, rows, cols]: [_; 3] = window.shape().try_into().unwrap();
        self.check_bounds(start + instants - 1, top + rows - 1, left + cols - 1);

        // SMELL: DRY: Implemntation nearly identical to fill_cell
        let mut futures = FuturesUnordered::new();
        let (mut span, mut instant) = self.find_span(start);
        let mut start = 0;
        let instants = window.shape()[0];
        while start < instants {
            let span_len = cmp::min(self.stride - instant, instants - start);
            let mut slice = window.slice_mut(s![start..start + span_len, .., ..]);
            let mut slice = unsafe { slice.raw_view_mut().deref_into_view_mut() };
            let cid = &self.spans[span];
            let resolver = Arc::clone(&self.resolver);
            let future = async move {
                let chunk = resolver.get_mmarray3(cid).await?;
                chunk.fill_window(instant, top, left, &mut slice).await
            };
            futures.push(future);

            instant = 0;
            span += 1;
            start += span_len;
        }

        while let Some(_) = futures.try_next().await? {
            // Just trying to flush out any errors that occurred in any of the futures
            continue;
        }

        Ok(())
    }

    /// Search a subarray for cells that fall in a given range.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        &self,
        bounds: geom::Cube,
        lower: N,
        upper: N,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let futures = FuturesUnordered::new();
        let (mut span, mut instant) = self.find_span(bounds.start);
        let mut start = 0;
        let instants = bounds.instants();
        while start < instants {
            let span_len = cmp::min(self.stride - instant, instants - start);
            let cid = self.spans[span].clone();
            let resolver = Arc::clone(&self.resolver);
            let bounds = geom::Cube::new(
                instant,
                instant + span_len,
                bounds.top,
                bounds.bottom,
                bounds.left,
                bounds.right,
            );
            let offset = span * self.stride;
            let future = async move {
                let result = match resolver.get_mmarray3(&cid).await {
                    Ok(chunk) => chunk
                        .search(bounds, lower, upper)
                        .map(move |result| {
                            result.and_then(|(instant, row, col)| Ok((instant + offset, row, col)))
                        })
                        .boxed(),
                    Err(err) => once(ready(Err(err))).boxed(),
                };

                result
            };
            futures.push(future);

            instant = 0;
            span += 1;
            start += span_len;
        }

        futures.flatten_unordered(None).boxed()
    }

    fn find_span(&self, instant: usize) -> (usize, usize) {
        (instant / self.stride, instant % self.stride)
    }

    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape;
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::Span: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

#[async_trait]
impl<N> Node<N> for Span<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    const NODE_TYPE: u8 = NODE_SPAN;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        _resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_u32(self.shape[0] as u32).await?;
        stream.write_u32(self.shape[1] as u32).await?;
        stream.write_u32(self.shape[2] as u32).await?;
        stream.write_u32(self.stride as u32).await?;
        stream.write_u32(self.spans.len() as u32).await?;
        for span in &self.spans {
            stream.write_cid(span).await?;
        }

        Ok(())
    }

    /// Load an object from a stream
    async fn load_from(
        resolver: &Arc<Resolver<N>>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let instants = stream.read_u32().await? as usize;
        let rows = stream.read_u32().await? as usize;
        let cols = stream.read_u32().await? as usize;
        let shape = [instants, rows, cols];
        let stride = stream.read_u32().await? as usize;
        let n_spans = stream.read_u32().await? as usize;
        let mut spans = Vec::with_capacity(n_spans);
        for _ in 0..n_spans {
            spans.push(stream.read_cid().await?);
        }

        Ok(Self {
            shape,
            stride,
            spans,
            resolver: Arc::clone(resolver),
        })
    }

    fn ls(&self) -> Vec<(String, Cid)> {
        self.spans
            .iter()
            .enumerate()
            .map(|(i, cid)| (i.to_string(), cid.clone()))
            .collect()
    }
}

impl<N> Cacheable for Span<N>
where
    N: Float + Debug + Send + Sync + 'static,
{
    fn size(&self) -> u64 {
        Resolver::<f32>::HEADER_SIZE
            + 4 * 3 // shape
            + 4     // stride
            + 4     // n_spans
            + self.spans.iter().map(|cid| cid.encoded_len()).sum::<usize>() as u64
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ndarray::Array2;
    use paste::paste;

    use super::*;
    use crate::{build::build_superchunk, fixed::Precise, testing, MMArray3};

    type SuperchunkResult = Result<(Vec<Array2<f32>>, MMArray3<f32>)>;

    macro_rules! test_all_the_things {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_get>]() -> Result<()> {
                    let (_, data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for instant in 0..instants {
                        let row = instant % rows;
                        let col = instant % cols;
                        let value = chunk.get(instant, row, col).await?;
                        assert_eq!(value, data[instant][[row, col]]);
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_get_time_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, _rows, _cols] = chunk.shape();

                    chunk.get(instants, 0, 0).await
                        .expect("This isn't what causes the panic");
                }

                #[should_panic]
                #[tokio::test]
                async fn [<$name _test_get_row_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [_instants, rows, _cols] = chunk.shape();

                    chunk.get(0, rows, 0).await
                        .expect("This isn't what causes the panic");
                }

                #[should_panic]
                #[tokio::test]
                async fn [<$name _test_get_col_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [_instants, _rows, cols] = chunk.shape();

                    chunk.get(0, 0, cols).await
                        .expect("This isn't what causes the panic");
                }

                #[tokio::test]
                async fn [<$name _test_fill_cell>]() -> Result<()> {
                    let (_, data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for row in 0..rows {
                        let col = cols - row - 1;
                        let start = row * 15;
                        let end = instants - start;
                        let values = chunk.get_cell(start, end, row, col).await?;
                        assert_eq!(values.len(), end - start);
                        for i in 0..values.len() {
                            assert_eq!(values[i], data[i + start][[row, col]]);
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

                #[should_panic]
                #[tokio::test]
                async fn [<$name _test_fill_cell_row_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();

                    let values = chunk.get_cell(0, instants, rows + 1, cols).await
                        .expect("This isn't what causes the panic");
                    assert_eq!(values.len(), instants + 1);
                }

                #[should_panic]
                #[tokio::test]
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
                            let start = top * 30;
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

                #[should_panic]
                #[tokio::test]
                async fn [<$name _test_fill_window_time_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[should_panic]
                #[tokio::test]
                async fn [<$name _test_fill_window_row_out_of_bounds>]() {
                    let (_, _, chunk) = $name().await.expect("this should work");
                    let [instants, rows, cols] = chunk.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    chunk.get_window(&bounds).await.expect("Unexpected error.");
                }

                #[should_panic]
                #[tokio::test]
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
                            let start = top * 30;
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

                    let span = match chunk {
                        MMArray3::Span(span) => span,
                        _ => { panic!("not a span"); }
                    };
                    assert_eq!(ls.len(), span.spans.len());

                    Ok(())
                }
            }
        };
    }

    async fn superchunk(resolver: &Arc<Resolver<f32>>, length: usize) -> SuperchunkResult {
        let data = testing::array(16)
            .into_iter()
            .cycle()
            .take(length)
            .collect::<Vec<Array2<f32>>>();
        let build = build_superchunk(
            data.clone().into_iter(),
            Arc::clone(&resolver),
            &[2, 2],
            2,
            Precise(3),
            8000,
        )
        .await
        .unwrap();

        Ok((data, build.data))
    }

    fn new(resolver: &Arc<Resolver<f32>>) -> Result<Span<f32>> {
        let span = Span::new([16, 16], 100, Arc::clone(resolver));

        assert_eq!(span.shape(), [0, 16, 16]);
        assert_eq!(span.stride, 100);
        assert_eq!(span.spans, vec![]);

        Ok(span)
    }

    #[test]
    fn test_new() -> Result<()> {
        let resolver = testing::resolver();
        new(&resolver)?;

        Ok(())
    }

    type DataArray = Result<(Arc<Resolver<f32>>, Vec<Array2<f32>>, MMArray3<f32>)>;

    async fn span_of_5() -> DataArray {
        let resolver = testing::resolver();
        let span = new(&resolver)?;
        let (mut alldata, chunk) = superchunk(&resolver, 100).await?;
        let mut span = span.append(chunk).await?;
        for _ in 0..4 {
            let (mut data, chunk) = superchunk(&resolver, 100).await?;
            span = span.append(chunk).await?;
            alldata.append(&mut data);
        }

        assert_eq!(span.shape(), [500, 16, 16]);

        Ok((resolver, alldata, MMArray3::Span(span)))
    }

    test_all_the_things!(span_of_5);

    async fn span_of_four_and_a_half() -> DataArray {
        let resolver = testing::resolver();
        let span = new(&resolver)?;
        let (mut alldata, chunk) = superchunk(&resolver, 100).await?;
        let mut span = span.append(chunk).await?;
        for length in [100, 100, 100, 50] {
            let (mut data, chunk) = superchunk(&resolver, length).await?;
            span = span.append(chunk).await?;
            alldata.append(&mut data);
        }

        assert_eq!(span.shape(), [450, 16, 16]);

        Ok((resolver, alldata, MMArray3::Span(span)))
    }

    test_all_the_things!(span_of_four_and_a_half);

    #[tokio::test]
    #[should_panic]
    async fn test_append_to_partially_filled_span() {
        let resolver = testing::resolver();
        let span = new(&resolver).unwrap();
        let (_, chunk) = superchunk(&resolver, 10).await.unwrap();
        let span = span.append(chunk).await.unwrap();

        let (_, chunk) = superchunk(&resolver, 10).await.unwrap();
        span.append(chunk).await.unwrap(); // Previous span isn't full
    }

    #[tokio::test]
    #[should_panic]
    async fn test_append_mismatched_shape() {
        let resolver = testing::resolver();
        let span = new(&resolver).unwrap();
        let data = testing::array(15);
        let build = build_superchunk(
            data.clone().into_iter(),
            Arc::clone(&resolver),
            &[2, 2],
            2,
            Precise(3),
            8000,
        )
        .await
        .unwrap();

        span.append(build.data).await.unwrap(); // Wrong shape
    }

    #[tokio::test]
    #[should_panic]
    async fn test_append_overly_large_chunk() {
        let resolver = testing::resolver();
        let span = new(&resolver).unwrap();
        let (_, chunk) = superchunk(&resolver, 101).await.unwrap();
        span.append(chunk).await.unwrap(); // too big
    }

    async fn nested_spans() -> DataArray {
        let resolver = testing::resolver();
        let mut data: Vec<Array2<f32>> = vec![];
        let mut span = Span::new([16, 16], 100, Arc::clone(&resolver));
        for _ in 0..10 {
            let mut subspan = Span::new([16, 16], 10, Arc::clone(&resolver));
            for _ in 0..10 {
                let (mut subdata, chunk) = superchunk(&resolver, 10).await?;
                data.append(&mut subdata);
                subspan = subspan.append(chunk).await?;
            }
            span = span.append(MMArray3::Span(subspan)).await?;
        }

        Ok((resolver, data, MMArray3::Span(span)))
    }

    test_all_the_things!(nested_spans);
}
