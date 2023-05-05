use std::{cmp, pin::Pin, sync::Arc};

use async_recursion::async_recursion;
use async_trait::async_trait;
use cid::Cid;
use futures::{
    future::ready,
    stream::{once, FuturesUnordered, Stream, StreamExt, TryStreamExt},
    AsyncRead, AsyncWrite,
};

use crate::{
    cache::Cacheable,
    errors::Result,
    extio::{ExtendedAsyncRead, ExtendedAsyncWrite},
    geom,
    mmbuffer::{MMBuffer0, MMBuffer1, MMBuffer3},
    mmstruct::{MMEncoding, MMStruct3},
    node::{Node, NODE_SPAN},
    resolver::Resolver,
};

/// A span of time
///

pub(crate) struct Span {
    pub(crate) shape: [usize; 3],
    pub(crate) stride: usize,
    pub(crate) spans: Vec<Cid>,
    resolver: Arc<Resolver>,
    pub encoding: MMEncoding,
}

impl Span {
    pub fn new(
        shape: [usize; 2],
        stride: usize,
        resolver: Arc<Resolver>,
        encoding: MMEncoding,
    ) -> Self {
        Self {
            shape: [0, shape[0], shape[1]],
            stride,
            spans: vec![],
            resolver,
            encoding,
        }
    }

    pub async fn append(&self, span: &MMStruct3) -> Result<Self> {
        // Can only append to this span if the last subspan is full
        if self.spans.len() > 0 {
            let last_span = self
                .resolver
                .get_mmstruct3(&self.spans[self.spans.len() - 1])
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
        let mut spans = self.spans.clone();
        spans.push(self.resolver.save(span).await?);

        Ok({
            Self {
                shape,
                stride: self.stride,
                spans,
                resolver: Arc::clone(&self.resolver),
                encoding: self.encoding,
            }
        })
    }

    // Replace the last subspan with new data
    //
    pub async fn update(&self, span: &MMStruct3) -> Result<Self> {
        let mut spans = self.spans.clone();
        spans.pop();

        let tmp = Self {
            shape: [spans.len() * self.stride, self.shape[1], self.shape[2]],
            stride: self.stride,
            spans,
            resolver: Arc::clone(&self.resolver),
            encoding: self.encoding,
        };

        tmp.append(span).await
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.shape
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
        let (span, instant) = self.find_span(instant);
        let cid = &self.spans[span];
        let chunk = self.resolver.get_mmstruct3(cid).await?;
        buffer.set_fractional_bits(chunk.fractional_bits());

        chunk.get(instant, row, col, buffer).await
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
        let mut futures = FuturesUnordered::new();
        let instants = end - start;
        let (mut span, mut instant) = self.find_span(start);

        let mut start = 0;
        while start < instants {
            let span_len = cmp::min(self.stride - instant, instants - start);
            let mut buffer = buffer.slice(start, start + span_len);
            let cid = &self.spans[span];
            let resolver = Arc::clone(&self.resolver);
            let future = async move {
                let chunk = resolver.get_mmstruct3(cid).await?;
                buffer.set_fractional_bits(chunk.fractional_bits());
                chunk
                    .fill_cell(instant, instant + span_len, row, col, &mut buffer)
                    .await
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
    pub(crate) async fn fill_window(
        &self,
        bounds: geom::Cube,
        buffer: &mut MMBuffer3<'_>,
    ) -> Result<()> {
        // SMELL: DRY: Implementation nearly identical to fill_cell
        let mut futures = FuturesUnordered::new();
        let (mut span, mut instant) = self.find_span(bounds.start);
        let mut start = 0;
        let [instants, rows, cols] = buffer.shape();
        while start < instants {
            let span_len = cmp::min(self.stride - instant, instants - start);
            let mut buffer = buffer.slice(start, start + span_len, 0, rows, 0, cols);
            let cid = &self.spans[span];
            let resolver = Arc::clone(&self.resolver);
            let future = async move {
                let chunk = resolver.get_mmstruct3(cid).await?;
                let span_bounds = geom::Cube::new(
                    instant,
                    instant + span_len,
                    bounds.top,
                    bounds.bottom,
                    bounds.left,
                    bounds.right,
                );
                buffer.set_fractional_bits(chunk.fractional_bits());
                chunk.fill_window(span_bounds, &mut buffer).await
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

    /// Search a subarray for cells that fall in a given mmarray.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        &self,
        bounds: geom::Cube,
        lower: i64,
        upper: i64,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
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
                let result = match resolver.get_mmstruct3(&cid).await {
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
}

#[async_trait]
impl Node for Span {
    const NODE_TYPE: u8 = NODE_SPAN;

    /// Save an object into the DAG
    ///
    async fn save_to(
        &self,
        _resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncWrite + Unpin + Send),
    ) -> Result<()> {
        stream.write_byte(self.encoding as u8).await?;
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
        resolver: &Arc<Resolver>,
        stream: &mut (impl AsyncRead + Unpin + Send),
    ) -> Result<Self> {
        let encoding = MMEncoding::try_from(stream.read_byte().await?)?;
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
            encoding,
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

impl Cacheable for Span {
    fn size(&self) -> u64 {
        Resolver::HEADER_SIZE
            + 1     // encoding
            + 4 * 3 // shape
            + 4     // stride
            + 4     // n_spans
            + self.spans.iter().map(|cid| cid.encoded_len()).sum::<usize>() as u64
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use ndarray::{s, Array1, Array3, Axis};
    use paste::paste;

    use super::*;
    use crate::{superchunk::Superchunk, testing};

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
                        let mut buffer = MMBuffer0::I64(0);
                        chunk.get(instant, row, col, &mut buffer).await?;
                        assert_eq!(i64::from(buffer), data[[instant, row, col]]);
                    }

                    Ok(())
                }

                #[tokio::test]
                async fn [<$name _test_fill_cell>]() -> Result<()> {
                    let (_, data, chunk) = $name().await?;
                    let [instants, rows, cols] = chunk.shape();
                    for row in 0..rows {
                        let col = cols - row - 1;
                        let start = row * 15;
                        let end = instants - start;
                        let mut array = Array1::zeros([end - start]);
                        let mut buffer = MMBuffer1::new_i64(array.view_mut());
                        chunk.fill_cell(start, end, row, col, &mut buffer).await?;

                        assert_eq!(array, data.slice(s![start..end, row, col]));
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
                            let start = top * 30;
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
                            let start = top * 30;
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
                async fn [<$name _test_ls>]() -> Result<()> {
                    let (_, _, chunk) = $name().await?;
                    let ls = chunk.ls();

                    let span = match chunk {
                        MMStruct3::Span(span) => span,
                        _ => { panic!("not a span"); }
                    };
                    assert_eq!(ls.len(), span.spans.len());

                    Ok(())
                }
            }
        };
    }

    async fn superchunks(
        resolver: &Arc<Resolver>,
        lens: &[usize],
    ) -> Result<(Array3<i64>, Vec<MMStruct3>)> {
        let length = lens.iter().sum();
        let data = testing::array(16);
        let mut data = Array3::from_shape_fn((length, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let mut start = 0;
        let mut chunks = vec![];
        for len in lens {
            let end = start + len;
            let array = data.slice_mut(s![start..end, .., ..]);
            let mut buffer = MMBuffer3::new_i64(array);
            let build = Superchunk::build(
                Arc::clone(resolver),
                &mut buffer,
                [*len, 16, 16],
                &[2, 2],
                2,
            )
            .await?;

            chunks.push(build.data);
            start = end;
        }

        Ok((data, chunks))
    }

    fn new(resolver: &Arc<Resolver>) -> Result<Span> {
        let span = Span::new([16, 16], 100, Arc::clone(resolver), MMEncoding::I64);

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

    type DataArray = Result<(Arc<Resolver>, Array3<i64>, MMStruct3)>;

    async fn span_of_5() -> DataArray {
        let resolver = testing::resolver();
        let mut span = new(&resolver)?;

        let (data, chunks) = superchunks(&resolver, &[100; 5]).await?;
        for chunk in chunks {
            span = span.append(&chunk).await?;
        }
        assert_eq!(span.shape(), [500, 16, 16]);

        Ok((resolver, data, MMStruct3::Span(span)))
    }

    test_all_the_things!(span_of_5);

    async fn span_of_four_and_a_half() -> DataArray {
        let resolver = testing::resolver();
        let mut span = new(&resolver)?;

        let (data, chunks) = superchunks(&resolver, &[100, 100, 100, 100, 50]).await?;
        for chunk in chunks {
            span = span.append(&chunk).await?;
        }
        assert_eq!(span.shape(), [450, 16, 16]);

        Ok((resolver, data, MMStruct3::Span(span)))
    }

    test_all_the_things!(span_of_four_and_a_half);

    #[tokio::test]
    #[should_panic]
    async fn test_append_to_partially_filled_span() {
        let resolver = testing::resolver();
        let span = new(&resolver).unwrap();
        let (_, chunks) = superchunks(&resolver, &[10]).await.unwrap();
        let span = span.append(&first(chunks)).await.unwrap();

        let (_, chunks) = superchunks(&resolver, &[10]).await.unwrap();
        span.append(&first(chunks)).await.unwrap(); // Previous span isn't full
    }

    #[tokio::test]
    #[should_panic]
    async fn test_append_mismatched_shape() {
        let resolver = testing::resolver();
        let span = new(&resolver).unwrap();
        let mut data = testing::array(15);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let build = Superchunk::build(
            Arc::clone(&resolver),
            &mut buffer,
            [100, 15, 15],
            &[2, 2],
            2,
        )
        .await
        .unwrap();

        span.append(&build.data).await.unwrap(); // Wrong shape
    }

    #[tokio::test]
    #[should_panic]
    async fn test_append_overly_large_chunk() {
        let resolver = testing::resolver();
        let span = new(&resolver).unwrap();
        let (_, chunks) = superchunks(&resolver, &[101]).await.unwrap();
        span.append(&first(chunks)).await.unwrap(); // too big
    }

    async fn nested_spans() -> DataArray {
        let resolver = testing::resolver();
        let (data, chunks) = superchunks(&resolver, &[10; 100]).await?;
        let mut chunks = chunks.into_iter();

        let mut span = Span::new([16, 16], 100, Arc::clone(&resolver), MMEncoding::I64);
        for _ in 0..10 {
            let mut subspan = Span::new([16, 16], 10, Arc::clone(&resolver), MMEncoding::I64);
            for _ in 0..10 {
                let chunk = chunks.next().unwrap();
                subspan = subspan.append(&chunk).await?;
            }
            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((resolver, data, MMStruct3::Span(span)))
    }

    test_all_the_things!(nested_spans);

    async fn update() -> DataArray {
        let resolver = testing::resolver();
        let mut span = new(&resolver)?;
        let (data, chunks) = superchunks(&resolver, &[100, 100, 100, 100, 50]).await?;
        for chunk in chunks {
            span = span.append(&chunk).await?;
        }

        let (new_data, chunks) = superchunks(&resolver, &[75]).await?;
        span = span.update(&first(chunks)).await?;

        let mut data = data.slice(s![..400, .., ..]).to_owned();
        data.append(Axis(0), new_data.view()).ok();

        assert_eq!(span.shape(), [475, 16, 16]);

        Ok((resolver, data, MMStruct3::Span(span)))
    }

    test_all_the_things!(update);

    #[tokio::test]
    async fn test_high_level_ls() -> Result<()> {
        let (resolver, _, span) = nested_spans().await?;
        let cid = resolver.save(&span).await?;
        let ls = resolver.ls(&cid).await?;
        assert_eq!(ls.len(), 10);
        assert_eq!(ls[0].name, "0");
        assert_eq!(ls[0].node_type.unwrap(), "Span");

        let ls = resolver.ls(&ls[0].cid).await?;
        assert_eq!(ls.len(), 10);
        assert_eq!(ls[0].name, "0");
        assert_eq!(ls[0].node_type.unwrap(), "Superchunk");

        let ls = resolver.ls(&ls[0].cid).await?;
        assert_eq!(ls.len(), 1);
        assert_eq!(ls[0].name, "subchunks");
        assert_eq!(ls[0].node_type.unwrap(), "Links");

        let ls = resolver.ls(&ls[0].cid).await?;
        assert_eq!(ls.len(), 4);
        assert_eq!(ls[0].name, "0");
        assert_eq!(ls[0].node_type.unwrap(), "Subchunk");

        Ok(())
    }

    fn first<T>(a: Vec<T>) -> T {
        a.into_iter().next().unwrap()
    }
}
