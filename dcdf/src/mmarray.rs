use std::{pin::Pin, sync::Arc};

use futures::stream::Stream;
use ndarray::{Array1, Array3};

use crate::{
    errors::Result,
    geom,
    mmbuffer::{MMBuffer0, MMBuffer1, MMBuffer3},
    mmstruct::{MMEncoding, MMStruct3},
    range::{FloatRange, IntRange},
};

pub enum MMArray1I32 {
    Range(IntRange<i32>),
}

impl MMArray1I32 {
    pub fn get(&self, index: usize) -> i32 {
        match self {
            Self::Range(mmarray) => mmarray.get(index),
        }
    }
    pub fn slice(&self, start: usize, stop: usize) -> Array1<i32> {
        match self {
            Self::Range(mmarray) => mmarray.slice(start, stop),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(mmarray) => mmarray.len(),
        }
    }

    pub fn shape(&self) -> [usize; 1] {
        match self {
            Self::Range(mmarray) => mmarray.shape(),
        }
    }
}
pub enum MMArray1I64 {
    Range(IntRange<i64>),
}

impl MMArray1I64 {
    pub fn get(&self, index: usize) -> i64 {
        match self {
            Self::Range(mmarray) => mmarray.get(index),
        }
    }
    pub fn slice(&self, start: usize, stop: usize) -> Array1<i64> {
        match self {
            Self::Range(mmarray) => mmarray.slice(start, stop),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(mmarray) => mmarray.len(),
        }
    }

    pub fn shape(&self) -> [usize; 1] {
        match self {
            Self::Range(mmarray) => mmarray.shape(),
        }
    }
}

pub enum MMArray1F32 {
    Range(FloatRange<f32>),
}

impl MMArray1F32 {
    pub fn get(&self, index: usize) -> f32 {
        match self {
            Self::Range(mmarray) => mmarray.get(index),
        }
    }
    pub fn slice(&self, start: usize, stop: usize) -> Array1<f32> {
        match self {
            Self::Range(mmarray) => mmarray.slice(start, stop),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(mmarray) => mmarray.len(),
        }
    }

    pub fn shape(&self) -> [usize; 1] {
        match self {
            Self::Range(mmarray) => mmarray.shape(),
        }
    }
}

pub enum MMArray1F64 {
    Range(FloatRange<f64>),
}

impl MMArray1F64 {
    pub fn get(&self, index: usize) -> f64 {
        match self {
            Self::Range(mmarray) => mmarray.get(index),
        }
    }
    pub fn slice(&self, start: usize, stop: usize) -> Array1<f64> {
        match self {
            Self::Range(mmarray) => mmarray.slice(start, stop),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(mmarray) => mmarray.len(),
        }
    }

    pub fn shape(&self) -> [usize; 1] {
        match self {
            Self::Range(mmarray) => mmarray.shape(),
        }
    }
}

pub struct MMArray3I32 {
    pub(crate) data: Arc<MMStruct3>,
}

impl MMArray3I32 {
    pub(crate) fn new(data: Arc<MMStruct3>) -> Self {
        if data.encoding() != MMEncoding::I32 {
            panic!("Expecting I32 data, found {:?}", data.encoding());
        }

        Self { data }
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.data.shape()
    }

    /// Get a cell's value at a particular time instant.
    ///
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<i32> {
        self.check_bounds(instant, row, col);

        let mut buffer = MMBuffer0::I32(0);
        self.data.get(instant, row, col, &mut buffer).await?;

        Ok(buffer.into())
    }

    /// Get a cell's value across time instants
    ///
    pub async fn cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<Array1<i32>> {
        self.check_bounds(end - 1, row, col);
        let mut array = Array1::zeros([end - start]);
        let mut buffer = MMBuffer1::new_i32(array.view_mut());
        self.data
            .fill_cell(start, end, row, col, &mut buffer)
            .await?;

        Ok(array)
    }

    /// Retreive a subarray of this MMArray
    ///
    pub async fn window(&self, bounds: geom::Cube) -> Result<Array3<i32>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut array = Array3::zeros([
            bounds.end - bounds.start,
            bounds.bottom - bounds.top,
            bounds.right - bounds.left,
        ]);

        let mut buffer = MMBuffer3::new_i32(array.view_mut());
        self.data.fill_window(bounds, &mut buffer).await?;

        Ok(array)
    }

    /// Search a subarray for cells that fall in a given mmarray.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        &self,
        bounds: geom::Cube,
        lower: i32,
        upper: i32,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        self.data.search(bounds, lower as i64, upper as i64)
    }

    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape();
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::MMArray3: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

pub struct MMArray3I64 {
    data: Arc<MMStruct3>,
}

impl MMArray3I64 {
    pub(crate) fn new(data: Arc<MMStruct3>) -> Self {
        if data.encoding() != MMEncoding::I64 {
            panic!("Expecting I64 data, found {:?}", data.encoding());
        }

        Self { data }
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.data.shape()
    }

    /// Get a cell's value at a particular time instant.
    ///
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<i64> {
        self.check_bounds(instant, row, col);

        let mut buffer = MMBuffer0::I64(0);
        self.data.get(instant, row, col, &mut buffer).await?;

        Ok(buffer.into())
    }

    /// Get a cell's value across time instants
    ///
    pub async fn cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<Array1<i64>> {
        self.check_bounds(end - 1, row, col);
        let mut array = Array1::zeros([end - start]);
        let mut buffer = MMBuffer1::new_i64(array.view_mut());
        self.data
            .fill_cell(start, end, row, col, &mut buffer)
            .await?;

        Ok(array)
    }

    /// Retreive a subarray of this MMArray
    ///
    pub async fn window(&self, bounds: geom::Cube) -> Result<Array3<i64>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut array = Array3::zeros([
            bounds.end - bounds.start,
            bounds.bottom - bounds.top,
            bounds.right - bounds.left,
        ]);

        let mut buffer = MMBuffer3::new_i64(array.view_mut());
        self.data.fill_window(bounds, &mut buffer).await?;

        Ok(array)
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
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        self.data.search(bounds, lower, upper)
    }

    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape();
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::MMArray3: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

pub struct MMArray3F32 {
    pub(crate) data: Arc<MMStruct3>,
    fractional_bits: usize,
}

impl MMArray3F32 {
    pub(crate) fn new(data: Arc<MMStruct3>) -> Self {
        if data.encoding() != MMEncoding::F32 {
            panic!("Expecting F32 data, found {:?}", data.encoding());
        }

        let fractional_bits = data.fractional_bits();
        Self {
            data,
            fractional_bits,
        }
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.data.shape()
    }

    /// Get a cell's value at a particular time instant.
    ///
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<f32> {
        self.check_bounds(instant, row, col);

        let mut buffer = MMBuffer0::F32((0.0, self.fractional_bits));
        self.data.get(instant, row, col, &mut buffer).await?;

        Ok(buffer.into())
    }

    /// Get a cell's value across time instants
    ///
    pub async fn cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<Array1<f32>> {
        self.check_bounds(end - 1, row, col);
        let mut array = Array1::zeros([end - start]);
        let mut buffer = MMBuffer1::new_f32(array.view_mut(), self.fractional_bits, false);
        self.data
            .fill_cell(start, end, row, col, &mut buffer)
            .await?;

        Ok(array)
    }

    /// Retreive a subarray of this MMArray
    ///
    pub async fn window(&self, bounds: geom::Cube) -> Result<Array3<f32>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut array = Array3::zeros([
            bounds.end - bounds.start,
            bounds.bottom - bounds.top,
            bounds.right - bounds.left,
        ]);

        let mut buffer = MMBuffer3::new_f32(array.view_mut(), self.fractional_bits, false);
        self.data.fill_window(bounds, &mut buffer).await?;

        Ok(array)
    }

    /// Search a subarray for cells that fall in a given mmarray.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        &self,
        _bounds: geom::Cube,
        _lower: f32,
        _upper: f32,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        // Need to figure out out how translate lower and upper bounds into appropriate fixed point
        // reprsentation given the local fractional bits in each underlying substruct that gets
        // searched.
        todo!();
    }

    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape();
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::MMArray3: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

pub struct MMArray3F64 {
    data: Arc<MMStruct3>,
    fractional_bits: usize,
}

impl MMArray3F64 {
    pub(crate) fn new(data: Arc<MMStruct3>) -> Self {
        if data.encoding() != MMEncoding::F64 {
            panic!("Expecting F64 data, found {:?}", data.encoding());
        }

        let fractional_bits = data.fractional_bits();
        Self {
            data,
            fractional_bits,
        }
    }

    /// Get the shape of the overall time series raster
    ///
    pub fn shape(&self) -> [usize; 3] {
        self.data.shape()
    }

    /// Get a cell's value at a particular time instant.
    ///
    pub async fn get(&self, instant: usize, row: usize, col: usize) -> Result<f64> {
        self.check_bounds(instant, row, col);

        let mut buffer = MMBuffer0::F64((0.0, self.fractional_bits));
        self.data.get(instant, row, col, &mut buffer).await?;

        Ok(buffer.into())
    }

    /// Get a cell's value across time instants
    ///
    pub async fn cell(
        &self,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> Result<Array1<f64>> {
        self.check_bounds(end - 1, row, col);
        let mut array = Array1::zeros([end - start]);
        let mut buffer = MMBuffer1::new_f64(array.view_mut(), self.fractional_bits, false);
        self.data
            .fill_cell(start, end, row, col, &mut buffer)
            .await?;

        Ok(array)
    }

    /// Retreive a subarray of this MMArray
    ///
    pub async fn window(&self, bounds: geom::Cube) -> Result<Array3<f64>> {
        self.check_bounds(bounds.end - 1, bounds.bottom - 1, bounds.right - 1);

        let mut array = Array3::zeros([
            bounds.end - bounds.start,
            bounds.bottom - bounds.top,
            bounds.right - bounds.left,
        ]);

        let mut buffer = MMBuffer3::new_f64(array.view_mut(), self.fractional_bits, false);
        self.data.fill_window(bounds, &mut buffer).await?;

        Ok(array)
    }

    /// Search a subarray for cells that fall in a given mmarray.
    ///
    /// Returns a boxed Stream that produces Vecs of coordinate triplets [instant, row, col] of
    /// matching cells.
    ///
    pub fn search(
        &self,
        _bounds: geom::Cube,
        _lower: f64,
        _upper: f64,
    ) -> Pin<Box<dyn Stream<Item = Result<(usize, usize, usize)>> + Send>> {
        // Need to figure out out how translate lower and upper bounds into appropriate fixed point
        // reprsentation given the local fractional bits in each underlying substruct that gets
        // searched.
        todo!();
    }

    /// Panics if given point is out of bounds for this chunk
    fn check_bounds(&self, instant: usize, row: usize, col: usize) {
        let [instants, rows, cols] = self.shape();
        if instant >= instants || row >= rows || col >= cols {
            panic!(
                "dcdf::MMArray3: index[{}, {}, {}] is out of bounds for array of shape {:?}",
                instant,
                row,
                col,
                [instants, rows, cols],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        chunk::Chunk, mmbuffer::MMBuffer3, resolver::Resolver, span::Span, superchunk::Superchunk,
        testing,
    };

    use std::collections::HashSet;

    use futures::StreamExt;
    use ndarray::{array, s, Array3};
    use paste::paste;

    macro_rules! mmarray1_int_tests {
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
                    assert_eq!(mmarray.get(mmarray.len()), 130); // Out of bounds
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
                    assert_eq!(mmarray.slice(start, end), array![125, 130]);
                }
            }
        };
    }

    type DataArray1I32 = Result<(Arc<Resolver>, Array1<i32>, MMArray1I32)>;

    async fn range_i32() -> DataArray1I32 {
        let resolver = testing::resolver();
        let data = Array1::from_iter((-20..130).step_by(5));
        let range = MMArray1I32::Range(IntRange::new(-20, 5, 30));

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        Ok((resolver, data, range))
    }

    mmarray1_int_tests!(range_i32);

    type DataArray1I64 = Result<(Arc<Resolver>, Array1<i64>, MMArray1I64)>;

    async fn range_i64() -> DataArray1I64 {
        let resolver = testing::resolver();
        let data = Array1::from_iter((-20..130).step_by(5));
        let range = MMArray1I64::Range(IntRange::new(-20, 5, 30));

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        Ok((resolver, data, range))
    }

    mmarray1_int_tests!(range_i64);

    macro_rules! mmarray1_float_tests {
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
            }
        };
    }

    type DataArray1F32 = Result<(Arc<Resolver>, Array1<f32>, MMArray1F32)>;

    async fn range_f32() -> DataArray1F32 {
        let resolver = testing::resolver();
        let data = Array1::range(-20.0, 130.0, 5.0);
        let range = MMArray1F32::Range(FloatRange::new(-20.0, 5.0, 30));

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        Ok((resolver, data, range))
    }

    mmarray1_float_tests!(range_f32);

    type DataArray1F64 = Result<(Arc<Resolver>, Array1<f64>, MMArray1F64)>;

    async fn range_f64() -> DataArray1F64 {
        let resolver = testing::resolver();
        let data = Array1::range(-20.0, 130.0, 5.0);
        let range = MMArray1F64::Range(FloatRange::new(-20.0, 5.0, 30));

        assert_eq!(range.len(), 30);
        assert_eq!(range.shape(), [30]);
        assert_eq!(range.shape(), range.slice(0, 30).shape());

        Ok((resolver, data, range))
    }

    mmarray1_float_tests!(range_f64);

    macro_rules! mmarray3_tests {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_get>]() -> Result<()> {
                    let (data, mmarray) = $name().await?;
                    let [instants, rows, cols] = mmarray.shape();
                    for instant in 0..instants {
                        for row in 0..rows {
                            for col in 0..cols {
                                assert_eq!(mmarray.get(instant, row, col).await?, data[[instant, row, col]]);
                            }
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_get_instant_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    mmarray.get(instants, rows - 1, cols - 1).await.unwrap();
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_get_row_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    mmarray.get(instants - 1, rows, cols - 1).await.unwrap();
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_get_col_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    mmarray.get(instants - 1, rows - 1, cols).await.unwrap();
                }

                #[tokio::test]
                async fn [<$name _test_cell>]() -> Result<()> {
                    let (data, mmarray) = $name().await?;
                    let [instants, rows, cols] = mmarray.shape();
                    for row in 0..rows {
                        for col in 0..cols {
                            let start = row + col;
                            let end = instants - start;
                            let cell = mmarray.cell(start, end, row, col).await?;
                            assert_eq!(cell, data.slice(s![start..end, row, col]));
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_cell_end_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    mmarray.cell(0, instants + 1, rows - 1, cols - 1).await.unwrap();
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_cell_row_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    mmarray.cell(0, instants - 1, rows, cols - 1).await.unwrap();
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_cell_col_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    mmarray.cell(0, instants - 1, rows - 1, cols).await.unwrap();
                }

                #[tokio::test]
                async fn [<$name _test_window>]() -> Result<()> {
                    let (data, mmarray) = $name().await?;
                    let [instants, rows, cols] = mmarray.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let window = mmarray.window(bounds).await?;

                            assert_eq!(window, data.slice(s![start..end, top..bottom, left..right]));
                        }
                    }

                    Ok(())
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_window_end_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    mmarray.window(bounds).await.unwrap();
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_window_bottom_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    mmarray.window(bounds).await.unwrap();
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_window_right_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    mmarray.window(bounds).await.unwrap();
                }
            }
        }
    }

    macro_rules! mmarray3_i32_search_tests {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_search>]() -> Result<()>{
                    let (data, mmarray) = $name().await?;
                    let [instants, rows, cols] = mmarray.shape();
                    for top in 0..rows / 2 {
                        let bottom = top + rows / 2;
                        for left in 0..cols / 2 {
                            let right = left + cols / 2;
                            let start = top + bottom;
                            let end = instants - start;
                            let lower = (start / 5) as i32;
                            let upper = (end / 10) as i32;

                            let bounds = geom::Cube::new(start, end, top, bottom, left, right);
                            let expected = testing::array_search_window3(
                                data.view(),
                                bounds,
                                lower,
                                upper,
                            ).into_iter().collect::<HashSet<_>>();

                            let results = mmarray
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
                #[should_panic]
                async fn [<$name _test_search_end_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    mmarray.search(bounds, 0, 42);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_search_bottom_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    mmarray.search(bounds, 0, 42);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_search_right_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    mmarray.search(bounds, 0, 42);
                }
            }
        };
    }

    macro_rules! mmarray3_i64_search_tests {
        ($name:ident) => {
            paste! {
                #[tokio::test]
                async fn [<$name _test_search>]() -> Result<()>{
                    let (data, mmarray) = $name().await?;
                    let [instants, rows, cols] = mmarray.shape();
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

                            let results = mmarray
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
                #[should_panic]
                async fn [<$name _test_search_end_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants + 1, 0, rows, 0, cols);
                    mmarray.search(bounds, 0, 42);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_search_bottom_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows + 1, 0, cols);
                    mmarray.search(bounds, 0, 42);
                }

                #[tokio::test]
                #[should_panic]
                async fn [<$name _test_search_right_out_of_bounds>]() {
                    let (_data, mmarray) = $name().await.unwrap();
                    let [instants, rows, cols] = mmarray.shape();
                    let bounds = geom::Cube::new(0, instants, 0, rows, 0, cols + 1);
                    mmarray.search(bounds, 0, 42);
                }
            }
        };
    }

    type DataArray3I32 = Result<(Array3<i32>, MMArray3I32)>;

    async fn chunk_i32() -> DataArray3I32 {
        let data = testing::array(16);
        let mut data = data.mapv(|v| v as i32);
        let mut buffer = MMBuffer3::new_i32(data.view_mut());
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;
        let mmarray = MMArray3I32::new(Arc::new(chunk));

        Ok((data, mmarray))
    }

    async fn span_i32() -> DataArray3I32 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::array(16);
        let data = data.mapv(|v| v as i32);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::I32,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::I32);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_i32(array);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[2, 2],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3I32::new(Arc::new(MMStruct3::Span(span)))))
    }

    async fn span_i32_shallow_superchunks() -> DataArray3I32 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::array(16);
        let data = data.mapv(|v| v as i32);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::I32,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::I32);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_i32(array);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[4, 0],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3I32::new(Arc::new(MMStruct3::Span(span)))))
    }

    #[test]
    #[should_panic]
    fn mmarray_i32_type_mismatch() {
        let mut data = testing::array(16);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;

        MMArray3I32::new(Arc::new(chunk));
    }

    mmarray3_tests!(chunk_i32);
    mmarray3_tests!(span_i32);
    mmarray3_tests!(span_i32_shallow_superchunks);
    mmarray3_i32_search_tests!(chunk_i32);
    mmarray3_i32_search_tests!(span_i32);

    type DataArray3I64 = Result<(Array3<i64>, MMArray3I64)>;

    async fn chunk_i64() -> DataArray3I64 {
        let mut data = testing::array(16);
        let mut buffer = MMBuffer3::new_i64(data.view_mut());
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;
        let mmarray = MMArray3I64::new(Arc::new(chunk));

        Ok((data, mmarray))
    }

    async fn span_i64() -> DataArray3I64 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::array(16);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::I64,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::I64);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_i64(array);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[2, 2],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3I64::new(Arc::new(MMStruct3::Span(span)))))
    }

    async fn span_i64_shallow_superchunks() -> DataArray3I64 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::array(16);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::I64,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::I64);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_i64(array);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[4, 0],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3I64::new(Arc::new(MMStruct3::Span(span)))))
    }

    #[test]
    #[should_panic]
    fn mmarray_i64_type_mismatch() {
        let data = testing::array(16);
        let mut data = data.mapv(|v| v as i32);
        let mut buffer = MMBuffer3::new_i32(data.view_mut());
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;

        MMArray3I64::new(Arc::new(chunk));
    }

    mmarray3_tests!(chunk_i64);
    mmarray3_tests!(span_i64);
    mmarray3_tests!(span_i64_shallow_superchunks);
    mmarray3_i64_search_tests!(chunk_i64);
    mmarray3_i64_search_tests!(span_i64);

    type DataArray3F32 = Result<(Array3<f32>, MMArray3F32)>;

    async fn chunk_f32() -> DataArray3F32 {
        let mut data = testing::farray(16);
        let mut buffer = MMBuffer3::new_f32(data.view_mut(), 3, false);
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;
        let mmarray = MMArray3F32::new(Arc::new(chunk));

        Ok((data, mmarray))
    }

    async fn span_f32() -> DataArray3F32 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::farray(16);
        let data = data.mapv(|v| v as f32);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::F32,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::F32);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_f32(array, 3, false);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[2, 2],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3F32::new(Arc::new(MMStruct3::Span(span)))))
    }

    async fn span_f32_shallow_superchunks() -> DataArray3F32 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::farray(16);
        let data = data.mapv(|v| v as f32);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::F32,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::F32);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_f32(array, 3, false);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[4, 0],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3F32::new(Arc::new(MMStruct3::Span(span)))))
    }

    #[test]
    #[should_panic]
    fn mmarray_f32_type_mismatch() {
        let data = testing::farray(16);
        let mut data = data.mapv(|v| v as f64);
        let mut buffer = MMBuffer3::new_f64(data.view_mut(), 3, false);
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;

        MMArray3F32::new(Arc::new(chunk));
    }

    mmarray3_tests!(chunk_f32);
    mmarray3_tests!(span_f32);
    mmarray3_tests!(span_f32_shallow_superchunks);

    type DataArray3F64 = Result<(Array3<f64>, MMArray3F64)>;

    async fn chunk_f64() -> DataArray3F64 {
        let data = testing::farray(16);
        let mut data = data.mapv(|v| v as f64);
        let mut buffer = MMBuffer3::new_f64(data.view_mut(), 3, false);
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;
        let mmarray = MMArray3F64::new(Arc::new(chunk));

        Ok((data, mmarray))
    }

    async fn span_f64() -> DataArray3F64 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::farray(16);
        let data = data.mapv(|v| v as f64);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::F64,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::F64);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_f64(array, 3, false);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[2, 2],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3F64::new(Arc::new(MMStruct3::Span(span)))))
    }

    async fn span_f64_shallow_superchunks() -> DataArray3F64 {
        let span_size = 500;
        let subspan_size = 100;
        let chunk_size = 20;

        let data = testing::farray(16);
        let data = data.mapv(|v| v as f64);
        let mut data = Array3::from_shape_fn((span_size, 16, 16), |(instant, row, col)| {
            data[[instant % 100, row, col]]
        });

        let resolver = testing::resolver();
        let mut span = Span::new(
            [16, 16],
            subspan_size,
            Arc::clone(&resolver),
            MMEncoding::F64,
        );
        for i in (0..span_size).step_by(subspan_size) {
            let mut subspan =
                Span::new([16, 16], chunk_size, Arc::clone(&resolver), MMEncoding::F64);
            for j in (0..subspan_size).step_by(chunk_size) {
                let start = i + j;
                let end = start + chunk_size;
                let array = data.slice_mut(s![start..end, .., ..]);
                let mut buffer = MMBuffer3::new_f64(array, 3, false);
                let build = Superchunk::build(
                    Arc::clone(&resolver),
                    &mut buffer,
                    [chunk_size, 16, 16],
                    &[4, 0],
                    2,
                )
                .await?;

                subspan = subspan.append(&build.data).await?;
            }

            span = span.append(&MMStruct3::Span(subspan)).await?;
        }

        Ok((data, MMArray3F64::new(Arc::new(MMStruct3::Span(span)))))
    }

    #[test]
    #[should_panic]
    fn mmarray_f64_type_mismatch() {
        let mut data = testing::farray(16);
        let mut buffer = MMBuffer3::new_f32(data.view_mut(), 3, false);
        let chunk = Chunk::build(&mut buffer, [100, 16, 16], 2).data;

        MMArray3F64::new(Arc::new(chunk));
    }

    mmarray3_tests!(chunk_f64);
    mmarray3_tests!(span_f64);
    mmarray3_tests!(span_f64_shallow_superchunks);
}
