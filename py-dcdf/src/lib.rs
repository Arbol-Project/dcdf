use std::{future::Future, str::FromStr, sync::Arc};

use cid::Cid;
use pyo3::{
    exceptions::{PyIOError, PyKeyError, PyNameError, PyValueError},
    prelude::*,
};

use dcdf::{self, geom, LsEntry, MMEncoding};
use numpy::{
    borrow::PyReadwriteArray3,
    datetime::{units, Datetime},
    IntoPyArray, PyArray1, PyArray3, ToPyArray,
};
use tokio::runtime::Runtime;

#[pyclass]
pub struct PyDataset(Arc<dcdf::Dataset>);

#[pymethods]
impl PyDataset {
    #[new]
    pub fn new(
        t_coord: PyCoordinate,
        y_coord: PyCoordinate,
        x_coord: PyCoordinate,
        shape: [usize; 2],
        resolver: &PyResolver,
    ) -> Self {
        Self(Arc::new(dcdf::Dataset::new(
            [t_coord.0, y_coord.0, x_coord.0],
            shape,
            Arc::clone(&resolver.0),
        )))
    }

    #[getter]
    fn coordinates(&self) -> Vec<PyCoordinate> {
        self.0
            .coordinates
            .iter()
            .map(|c| PyCoordinate(c.clone()))
            .collect()
    }

    #[getter]
    fn variables(&self) -> Vec<PyVariable> {
        self.0
            .variables
            .iter()
            .map(|c| PyVariable(c.clone()))
            .collect()
    }

    #[getter]
    fn cid(&self) -> Option<String> {
        self.0.cid.and_then(|cid| Some(cid.to_string()))
    }

    #[getter]
    fn prev(&self) -> Option<String> {
        self.0.prev.and_then(|cid| Some(cid.to_string()))
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.0.shape[0], self.0.shape[1])
    }

    pub fn commit(&self) -> PyResult<String> {
        let cid = block_on_result(self.0.commit())?;

        Ok(cid.to_string())
    }

    pub fn add_variable(
        &self,
        name: String,
        span_size: usize,
        chunk_size: usize,
        k2_levels: Vec<u32>,
        round: bool,
        fractional_bits: usize,
        encoding: u8,
    ) -> PyResult<Self> {
        let round = if round { Some(fractional_bits) } else { None };
        let encoding = MMEncoding::try_from(encoding).map_err(convert_error)?;
        let dataset = block_on_result(
            self.0
                .add_variable(name, round, span_size, chunk_size, k2_levels, encoding),
        )?;

        Ok(Self(Arc::new(dataset)))
    }

    pub fn append_i32(&self, name: &str, mut data: PyReadwriteArray3<i32>) -> PyResult<Self> {
        let data = data.as_array_mut();
        let dataset = block_on_result(self.0.append_i32(name, data))?;

        Ok(Self(Arc::new(dataset)))
    }

    pub fn append_i64(&self, name: &str, mut data: PyReadwriteArray3<i64>) -> PyResult<Self> {
        let data = data.as_array_mut();
        let dataset = block_on_result(self.0.append_i64(name, data))?;

        Ok(Self(Arc::new(dataset)))
    }

    pub fn append_f32(&self, name: &str, mut data: PyReadwriteArray3<f32>) -> PyResult<Self> {
        let data = data.as_array_mut();
        let dataset = block_on_result(self.0.append_f32(name, data))?;

        Ok(Self(Arc::new(dataset)))
    }

    pub fn append_f64(&self, name: &str, mut data: PyReadwriteArray3<f64>) -> PyResult<Self> {
        let data = data.as_array_mut();
        let dataset = block_on_result(self.0.append_f64(name, data))?;

        Ok(Self(Arc::new(dataset)))
    }

    pub fn get_coordinate(&self, name: &str) -> Option<PyCoordinate> {
        self.0
            .get_coordinate(name)
            .and_then(|coordinate| Some(PyCoordinate(coordinate.clone())))
    }

    pub fn get_variable(&self, name: &str) -> Option<PyVariable> {
        self.0
            .get_variable(name)
            .and_then(|variable| Some(PyVariable(variable.clone())))
    }
}

#[pyclass]
pub struct PyResolver(Arc<dcdf::Resolver>);

#[pymethods]
impl PyResolver {
    #[new]
    pub fn new(cache_bytes: u64) -> Self {
        Self(Arc::new(dcdf::Resolver::new(
            Box::new(dcdf_ipfs::IpfsMapper::new()),
            cache_bytes,
        )))
    }

    pub fn get_dataset(&self, cid: &str) -> PyResult<PyDataset> {
        Ok(PyDataset(block_on_result(
            self.0.get_dataset(&parse_cid(cid)?),
        )?))
    }

    pub fn ls(&self, cid: &str) -> PyResult<Vec<PyLsEntry>> {
        let ls = block_on_result(self.0.ls(&parse_cid(cid)?))?;

        Ok(ls.into_iter().map(|entry| PyLsEntry::from(entry)).collect())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCoordinate(dcdf::Coordinate);

#[pymethods]
impl PyCoordinate {
    #[staticmethod]
    pub fn range_i32(name: &str, start: i32, step: i32, steps: usize) -> Self {
        Self(dcdf::Coordinate::range_i32(name, start, step, steps))
    }

    #[staticmethod]
    pub fn range_i64(name: &str, start: i64, step: i64, steps: usize) -> Self {
        Self(dcdf::Coordinate::range_i64(name, start, step, steps))
    }

    #[staticmethod]
    pub fn range_f32(name: &str, start: f32, step: f32, steps: usize) -> Self {
        Self(dcdf::Coordinate::range_f32(name, start, step, steps))
    }

    #[staticmethod]
    pub fn range_f64(name: &str, start: f64, step: f64, steps: usize) -> Self {
        Self(dcdf::Coordinate::range_f64(name, start, step, steps))
    }

    #[staticmethod]
    pub fn time(name: &str, start: i64, step: i64) -> Self {
        Self(dcdf::Coordinate::time(name, start, step))
    }

    #[getter]
    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    pub fn encoding(&self) -> usize {
        let encoding = match self.0.kind {
            dcdf::CoordinateKind::Time(_) => MMEncoding::Time,
            dcdf::CoordinateKind::I32(_) => MMEncoding::I32,
            dcdf::CoordinateKind::I64(_) => MMEncoding::I64,
            dcdf::CoordinateKind::F32(_) => MMEncoding::F32,
            dcdf::CoordinateKind::F64(_) => MMEncoding::F64,
        };

        encoding as usize
    }

    pub fn len(&self) -> PyResult<usize> {
        self.0.len().map_err(convert_error)
    }

    pub fn data_time(&self) -> PyTimeRange {
        PyTimeRange(self.0.data_time().clone())
    }

    pub fn data_i32(&self) -> PyMMArray1I32 {
        PyMMArray1I32(self.0.data_i32().clone())
    }

    pub fn data_i64(&self) -> PyMMArray1I64 {
        PyMMArray1I64(self.0.data_i64().clone())
    }

    pub fn data_f32(&self) -> PyMMArray1F32 {
        PyMMArray1F32(self.0.data_f32().clone())
    }

    pub fn data_f64(&self) -> PyMMArray1F64 {
        PyMMArray1F64(self.0.data_f64().clone())
    }
}

#[pyclass]
pub struct PyTimeRange(dcdf::TimeRange);

#[pymethods]
impl PyTimeRange {
    pub fn get(&self, index: usize) -> i64 {
        self.0.get(index)
    }

    pub fn slice<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
    ) -> &'py PyArray1<Datetime<units::Seconds>> {
        let slice = self.0.slice(start, stop);

        slice.map(|n| Datetime::from(*n)).into_pyarray(py)
    }
}

#[pyclass]
pub struct PyMMArray1I32(dcdf::MMArray1I32);

#[pymethods]
impl PyMMArray1I32 {
    pub fn get(&self, index: usize) -> i32 {
        self.0.get(index)
    }

    pub fn slice<'py>(&self, py: Python<'py>, start: usize, stop: usize) -> &'py PyArray1<i32> {
        self.0.slice(start, stop).into_pyarray(py)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn shape(&self) -> [usize; 1] {
        self.0.shape()
    }
}

#[pyclass]
pub struct PyMMArray1I64(dcdf::MMArray1I64);

#[pymethods]
impl PyMMArray1I64 {
    pub fn get(&self, index: usize) -> i64 {
        self.0.get(index)
    }

    pub fn slice<'py>(&self, py: Python<'py>, start: usize, stop: usize) -> &'py PyArray1<i64> {
        self.0.slice(start, stop).into_pyarray(py)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn shape(&self) -> [usize; 1] {
        self.0.shape()
    }
}

#[pyclass]
pub struct PyMMArray1F32(dcdf::MMArray1F32);

#[pymethods]
impl PyMMArray1F32 {
    pub fn get(&self, index: usize) -> f32 {
        self.0.get(index)
    }

    pub fn slice<'py>(&self, py: Python<'py>, start: usize, stop: usize) -> &'py PyArray1<f32> {
        self.0.slice(start, stop).into_pyarray(py)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn shape(&self) -> [usize; 1] {
        self.0.shape()
    }
}

#[pyclass]
pub struct PyMMArray1F64(dcdf::MMArray1F64);

#[pymethods]
impl PyMMArray1F64 {
    pub fn get(&self, index: usize) -> f64 {
        self.0.get(index)
    }

    pub fn slice<'py>(&self, py: Python<'py>, start: usize, stop: usize) -> &'py PyArray1<f64> {
        self.0.slice(start, stop).into_pyarray(py)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn shape(&self) -> [usize; 1] {
        self.0.shape()
    }
}

#[pyclass]
pub struct PyVariable(dcdf::Variable);

#[pymethods]
impl PyVariable {
    #[getter]
    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    #[getter]
    pub fn round(&self) -> bool {
        self.0.round.is_some()
    }

    #[getter]
    pub fn fractional_bits(&self) -> usize {
        match self.0.round {
            Some(bits) => bits,
            None => 0,
        }
    }

    #[getter]
    pub fn span_size(&self) -> usize {
        self.0.span_size
    }

    #[getter]
    pub fn chunk_size(&self) -> usize {
        self.0.chunk_size
    }

    #[getter]
    pub fn k2_levels(&self) -> Vec<u32> {
        self.0.k2_levels.clone()
    }

    #[getter]
    pub fn encoding(&self) -> usize {
        self.0.encoding as usize
    }

    #[getter]
    pub fn cid(&self) -> String {
        self.0.cid.to_string()
    }

    pub fn data_i32(&self) -> PyResult<PyMMArray3I32> {
        Ok(PyMMArray3I32(block_on_result(self.0.data_i32())?.clone()))
    }

    pub fn data_i64(&self) -> PyResult<PyMMArray3I64> {
        Ok(PyMMArray3I64(block_on_result(self.0.data_i64())?.clone()))
    }

    pub fn data_f32(&self) -> PyResult<PyMMArray3F32> {
        Ok(PyMMArray3F32(block_on_result(self.0.data_f32())?.clone()))
    }

    pub fn data_f64(&self) -> PyResult<PyMMArray3F64> {
        Ok(PyMMArray3F64(block_on_result(self.0.data_f64())?.clone()))
    }
}

#[pyclass]
pub struct PyMMArray3I32(dcdf::MMArray3I32);

#[pymethods]
impl PyMMArray3I32 {
    pub fn shape(&self) -> [usize; 3] {
        self.0.shape()
    }

    pub fn get(&self, instant: usize, row: usize, col: usize) -> PyResult<i32> {
        block_on_result(self.0.get(instant, row, col))
    }

    pub fn cell<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> PyResult<&'py PyArray1<i32>> {
        let array = block_on_result(self.0.cell(start, end, row, col))?;

        Ok(array.to_pyarray(py))
    }

    pub fn window<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> PyResult<&'py PyArray3<i32>> {
        let bounds = geom::Cube::new(start, end, top, bottom, left, right);
        let array = block_on_result(self.0.window(bounds))?;

        Ok(array.to_pyarray(py))
    }
}

#[pyclass]
pub struct PyMMArray3I64(dcdf::MMArray3I64);

#[pymethods]
impl PyMMArray3I64 {
    pub fn shape(&self) -> [usize; 3] {
        self.0.shape()
    }

    pub fn get(&self, instant: usize, row: usize, col: usize) -> PyResult<i64> {
        block_on_result(self.0.get(instant, row, col))
    }

    pub fn cell<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> PyResult<&'py PyArray1<i64>> {
        let array = block_on_result(self.0.cell(start, end, row, col))?;

        Ok(array.to_pyarray(py))
    }

    pub fn window<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> PyResult<&'py PyArray3<i64>> {
        let bounds = geom::Cube::new(start, end, top, bottom, left, right);
        let array = block_on_result(self.0.window(bounds))?;

        Ok(array.to_pyarray(py))
    }
}

#[pyclass]
pub struct PyMMArray3F32(dcdf::MMArray3F32);

#[pymethods]
impl PyMMArray3F32 {
    pub fn shape(&self) -> [usize; 3] {
        self.0.shape()
    }

    pub fn get(&self, instant: usize, row: usize, col: usize) -> PyResult<f32> {
        block_on_result(self.0.get(instant, row, col))
    }

    pub fn cell<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let array = block_on_result(self.0.cell(start, end, row, col))?;

        Ok(array.to_pyarray(py))
    }

    pub fn window<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> PyResult<&'py PyArray3<f32>> {
        let bounds = geom::Cube::new(start, end, top, bottom, left, right);
        let array = block_on_result(self.0.window(bounds))?;

        Ok(array.to_pyarray(py))
    }
}

#[pyclass]
pub struct PyMMArray3F64(dcdf::MMArray3F64);

#[pymethods]
impl PyMMArray3F64 {
    pub fn shape(&self) -> [usize; 3] {
        self.0.shape()
    }

    pub fn get(&self, instant: usize, row: usize, col: usize) -> PyResult<f64> {
        block_on_result(self.0.get(instant, row, col))
    }

    pub fn cell<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> PyResult<&'py PyArray1<f64>> {
        let array = block_on_result(self.0.cell(start, end, row, col))?;

        Ok(array.to_pyarray(py))
    }

    pub fn window<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> PyResult<&'py PyArray3<f64>> {
        let bounds = geom::Cube::new(start, end, top, bottom, left, right);
        let array = block_on_result(self.0.window(bounds))?;

        Ok(array.to_pyarray(py))
    }
}

#[pyclass]
pub struct PyLsEntry {
    #[pyo3(get)]
    pub cid: String,

    #[pyo3(get)]
    pub name: String,

    #[pyo3(get)]
    pub node_type: Option<&'static str>,

    #[pyo3(get)]
    pub size: Option<u64>,
}

impl From<LsEntry> for PyLsEntry {
    fn from(entry: LsEntry) -> Self {
        Self {
            cid: entry.cid.to_string(),
            name: entry.name,
            node_type: entry.node_type,
            size: entry.size,
        }
    }
}

#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyResolver>()?;
    m.add_class::<PyDataset>()?;
    m.add_class::<PyCoordinate>()?;
    m.add_class::<PyMMArray1I32>()?;
    m.add_class::<PyMMArray1I64>()?;
    m.add_class::<PyMMArray1F32>()?;
    m.add_class::<PyMMArray1F64>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyMMArray3I32>()?;
    m.add_class::<PyMMArray3I64>()?;
    m.add_class::<PyMMArray3F32>()?;
    m.add_class::<PyMMArray3F64>()?;

    Ok(())
}

fn block_on_result<F, T>(task: F) -> PyResult<T>
where
    F: Future<Output = dcdf::Result<T>>,
{
    let rt = Runtime::new().unwrap();
    rt.block_on(task).map_err(convert_error)
}

fn parse_cid(cid: &str) -> PyResult<Cid> {
    Cid::from_str(cid)
        .map_err(|e| dcdf::Error::Cid(e))
        .map_err(convert_error)
}

fn convert_error(cause: dcdf::Error) -> PyErr {
    match cause {
        dcdf::Error::Io(cause) => cause.into(),
        dcdf::Error::Cid(cause) => PyValueError::new_err(format!("{cause}")),
        dcdf::Error::UnsignedVarint(cause) => PyIOError::new_err(format!("{cause}")),
        dcdf::Error::Load => PyIOError::new_err("unable to load object"),
        dcdf::Error::BadValue => PyValueError::new_err("bad value"), // TODO get the value
        dcdf::Error::BadName(name) => PyNameError::new_err(format!("Name error: {name}")),
        dcdf::Error::NotFound(cid) => PyKeyError::new_err(format!("Could not find cid: {cid}")),
        dcdf::Error::TimeIsInfinite => PyValueError::new_err("time coordinates are infinite"),
    }
}
