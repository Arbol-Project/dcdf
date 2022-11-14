use std::mem;
use std::str::FromStr;
use std::sync::Arc;

use cid::Cid;
use ndarray::Array;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray2};

use pyo3::exceptions::{PyIndexError, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use dcdf;
use dcdf_ipfs::IpfsMapper;

use super::helpers::convert_error;

#[pyfunction]
pub fn new_ipfs_resolver_f32(cache_bytes: u64) -> PyResolverF32 {
    let mapper = Box::new(IpfsMapper::new());
    PyResolverF32 {
        inner: Arc::new(dcdf::Resolver::new(mapper, cache_bytes)),
    }
}

#[pyclass]
pub struct PyResolverF32 {
    inner: Arc<dcdf::Resolver<f32>>,
}

#[pymethods]
impl PyResolverF32 {
    fn get_folder(&self, cid: &str) -> PyResult<PyFolderF32> {
        let cid = Cid::from_str(cid).expect("Invalid cid");
        Ok(PyFolderF32::wrap(
            self.inner.get_folder(&cid).map_err(convert_error)?,
        ))
    }

    fn get_commit(&self, cid: &str) -> PyResult<PyCommitF32> {
        let cid = Cid::from_str(cid).expect("Invalid cid");
        Ok(PyCommitF32::wrap(
            self.inner.get_commit(&cid).map_err(convert_error)?,
        ))
    }

    fn get_superchunk(&self, cid: &str) -> PyResult<PySuperchunkF32> {
        let cid = Cid::from_str(cid).expect("Invalid cid");
        Ok(PySuperchunkF32::wrap(
            self.inner.get_superchunk(&cid).map_err(convert_error)?,
        ))
    }

    pub fn load_object<'py>(&self, py: Python<'py>, cid: &str) -> PyResult<Option<PyObject>> {
        let cid = Cid::from_str(cid).expect("Invalid cid");
        Ok(match self.inner.load(&cid) {
            Some(mut stream) => {
                let mut object = Vec::new();
                stream.read_to_end(&mut object)?;

                Some(PyBytes::new(py, &object).into())
            }
            None => None,
        })
    }

    pub fn store_object(&self, object: &[u8]) -> PyResult<String> {
        let mut stream = self.inner.store();
        stream.write_all(object)?;

        Ok(stream.finish().to_string())
    }

    pub fn insert(&self, root: Option<&str>, path: &str, object: &str) -> PyResult<String> {
        let root = match root {
            None => Arc::new(dcdf::Folder::new(&self.inner)),
            Some(cid) => {
                let cid = Cid::from_str(cid).expect("Invalid cid");
                self.inner.get_folder(&cid).map_err(convert_error)?
            }
        };

        let object = Cid::from_str(object).expect("Invalid cid");
        let root = self._insert(root, path, object).map_err(convert_error)?;

        Ok(root.to_string())
    }

    pub fn commit(
        &self,
        message: &str,
        root: Option<&str>,
        prev: Option<&str>,
    ) -> PyResult<String> {
        let root = match root {
            Some(root) => Cid::from_str(root).expect("Invalid cid"),
            None => self
                .inner
                .save(dcdf::Folder::new(&self.inner))
                .map_err(convert_error)?,
        };
        let prev = match prev {
            Some(cid) => Some(Cid::from_str(cid).expect("Invalid cid")),
            None => None,
        };
        let commit = dcdf::Commit::new(message, root, prev, &self.inner);
        let cid = self.inner.save(commit).map_err(convert_error)?;

        Ok(cid.to_string())
    }
}

impl PyResolverF32 {
    fn _insert(
        &self,
        folder: Arc<dcdf::Folder<f32>>,
        path: &str,
        object: Cid,
    ) -> dcdf::Result<Cid> {
        let folder = match path.split_once("/") {
            None => folder.update(path, object),
            Some((name, rest)) => {
                let child = match folder.get(name) {
                    None => Arc::new(dcdf::Folder::new(&self.inner)),
                    Some(cid) => self.inner.get_folder(&cid)?,
                };
                folder.update(name, self._insert(child, rest, object)?)
            }
        };

        self.inner.save(folder)
    }
}

#[pyclass]
pub struct PyFolderF32 {
    inner: Arc<dcdf::Folder<f32>>,
}

impl PyFolderF32 {
    fn wrap(inner: Arc<dcdf::Folder<f32>>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFolderF32 {
    fn get(&self, key: &str) -> Option<String> {
        match self.inner.get(key) {
            Some(cid) => Some(cid.to_string()),
            None => None,
        }
    }

    fn __getitem__(&self, key: &str) -> PyResult<String> {
        match self.get(key) {
            Some(item) => Ok(item),
            None => Err(PyKeyError::new_err(format!("{}", key))),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        self.inner.get(key).is_some()
    }
}

#[pyclass]
pub struct PyCommitF32 {
    inner: Arc<dcdf::Commit<f32>>,
}

impl PyCommitF32 {
    fn wrap(inner: Arc<dcdf::Commit<f32>>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCommitF32 {
    #[getter]
    fn message(&self) -> &str {
        self.inner.message()
    }

    #[getter]
    fn prev(&self) -> PyResult<Option<Self>> {
        let prev = match self.inner.prev().map_err(convert_error)? {
            Some(inner) => Some(Self { inner }),
            None => None,
        };

        Ok(prev)
    }

    #[getter]
    fn root(&self) -> PyFolderF32 {
        PyFolderF32::wrap(self.inner.root())
    }

    #[getter]
    fn root_cid(&self) -> String {
        self.inner.root.to_string()
    }
}

#[pyclass]
pub struct PySuperchunkBuilderF32 {
    resolver: Arc<dcdf::Resolver<f32>>,
    inner: Option<dcdf::SuperchunkBuilder<f32>>,
}

#[pymethods]
impl PySuperchunkBuilderF32 {
    #[new]
    fn new(
        first: PyReadonlyArray2<f32>,
        k: i32,
        fraction: usize,
        round: bool,
        levels: usize,
        resolver: &PyResolverF32,
        local_threshold: u64,
    ) -> Self {
        let fraction = if round {
            dcdf::Round(fraction)
        } else {
            dcdf::Precise(fraction)
        };
        let first = first.to_owned_array();
        let resolver = Arc::clone(&resolver.inner);
        Self {
            resolver: Arc::clone(&resolver),
            inner: Some(dcdf::SuperchunkBuilder::new(
                first,
                k,
                fraction,
                levels,
                resolver,
                local_threshold,
            )),
        }
    }

    fn push(&mut self, instant: PyReadonlyArray2<f32>) {
        let instant = instant.to_owned_array();
        if let Some(inner) = &mut self.inner {
            inner.push(instant);
        }
    }

    fn finish(&mut self) -> PyResult<String> {
        let build = mem::replace(&mut self.inner, None).expect("finish called twice");
        let chunk = build.finish().map_err(convert_error)?;
        let cid = self.resolver.save(chunk).map_err(convert_error)?;

        Ok(cid.to_string())
    }
}

#[pyclass]
pub struct PySuperchunkF32 {
    inner: Arc<dcdf::Superchunk<f32>>,
}

impl PySuperchunkF32 {
    fn wrap(inner: Arc<dcdf::Superchunk<f32>>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySuperchunkF32 {
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let [instants, rows, cols] = self.inner.shape();
        (instants, rows, cols)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: Vec<usize>) -> PyResult<PyObject> {
        if let [instant, row, col] = key[0..3] {
            Ok(self.get(instant, row, col)?.to_object(py))
        } else {
            Err(PyIndexError::new_err(
                "Wrong number of coordinates for three dimensional array",
            ))
        }
    }

    fn get(&self, instant: usize, row: usize, col: usize) -> PyResult<f32> {
        self.inner.get(instant, row, col).map_err(convert_error)
    }

    fn cell<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut a = Array::zeros(end - start);
        let cells = self
            .inner
            .iter_cell(start, end, row, col)
            .map_err(convert_error)?;
        for (i, n) in cells.enumerate() {
            a[i] = n;
        }
        Ok(a.into_pyarray(py))
    }

    fn window<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> PyResult<&'py PyArray3<f32>> {
        let bounds = dcdf::Cube::new(start, end, top, bottom, left, right);
        let a = self.inner.get_window(&bounds).map_err(convert_error)?;

        Ok(a.into_pyarray(py))
    }

    fn search(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: f32,
        upper: f32,
    ) -> PyResult<Vec<(usize, usize, usize)>> {
        let bounds = dcdf::Cube::new(start, end, top, bottom, left, right);
        let results = self
            .inner
            .iter_search(&bounds, lower, upper)
            .map_err(convert_error)?;
        let mut py_results = Vec::new();
        for result in results {
            py_results.push(result.map_err(convert_error)?);
        }

        Ok(py_results)
    }
}
