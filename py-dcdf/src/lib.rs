use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use dcdf;

#[pyclass]
struct PySnapshot32 {
    inner: dcdf::Snapshot,
}

#[pymethods]
impl PySnapshot32 {
    #[new]
    fn new(data: PyReadonlyArray2<i32>, k: i32) -> Self {
        let data = data.as_array();
        PySnapshot32 {
            inner: dcdf::Snapshot::from_array(data, k),
        }
    }

    fn get(&self, row: usize, col: usize) -> i32 {
        self.inner.get(row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<i32> {
        self.inner
            .get_window(top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i32,
        upper: i32,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(top, bottom, left, right, lower, upper)
    }
}

#[pyclass]
struct PySnapshotU32 {
    inner: dcdf::Snapshot,
}

#[pymethods]
impl PySnapshotU32 {
    #[new]
    fn new(data: PyReadonlyArray2<u32>, k: i32) -> Self {
        let data = data.as_array();
        PySnapshotU32 {
            inner: dcdf::Snapshot::from_array(data, k),
        }
    }

    fn get(&self, row: usize, col: usize) -> u32 {
        self.inner.get(row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<u32> {
        self.inner
            .get_window(top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: u32,
        upper: u32,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(top, bottom, left, right, lower, upper)
    }
}

#[pyclass]
struct PySnapshot64 {
    inner: dcdf::Snapshot,
}

#[pymethods]
impl PySnapshot64 {
    #[new]
    fn new(data: PyReadonlyArray2<i64>, k: i32) -> Self {
        let data = data.as_array();
        PySnapshot64 {
            inner: dcdf::Snapshot::from_array(data, k),
        }
    }

    fn get(&self, row: usize, col: usize) -> i64 {
        self.inner.get(row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<i64> {
        self.inner
            .get_window(top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i64,
        upper: i64,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(top, bottom, left, right, lower, upper)
    }
}

#[pyclass]
struct PySnapshotU64 {
    inner: dcdf::Snapshot,
}

#[pymethods]
impl PySnapshotU64 {
    #[new]
    fn new(data: PyReadonlyArray2<u64>, k: i32) -> Self {
        let data = data.as_array();
        PySnapshotU64 {
            inner: dcdf::Snapshot::from_array(data, k),
        }
    }

    fn get(&self, row: usize, col: usize) -> u64 {
        self.inner.get(row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<u64> {
        self.inner
            .get_window(top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: u64,
        upper: u64,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(top, bottom, left, right, lower, upper)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySnapshot32>()?;
    m.add_class::<PySnapshotU32>()?;
    m.add_class::<PySnapshot64>()?;
    m.add_class::<PySnapshotU64>()?;
    Ok(())
}
