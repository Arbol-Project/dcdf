use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use dcdf;

#[pyclass]
struct PySnapshot32 {
    inner: dcdf::Snapshot<i32>,
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
    inner: dcdf::Snapshot<u32>,
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
    inner: dcdf::Snapshot<i64>,
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
    inner: dcdf::Snapshot<u64>,
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

#[pyclass]
struct PyLog32 {
    inner: dcdf::Log<i32>,
}

#[pymethods]
impl PyLog32 {
    #[new]
    fn new(snapshot: PyReadonlyArray2<i32>, log: PyReadonlyArray2<i32>, k: i32) -> Self {
        let snapshot = snapshot.as_array();
        let log = log.as_array();

        PyLog32 {
            inner: dcdf::Log::from_arrays(snapshot, log, k),
        }
    }

    fn get(&self, snapshot: &PySnapshot32, row: usize, col: usize) -> i32 {
        self.inner.get(&snapshot.inner, row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        snapshot: &PySnapshot32,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<i32> {
        self.inner
            .get_window(&snapshot.inner, top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        snapshot: &PySnapshot32,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i32,
        upper: i32,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(&snapshot.inner, top, bottom, left, right, lower, upper)
    }
}

#[pyclass]
struct PyLogU32 {
    inner: dcdf::Log<u32>,
}

#[pymethods]
impl PyLogU32 {
    #[new]
    fn new(snapshot: PyReadonlyArray2<u32>, log: PyReadonlyArray2<u32>, k: i32) -> Self {
        let snapshot = snapshot.as_array();
        let log = log.as_array();

        PyLogU32 {
            inner: dcdf::Log::from_arrays(snapshot, log, k),
        }
    }

    fn get(&self, snapshot: &PySnapshotU32, row: usize, col: usize) -> u32 {
        self.inner.get(&snapshot.inner, row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        snapshot: &PySnapshotU32,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<u32> {
        self.inner
            .get_window(&snapshot.inner, top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        snapshot: &PySnapshotU32,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: u32,
        upper: u32,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(&snapshot.inner, top, bottom, left, right, lower, upper)
    }
}


#[pyclass]
struct PyLog64 {
    inner: dcdf::Log<i64>,
}

#[pymethods]
impl PyLog64 {
    #[new]
    fn new(snapshot: PyReadonlyArray2<i64>, log: PyReadonlyArray2<i64>, k: i32) -> Self {
        let snapshot = snapshot.as_array();
        let log = log.as_array();

        PyLog64 {
            inner: dcdf::Log::from_arrays(snapshot, log, k),
        }
    }

    fn get(&self, snapshot: &PySnapshot64, row: usize, col: usize) -> i64 {
        self.inner.get(&snapshot.inner, row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        snapshot: &PySnapshot64,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<i64> {
        self.inner
            .get_window(&snapshot.inner, top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        snapshot: &PySnapshot64,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i64,
        upper: i64,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(&snapshot.inner, top, bottom, left, right, lower, upper)
    }
}

#[pyclass]
struct PyLogU64 {
    inner: dcdf::Log<u64>,
}

#[pymethods]
impl PyLogU64 {
    #[new]
    fn new(snapshot: PyReadonlyArray2<u64>, log: PyReadonlyArray2<u64>, k: i32) -> Self {
        let snapshot = snapshot.as_array();
        let log = log.as_array();

        PyLogU64 {
            inner: dcdf::Log::from_arrays(snapshot, log, k),
        }
    }

    fn get(&self, snapshot: &PySnapshotU64, row: usize, col: usize) -> u64 {
        self.inner.get(&snapshot.inner, row, col)
    }

    fn get_window<'py>(
        &self,
        py: Python<'py>,
        snapshot: &PySnapshotU64,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    ) -> &'py PyArray2<u64> {
        self.inner
            .get_window(&snapshot.inner, top, bottom, left, right)
            .to_pyarray(py)
    }

    fn search_window(
        &self,
        snapshot: &PySnapshotU64,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: u64,
        upper: u64,
    ) -> Vec<(usize, usize)> {
        self.inner
            .search_window(&snapshot.inner, top, bottom, left, right, lower, upper)
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySnapshot32>()?;
    m.add_class::<PySnapshotU32>()?;
    m.add_class::<PySnapshot64>()?;
    m.add_class::<PySnapshotU64>()?;
    m.add_class::<PyLog32>()?;
    m.add_class::<PyLogU32>()?;
    m.add_class::<PyLog64>()?;
    m.add_class::<PyLogU64>()?;
    Ok(())
}
