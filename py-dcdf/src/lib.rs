use ndarray::Array;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use std::mem;

use dcdf;

#[pyclass]
struct PyBuilderI32 {
    inner: Option<dcdf::Builder<i32>>,
}

#[pymethods]
impl PyBuilderI32 {
    #[new]
    fn new(first: PyReadonlyArray2<i32>, k: i32) -> Self {
        let first = first.to_owned_array();
        Self {
            inner: Some(dcdf::Builder::new(first, k)),
        }
    }

    fn push(&mut self, instant: PyReadonlyArray2<i32>) {
        let instant = instant.to_owned_array();
        if let Some(inner) = &mut self.inner {
            inner.push(instant);
        }
    }

    fn finish(&mut self, py: Python) -> PyResult<PyBuildI32> {
        let inner = mem::replace(&mut self.inner, None);
        match inner {
            Some(build) => PyBuildI32::new(py, build.finish()),
            None => panic!("finish called twice"),
        }
    }
}

#[pyclass]
struct PyBuildI32 {
    #[pyo3(get)]
    pub data: Py<PyChunkI32>,

    #[pyo3(get)]
    pub logs: usize,

    #[pyo3(get)]
    pub snapshots: usize,

    #[pyo3(get)]
    pub compression: f32,
}

impl PyBuildI32 {
    fn new(py: Python, build: dcdf::Build<i32>) -> PyResult<Self> {
        Ok(Self {
            data: Py::new(py, PyChunkI32::new(build.data))?,
            logs: build.logs,
            snapshots: build.snapshots,
            compression: build.compression,
        })
    }
}

#[pyclass]
struct PyChunkI32 {
    inner: dcdf::Chunk<i32>,
}

impl PyChunkI32 {
    fn new(inner: dcdf::Chunk<i32>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyChunkI32 {
    // The original plan here was to wrap and expose the iterator implementations in dcdf::codec,
    // but it turns out that's really hard to do in the current state of pyo3. The crux of the
    // issue is that because the iterators hold a reference to the Chunk, they need a lifetime
    // annotation to indicate that they can't outlive the Chunk. Since Python has no concept of
    // lifetimes or ownership, pyo3 doesn't support lifetime annotations. There is a lot of detail
    // about this issue, here, and you can see the pyo3 developers are working towards a solution:
    //
    // https://github.com/PyO3/pyo3/issues/1085
    //

    fn cell<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        end: usize,
        row: usize,
        col: usize,
    ) -> &'py PyArray1<i32> {
        let mut a = Array::zeros(end - start);
        for (i, n) in self.inner.iter_cell(start, end, row, col).enumerate() {
            a[i] = n;
        }
        a.into_pyarray(py)
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
    ) -> &'py PyArray3<i32> {
        let rows = bottom - top;
        let cols = right - left;
        let mut a = Array::zeros((end - start, rows, cols));
        for (i, w) in self
            .inner
            .iter_window(start, end, top, bottom, left, right)
            .enumerate()
        {
            // There must be a better way to do this
            for row in 0..rows {
                for col in 0..cols {
                    a[[i, row, col]] = w[[row, col]];
                }
            }
        }
        a.into_pyarray(py)
    }

    fn search(
        &self,
        start: usize,
        end: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        lower: i32,
        upper: i32,
    ) -> Vec<Vec<(usize, usize)>> {
        self.inner
            .iter_search(start, end, top, bottom, left, right, lower, upper)
            .collect()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBuilderI32>()?;
    m.add_class::<PyBuildI32>()?;
    m.add_class::<PyChunkI32>()?;
    Ok(())
}
