use ndarray::Array;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use std::fs::File;
use std::mem;
use std::path::Path;

use dcdf;

// =================== Integers ==============

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

    fn finish(&mut self) -> PyResult<PyBuildI32> {
        let inner = mem::replace(&mut self.inner, None);
        match inner {
            Some(build) => PyBuildI32::new(build.finish()),
            None => panic!("finish called twice"),
        }
    }
}

#[pyclass]
struct PyBuildI32 {
    #[pyo3(get)]
    pub logs: usize,

    #[pyo3(get)]
    pub snapshots: usize,

    #[pyo3(get)]
    pub compression: f32,

    inner: dcdf::Build<i32>,
}

impl PyBuildI32 {
    fn new(build: dcdf::Build<i32>) -> PyResult<Self> {
        Ok(Self {
            logs: build.logs,
            snapshots: build.snapshots,
            compression: build.compression,
            inner: build,
        })
    }
}

#[pymethods]
impl PyBuildI32 {
    fn save(&self, path: &str) -> PyResult<()> {
        let path = Path::new(path);
        let mut file = File::create(&path)?;
        self.inner.save(&mut file)?;

        Ok(())
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

// =================== Floats ==============

#[pyclass]
struct PyBuilderF32 {
    inner: Option<dcdf::FBuilder<f32>>,
}

#[pymethods]
impl PyBuilderF32 {
    #[new]
    fn new(first: PyReadonlyArray2<f32>, k: i32, fraction: usize, round: bool) -> Self {
        let fraction = if round {
            dcdf::Round(fraction)
        } else {
            dcdf::Precise(fraction)
        };
        let first = first.to_owned_array();
        Self {
            inner: Some(dcdf::FBuilder::new(first, k, fraction)),
        }
    }

    fn push(&mut self, instant: PyReadonlyArray2<f32>) {
        let instant = instant.to_owned_array();
        if let Some(inner) = &mut self.inner {
            inner.push(instant);
        }
    }

    fn finish(&mut self) -> PyResult<PyBuildF32> {
        let inner = mem::replace(&mut self.inner, None);
        match inner {
            Some(build) => PyBuildF32::new(build.finish()),
            None => panic!("finish called twice"),
        }
    }
}

#[pyclass]
struct PyBuildF32 {
    #[pyo3(get)]
    pub logs: usize,

    #[pyo3(get)]
    pub snapshots: usize,

    #[pyo3(get)]
    pub compression: f32,

    inner: dcdf::FBuild<f32>,
}

impl PyBuildF32 {
    fn new(build: dcdf::FBuild<f32>) -> PyResult<Self> {
        Ok(Self {
            logs: build.logs,
            snapshots: build.snapshots,
            compression: build.compression,
            inner: build,
        })
    }
}

#[pymethods]
impl PyBuildF32 {
    fn save(&self, path: &str) -> PyResult<()> {
        let path = Path::new(path);
        let mut file = File::create(&path)?;
        self.inner.save(&mut file)?;

        Ok(())
    }
}

#[pyclass]
struct PyChunkF32 {
    inner: dcdf::FChunk<f32>,
}

impl PyChunkF32 {
    fn new(inner: dcdf::FChunk<f32>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyChunkF32 {
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
    ) -> &'py PyArray1<f32> {
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
    ) -> &'py PyArray3<f32> {
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
        lower: f32,
        upper: f32,
    ) -> Vec<Vec<(usize, usize)>> {
        self.inner
            .iter_search(start, end, top, bottom, left, right, lower, upper)
            .collect()
    }
}

#[pyclass]
struct PyFractionSuggesterF32 {
    inner: Option<dcdf::FractionSuggester<f32>>,
}

#[pymethods]
impl PyFractionSuggesterF32 {
    #[new]
    fn new(max_value: f32) -> Self {
        Self {
            inner: Some(dcdf::FractionSuggester::new(max_value)),
        }
    }

    fn push(&mut self, instant: PyReadonlyArray2<f32>) -> bool {
        let instant = instant.to_owned_array();
        match &mut self.inner {
            Some(inner) => match inner.push(instant) {
                dcdf::Continue::Yes => true,
                dcdf::Continue::No => false,
            },
            None => false,
        }
    }

    fn finish(&mut self) -> PyResult<(usize, bool)> {
        let inner = mem::replace(&mut self.inner, None);
        match inner {
            Some(inner) => match inner.finish() {
                dcdf::Precise(fraction_bits) => Ok((fraction_bits, false)),
                dcdf::Round(fraction_bits) => Ok((fraction_bits, true)),
            },
            None => panic!("finish called twice"),
        }
    }
}

#[pyfunction]
fn load(py: Python, path: &str) -> PyResult<PyObject> {
    let path = Path::new(path);
    let mut file = File::open(&path)?;
    let inner = dcdf::load(&mut file)?;
    let chunk = match inner {
        dcdf::I32(chunk) => Py::new(py, PyChunkI32::new(chunk))?.to_object(py),
        dcdf::F32(chunk) => Py::new(py, PyChunkF32::new(chunk))?.to_object(py),
        _ => panic!("Unsupported data type."),
    };

    Ok(chunk)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;

    m.add_class::<PyBuilderI32>()?;
    m.add_class::<PyBuildI32>()?;
    m.add_class::<PyChunkI32>()?;

    m.add_class::<PyBuilderF32>()?;
    m.add_class::<PyBuildF32>()?;
    m.add_class::<PyChunkF32>()?;
    m.add_class::<PyFractionSuggesterF32>()?;

    Ok(())
}
