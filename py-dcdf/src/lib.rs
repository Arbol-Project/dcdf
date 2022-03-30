use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use dcdf;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn encode_snapshot(_py: Python, data: PyReadonlyArray2<i32>) -> PyResult<Vec<u8>> {
    let data = data.as_array();
    Ok(dcdf::encode_snapshot(data))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(encode_snapshot, m)?)?;
    Ok(())
}
