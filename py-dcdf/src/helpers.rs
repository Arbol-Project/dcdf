use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use dcdf;

pub(crate) fn convert_error(cause: dcdf::Error) -> PyErr {
    match cause {
        dcdf::Error::IO(cause) => cause.into(),
        dcdf::Error::CID(cause) => PyValueError::new_err(format!("{cause}")),
    }
}
