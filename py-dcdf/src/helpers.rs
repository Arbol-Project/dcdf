use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use dcdf;

pub(crate) fn convert_error(cause: dcdf::Error) -> PyErr {
    match cause {
        dcdf::Error::Io(cause) => cause.into(),
        dcdf::Error::Cid(cause) => PyValueError::new_err(format!("{cause}")),
        dcdf::Error::UnsignedVarint(cause) => PyIOError::new_err(format!("{cause}")),
        dcdf::Error::Load => PyIOError::new_err("unable to load object"),
    }
}
