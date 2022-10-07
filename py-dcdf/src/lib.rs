mod simple;

pub use simple::{
    load, load_from, PyBuildF32, PyBuildI32, PyBuilderF32, PyBuilderI32, PyChunkF32, PyChunkI32,
    PyFractionSuggesterF32,
};

mod dag;

pub use dag::{
    commit_f32, new_ipfs_resolver, PyCommitF32, PyFolderF32, PyFolderItem, PyResolverF32,
    PySuperchunkBuilderF32, PySuperchunkF32,
};

use pyo3::prelude::*;

#[pymodule]
fn _dcdf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_from, m)?)?;

    m.add_class::<PyBuilderI32>()?;
    m.add_class::<PyBuildI32>()?;
    m.add_class::<PyChunkI32>()?;

    m.add_class::<PyBuilderF32>()?;
    m.add_class::<PyBuildF32>()?;
    m.add_class::<PyChunkF32>()?;
    m.add_class::<PyFractionSuggesterF32>()?;

    m.add_class::<PyCommitF32>()?;
    m.add_class::<PyFolderF32>()?;
    m.add_class::<PyResolverF32>()?;
    m.add_class::<PySuperchunkF32>()?;
    m.add_class::<PySuperchunkBuilderF32>()?;
    m.add_class::<PyFolderItem>()?;

    m.add_function(wrap_pyfunction!(commit_f32, m)?)?;
    m.add_function(wrap_pyfunction!(new_ipfs_resolver, m)?)?;

    Ok(())
}
