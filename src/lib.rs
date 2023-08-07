mod basis_iterator;
mod basis_hashing;
mod search;
mod hamiltonian;
mod basis;

use hamiltonian::{
    Hamiltonian,
    prepare_state,
    hilbert_space_dimension,
};
use basis::{
  get_basis,
  particles_per_mode,
  n_bosons_weight,
};

use pyo3::{
    pymodule,
    Python,
    PyResult,
    wrap_pyfunction,
};
use pyo3::types::PyModule;

#[pymodule]
fn bose_hubbard(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
  m.add_class::<Hamiltonian>()?;
  m.add_function(wrap_pyfunction!(prepare_state, m)?)?;
  m.add_function(wrap_pyfunction!(hilbert_space_dimension, m)?)?;
  m.add_function(wrap_pyfunction!(get_basis, m)?)?;
  m.add_function(wrap_pyfunction!(particles_per_mode, m)?)?;
  m.add_function(wrap_pyfunction!(n_bosons_weight, m)?)?;
  Ok(())
}
