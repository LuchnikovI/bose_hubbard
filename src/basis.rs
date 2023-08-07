use num_complex::{Complex64, ComplexFloat};
use numpy::{PyArray2, PyArray1};
use pyo3::{Python, pyfunction};
use num_cpus::get_physical;
use rayon::ThreadPoolBuilder;
use crate::basis_iterator::BasisIterator;
use crate::hamiltonian::hilbert_space_dimension;

fn get_basis_generic<'py, const N: usize>(
    py: Python<'py>,
    particles_number: u8,
    constraints: &[u8],
) -> &'py PyArray2<u8>
{
    let hs_dim = hilbert_space_dimension(N as u8, particles_number, constraints.to_owned());
    let basis = unsafe { PyArray2::new(py, [hs_dim, N], false) };
    let iter = BasisIterator::<N>::new(particles_number, constraints);
    for (slice, basis_elem) in unsafe { basis.as_slice_mut().unwrap() }.chunks_mut(N).zip(iter)
    {
        for (dst, src) in slice.iter_mut().zip(basis_elem)
        {
            *dst = src;
        }
    }
    basis
}

/// Returns all basis vectors in lexicographical order
/// (the same order is used in all computations)
/// Args:
///     modes_number (int): number of bosonic modes
///     particles_number (int): number of particles
///     constraints (List[int]): maximal number of bosons per side
/// Return:
///     Two dimensional bytes valued np.ndarray representing
///     all basis vectors
#[pyfunction]
#[pyo3(signature = (modes_number, particles_number, constraints))]
pub fn get_basis<'py>(
    py: Python<'py>,
    modes_number: u8,
    particles_number: u8,
    constraints: Vec<u8>,
) -> &'py PyArray2<u8>
{
    match modes_number {
        0 | 1 => unimplemented!("Basis is not implemented for less than 2 bosonic modes"),
        2  => get_basis_generic::<2>(py, particles_number, &constraints),
        3  => get_basis_generic::<3>(py, particles_number, &constraints),
        4  => get_basis_generic::<4>(py, particles_number, &constraints),
        5  => get_basis_generic::<5>(py, particles_number, &constraints),
        6  => get_basis_generic::<6>(py, particles_number, &constraints),
        7  => get_basis_generic::<7>(py, particles_number, &constraints),
        8  => get_basis_generic::<8>(py, particles_number, &constraints),
        9  => get_basis_generic::<9>(py, particles_number, &constraints),
        10 => get_basis_generic::<10>(py, particles_number, &constraints),
        11 => get_basis_generic::<11>(py, particles_number, &constraints),
        12 => get_basis_generic::<12>(py, particles_number, &constraints),
        13 => get_basis_generic::<13>(py, particles_number, &constraints),
        14 => get_basis_generic::<14>(py, particles_number, &constraints),
        15 => get_basis_generic::<15>(py, particles_number, &constraints),
        16 => get_basis_generic::<16>(py, particles_number, &constraints),
        17 => get_basis_generic::<17>(py, particles_number, &constraints),
        18 => get_basis_generic::<18>(py, particles_number, &constraints),
        19 => get_basis_generic::<19>(py, particles_number, &constraints),
        20 => get_basis_generic::<20>(py, particles_number, &constraints),
        21 => get_basis_generic::<21>(py, particles_number, &constraints),
        22 => get_basis_generic::<22>(py, particles_number, &constraints),
        23 => get_basis_generic::<23>(py, particles_number, &constraints),
        24 => get_basis_generic::<24>(py, particles_number, &constraints),
        25 => get_basis_generic::<25>(py, particles_number, &constraints),
        26 => get_basis_generic::<26>(py, particles_number, &constraints),
        27 => get_basis_generic::<27>(py, particles_number, &constraints),
        28 => get_basis_generic::<28>(py, particles_number, &constraints),
        29 => get_basis_generic::<29>(py, particles_number, &constraints),
        30 => get_basis_generic::<30>(py, particles_number, &constraints),
        31 => get_basis_generic::<31>(py, particles_number, &constraints),
        32 => get_basis_generic::<32>(py, particles_number, &constraints),
        other => unimplemented!("Basis is not implemented for {} modes", other),
    }
}

fn diagonal_statistics<T>(
    state: &PyArray1<Complex64>,
    basis: &PyArray2<u8>,
    dst: &mut [f64],
    statistics_fn: T,
)
where
    T: Fn(&[u8], f64, &mut [f64]) + Clone + Send + Sync
{
    let threads_number = get_physical() + 1;
    let pool = ThreadPoolBuilder::new()
        .num_threads(get_physical() + 1)
        .build()
        .unwrap();
    let mut dst_pieces = vec![vec![0f64; dst.len()]; threads_number];
    let size = basis.shape()[0];
    let stride = basis.shape()[1];
    let piece_size = size / threads_number + 1;
    let basis = unsafe { basis.as_slice().unwrap() };
    let state = unsafe { state.as_slice().unwrap() };
    let basis_chunks = basis.chunks(stride * piece_size);
    let state_chunks = state.chunks(piece_size);
    let dst_pieces_iter = dst_pieces.iter_mut();
    let iter = basis_chunks.zip(state_chunks.zip(dst_pieces_iter));
    pool.scope(|s| {
        for (basis_piece, (state_piece, dst_piece)) in iter
        {
            let basis_elems_iter = basis_piece.chunks(stride);
            let psi_iter = state_piece.iter();
            s.spawn(|_| {
                for (psi, elem) in psi_iter.zip(basis_elems_iter) {
                    let prob = psi.abs().powi(2);
                    statistics_fn(elem, prob, dst_piece)
                }
            });
        }
    });
    for dst_piece in dst_pieces {
        for (dst, src) in dst.iter_mut().zip(dst_piece)
        {
            *dst += src;
        }
    }
}

/// Computes average number of particles per mode.
/// Args:
///     basis (np.ndarray): basis elements
///     state (np.ndarray): state vector
/// Returns:
///     np.ndarray representing number of particles per mode
#[pyfunction]
#[pyo3(signature = (basis, state))]
pub fn particles_per_mode<'py>(
    py: Python<'py>,
    basis: &PyArray2<u8>,
    state: &PyArray1<Complex64>,
) -> &'py PyArray1<f64>
{
    let dst = PyArray1::zeros(py, [basis.shape()[1]], false);
    diagonal_statistics(
        state,
        basis, 
        unsafe { dst.as_slice_mut().unwrap() },
        |basis_elements, prob, dst_elements| {
            for (elem, dst) in basis_elements.into_iter().zip(dst_elements)
            {
                *dst += (*elem as f64) * prob;
            }
        },
    );
    dst
}

/// Computes weight of n boson states for all n.
/// Args:
///     basis (np.ndarray): basis elements
///     state (np.ndarray): state vector
/// Returns:
///     np.ndarray weights of n boson states for all n
#[pyfunction]
#[pyo3(signature = (basis, state))]
pub fn n_bosons_weight<'py>(
    py: Python<'py>,
    basis: &PyArray2<u8>,
    state: &PyArray1<Complex64>,
) -> &'py PyArray1<f64>
{
    let modes_number = basis.shape()[1];
    let particles_number: usize = unsafe { basis
        .as_slice() }
        .unwrap()
        .into_iter()
        .take(modes_number)
        .map(|x| *x as usize)
        .sum();
    let dst = PyArray1::zeros(py, [particles_number + 1], false);
    diagonal_statistics(
        state,
        basis, 
        unsafe { dst.as_slice_mut().unwrap() },
        |basis_elements, prob, dst_elements| {
            for elem in basis_elements {
                unsafe {
                    *dst_elements.get_unchecked_mut(*elem as usize) += prob;
                }
            }
        },
    );
    dst
}