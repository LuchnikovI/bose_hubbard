use std::collections::HashMap;
use num_complex::Complex64;
use rayon::{
    prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelIterator},
    slice::{ParallelSliceMut, ParallelSlice},
};
use pyo3::{
    pyclass,
    pymethods,
    pyfunction,
    Python,
};
use numpy::{PyArray1, PyArray0};
use crate::basis_iterator::BasisIterator;
use crate::basis_hashing::hash_state;
use crate::search::ValuePositionMap;

// ----------------------------------------------------------------------------------------

fn _hilbert_space_dimension<'b, 'a: 'b>(
    modes_number: u8,
    particles_number: u8,
    constraints: &'a [u8],
    capacity: u8,
    cache: &'b mut HashMap<(u8, u8, &'a [u8]), usize>
) -> usize
{
    if  capacity < particles_number {
        0
    } else if modes_number == 1 || particles_number == 0 {
        1
    } else {
        if let Some(number) = cache.get(&(modes_number, particles_number, constraints))
        {
            return *number;
        }
        let mut number = 0;
        let cnstr = unsafe { *constraints.get_unchecked(0) };
        for i in 0..=std::cmp::min(particles_number, cnstr) {
            number += _hilbert_space_dimension(
                modes_number - 1,
                particles_number - i,
                &constraints[1..],
                capacity - cnstr,
                cache,
            );
        }
        cache.insert((modes_number, particles_number, constraints), number);
        number
    }
}

/// Computes dimension of a hilbert space.
/// Args:
///     modes_number (int): number of bosonic modes
///     particles_number (int): number of particles
///     constraints (List[int]): maximal number of particles per mode
/// Returns:
///     hilbert space dimension
#[pyfunction]
#[pyo3(signature = (modes_number, particles_number, constraints))]
pub fn hilbert_space_dimension(
    modes_number: u8,
    particles_number: u8,
    constraints: Vec<u8>,
) -> usize
{
    assert_eq!(modes_number, constraints.len() as u8);
    let mut cache: HashMap<(u8, u8, &[u8]), usize> = HashMap::new();
    let capacity = constraints.iter().sum();
    _hilbert_space_dimension(
        modes_number,
        particles_number,
        &constraints,
        capacity,
        &mut cache,
    )
}

// ----------------------------------------------------------------------------------------

fn prepare_state_generic<'py, const N: usize>(
    py: Python<'py>,
    particles_per_mode: &[u8],
    constraints: &[u8],
) -> &'py PyArray1<Complex64>
{
    let particles_number: u8 = particles_per_mode.iter().sum();
    let size = hilbert_space_dimension(N as u8, particles_number, constraints.to_owned());
    let state = PyArray1::<Complex64>::zeros(py, [size], false);
    let state_iter = BasisIterator::<N>::new(particles_number, constraints);
    for (elem, s) in unsafe { state.as_slice_mut().unwrap() }.iter_mut().zip(state_iter)
    {
        if s == particles_per_mode {
            *elem = Complex64::new(1f64, 0f64);
            break;
        }
    }
    state
}

/// Prepares a state with fixed number of particles per bosonic mode.
/// Args:
///     particles_per_mode (List[int]): number of particles per mode
///     constraints (List[int]): maximal number of particles per mode
/// Returns:
///     np.complex128 valued np array representing a state
#[pyfunction]
#[pyo3(signature = (particles_per_mode, constraints))]
pub fn prepare_state<'py>(
    py: Python<'py>,
    particles_per_mode: Vec<u8>,
    constraints: Vec<u8>
) -> &'py PyArray1<Complex64>
{
    let modes_number = particles_per_mode.len();
    // TODO: maybe reduce code repetition by a macro
    match modes_number {
        0 | 1 => unimplemented!("State preparation is not defined for less than 2 modes"),
        2 => prepare_state_generic::<2>(py, &particles_per_mode, &constraints),
        3 => prepare_state_generic::<3>(py, &particles_per_mode, &constraints),
        4 => prepare_state_generic::<4>(py, &particles_per_mode, &constraints),
        5 => prepare_state_generic::<5>(py, &particles_per_mode, &constraints),
        6 => prepare_state_generic::<6>(py, &particles_per_mode, &constraints),
        7 => prepare_state_generic::<7>(py, &particles_per_mode, &constraints),
        8 => prepare_state_generic::<8>(py, &particles_per_mode, &constraints),
        9 => prepare_state_generic::<9>(py, &particles_per_mode, &constraints),
        10 => prepare_state_generic::<10>(py, &particles_per_mode, &constraints),
        11 => prepare_state_generic::<11>(py, &particles_per_mode, &constraints),
        12 => prepare_state_generic::<12>(py, &particles_per_mode, &constraints),
        13 => prepare_state_generic::<13>(py, &particles_per_mode, &constraints),
        14 => prepare_state_generic::<14>(py, &particles_per_mode, &constraints),
        15 => prepare_state_generic::<15>(py, &particles_per_mode, &constraints),
        16 => prepare_state_generic::<16>(py, &particles_per_mode, &constraints),
        17 => prepare_state_generic::<17>(py, &particles_per_mode, &constraints),
        18 => prepare_state_generic::<18>(py, &particles_per_mode, &constraints),
        19 => prepare_state_generic::<19>(py, &particles_per_mode, &constraints),
        20 => prepare_state_generic::<20>(py, &particles_per_mode, &constraints),
        21 => prepare_state_generic::<21>(py, &particles_per_mode, &constraints),
        22 => prepare_state_generic::<22>(py, &particles_per_mode, &constraints),
        23 => prepare_state_generic::<23>(py, &particles_per_mode, &constraints),
        24 => prepare_state_generic::<24>(py, &particles_per_mode, &constraints),
        25 => prepare_state_generic::<25>(py, &particles_per_mode, &constraints),
        26 => prepare_state_generic::<26>(py, &particles_per_mode, &constraints),
        27 => prepare_state_generic::<27>(py, &particles_per_mode, &constraints),
        28 => prepare_state_generic::<28>(py, &particles_per_mode, &constraints),
        29 => prepare_state_generic::<29>(py, &particles_per_mode, &constraints),
        30 => prepare_state_generic::<30>(py, &particles_per_mode, &constraints),
        31 => prepare_state_generic::<31>(py, &particles_per_mode, &constraints),
        32 => prepare_state_generic::<32>(py, &particles_per_mode, &constraints),
        other => unimplemented!("State preparation is not implemented for {} modes", other),
    }
}

// ----------------------------------------------------------------------------------------

#[pyclass]
#[derive(Debug, Clone)]
pub struct Hamiltonian
{
    stride: usize,
    buff: Vec<(Complex64, usize)>,
}

impl Hamiltonian
{
    fn generic_new<const N: usize>(
        particles_number: u8,
        constraints: &[u8],
        hoppings: &[(usize, usize)],
        kinetic_ampl: f64,
        interaction_ampl: f64,
    ) -> Self
    {
        for &(lhs, rhs) in hoppings {
            assert!(lhs < N, "hopping out of bound, number of modes {}, out of bound mode number {}", N, lhs);
            assert!(rhs < N, "hopping out of bound, number of modes {}, out of bound mode number {}", N, rhs);
            assert!(lhs != rhs, "hopping to the same mode {}", lhs);
        }
        let states = BasisIterator::<N>::new(particles_number, constraints).collect::<Vec<_>>();
        let val_pos_map = ValuePositionMap::<N>::new(particles_number, constraints);
        let hoppings_number = hoppings.len();
        let mut hamiltonian: Vec<(Complex64, usize)> = Vec::with_capacity(
            states.len() * (hoppings_number + 1)
        );
        unsafe { hamiltonian.set_len(states.len() * (hoppings_number + 1)) };
        hamiltonian
            .par_chunks_mut(hoppings_number + 1)
            .zip(states)
            .enumerate()
            .for_each(|(iter, (row, final_state))| {
                for (&(src, dst), (val, pos)) in hoppings.iter().zip(row.into_iter()) {
                    unsafe {
                        let predicate = (*final_state.get_unchecked(dst) != 0) &&
                            (*final_state.get_unchecked(src) < *constraints.get_unchecked(src));
                        if predicate {
                            let mut input_state = final_state;
                            *input_state.get_unchecked_mut(dst) -= 1;
                            *input_state.get_unchecked_mut(src) += 1;
                            *pos = val_pos_map.get_pos(hash_state(&input_state));
                            let elem = ((*input_state.get_unchecked(src) as f64) *
                            ((*input_state.get_unchecked(dst) + 1) as f64)).sqrt();
                            *val = Complex64::new(elem, 0f64);
                            *val *= kinetic_ampl;
                        } else {
                            *pos = iter;
                            *val = Complex64::new(0f64, 0f64);
                        }
                    }
                }
                let diag_elem = interaction_ampl * final_state.into_iter().map(|x| ((x * x) as f64)).sum::<f64>() / 2f64;
                let last_elem = row.last_mut().unwrap();
                last_elem.0 = Complex64::new(diag_elem, 0f64);
                last_elem.1 = iter;
            });
        Self { stride: hoppings_number + 1, buff: hamiltonian }
    }
}

#[pymethods]
impl Hamiltonian
{
    /// Creates a new Bose-Hubbard hamiltonian instance.
    /// Args:
    ///     modes_number (int): number of bosonic modes
    ///     particles_number (int): number of bosons
    ///     hoppings (List[Tuple[int, int]]): hopping connectivity between modes
    ///     constraints (List[int]): maximal number of particles per mode
    ///     kinetic_ampl (float): amplitude of the kinetic part of a hamiltonian
    ///     interaction_ampl (float): amplitude of the interaction path of a hamiltonian
    /// Returns:
    ///     Bose-Hubbard hamiltonian
    #[new]
    #[pyo3(signature = (
        modes_number,
        particles_number,
        hoppings,
        constraints,
        kinetic_ampl,
        interaction_ampl,
    ))]
    pub fn new(
        modes_number: u8,
        particles_number: u8,
        mut hoppings: Vec<(usize, usize)>,
        constraints: Vec<u8>,
        kinetic_ampl: f64,
        interaction_ampl: f64,
    ) -> Self
    {
        let mut flipped_hoppings: Vec<_> = hoppings.iter().map(|x| (x.1, x.0)).collect();
        hoppings.append(&mut flipped_hoppings);
        // TODO: maybe reduce code repetition by a macro
        match modes_number as usize {
            0 | 1 => unimplemented!("Hamiltonian is not defined for less than 2 modes"),
            2  => Hamiltonian::generic_new::<2>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            3  => Hamiltonian::generic_new::<3>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            4  => Hamiltonian::generic_new::<4>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            5  => Hamiltonian::generic_new::<5>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            6  => Hamiltonian::generic_new::<6>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            7  => Hamiltonian::generic_new::<7>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            8  => Hamiltonian::generic_new::<8>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            9  => Hamiltonian::generic_new::<9>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            10 => Hamiltonian::generic_new::<10>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            11 => Hamiltonian::generic_new::<11>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            12 => Hamiltonian::generic_new::<12>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            13 => Hamiltonian::generic_new::<13>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            14 => Hamiltonian::generic_new::<14>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            15 => Hamiltonian::generic_new::<15>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            16 => Hamiltonian::generic_new::<16>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            17 => Hamiltonian::generic_new::<17>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            18 => Hamiltonian::generic_new::<18>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            19 => Hamiltonian::generic_new::<19>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            20 => Hamiltonian::generic_new::<20>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            21 => Hamiltonian::generic_new::<21>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            22 => Hamiltonian::generic_new::<22>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            23 => Hamiltonian::generic_new::<23>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            24 => Hamiltonian::generic_new::<24>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            25 => Hamiltonian::generic_new::<25>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            26 => Hamiltonian::generic_new::<26>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            27 => Hamiltonian::generic_new::<27>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            28 => Hamiltonian::generic_new::<28>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            29 => Hamiltonian::generic_new::<29>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            30 => Hamiltonian::generic_new::<30>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            31 => Hamiltonian::generic_new::<31>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            32 => Hamiltonian::generic_new::<32>(particles_number, &constraints, &hoppings, kinetic_ampl, interaction_ampl),
            other => unimplemented!("Hamiltonian is not defined for {} modes", other),
        }
    }

    /// Applies a hamiltonian to a state.
    /// Args:
    ///     src (np.ndarray): a one dimensional np.complex128 valued array
    ///         representing a quantum state
    /// Returns:
    ///     a one dimensional np.complex128 valued array representing matvec
    ///     of a state and a hamiltonian
    #[pyo3(signature = (src))]
    pub fn apply<'py>(
        &self,
        py: Python<'py>,
        src: &PyArray1<Complex64>,
    ) -> &'py PyArray1<Complex64>
    {
        let dst = unsafe { PyArray1::new(py, src.shape()[0], false) };
        let src = unsafe { src.as_slice() }.unwrap();
        let state_size = self.buff.len() / self.stride;
        assert_eq!(state_size, src.len(), "src state size does not match the hamiltonian size");
        assert_eq!(state_size, dst.len(), "dst state size does not match the hamiltonian size");
        self.buff
            .par_chunks(self.stride)
            .zip(unsafe { dst.as_slice_mut() }.unwrap())
            .for_each(|(slice, elem)| {
                *elem = Complex64::new(0f64, 0f64);
                for (ampl, pos) in slice {
                    *elem += unsafe { *ampl * src.get_unchecked(*pos) }
                }
            });
        dst
    }

    /// Returns trace of a hamiltonian
    pub fn trace<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray0<Complex64>
    {
        let tr: Complex64 = self.buff[(self.stride - 1)..]
            .into_par_iter()
            .skip(self.stride)
            .map(|x| x.0)
            .sum();
        let tr_arr = unsafe { PyArray0::new(py, [], false) };
        *unsafe { tr_arr.get_mut([]) }.unwrap() = tr;
        tr_arr
    }

    /// Returns hilbert space dimension
    pub fn hilbert_space_dim<'py>(
        &self,
    ) -> usize
    {
        self.buff.len() / self.stride
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use pyo3::Python;
    use crate::hamiltonian::prepare_state;
    use crate::hamiltonian::BasisIterator;
    use super::Hamiltonian;

    fn _test_hamiltonian<'py, const N: usize>(
        py: Python<'py>,
        hoppings: Vec<(usize, usize)>,
        constraints: Vec<u8>,
        initial_state: Vec<u8>,
    )
    {
        let particles_number: u8 = initial_state.iter().sum();
        let src = prepare_state(py, initial_state.clone(), constraints.clone());
        let hamiltonian = Hamiltonian::new(
            N as u8,
            particles_number,
            hoppings.clone(),
            constraints.clone(),
            1.23,
            2.34,
        );
        let state_iter = BasisIterator::<N>::new(particles_number, &constraints);
        let dst = hamiltonian.apply(py, src);
        for (s, elem) in state_iter.zip(unsafe { dst.as_slice() }.unwrap().iter())
        {
            let mut correct_elem = Complex64::new(0f64, 0f64);
            if initial_state.as_slice() == s {
                correct_elem = Complex64::new(s.into_iter().map(|x| (2.34 / 2f64) * ((x * x) as f64)).sum(), 0f64);
            }
            for (src, dst) in hoppings.iter().map(|x| *x).chain(hoppings.iter().map(|x| (x.1, x.0))) {
                if s[dst] != 0 && s[src] < constraints[src] {
                    let mut s_clone = s.clone();
                    s_clone[src] += 1;
                    s_clone[dst] -= 1;
                    if s_clone == initial_state.as_slice() {
                        correct_elem += Complex64::new(1.23 * ((s_clone[src] * (s_clone[dst] + 1)) as f64).sqrt(), 0f64);
                    }
                }
            }
            assert_eq!(*elem, correct_elem);
        }
    }

    #[test]
    fn test_hamiltonian()
    {
        Python::with_gil(|py|
        {
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![0, 1, 2, 3, 2, 1, 0, 1, 1, 1],
                vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                vec![0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![3, 1, 0, 1, 3, 1, 3, 1, 3, 0],
                vec![0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                vec![0, 2, 0, 0, 0, 2, 0, 1, 0, 0],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![1, 2, 3, 5, 4, 0, 2, 5, 1, 5],
                vec![0, 2, 0, 0, 0, 2, 0, 1, 0, 0],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                ],
                vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                vec![1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
            );
            _test_hamiltonian::<10>(
                py,
                vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                (1, 4), (3, 8), (0, 9),
                ],
                vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                vec![0, 0, 0, 2, 0, 1, 0, 0, 2, 0],
            );
            _test_hamiltonian::<3>(
                py,
                vec![(0, 1), (1, 2)],
                vec![10, 10, 10],
                vec![3, 4, 3],
            );
            _test_hamiltonian::<3>(
                py,
                vec![(0, 1), (1, 2)],
                vec![10, 10, 10],
                vec![10, 0, 0],
            );
            _test_hamiltonian::<3>(
                py,
                vec![(0, 1), (1, 2), (0, 2)],
                vec![10, 10, 10],
                vec![0, 5, 5],
            );
            _test_hamiltonian::<3>(
                py,
                vec![(0, 1), (1, 2)],
                vec![0, 0, 0],
                vec![0, 0, 0],
            );
            _test_hamiltonian::<3>(
                py,
                vec![(0, 1), (1, 2)],
                vec![1, 1, 1],
                vec![0, 1, 0],
            );
            _test_hamiltonian::<3>(
                py,
                vec![(0, 1), (1, 2)],
                vec![1, 0, 1],
                vec![0, 1, 0],
            );
            _test_hamiltonian::<2>(
                py,
                vec![(0, 1)],
                vec![1, 1],
                vec![0, 1],
            );
            _test_hamiltonian::<2>(
                py,
                vec![(0, 1)],
                vec![1, 0],
                vec![0, 1],
            );
            _test_hamiltonian::<2>(
                py,
                vec![(0, 1)],
                vec![0, 0],
                vec![0, 0],
            );
        })
    }
}
