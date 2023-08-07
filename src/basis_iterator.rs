#[derive(Clone, Copy)]
struct AutomatonState {
    bss: *mut u8,
    cnstr: *const u8,
    particles_counter: u8,
    particles_number: u8,
    pos_from_end: usize,
}

enum BasisAutomaton {
    PositionsOccupied(AutomatonState),
    StepPossible(AutomatonState),
    StepPointFound(AutomatonState),
}

impl BasisAutomaton {

    #[inline(always)]
    fn new(basis: &mut [u8], constraints: &[u8], particles_number: u8) -> Self
    {
        let bss = basis.as_mut_ptr_range().end;
        let cnstr = constraints.as_ptr_range().end;
        let particles_counter = 0u8;
        let state = AutomatonState {
            bss, cnstr, particles_counter, particles_number, pos_from_end: 0,
        };
        BasisAutomaton::PositionsOccupied(state)
    }

    #[inline(always)]
    unsafe fn step(&mut self) -> Option<()>
    {
        match self {
            BasisAutomaton::PositionsOccupied(state) => {
                state.bss = state.bss.sub(1);
                state.cnstr = state.cnstr.sub(1);
                state.pos_from_end += 1;
                let local_particles_number = *state.bss;
                let local_constraint = *state.cnstr;
                state.particles_counter += local_particles_number;
                if state.particles_counter  == state.particles_number {
                    return None;
                }
                if local_particles_number < local_constraint {
                    *self = BasisAutomaton::StepPossible(*state);
                }
                Some(())
            },
            BasisAutomaton::StepPossible(state) => {
                state.bss = state.bss.sub(1);
                state.cnstr = state.cnstr.sub(1);
                state.pos_from_end += 1;
                if *state.bss != 0 {
                    *self = BasisAutomaton::StepPointFound(*state);
                }
                Some(())
            },
            BasisAutomaton::StepPointFound(state) => {
                *state.bss -= 1;
                state.particles_counter += 1;
                while state.pos_from_end > 1 {
                    state.bss = state.bss.add(1);
                    state.cnstr = state.cnstr.add(1);
                    state.pos_from_end -= 1;
                    if state.particles_counter > *state.cnstr {
                        *state.bss = *state.cnstr;
                        state.particles_counter -= *state.cnstr;
                    } else {
                        *state.bss = state.particles_counter;
                        state.particles_counter = 0;
                    }
                }
                None
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(super) struct BasisIterator<'a, const N: usize> {
    basis: Option<[u8; N]>,
    constraints: &'a [u8],
    particles_number: u8,
}

impl<'a, const N: usize> BasisIterator<'a, N> {

    pub(super) fn new(
        particles_number: u8,
        constraints: &'a [u8],
    ) -> Self
    {
        assert!(N > 1, "State must contain more than 1 bosonic mode");
        let mut basis = [0; N];
        let iter_basis = basis.iter_mut();
        let iter_constraints = constraints.into_iter();
        let mut particles_number_clone = particles_number;
        for (s, c) in iter_basis.zip(iter_constraints) {
            if particles_number_clone > *c {
                *s = *c;
                particles_number_clone -= c;
            } else {
                *s = particles_number_clone;
                return Self { basis: Some(basis), constraints, particles_number: particles_number };
            }
        }
        panic!("Number of particles is inconsistent with the given constraints")
    }

}

impl<'a, const N: usize> Iterator for BasisIterator<'a, N> {
    type Item = [u8; N];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(state) = &mut self.basis {
            let to_return = *state;
            let mut automaton = BasisAutomaton::new(state, &self.constraints, self.particles_number);
            while unsafe { automaton.step().is_some() } {};
            if let BasisAutomaton::PositionsOccupied(_) = automaton {
                self.basis = None;
            }
            Some(to_return)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::hamiltonian::hilbert_space_dimension;

    use super::BasisIterator;

    fn _test_basis_iterator<const N: usize>(particles_number: u8, consraints: &[u8])
    {
        let basis_iterator = BasisIterator::<N>::new(particles_number, consraints);
        let mut states_counter = 0usize;
        for elem in basis_iterator {
            states_counter += 1;
            let mut aggregated_particles_number = 0;
            for (e, c) in elem.iter().zip(consraints) {
                aggregated_particles_number += e;
                assert!(*e <= *c);
            }
            assert_eq!(particles_number, aggregated_particles_number);
        }
        let correct_states_count = hilbert_space_dimension(N as u8, particles_number, consraints.to_owned());
        assert_eq!(correct_states_count, states_counter);
    }

    #[test]
    fn test_basis_iterator()
    {
        _test_basis_iterator::<3>(10, &[10, 10, 10]);
        _test_basis_iterator::<3>(10, &[3, 2, 5]);
        _test_basis_iterator::<10>(3, &[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]);
        _test_basis_iterator::<10>(3, &[1, 1, 3, 1, 1, 1, 2, 1, 1, 1]);
        _test_basis_iterator::<10>(1, &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        _test_basis_iterator::<10>(1, &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        _test_basis_iterator::<2>(12, &[12, 12]);
        _test_basis_iterator::<2>(12, &[3, 10]);
        _test_basis_iterator::<10>(10, &[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]);
        _test_basis_iterator::<10>(10, &[1, 2, 3, 4, 3, 2, 1, 0, 1, 2]);
    }
}