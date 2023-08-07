import numpy as np
import pytest
from bose_hubbard import get_basis, particles_per_mode
from pysrc.utils import hamiltonian2lin_op

def particles_per_mode_validation(
        basis: np.ndarray,
        state: np.ndarray,
) -> np.ndarray:
    return ((np.abs(state.reshape((-1, 1))) ** 2) * basis).sum(0)


@pytest.mark.parametrize("modes_number,particles_number,constraints", [
        (2, 2, [2, 2]),
        (2, 2, [1, 2]),
        (2, 2, [2, 1]),
        (10, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (10, 1, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        (10, 2, [1, 2, 0, 2, 1, 2, 1, 2, 1, 2]),
        (10, 2, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        (2, 10, [10, 10]),
        (2, 10, [3, 7]),
        (2, 10, [5, 5]),
])
def test_particles_per_mode(
        modes_number,
        particles_number,
        constraints,
):
    basis = get_basis(modes_number, particles_number, constraints)
    size = basis.shape[0]
    state = np.random.normal(size = (size, 2))
    state = state[:, 0] + 1j * state[:, 1]
    distr_trial = particles_per_mode(basis, state)
    distr_validation = particles_per_mode_validation(basis, state)
    assert np.linalg.norm(distr_validation - distr_trial) < 1e-6
