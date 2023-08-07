import os
from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass
import yaml
import numpy as np
import h5py
import datetime
from scipy.sparse.linalg import LinearOperator, _expm_multiply
from tqdm import tqdm
from bose_hubbard import (
    Hamiltonian,
    prepare_state,
    particles_per_mode,
    n_bosons_weight,
    get_basis,
)


def hamiltonian2lin_op(
        hamiltonian: Hamiltonian,
) -> LinearOperator:
    def matvec(x: np.ndarray):
        if x.dtype != np.complex128:
            x = x.astype(np.complex128)
        return -1j * hamiltonian.apply(np.array(x).reshape((-1,)))
    def rmatvec(x: np.ndarray):
        if x.dtype != np.complex128:
            x = x.astype(np.complex128)
        return 1j * hamiltonian.apply(np.array(x).reshape((-1,)))
    dim = hamiltonian.hilbert_space_dim()
    lin_op = LinearOperator(
        dtype=np.complex128,
        shape=(dim, dim),
        matvec=matvec,
        rmatvec=rmatvec,
    )
    return lin_op


def write2hdf(path: str, name: str, data_point: np.ndarray):
    with h5py.File(path, 'a') as data:
        match data.get(name):
            case None:
                data.create_dataset(name, (1,) + data_point.shape, data=data_point[np.newaxis],
                                    chunks=(1, *data_point.shape),
                                    maxshape=(None, *data_point.shape),
                )
            case arr:
                updated_arr = np.concatenate([arr, data_point[np.newaxis]], axis=0)
                arr.resize(arr.shape[0] + 1, 0)
                arr[:] = updated_arr


@dataclass
class Config:
    name: str
    hopping_graph: List[Tuple[int, int]]
    hopping_amplitude: float
    interaction_amplitude: float
    initial_state: np.ndarray
    diagonal_statistics: Dict[str, Callable[[np.ndarray], np.ndarray]]
    total_time_steps: int
    time_step_size: float
    basis: np.ndarray
    constraints: List[int]

    def to_yaml(self) -> str:
        repr = {
            "config":
            {
                "name": self.name,
                "hopping_graph": self.hopping_graph,
                "hopping_amplitude": self.hopping_amplitude,
                "interaction_amplitude": self.interaction_amplitude,
                "initial_state": [int(v) for v in self.initial_state],
                "diagonal_statistics": [state for state, _ in self.diagonal_statistics.items()],
                "total_time_steps": self.total_time_steps,
                "time_step_size": self.time_step_size,
                "constraints": self.constraints,
            }
        }
        return yaml.safe_dump(repr)

    def __init__(self, config: str, name: str):
        self.name = name
        config = yaml.safe_load(config)
        self.hopping_graph = tuple(map(tuple, config["hopping_graph"]))
        self.hopping_amplitude = float(config["hopping_amplitude"])
        self.interaction_amplitude = float(config["interaction_amplitude"])
        self.initial_state = np.array(config["initial_state"])
        self.total_time_steps = int(config["total_time_steps"])
        self.time_step_size = float(config["time_step_size"])
        self.constraints = tuple(config["constraints"])
        self.diagonal_statistics = None
        self.basis = get_basis(len(self.initial_state), self.initial_state.sum(), self.constraints)
        for statistics_name in config["diagonal_statistics"]:
            match statistics_name:
                case "particles_per_mode":
                    if self.diagonal_statistics is None:
                        self.diagonal_statistics = { statistics_name: particles_per_mode }
                    else:
                        self.diagonal_statistics[statistics_name] = particles_per_mode
                case "n_bosons_weight":
                    if self.diagonal_statistics is None:
                        self.diagonal_statistics = { statistics_name: n_bosons_weight }
                    else:
                        self.diagonal_statistics[statistics_name] = n_bosons_weight
                case other:
                    print(f"Error todo {other}")

    def run_bose_hubbard_dynamics(self, result_dir: str):
        modes_number = self.initial_state.shape[0]
        particles_number = self.initial_state.sum()
        state = prepare_state(
            self.initial_state,
            self.constraints
        )
        print(f"hilbert_space_dimension: {state.shape[0]}")
        hamiltonian = Hamiltonian(
            modes_number,
            particles_number,
            self.hopping_graph,
            self.constraints,
            self.hopping_amplitude,
            self.interaction_amplitude,
        )
        hamiltonian_trace = hamiltonian.trace()
        hamiltonian = hamiltonian2lin_op(hamiltonian)
        experiment_dir = f"{result_dir}/{self.name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        log_file = f"{experiment_dir}/log.yaml"
        data_file = f"{experiment_dir}/data.hdf5"
        os.makedirs(experiment_dir)
        with open(log_file, 'a') as f:
            f.write(self.to_yaml())
        for name, diagonal_statistic in self.diagonal_statistics.items():
            s = diagonal_statistic(self.basis, state)
            write2hdf(data_file, name, s)
        for _ in tqdm(range(self.total_time_steps)):
            state = _expm_multiply._expm_multiply_simple(
                hamiltonian,
                state,
                self.time_step_size,
                traceA=hamiltonian_trace,
            )
            for name, diagonal_statistic in self.diagonal_statistics.items():
                s = diagonal_statistic(self.basis, state)
                write2hdf(data_file, name, s)
