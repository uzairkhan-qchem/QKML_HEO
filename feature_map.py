import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import HGate, RZGate, RYGate, iSwapGate
from qiskit.circuit.library.n_local.n_local import NLocal


class PetersFeatureMap(NLocal):
    """
    A paramterized feature map quantum circuit based on the Qiskit NLocal
    circuit which implements the feature map described in the paper "Machine
    learning of high dimensional data on a noisy quantum processor" by Peters
    et al. (2021).
    """

    def __init__(self, feature_dimension: int, reps: int):
        """
        Constructor for the feature map described be Peters et al. (2021)

        Args:
            feature_dimension: The number of dimensions of the input data
        """

        # calculate the minimum number of qubits necessary to have a parameter
        # for feature dimension
        num_qubits = int(np.ceil(feature_dimension / (3 * (reps + 1))))

        # create 6 Qiskit parameter instances (3 tunable parameters and 3 data
        # points)
        params = ParameterVector(name="_", length=3)

        rotation_blocks = [
            HGate(),
            RZGate(params[0]),
            RYGate(params[1]),
            RZGate(params[2]),
        ]

        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement=self._get_entanglement(num_qubits),
            entanglement_blocks=[iSwapGate().power(1 / 2)],
            reps=reps,
            name="PetersFeatureMap",
        )

    def _get_entanglement(self, n: int) -> list[list[int]]:
        """
        Given a number of qubits n, return a list of sequential indices
        indicating which qubits should be connected via the entangling
        block.

        Args:
            n: the number of qubits in the feature map circuit
        """

        indices = []

        for i in range(0, n, 2):
            if i + 1 < n:
                indices.append([i, i + 1])

        for i in range(1, n, 2):
            if i + 1 < n:
                indices.append([i, i + 1])

        return indices
