# evaluate_heo_kernel_32_v2.py
import json
import numpy as np
from feature_map import PetersFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import BackendSampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_ionq import IonQProvider

# ---------- Load prepared HEO data ----------
data = np.load("heo_binary_data.npz", allow_pickle=True)
xrd_quantum = data["xrd_quantum"]       # <-- quantum preprocessing
labels = data["labels"]
indices = data["selected_indices"]

n_samples = xrd_quantum.shape[0]
print(f"Loaded {n_samples} normalized patterns, {xrd_quantum.shape[1]} features each")
print(f"Labels: {labels}")
print(f"Indices: {indices}")

# ---------- Quantum kernel setup ----------
num_features = xrd_quantum.shape[1]
feature_map = PetersFeatureMap(feature_dimension=num_features, reps=1)

import os
api_key = os.environ.get("IONQ_API_KEY")
if api_key is None:
    with open("api_key.txt") as f:
        api_key = f.read().strip()

provider = IonQProvider(api_key)
backend = provider.get_backend("ionq_simulator")
sampler = BackendSampler(backend=backend, options={"noise_model": "aria-1", "shots": 1024})

fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# ---------- Evaluate kernel ----------
n_pairs = n_samples * (n_samples - 1) // 2
print(f"Submitting {n_pairs} kernel jobs to IonQ simulator...")
kernel_matrix = kernel.evaluate(xrd_quantum)

# ---------- Save to JSON ----------
kernel_entries = {}
for i in range(n_samples):
    for j in range(i + 1, n_samples):
        kernel_entries[str((i, j))] = {"result": {"0": float(kernel_matrix[i, j])}}

output_data = {
    "kernel_data": {"num_features": num_features, "num_repetitions": 1},
    "xrd_data": {"xrd": xrd_quantum.tolist(), "theta": []},
    "kernel_entries": kernel_entries,
    "labels": labels.tolist(),
    "selected_indices": indices.tolist()
}

output_path = "heo_quantum_kernel_32.json"
with open(output_path, "w") as f:
    json.dump(output_data, f)

print(f"Kernel matrix saved to {output_path}")