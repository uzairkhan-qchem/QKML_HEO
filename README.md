# Quantum kernel Machine Learning for XRD Phase Classification in High‑Entropy Oxides

Code and data for reproducing all results in the manuscript. The pipeline applies the quantum kernel machine learning to a high-entropy oxide XRD dataset.

---

## Repository structure

| File / Folder | Purpose |
|---------------|---------|
| `prepare_heo_binary_data_v2.py` | Build the 32-sample balanced dataset (16 Fm-3m, 16 Ia-3) |
| `evaluate_heo_kernel_32_v2.py` | Compute quantum kernel on IonQ Aria-1 noisy simulator |
| `compute_classical_kernels.py` | Build all classical kernel matrices on L2-normalised data |
| `compute_metrics_32.py` | Calculate geometric differences and model complexities |
| `run_gp_multi_seed_32.py` | Multi-seed GP classification (75 paired splits per training size) |
| `wilcoxon_significance.py` | Wilcoxon signed-rank tests on pooled accuracies |
| `heo_pooled_results_32_figures.R` | Generate publication figures (Figure 4a-c) with ggplot2 |
| `feature_map.py` | PetersFeatureMap quantum circuit definition |
| `kernels.py` | Classical kernel functions and FixedPrecomputedGPKernel |
| `metrics.py` | Geometric difference, model complexity |
| `api_key.txt.example` | Template for the IonQ API key (copy to `api_key.txt` and insert your own key, or set `IONQ_API_KEY`) |
| `heo_binary_data.npz` | Processed dataset (X_classical, X_quantum, labels) |
| `heo_quantum_kernel_32.json` | Precomputed quantum kernel matrix |
| `heo_classical_kernels_32.npz` | All classical kernel matrices |
| `heo_pooled_results_32.npz` | Pooled GP classification accuracies (75 splits) |
| `ML4HEOs/` | Snapshot of the original Velasco et al. dataset ([source](https://github.com/aimat-lab/ML4HEOs)) |
| `requirements_gpflow.txt` | Python 3.12 package list for the GPflow environment |
| `requirements_qiskit.txt` | Python 3.9 package list for the Qiskit environment |
| `LICENSE` | MIT License |
| `.gitignore` | Prevents accidental commits of virtual environments, caches, and API keys |

---

## Dependencies

Two separate Python virtual environments are required:

1. **Quantum kernel evaluation** (Python 3.9)
   - qiskit, qiskit-machine-learning, qiskit-ionq, numpy, scipy
   - see `requirements_qiskit.txt`

2. **Classical computation & GP classification** (Python 3.12)
   - gpflow, tensorflow (2.16), tensorflow-probability (0.24), numpy, pandas, scipy, matplotlib, tqdm
   - see `requirements_gpflow.txt`

Install with:

```bash
pip install -r requirements_gpflow.txt   # GPflow environment
pip install -r requirements_qiskit.txt   # Qiskit environment
```

Figure generation additionally requires **R** with the `ggplot2` package.

---

## Step-by-step reproduction

> **Note:** To skip the quantum kernel evaluation, use the precomputed
> `heo_quantum_kernel_32.json` provided in this repository. Step 2 can be
> omitted, and you can start from Step 3.

### Step 1 - Prepare the dataset

```bash
# Activate the GPflow (Python 3.12) environment
source venv_gpflow/bin/activate
python prepare_heo_binary_data_v2.py
```

Produces: `heo_binary_data.npz`

### Step 2 - Evaluate the quantum kernel (requires IonQ API key)

```bash
# Activate the Qiskit (Python 3.9) environment
source venv_qiskit/bin/activate
python evaluate_heo_kernel_32_v2.py
```

Produces: `heo_quantum_kernel_32.json`
Time: ~5 minutes on the IonQ Aria-1 noisy simulator.

### Step 3 - Build classical kernel matrices

```bash
# Activate the GPflow environment
source venv_gpflow/bin/activate
python compute_classical_kernels.py
```

Produces: `heo_classical_kernels_32.npz`

### Step 4 - Compute geometric differences and model complexities

```bash
python compute_metrics_32.py
```

Prints: values reported in Tables 2 and 3 of the manuscript.

### Step 5 - Run multi-seed GP classification

```bash
python run_gp_multi_seed_32.py
```

Produces: `heo_pooled_results_32.npz`
Time: ~1.5 hours (three seeds, four kernels, 27 training sizes, paired splits).

### Step 6 - Statistical tests

```bash
python wilcoxon_significance.py
```

Prints: mean accuracy differences and p-values for all training sizes
(Supplementary Table S1).

### Step 7 - Generate publication figures

```bash
Rscript heo_pooled_results_32_figures.R
```

Produces: Figure 4a-c of the manuscript. Requires R with `ggplot2`; see the
script header for input expectations.

---

## Data availability

The high-entropy oxide XRD dataset is from Velasco et al. (2021) and is
publicly available at <https://github.com/aimat-lab/ML4HEOs>. The `ML4HEOs/`
folder in this repository is a snapshot of that dataset. All derived data
files (`heo_binary_data.npz`, `heo_quantum_kernel_32.json`,
`heo_classical_kernels_32.npz`, `heo_pooled_results_32.npz`) are also
provided, so every result can be reproduced without external downloads.

## License

This project is licensed under the MIT License - see the `LICENSE` file for
details.
