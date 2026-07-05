# Toward Quantum Advantage in Low-Data Materials Discovery: XRD Phase Classification of High-Entropy Oxides

This repository contains the complete code and data for reproducing all results reported in the manuscript. The pipeline follows the quantum kernel machine learning methodology of Adams et al. (2026) and applies it to a high‑entropy oxide XRD dataset.

---

## Repository structure

| File / Folder | Purpose |
|---------------|---------|
| `prepare_heo_binary_data_v2.py` | Build the 32‑sample balanced dataset (16 Fm‑3m, 16 Ia‑3) |
| `evaluate_heo_kernel_32_v2.py` | Compute quantum kernel on IonQ Aria‑1 noisy simulator |
| `compute_classical_kernels.py` | Build all classical kernel matrices on L₂‑normalised data |
| `compute_metrics_32.py` | Calculate geometric differences and model complexities |
| `run_gp_multi_seed_32.py` | Multi‑seed GP classification (75 paired splits per training size) |
| `wilcoxon_significance.py` | Wilcoxon signed‑rank tests on pooled accuracies |
| `feature_map.py` | PetersFeatureMap quantum circuit definition |
| `kernels.py` | Classical kernel functions and FixedPrecomputedGPKernel |
| `metrics.py` | Geometric difference, model complexity |
| `api_key.txt` | IonQ API key (place your own key here, or set `IONQ_API_KEY`) |
| `heo_binary_data.npz` | Processed dataset (X_classical, X_quantum, labels) |
| `heo_quantum_kernel_32.json` | Precomputed quantum kernel matrix |
| `heo_classical_kernels_32.npz` | All classical kernel matrices |
| `heo_pooled_results_32.npz` | Pooled GP classification accuracies (75 splits) |
| `heo_relative_accuracy_pooled.png` | Main publication figure |
| `ML4HEOs/` | Original Velasco et al. dataset (also available at [GitHub](https://github.com/aimat-lab/ML4HEOs)) |
| `requirements_gpflow.txt` | Python 3.12 package list for GPFlow environment |
| `requirements_qiskit.txt` | Python 3.9 package list for Qiskit environment |
| `LICENSE` | MIT License |
| `.gitignore` | Prevents accidental commits of virtual environments and caches |

**Note:** Publication figures (Figure 4a–c) were generated from `heo_pooled_results_32.npz` using R (ggplot2). The R plotting script is available upon request.

---

## Dependencies

Two separate Python virtual environments are required:

1. **Quantum kernel evaluation** (Python 3.9)  
   - qiskit, qiskit‑machine‑learning, qiskit‑ionq, numpy, scipy  
   - (see `requirements_qiskit.txt`)

2. **Classical computation & GP classification** (Python 3.12)  
   - gpflow, tensorflow (2.16), tensorflow‑probability (0.24), numpy, pandas, scipy, matplotlib, tqdm  
   - (see `requirements_gpflow.txt`)

The exact packages used in this work can be installed by:
```bash
pip install -r requirements_gpflow.txt   # for the GPFlow environment
pip install -r requirements_qiskit.txt   # for the Qiskit environment
Step‑by‑step reproduction
Note: If you want to skip the time‑consuming quantum kernel evaluation, the precomputed kernel heo_quantum_kernel_32.json is already provided. Steps 2 can be omitted, and you can start from Step 3.

Step 1 – Prepare the dataset
bash
# Activate the GPFlow (Python 3.12) environment
source venv_gpflow/bin/activate
python prepare_heo_binary_data_v2.py
Produces: heo_binary_data.npz

Step 2 – Evaluate the quantum kernel (requires IonQ API key)
bash
# Activate the Qiskit (Python 3.9) environment
source venv_qiskit/bin/activate
python evaluate_heo_kernel_32_v2.py
Produces: heo_quantum_kernel_32.json
Time: ~5 minutes on the IonQ Aria‑1 noisy simulator.

Step 3 – Build classical kernel matrices
bash
# Activate GPFlow environment
source venv_gpflow/bin/activate
python compute_classical_kernels.py
Produces: heo_classical_kernels_32.npz

Step 4 – Compute geometric differences and model complexities
bash
python compute_metrics_32.py
Prints: Tables 2 and 3 values.

Step 5 – Run multi‑seed GP classification
bash
python run_gp_multi_seed_32.py
Produces: heo_pooled_results_32.npz
Time: ~1.5 hours (three seeds, four kernels, 27 training sizes, paired splits).

Step 6 – Statistical tests
bash
python wilcoxon_significance.py
Prints: Mean accuracy differences and p‑values for all training sizes.

Step 7 – Generate publication figures (optional)
Figures in the manuscript were produced with R (ggplot2) from heo_pooled_results_32.npz. A Python‑based alternative can be provided upon request.

Data availability
The high‑entropy oxide XRD dataset is from Velasco et al. (2021) and is publicly available at https://github.com/aimat-lab/ML4HEOs. The ML4HEOs/ folder included in this repository is a snapshot of that dataset. All derived data files (heo_binary_data.npz, heo_quantum_kernel_32.json, etc.) are also provided.

License
This project is licensed under the MIT License – see the LICENSE file for details.

text

This version is accurate, complete, and submission‑ready. The only remaining task is to push the repository and insert your actual GitHub link into the manuscript.