# compute_metrics_32.py  (final, robust version)
import json
import numpy as np
from ast import literal_eval
import metrics

# ---------- 1. Load quantum kernel ----------
with open("heo_quantum_kernel_32.json") as f:
    qdata = json.load(f)

n_q = len(qdata["xrd_data"]["xrd"])
K_q = np.ones((n_q, n_q))
for key, val in qdata["kernel_entries"].items():
    i, j = literal_eval(key)
    v = val["result"]["0"]
    K_q[i, j] = v
    K_q[j, i] = v
jitter = 1e-6
K_q += jitter * np.eye(n_q)
print(f"Quantum kernel loaded: shape {K_q.shape}")

# ---------- 2. Load classical kernels and labels ----------
cl = np.load("heo_classical_kernels_32.npz", allow_pickle=True)
labels_str = cl["labels_str"]
y_true = np.array([0 if l == "Fm-3m" else 1 for l in labels_str])

# Identify all Gaussian RBF keys
gauss_keys = [k for k in cl.keys() if k.startswith("K_gauss_")]
# Sort them by lengthscale value
gauss_keys.sort(key=lambda k: float(k.replace("K_gauss_", "").replace("p", ".")))

# ---------- 3. Geometric differences ----------
print("\nGeometric differences g_CQ:")
print("Classical kernel              g_CQ")
print("-" * 40)

def print_gcq(name, K_c):
    g_val, _, _, _, _ = metrics.geometric_difference(K_c, K_q)
    print(f"{name:<28s} {g_val:10.4f}")
    return g_val

print_gcq("Cosine",                cl["K_cos"])
print_gcq("Angular RBF",           cl["K_arbf"])
print_gcq("Cosine RBF",            cl["K_crbf"])
print_gcq("Cosine‑dist exp",       cl["K_cde"])
for key in gauss_keys:
    ell = key.replace("K_gauss_", "").replace("p", ".")
    print_gcq(f"Gaussian RBF (ℓ={ell})", cl[key])

# ---------- 4. Model complexities ----------
print("\nModel complexities s_K (binary Fm‑3m vs. Ia‑3):")
print("Kernel                         s_K")
print("-" * 40)

def print_sk(name, K):
    invK = np.linalg.inv(K)
    s_val = metrics.model_complexity(invK, y_true)
    print(f"{name:<28s} {s_val:12.4f}")
    return s_val

print_sk("Quantum (sim.)",             K_q)
print_sk("Cosine",                     cl["K_cos"])
print_sk("Angular RBF",                cl["K_arbf"])
print_sk("Cosine RBF",                 cl["K_crbf"])
print_sk("Cosine‑dist exp",            cl["K_cde"])
for key in gauss_keys:
    ell = key.replace("K_gauss_", "").replace("p", ".")
    print_sk(f"Gaussian RBF (ℓ={ell})", cl[key])