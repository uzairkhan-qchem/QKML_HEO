# compute_classical_kernels.py
import numpy as np
import kernels

# ---------- 1. Load data ----------
d = np.load("heo_binary_data.npz", allow_pickle=True)
X_classical = d["xrd_classical"]     # L2‑normalised – for classical kernels
labels_str  = d["labels"]

# ---------- 2. Helper to build a classical kernel ----------
jitter = 1e-6
def compute_classical_kernel(X, kernel_func, *args):
    n = X.shape[0]
    K = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            val = kernel_func(X[i], X[j], *args)
            K[i, j] = val
            K[j, i] = val
    K += jitter * np.eye(n)
    return K

# ---------- 3. Build standard classical kernels ----------
print("Computing classical kernel matrices on L2‑normalised data...")

K_cos   = compute_classical_kernel(X_classical, kernels.cosine_kernel_function)
K_arbf  = compute_classical_kernel(X_classical, kernels.angular_rbf_kernel_function)
K_crbf  = compute_classical_kernel(X_classical, kernels.cosine_rbf_kernel_function)
K_cde   = compute_classical_kernel(X_classical, kernels.cosine_distance_exponential_kernel_function)

# ---------- 4. Gaussian RBF with multiple lengthscales ----------
lengthscales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
K_gauss = {}
for ell in lengthscales:
    K_gauss[ell] = compute_classical_kernel(
        X_classical, kernels.gaussian_rbf_euclidean_kernel_function, ell)
    print(f"  Gaussian RBF (ℓ={ell:.2f}) computed")

# ---------- 5. Save all matrices ----------
np.savez("heo_classical_kernels_32.npz",
         K_cos=K_cos, K_arbf=K_arbf, K_crbf=K_crbf, K_cde=K_cde,
         K_gauss_0p5=K_gauss[0.5],
         K_gauss_0p75=K_gauss[0.75],
         K_gauss_1p0=K_gauss[1.0],
         K_gauss_1p5=K_gauss[1.5],
         K_gauss_2p0=K_gauss[2.0],
         K_gauss_3p0=K_gauss[3.0],
         K_gauss_4p0=K_gauss[4.0],
         K_gauss_5p0=K_gauss[5.0],
         K_gauss_7p0=K_gauss[7.0],
         K_gauss_10p0=K_gauss[10.0],
         lengthscales=np.array(lengthscales),
         labels_str=labels_str)
print("\nAll classical kernel matrices saved to heo_classical_kernels_32.npz")
print("Keys: K_cos, K_arbf, K_crbf, K_cde, K_gauss_* for each lengthscale")