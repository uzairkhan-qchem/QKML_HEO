# metrics.py
import numpy as np

def geometric_difference(K_classical, K_quantum):
# NOTE: This SVD-based inversion assumes K_classical is positive definite.
# For indefinite kernels (e.g., Cosine RBF), g_CQ and s_K are not meaningful.
    Uc, Sc, Vhc = np.linalg.svd(K_classical, hermitian=True)
    Uq, Sq, Vhq = np.linalg.svd(K_quantum, hermitian=True)
    sqrt_Kq = Uq @ np.diag(np.sqrt(Sq)) @ Vhq
    inv_Kc = Vhc.T @ np.diag(1.0 / Sc) @ Uc.T
    M = sqrt_Kq @ inv_Kc @ sqrt_Kq
    sing_vals = np.linalg.svd(M, compute_uv=False)
    eig_vals, eig_vecs = np.linalg.eig(M)
    g_cq = np.sqrt(np.max(sing_vals))
    return g_cq, sing_vals, eig_vals, eig_vecs, sqrt_Kq

def model_complexity(K_inv, labels):
    s = 0.0
    n = K_inv.shape[0]
    for i in range(n):
        for j in range(n):
            s += K_inv[i, j] * (1.0 if labels[i] == labels[j] else -1.0)
    return s

def engineered_labels(sqrt_Kq, eig_vecs):
    # Select eigenvector corresponding to the largest eigenvalue
    # (np.linalg.eig does not guarantee ordering)
    idx = np.argmax(eig_vals.real)
    leading_eigvec = eig_vecs[:, idx]
    vec = sqrt_Kq @ leading_eigvec
    return np.array([0 if x < 0 else 1 for x in vec])