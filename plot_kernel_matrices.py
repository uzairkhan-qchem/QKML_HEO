# plot_kernel_matrices.py
import json, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from ast import literal_eval

# ---------- 1. Load quantum kernel ----------
with open("heo_quantum_kernel_32.json") as f:
    qdata = json.load(f)
n = len(qdata["xrd_data"]["xrd"])
K_q = np.ones((n, n))
for key, val in qdata["kernel_entries"].items():
    i, j = literal_eval(key)
    v = val["result"]["0"]
    K_q[i, j] = v
    K_q[j, i] = v
# jitter already included in saved file? We add just in case
K_q += 1e-6 * np.eye(n)

# ---------- 2. Load classical kernels ----------
cl = np.load("heo_classical_kernels_32.npz", allow_pickle=True)
K_arbf = cl["K_arbf"]
K_cde  = cl["K_cde"]
K_g1   = cl["K_gauss_1p0"]

# ---------- 3. Plotting ----------
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# ---- helper to draw a class-separation dashed line ----
def add_class_line(ax, n, offset=-0.5):
    ax.axhline(y=n + offset, color='white', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axvline(x=n + offset, color='white', linestyle='--', linewidth=1.2, alpha=0.8)

# ---- panel 1: quantum kernel (log scale) ----
im0 = axes[0].imshow(K_q, cmap='magma', norm=LogNorm(vmin=K_q[K_q > 0].min(), vmax=K_q.max()))
axes[0].set_title("Quantum (sim.)", fontweight='bold')
add_class_line(axes[0], 16)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# ---- panel 2: angular RBF ----
im1 = axes[1].imshow(K_arbf, cmap='viridis')
axes[1].set_title("Angular RBF", fontweight='bold')
add_class_line(axes[1], 16)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# ---- panel 3: cosine-dist exp ----
im2 = axes[2].imshow(K_cde, cmap='viridis')
axes[2].set_title("Cosine-dist exp", fontweight='bold')
add_class_line(axes[2], 16)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

# ---- panel 4: Gaussian RBF (l=1) ----
im3 = axes[3].imshow(K_g1, cmap='viridis')
axes[3].set_title("Gaussian RBF (l=1)", fontweight='bold')
add_class_line(axes[3], 16)
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

# ---- common labels ----
for ax in axes:
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.set_yticks([0, 8, 16, 24, 31])

plt.tight_layout(pad=1.2)
plt.savefig("kernel_matrices_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: kernel_matrices_comparison.png")