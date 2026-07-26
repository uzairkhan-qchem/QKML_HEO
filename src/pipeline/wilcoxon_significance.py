# wilcoxon_significance.py  (corrected – uses the paired-split pooled file)
import numpy as np
from scipy.stats import wilcoxon

data = np.load("heo_pooled_results_32.npz", allow_pickle=True)
pooled = data["pooled"].item()
train_sizes = data["train_sizes"]

q_key = "Quantum (sim.)"
classical_keys = ["Angular RBF", "Cosine‑dist exp", "Gaussian RBF (l=1)"]

print("Wilcoxon signed‑rank test (quantum accuracy – classical accuracy):")
print("Size  |  Angular RBF         |  Cosine‑dist exp      |  Gaussian RBF (l=1)")
print("-" * 75)

for ts in train_sizes:
    acc_q = np.array(pooled[q_key][ts])
    row = f"  {ts:2d}  |"
    for ck in classical_keys:
        acc_c = np.array(pooled[ck][ts])
        diff = acc_q - acc_c
        if np.all(diff == 0):
            p = 1.0
        else:
            _, p = wilcoxon(diff)
        mean_diff = np.mean(diff)
        row += f"  {mean_diff:+.3f} (p={p:.4f})  |"
    print(row)