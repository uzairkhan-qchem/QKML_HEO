# random_split_accuracy.py
import numpy as np, matplotlib.pyplot as plt

# ---------- 1. Load pooled results ----------
data = np.load("heo_pooled_results_32.npz", allow_pickle=True)
pooled = data["pooled"].item()
train_sizes = data["train_sizes"]

# ---------- 2. Kernel display names -> exact keys in .npz ----------
kernel_keys = {
    'Quantum (sim.)':     'Quantum (sim.)',
    'Angular RBF':        'Angular RBF',
    'Cosine‑dist exp':    'Cosine‑dist exp',      # non‑breaking hyphen
    'Gaussian RBF (l=1)': 'Gaussian RBF (l=1)',
}

colors = {
    'Quantum (sim.)': '#2166AC', 'Angular RBF': '#D6604D',
    'Cosine‑dist exp': '#4DAF4A', 'Gaussian RBF (l=1)': '#984EA3',
}
linestyles = {
    'Quantum (sim.)': '-', 'Angular RBF': '--',
    'Cosine‑dist exp': '-.', 'Gaussian RBF (l=1)': ':',
}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 13,
    'legend.fontsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

fig, ax = plt.subplots(figsize=(8, 5.5))
for label, key in kernel_keys.items():
    # trim to training sizes ≤ 30 (avoid the degenerate tail)
    mask = train_sizes <= 30
    ts = train_sizes[mask]
    means = [np.mean(pooled[key][t]) for t in ts]
    stds  = [np.std(pooled[key][t])  for t in ts]
    ax.plot(ts, means, color=colors[label], linestyle=linestyles[label],
            linewidth=2.2, label=label)
    ax.fill_between(ts, np.array(means)-np.array(stds),
                    np.array(means)+np.array(stds), alpha=0.15, color=colors[label])

ax.set_xlabel("Number of training points", labelpad=8)
ax.set_ylabel("Accuracy", labelpad=8)
ax.set_title("Few-Shot Classification on Crystal-Structure Labels", pad=12)

# Manuscript‑style ticks: multiples of 5
ax.set_xticks([5, 10, 15, 20, 25, 30])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8); ax.spines[spine].set_color('grey')
ax.tick_params(length=4, width=0.8, colors='grey')
ax.grid(False)
ax.legend(frameon=False, loc='lower right')

plt.tight_layout(pad=1.3)
plt.savefig("random_split_accuracy.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved: random_split_accuracy.png")