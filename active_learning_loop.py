# active_learning_loop.py  (final – larger test set, more seeds, ASCII-safe)
import json, numpy as np, gpflow, kernels, matplotlib.pyplot as plt
from ast import literal_eval
from tqdm import tqdm

# ---------- 1. Load data and kernels ----------
data = np.load("heo_binary_data.npz", allow_pickle=True)
y_true = np.array([0 if l == "Fm-3m" else 1 for l in data["labels"]])
N_total = len(y_true)

with open("heo_quantum_kernel_32.json") as f:
    qdata = json.load(f)
K_q = np.ones((N_total, N_total))
for key, val in qdata["kernel_entries"].items():
    i, j = literal_eval(key)
    v = val["result"]["0"]
    K_q[i, j] = v; K_q[j, i] = v
K_q += 1e-6 * np.eye(N_total)

cl = np.load("heo_classical_kernels_32.npz", allow_pickle=True)
kernel_list = [
    ("Quantum (sim.)",         K_q),
    ("Angular RBF",            cl["K_arbf"]),
    ("Cosine-dist exp",        cl["K_cde"]),
    ("Gaussian RBF (l=1)",     cl["K_gauss_1p0"]),
]

# ---------- 2. Active-learning loop with larger hold-out test set ----------
def run_active_learning(K_full, seeds, test_size=10, max_labeled=22):
    """
    For each seed:
      - Randomly select `test_size` hold-out samples (never seen).
      - Start with 2 labelled samples from the remaining pool.
      - At each iteration, train Bernoulli VGP, predict on the fixed test set,
        record accuracy, then acquire the pool sample with highest predictive variance.
    Returns (mean_accuracy_array, std_accuracy_array) over seeds.
    """
    all_accs = []
    for seed in tqdm(seeds, desc="  Seed", leave=False):
        np.random.seed(seed)
        idx = np.random.permutation(N_total)
        test_idx = idx[:test_size]                # fixed hold-out set
        pool_idx = list(idx[test_size:])          # everything else
        train_idx = [pool_idx.pop(0), pool_idx.pop(0)]   # start with 2 labelled
        accs = []
        for n_labeled in range(2, max_labeled + 1):
            y_train = y_true[train_idx]
            kernel = kernels.FixedPrecomputedGPKernel(K_full)
            model = gpflow.models.VGP(
                (np.array(train_idx).reshape(-1,1).astype(float), y_train.reshape(-1,1)),
                kernel, gpflow.likelihoods.Bernoulli())
            gpflow.optimizers.Scipy().minimize(
                model.training_loss, model.trainable_variables,
                compile=False, options=dict(maxiter=1000))   # more iterations for convergence

            # Evaluate on the fixed hold-out test set
            y_pred, _ = model.predict_y(test_idx.reshape(-1,1).astype(float))
            preds = np.round(y_pred.numpy().flatten()).astype(int)
            acc = np.mean(preds == y_true[test_idx])
            accs.append(acc)

            # Acquire next point (uncertainty sampling)
            if len(pool_idx) > 0 and n_labeled < max_labeled:
                _, y_var = model.predict_y(np.array(pool_idx).reshape(-1,1).astype(float))
                next_pt = pool_idx.pop(np.argmax(y_var.numpy().flatten()))
                train_idx.append(next_pt)
        all_accs.append(accs)

    min_len = min(len(a) for a in all_accs)
    aligned = np.array([a[:min_len] for a in all_accs])
    return np.mean(aligned, axis=0), np.std(aligned, axis=0)

# ---------- 3. Run experiment ----------
seeds = list(range(20))          # 20 seeds for stable error bars
max_labeled = 22                 # 10 test + 22 train = 32 total

results = {}
for name, K_full in tqdm(kernel_list, desc="Active learning"):
    mean_acc, std_acc = run_active_learning(K_full, seeds, max_labeled=max_labeled)
    results[name] = (mean_acc, std_acc)

# Random baseline (same protocol, random acquisition)
random_accs = []
for seed in tqdm(seeds, desc="Random baseline", leave=False):
    np.random.seed(seed)
    idx = np.random.permutation(N_total)
    test_idx = idx[:10]
    pool_idx = list(idx[10:])
    train_idx = [pool_idx.pop(0), pool_idx.pop(0)]
    accs = []
    for n_labeled in range(2, max_labeled + 1):
        y_train = y_true[train_idx]
        kernel = kernels.FixedPrecomputedGPKernel(K_q)
        model = gpflow.models.VGP(
            (np.array(train_idx).reshape(-1,1).astype(float), y_train.reshape(-1,1)),
            kernel, gpflow.likelihoods.Bernoulli())
        gpflow.optimizers.Scipy().minimize(
            model.training_loss, model.trainable_variables,
            compile=False, options=dict(maxiter=1000))
        y_pred, _ = model.predict_y(test_idx.reshape(-1,1).astype(float))
        preds = np.round(y_pred.numpy().flatten()).astype(int)
        acc = np.mean(preds == y_true[test_idx])
        accs.append(acc)
        if len(pool_idx) > 0:
            next_pt = pool_idx.pop(np.random.randint(len(pool_idx)))
            train_idx.append(next_pt)
    random_accs.append(accs)

min_len = min(len(a) for a in random_accs)
aligned_rand = np.array([a[:min_len] for a in random_accs])
results["Random acquisition"] = (np.mean(aligned_rand, axis=0),
                                 np.std(aligned_rand, axis=0))

# ---------- 4. Publication-quality plot (ASCII-safe) ----------
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'legend.fontsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

colors = {'Quantum (sim.)': '#2166AC', 'Angular RBF': '#D6604D',
          'Cosine-dist exp': '#4DAF4A', 'Gaussian RBF (l=1)': '#984EA3',
          'Random acquisition': '#FF7F00'}
linestyles = {'Quantum (sim.)': '-', 'Angular RBF': '--',
              'Cosine-dist exp': '-.', 'Gaussian RBF (l=1)': ':',
              'Random acquisition': '-'}

fig, ax = plt.subplots(figsize=(8, 5.5))
for name, (mean, std) in results.items():
    iters = np.arange(2, 2 + len(mean))
    ax.plot(iters, mean, color=colors[name], linestyle=linestyles[name],
            linewidth=2.2, label=name)
    ax.fill_between(iters, mean - std, mean + std, alpha=0.12, color=colors[name])

ax.axhline(0.90, color='grey', linestyle=':', linewidth=1.2, alpha=0.8, label='90% target')
ax.set_xlabel("Number of labelled samples", labelpad=8)
ax.set_ylabel("Accuracy on held-out test set", labelpad=8)
ax.set_title("Active Learning for HEO Crystal-Structure Classification", pad=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8); ax.spines[spine].set_color('grey')
ax.tick_params(length=4, width=0.8, colors='grey')
ax.grid(False)
ax.legend(frameon=False, loc='lower right')

plt.tight_layout(pad=1.3)
plt.savefig("active_learning_convergence.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Figure saved: active_learning_convergence.png")