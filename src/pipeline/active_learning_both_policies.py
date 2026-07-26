# active_learning_both_policies.py  (CSV export, no matplotlib)
import json, numpy as np, gpflow, kernels, csv
import os
from ast import literal_eval
from tqdm import tqdm

# ---------- 1. Load data and kernels ----------
data = np.load(os.path.join("data","processed","heo_binary_data.npz"), allow_pickle=True)
y_true = np.array([0 if l == "Fm-3m" else 1 for l in data["labels"]])
N_total = len(y_true)

with open(os.path.join("data","processed","heo_quantum_kernel_32.json")) as f:
    qdata = json.load(f)
K_q = np.ones((N_total, N_total))
for key, val in qdata["kernel_entries"].items():
    i, j = literal_eval(key)
    v = val["result"]["0"]
    K_q[i, j] = v; K_q[j, i] = v
K_q += 1e-6 * np.eye(N_total)

cl = np.load(os.path.join("data","processed","heo_classical_kernels_32.npz"), allow_pickle=True)
kernel_list = [
    ("Quantum (sim.)",         K_q),
    ("Angular RBF",            cl["K_arbf"]),
    ("Cosine-dist exp",        cl["K_cde"]),
    ("Gaussian RBF (l=1)",     cl["K_gauss_1p0"]),
]

# ---------- 2. Active-learning loop with acquisition policy ----------
def run_active_learning(K_full, seeds, test_size=10, max_labeled=22,
                        acquisition='uncertainty'):
    """
    acquisition : 'uncertainty' or 'random'
    """
    all_accs = []
    for seed in tqdm(seeds, desc="  Seed", leave=False):
        np.random.seed(seed)
        idx = np.random.permutation(N_total)
        test_idx = idx[:test_size]
        pool_idx = list(idx[test_size:])
        train_idx = [pool_idx.pop(0), pool_idx.pop(0)]
        accs = []
        for n_labeled in range(2, max_labeled + 1):
            y_train = y_true[train_idx]
            kernel = kernels.FixedPrecomputedGPKernel(K_full)
            model = gpflow.models.VGP(
                (np.array(train_idx).reshape(-1,1).astype(float),
                 y_train.reshape(-1,1)),
                kernel, gpflow.likelihoods.Bernoulli())
            gpflow.optimizers.Scipy().minimize(
                model.training_loss, model.trainable_variables,
                compile=False, options=dict(maxiter=1000))
            y_pred, _ = model.predict_y(test_idx.reshape(-1,1).astype(float))
            preds = np.round(y_pred.numpy().flatten()).astype(int)
            acc = np.mean(preds == y_true[test_idx])
            accs.append(acc)
            if len(pool_idx) > 0 and n_labeled < max_labeled:
                if acquisition == 'uncertainty':
                    _, y_var = model.predict_y(
                        np.array(pool_idx).reshape(-1,1).astype(float))
                    next_pt = pool_idx.pop(np.argmax(y_var.numpy().flatten()))
                else:  # random acquisition
                    next_pt = pool_idx.pop(np.random.randint(len(pool_idx)))
                train_idx.append(next_pt)
        all_accs.append(accs)
    min_len = min(len(a) for a in all_accs)
    aligned = np.array([a[:min_len] for a in all_accs])
    return np.mean(aligned, axis=0), np.std(aligned, axis=0)

# ---------- 3. Run all experiments ----------
seeds = list(range(20))
results = {}

# Uncertainty sampling for each kernel
for name, K_full in tqdm(kernel_list, desc="Uncertainty sampling"):
    mean_acc, std_acc = run_active_learning(K_full, seeds,
                                            acquisition='uncertainty')
    results[name] = (mean_acc, std_acc)

# Random acquisition – uses the quantum kernel as the classifier but acquires randomly
# (same as the original random baseline in active_learning_loop.py)
random_key = "Random acquisition"
mean_rnd, std_rnd = run_active_learning(K_q, seeds,
                                         acquisition='random')
results[random_key] = (mean_rnd, std_rnd)

# ---------- 4. Export CSV ----------
with open(os.path.join("data","processed","active_learning_data.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["training_points", "kernel", "mean_accuracy",
                     "std_accuracy"])
    for kernel_name, (mean_arr, std_arr) in results.items():
        for i, (m, s) in enumerate(zip(mean_arr, std_arr)):
            n_train = 2 + i
            writer.writerow([n_train, kernel_name, m, s])

print("Data saved to active_learning_data.csv")
print("Use the R script to generate the publication-quality figure.")