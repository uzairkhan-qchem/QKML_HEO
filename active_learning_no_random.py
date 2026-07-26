# active_learning_no_random.py  (CSV export only, no matplotlib)
import json, numpy as np, gpflow, kernels, csv
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

# ---------- 2. Active-learning loop ----------
def run_active_learning(K_full, seeds, test_size=10, max_labeled=22):
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
                (np.array(train_idx).reshape(-1,1).astype(float), y_train.reshape(-1,1)),
                kernel, gpflow.likelihoods.Bernoulli())
            gpflow.optimizers.Scipy().minimize(
                model.training_loss, model.trainable_variables,
                compile=False, options=dict(maxiter=1000))
            y_pred, _ = model.predict_y(test_idx.reshape(-1,1).astype(float))
            preds = np.round(y_pred.numpy().flatten()).astype(int)
            acc = np.mean(preds == y_true[test_idx])
            accs.append(acc)
            if len(pool_idx) > 0 and n_labeled < max_labeled:
                _, y_var = model.predict_y(np.array(pool_idx).reshape(-1,1).astype(float))
                next_pt = pool_idx.pop(np.argmax(y_var.numpy().flatten()))
                train_idx.append(next_pt)
        all_accs.append(accs)
    min_len = min(len(a) for a in all_accs)
    aligned = np.array([a[:min_len] for a in all_accs])
    return np.mean(aligned, axis=0), np.std(aligned, axis=0)

# ---------- 3. Run ----------
seeds = list(range(20))
results = {}
for name, K_full in tqdm(kernel_list, desc="Active learning"):
    mean_acc, std_acc = run_active_learning(K_full, seeds)
    results[name] = (mean_acc, std_acc)

# ---------- 4. Export CSV ----------
with open("active_learning_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["training_points", "kernel", "mean_accuracy", "std_accuracy"])
    for kernel_name, (mean_arr, std_arr) in results.items():
        for i, (m, s) in enumerate(zip(mean_arr, std_arr)):
            n_train = 2 + i          # starting from 2 labelled samples
            writer.writerow([n_train, kernel_name, m, s])

print("Data saved to active_learning_data.csv")
print("Use the R script to generate the publication-quality figure.")