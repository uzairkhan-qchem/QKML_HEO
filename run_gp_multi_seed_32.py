# run_gp_multi_seed_32.py  (corrected – shared splits, nan‑safe)
import json, numpy as np, gpflow, kernels
from ast import literal_eval
from scipy.stats import norm, wilcoxon
from tqdm import tqdm

# ---------- Load labels ----------
data = np.load("heo_binary_data.npz", allow_pickle=True)
labels_str = data["labels"]
y_true = np.array([0 if l == "Fm-3m" else 1 for l in labels_str])
N_total = len(y_true)

# ---------- Quantum kernel ----------
with open("heo_quantum_kernel_32.json") as f:
    qdata = json.load(f)
K_q = np.ones((N_total, N_total))
for key, val in qdata["kernel_entries"].items():
    i, j = literal_eval(key)
    v = val["result"]["0"]
    K_q[i, j] = v; K_q[j, i] = v
K_q += 1e-6 * np.eye(N_total)

# ---------- Classical kernels ----------
cl = np.load("heo_classical_kernels_32.npz", allow_pickle=True)
kernels_to_test = [
    ("Quantum (sim.)",         K_q),
    ("Angular RBF",            cl["K_arbf"]),
    ("Cosine‑dist exp",        cl["K_cde"]),
    ("Gaussian RBF (l=1)",     cl["K_gauss_1p0"]),
]

def fit_and_predict(K_full, train_idx, test_idx, y_train):
    kernel = kernels.FixedPrecomputedGPKernel(K_full)
    model = gpflow.models.VGP(
        (train_idx.reshape(-1,1).astype(float), y_train.reshape(-1,1)),
        kernel, gpflow.likelihoods.Bernoulli())
    gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables,
                                       compile=False, options=dict(maxiter=500))
    y_mean, _ = model.predict_y(test_idx.reshape(-1,1).astype(float))
    return np.round(y_mean.numpy().flatten()).astype(int)

seeds = [0, 1, 2]
train_sizes = list(range(5, 32))
n_repeats = 25

# ---------- Precompute all splits once ----------
# splits[seed][ts][rep] = (train_idx, test_idx)
splits = {}
for seed in seeds:
    np.random.seed(seed)
    splits[seed] = {}
    for ts in train_sizes:
        splits[seed][ts] = []
        for _ in range(n_repeats):
            idx = np.random.permutation(N_total)
            splits[seed][ts].append((idx[:ts], idx[ts:]))

# ---------- Evaluate each kernel on the same splits ----------
all_acc = {name: {s: {} for s in seeds} for name, _ in kernels_to_test}
error_log = []      # <-- new: collects any fitting exceptions

for name, K_full in tqdm(kernels_to_test, desc="Kernel"):
    for seed in tqdm(seeds, desc=" Seed", leave=False):
        for ts in tqdm(train_sizes, desc="  Train size", leave=False):
            accs = []
            for train_idx, test_idx in splits[seed][ts]:
                y_train = y_true[train_idx]
                try:
                    preds = fit_and_predict(K_full, train_idx, test_idx, y_train)
                    accs.append(np.mean(preds == y_true[test_idx]))
                except Exception as e:
                    # Log the exception and store NaN instead of 0.0
                    error_log.append({
                        "kernel": name,
                        "seed": seed,
                        "train_size": ts,
                        "error": repr(e)
                    })
                    accs.append(np.nan)          # <-- NaN replaces 0.0
            all_acc[name][seed][ts] = accs

# ---------- Report any fitting failures ----------
if error_log:
    print(f"WARNING: {len(error_log)} GP fitting failure(s) occurred:")
    for err in error_log:
        print(err)
else:
    print("No GP fitting failures detected (error log empty).")

# ---------- Pool across seeds (ignoring NaN entries) ----------
pooled = {}
for name, _ in kernels_to_test:
    pooled[name] = {ts: [] for ts in train_sizes}
    for seed in seeds:
        for ts in train_sizes:
            # Keep only finite accuracies when pooling
            valid = [v for v in all_acc[name][seed][ts] if not np.isnan(v)]
            pooled[name][ts].extend(valid)

# ---------- Relative accuracy (pooled) ----------
def katz_ratio_ci(acc_q, acc_c):
    eps = 1e-9
    q = np.clip(acc_q, eps, 1); c = np.clip(acc_c, eps, 1)
    log_r = np.log(q / c)
    m = np.mean(log_r); se = np.std(log_r, ddof=1)/np.sqrt(len(log_r))
    z = norm.ppf(0.975)
    return np.exp(m), np.exp(m - z*se), np.exp(m + z*se)

print("\nPooled relative accuracy (75 splits per size):")
print("Size | Angular RBF         | Cosine‑dist exp      | Gaussian RBF (l=1)")
print("-"*75)
for ts in train_sizes:
    acc_q = np.array(pooled["Quantum (sim.)"][ts])
    r_a, la, ha = katz_ratio_ci(acc_q, np.array(pooled["Angular RBF"][ts]))
    r_c, lc, hc = katz_ratio_ci(acc_q, np.array(pooled["Cosine‑dist exp"][ts]))
    r_g, lg, hg = katz_ratio_ci(acc_q, np.array(pooled["Gaussian RBF (l=1)"][ts]))
    print(f" {ts:2d} | {r_a:.3f} ({la:.3f},{ha:.3f})  | {r_c:.3f} ({lc:.3f},{hc:.3f})  | {r_g:.3f} ({lg:.3f},{hg:.3f})")

# ---------- Wilcoxon pooled (NOW CORRECTLY PAIRED) ----------
print("\nWilcoxon (pooled, 75 PAIRED splits):")
print("Size | Angular RBF          | Cosine‑dist exp       | Gaussian RBF (l=1)")
print("-"*75)
for ts in train_sizes:
    acc_q = np.array(pooled["Quantum (sim.)"][ts])
    row = f" {ts:2d} |"
    for ck in ["Angular RBF", "Cosine‑dist exp", "Gaussian RBF (l=1)"]:
        acc_c = np.array(pooled[ck][ts])
        diff = acc_q - acc_c
        if np.all(diff == 0):
            p = 1.0
        else:
            _, p = wilcoxon(diff)
        row += f" {np.mean(diff):+.3f} (p={p:.4f}) |"
    print(row)

# Save corrected pooled data
np.savez("heo_pooled_results_32.npz",
         train_sizes=train_sizes, pooled=pooled)
print("\nPooled results saved to heo_pooled_results_32.npz")