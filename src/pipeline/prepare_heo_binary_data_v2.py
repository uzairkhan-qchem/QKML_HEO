# prepare_heo_binary_data_v2.py
import pandas as pd
import numpy as np
import re
import os

# ---------- 1. identify sample indices ----------
df = pd.read_csv(
    os.path.join("ML4HEOs", "data", "chemical_composition_table.csv"),
    header=None, skiprows=3
)
sample_ids = pd.to_numeric(df.iloc[:, 1], errors='coerce')
phase_str = df.iloc[:, 12].fillna("")
valid = ~sample_ids.isna()
sample_ids = sample_ids[valid].astype(int)
phase_str = phase_str[valid]

fluorite, bixbyite = [], []
for idx, pstr in zip(sample_ids, phase_str):
    pstr_clean = pstr.strip().strip('"').replace("\n", " ")
    parts = re.findall(r'([A-Za-z0-9/-]+)\s*\((\d+\.?\d*)\s*%\)', pstr_clean)
    if not parts:
        ph = pstr_clean.strip()
        if ph == "Fm-3m":      fluorite.append(idx)
        elif ph == "Ia-3":     bixbyite.append(idx)
        continue
    phases = set()
    purity_map = {}
    for ph_name, pur_str in parts:
        pur = float(pur_str)
        phases.add(ph_name)
        purity_map[ph_name] = pur
    if phases - {"Fm-3m", "Ia-3"}:
        continue
    if len(phases) == 1:
        ph = list(phases)[0]
        if purity_map[ph] >= 98.0:
            if ph == "Fm-3m":      fluorite.append(idx)
            else:                   bixbyite.append(idx)

np.random.seed(42)
fluorite_sel = sorted(np.random.choice(fluorite, size=16, replace=False))
bixbyite_sel = sorted(bixbyite)
all_indices = np.array(fluorite_sel + bixbyite_sel)
labels = np.array(["Fm-3m"] * 16 + ["Ia-3"] * 16)

print("Selected indices:", all_indices)

# ---------- 2. load raw XRD data ----------
df_xrd = pd.read_csv(
    os.path.join("ML4HEOs", "data", "XRDdata.csv"),
    header=None, skiprows=1
)
angles = df_xrd.iloc[:, 0].values.astype(float)
intensity_cols = [i for i in range(1, df_xrd.shape[1]) if i % 2 == 1]
X_all = df_xrd.iloc[:, intensity_cols].values.T.astype(float)

# ---------- 3. subsample to 150 angles (raw intensities) ----------
num_features = 150
theta_idx = np.linspace(0, len(angles)-1, num_features, dtype=int)
X_raw = X_all[all_indices - 1][:, theta_idx]

# ---------- 4. classical version: L2-normalise each row ----------
X_classical = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)

# ---------- 5. quantum version: full preprocessing ----------
normalize_scale = 0.001
X_quantum = 1.0 / (1.0 + np.exp(-normalize_scale * X_raw))
X_quantum = X_quantum / np.sum(X_quantum, axis=1, keepdims=True)
x_min, x_max = np.min(X_quantum), np.max(X_quantum)
X_quantum = np.pi * (X_quantum - x_min) / (x_max - x_min)

# ---------- 6. save ----------
np.savez("heo_binary_data.npz",
         xrd_classical=X_classical,
         xrd_quantum=X_quantum,
         labels=labels,
         selected_indices=all_indices,
         angles_sub=angles[theta_idx])
print(f"Saved heo_binary_data.npz")
print(f"Classical shape: {X_classical.shape}, Quantum shape: {X_quantum.shape}")