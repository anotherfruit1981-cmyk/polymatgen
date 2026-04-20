"""
overnight2.py — GCN and Equivariant extended epoch runs
Tests 2000 and 3000 epochs to find the practical ceiling.
Results saved to overnight2_results.txt
"""

import os, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

LOG = "overnight2_results.txt"

def log(msg=""):
    print(msg)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def eval_equivariant(eq, test_smiles, y_test):
    y_true, y_pred = [], []
    for smi, y in zip(test_smiles, y_test):
        try:
            pred = eq.predict(smi)
            y_true.append(y)
            y_pred.append(pred)
        except Exception:
            continue
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "n_test": len(y_true),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(np.mean((y_true-y_pred)**2))), 2),
    }

# Load Tg dataset
log("Loading Tg dataset...")
TG_PATH = os.path.join("src", "polymatgen", "data",
    "LAMALAB_CURATED_Tg_structured_polymerclass_with_embeddings.csv")
df = pd.read_csv(TG_PATH)
df = df[df["meta.reliability"] != "red"]
df = df[df["labels.Exp_Tg(K)"].notna()]
df = df[df["PSMILES"].notna()]
smiles_all = df["PSMILES"].tolist()
y_all = df["labels.Exp_Tg(K)"].tolist()
_, test_smiles, _, y_test = train_test_split(
    smiles_all, y_all, test_size=0.2, random_state=42)
log(f"Dataset: {len(smiles_all)} total, {len(test_smiles)} test\n")

# -------------------------------------------------------
# GCN — 2000 and 3000 epochs
# -------------------------------------------------------
log("=" * 55)
log("GCN extended epochs (continuing from 1000)")
log("=" * 55)
from polymatgen.ml.gcn_predictor import GCNTgPredictor
for epochs in [2000, 3000]:
    log(f"\n  GCN epochs={epochs}")
    t0 = time.time()
    gcn = GCNTgPredictor(epochs=epochs)
    gcn.train()
    m = gcn.evaluate_default()
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} K  RMSE={m['rmse']:.2f} K  "
        f"time={time.time()-t0:.0f}s")

# -------------------------------------------------------
# Equivariant — 2000 and 3000 epochs (7000 samples)
# -------------------------------------------------------
log("\n" + "=" * 55)
log("Equivariant extended epochs (7000 samples)")
log("=" * 55)
from polymatgen.ml.equivariant_predictor import EquivariantTgPredictor
for epochs in [2000, 3000]:
    log(f"\n  Equivariant epochs={epochs}, samples=7000")
    t0 = time.time()
    eq = EquivariantTgPredictor(epochs=epochs, max_train_samples=7000)
    eq.train()
    m = eval_equivariant(eq, test_smiles, y_test)
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} K  RMSE={m['rmse']:.2f} K  "
        f"time={time.time()-t0:.0f}s")

log("\n" + "=" * 55)
log("Done. Results saved to overnight2_results.txt")
log("=" * 55)