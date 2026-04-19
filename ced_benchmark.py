"""
CED benchmark — RF vs GCN vs Equivariant
Cohesive Energy Density (MPa), 294 polymers, polyVERSE
"""

import os, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

LOG = "ced_results.txt"

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

# Load CED dataset
CED_PATH = os.path.join("src", "polymatgen", "data",
    "Cohesive_energy_density_2025_06_23.csv")
df = pd.read_csv(CED_PATH)
df = df[df["value_COE"].notna() & df["smiles1"].notna()]
smiles_all = df["smiles1"].tolist()
y_all = df["value_COE"].tolist()
_, test_smiles, _, y_test = train_test_split(
    smiles_all, y_all, test_size=0.2, random_state=42)

log("=" * 55)
log("CED Benchmark — RF vs GCN vs Equivariant")
log(f"Dataset: {len(smiles_all)} total, {len(test_smiles)} test")
log("=" * 55)

# RF
log("\n--- RF (200 trees) ---")
from polymatgen.ml.predictors import CohesiveEnergyPredictor
t0 = time.time()
rf = CohesiveEnergyPredictor(n_estimators=200)
rf.train()
m = rf.evaluate()
log(f"  R2={m['r2']}  MAE={m['mae']:.2f} MPa  RMSE={m['rmse']:.2f} MPa  "
    f"time={time.time()-t0:.0f}s")

# GCN sweep
log("\n--- GCN epochs sweep ---")
from polymatgen.ml.gcn_predictor import GCNCohesiveEnergyPredictor
for epochs in [200, 500, 1000, 2000, 3000]:
    log(f"\n  GCN epochs={epochs}")
    t0 = time.time()
    gcn = GCNCohesiveEnergyPredictor(epochs=epochs)
    gcn.train()
    m = gcn.evaluate_default()
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} MPa  RMSE={m['rmse']:.2f} MPa  "
        f"time={time.time()-t0:.0f}s")

# Equivariant — note CED only has 294 entries so use all of them
log("\n--- Equivariant epochs sweep (250 samples = ~85% of dataset) ---")
from polymatgen.ml.equivariant_predictor import EquivariantCohesiveEnergyPredictor
for epochs in [200, 500, 1000, 2000, 3000]:
    log(f"\n  Equivariant epochs={epochs}, samples=250")
    t0 = time.time()
    eq = EquivariantCohesiveEnergyPredictor(
        epochs=epochs, max_train_samples=250)
    eq.train()
    m = eval_equivariant(eq, test_smiles, y_test)
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} MPa  RMSE={m['rmse']:.2f} MPa  "
        f"time={time.time()-t0:.0f}s")

log("\n" + "=" * 55)
log("Done. Results saved to ced_results.txt")
log("=" * 55)