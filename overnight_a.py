"""
Overnight benchmark for polymatgen.
Runs automatically and saves all results to overnight_results.txt.
"""

import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

LOG = "overnight_results.txt"

def log(msg=""):
    print(msg)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def eval_equivariant(eq, test_smiles, y_test):
    y_true, y_pred = [], []
    for smi, y in zip(test_smiles, y_test):
        try:
            pred = eq.predict(smi)
            y_true.append(y)    # only append if predict succeeds
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

# Load Tg dataset once
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
# Experiment 1: RF with more trees
# -------------------------------------------------------
log("=" * 55)
log("Experiment 1: RF n_estimators sweep")
log("=" * 55)
from polymatgen.ml.predictors import TgPredictor
for n in [100, 200, 500]:
    log(f"\n  RF n_estimators={n}")
    t0 = time.time()
    rf = TgPredictor(n_estimators=n)
    rf.train()
    m = rf.evaluate()
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} K  RMSE={m['rmse']:.2f} K  "
        f"time={time.time()-t0:.0f}s")

# -------------------------------------------------------
# Experiment 2: GCN epochs sweep
# -------------------------------------------------------
log("\n" + "=" * 55)
log("Experiment 2: GCN epochs sweep")
log("=" * 55)
from polymatgen.ml.gcn_predictor import GCNTgPredictor
for epochs in [500, 1000]:
    log(f"\n  GCN epochs={epochs}")
    t0 = time.time()
    gcn = GCNTgPredictor(epochs=epochs)
    gcn.train()
    m = gcn.evaluate_default()
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} K  RMSE={m['rmse']:.2f} K  "
        f"time={time.time()-t0:.0f}s")

# -------------------------------------------------------
# Experiment 3: Equivariant — epochs sweep (7000 samples)
# -------------------------------------------------------
log("\n" + "=" * 55)
log("Experiment 3: Equivariant epochs sweep (7000 samples)")
log("=" * 55)
from polymatgen.ml.equivariant_predictor import EquivariantTgPredictor
for epochs in [100, 200, 300, 500, 1000]:
    log(f"\n  Equivariant epochs={epochs}, samples=7000")
    t0 = time.time()
    eq = EquivariantTgPredictor(epochs=epochs, max_train_samples=7000)
    eq.train()
    m = eval_equivariant(eq, test_smiles, y_test)
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} K  RMSE={m['rmse']:.2f} K  "
        f"time={time.time()-t0:.0f}s")

# -------------------------------------------------------
# Experiment 4: Equivariant — data size sweep (100 epochs)
# -------------------------------------------------------
log("\n" + "=" * 55)
log("Experiment 4: Equivariant data size sweep (100 epochs)")
log("=" * 55)
for n_samples in [500, 1000, 2000, 4000]:
    log(f"\n  Equivariant epochs=100, samples={n_samples}")
    t0 = time.time()
    eq = EquivariantTgPredictor(epochs=100, max_train_samples=n_samples)
    eq.train()
    m = eval_equivariant(eq, test_smiles, y_test)
    log(f"  R2={m['r2']}  MAE={m['mae']:.2f} K  RMSE={m['rmse']:.2f} K  "
        f"time={time.time()-t0:.0f}s")

# -------------------------------------------------------
# Experiment 5: Bandgap — all three models
# -------------------------------------------------------
log("\n" + "=" * 55)
log("Experiment 5: Bandgap — RF vs GCN vs Equivariant")
log("=" * 55)
from polymatgen.ml.predictors import BandgapPredictor
from polymatgen.ml.gcn_predictor import GCNBandgapPredictor
from polymatgen.ml.equivariant_predictor import EquivariantBandgapPredictor

log("\n  RF Bandgap")
t0 = time.time()
rf_bg = BandgapPredictor(n_estimators=200)
rf_bg.train()
m = rf_bg.evaluate()
log(f"  R2={m['r2']}  MAE={m['mae']:.3f} eV  RMSE={m['rmse']:.3f} eV  "
    f"time={time.time()-t0:.0f}s")

log("\n  GCN Bandgap (200 epochs)")
t0 = time.time()
gcn_bg = GCNBandgapPredictor(epochs=200)
gcn_bg.train()
m = gcn_bg.evaluate_default()
log(f"  R2={m['r2']}  MAE={m['mae']:.3f} eV  RMSE={m['rmse']:.3f} eV  "
    f"time={time.time()-t0:.0f}s")

log("\n  Equivariant Bandgap (100 epochs, 2000 samples)")
t0 = time.time()
eq_bg = EquivariantBandgapPredictor(epochs=100, max_train_samples=2000)
eq_bg.train()
BANDGAP_PATH = os.path.join("src", "polymatgen", "data", "bandgap_chain.csv")
df_bg = pd.read_csv(BANDGAP_PATH)
df_bg = df_bg[df_bg["bandgap_chain"].notna() & df_bg["smiles"].notna()]
_, ts, _, yt = train_test_split(
    df_bg["smiles"].tolist(), df_bg["bandgap_chain"].tolist(),
    test_size=0.2, random_state=42)
m = eval_equivariant(eq_bg, ts, yt)
log(f"  R2={m['r2']}  MAE={m['mae']:.3f} eV  RMSE={m['rmse']:.3f} eV  "
    f"time={time.time()-t0:.0f}s")

log("\n" + "=" * 55)
log("All experiments complete. Results saved to overnight_results.txt")
log("=" * 55)