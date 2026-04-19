import os, time
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

from polymatgen.ml.predictors import BandgapPredictor
from polymatgen.ml.gcn_predictor import GCNBandgapPredictor
from polymatgen.ml.equivariant_predictor import EquivariantBandgapPredictor

BANDGAP_PATH = os.path.join("src", "polymatgen", "data", "bandgap_chain.csv")
df_bg = pd.read_csv(BANDGAP_PATH)
df_bg = df_bg[df_bg["bandgap_chain"].notna() & df_bg["smiles"].notna()]
_, ts, _, yt = train_test_split(
    df_bg["smiles"].tolist(), df_bg["bandgap_chain"].tolist(),
    test_size=0.2, random_state=42)

log("\n" + "=" * 55)
log("Experiment 5 (continued): Bandgap — GCN + Equivariant")
log("=" * 55)

log("\n  GCN Bandgap (1000 epochs)")
t0 = time.time()
gcn_bg = GCNBandgapPredictor(epochs=1000)
gcn_bg.train()
m = gcn_bg.evaluate_default()
log(f"  R2={m['r2']}  MAE={m['mae']:.3f} eV  RMSE={m['rmse']:.3f} eV  "
    f"time={time.time()-t0:.0f}s")

log("\n  Equivariant Bandgap (1000 epochs, 4000 samples)")
t0 = time.time()
eq_bg = EquivariantBandgapPredictor(epochs=1000, max_train_samples=4000)
eq_bg.train()
m = eval_equivariant(eq_bg, ts, yt)
log(f"  R2={m['r2']}  MAE={m['mae']:.3f} eV  RMSE={m['rmse']:.3f} eV  "
    f"time={time.time()-t0:.0f}s")

log("\nAll done.")