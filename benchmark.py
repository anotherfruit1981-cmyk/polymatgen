from polymatgen.ml.predictors import TgPredictor
from polymatgen.ml.gcn_predictor import GCNTgPredictor
from polymatgen.ml.equivariant_predictor import EquivariantTgPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import os

print("=" * 50)
print("Benchmark: RF vs GCN vs Equivariant — Tg")
print("=" * 50)

print("\n--- Random Forest ---")
rf = TgPredictor(n_estimators=200)
rf.train()
m = rf.evaluate()
rf_r2, rf_mae, rf_rmse = m['r2'], m['mae'], m['rmse']
print(f"  n_train : {m['n_train']}")
print(f"  n_test  : {m['n_test']}")
print(f"  R2      : {rf_r2}")
print(f"  MAE     : {rf_mae:.2f} K")
print(f"  RMSE    : {rf_rmse:.2f} K")

print("\n--- GCN (100 epochs) ---")
gcn = GCNTgPredictor(epochs=100)
gcn.train()
m = gcn.evaluate_default()
gcn_r2, gcn_mae, gcn_rmse = m['r2'], m['mae'], m['rmse']
print(f"  n_test  : {m['n_test']}")
print(f"  R2      : {gcn_r2}")
print(f"  MAE     : {gcn_mae:.2f} K")
print(f"  RMSE    : {gcn_rmse:.2f} K")

print("\n--- E(3)-Equivariant GNN (50 epochs, 2000 samples) ---")
eq = EquivariantTgPredictor(epochs=50, max_train_samples=2000)
eq.train()

# Manual evaluate on held-out 20%
df = pd.read_csv(os.path.join("src", "polymatgen", "data",
    "LAMALAB_CURATED_Tg_structured_polymerclass_with_embeddings.csv"))
df = df[df["meta.reliability"] != "red"]
df = df[df["labels.Exp_Tg(K)"].notna()]
df = df[df["PSMILES"].notna()]

smiles_all = df["PSMILES"].tolist()
y_all = df["labels.Exp_Tg(K)"].tolist()
_, test_smiles, _, y_test = train_test_split(
    smiles_all, y_all, test_size=0.2, random_state=42
)

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
eq_r2 = round(float(r2_score(y_true, y_pred)), 4)
eq_mae = round(float(mean_absolute_error(y_true, y_pred)), 2)
eq_rmse = round(float(np.sqrt(np.mean((y_true - y_pred)**2))), 2)
print(f"  n_test  : {len(y_true)}")
print(f"  R2      : {eq_r2}")
print(f"  MAE     : {eq_mae:.2f} K")
print(f"  RMSE    : {eq_rmse:.2f} K")

print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"  RF           R2={rf_r2:.4f}  MAE={rf_mae:.2f} K  RMSE={rf_rmse:.2f} K")
print(f"  GCN          R2={gcn_r2:.4f}  MAE={gcn_mae:.2f} K  RMSE={gcn_rmse:.2f} K")
print(f"  Equivariant  R2={eq_r2:.4f}  MAE={eq_mae:.2f} K  RMSE={eq_rmse:.2f} K")
print("\nDone.")