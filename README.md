# polymatgen

A polymer-first materials informatics library for Python — built from scratch. Designed around polymer topology and chemistry rather than crystallographic symmetry, polymatgen provides tools for representing, analysing, and querying polymer systems.

## Features

- **Core data model** — `Monomer`, `Chain`, `Polymer` classes with RDKit integration
- **Property calculators** — Fox equation (Tg), Flory-Huggins χ, Hildebrand solubility parameter, contour length, radius of gyration
- **IO** — JSON, LAMMPS summary, and CSV read/write
- **Analysis** — molecular weight distribution, sequence analysis, chain statistics
- **Databases** — built-in reference data, PI1M (1M polymer SMILES), polyVERSE (chi parameter, bandgap, gas permeability, cohesive energy density, experimental Tg for 7,367 polymers)
- **ML predictors** — three backends for Tg, bandgap, and cohesive energy density:
  - Morgan fingerprint + Random Forest (fast baseline)
  - Graph Convolutional Network (learned 2D graph representations)
  - E(3)-Equivariant GNN (3D conformer-based, rotation/translation invariant)
- **Uncertainty quantification** — `predict_with_uncertainty()` on all RF predictors via tree variance
- **Inverse design** — genetic algorithm over PI1M space for multi-property constrained polymer search

## Benchmark Results

All models evaluated on held-out 20% test sets. Tg dataset: 7,363 polymers (PolyMetriX). Bandgap dataset: 4,209 polymers (polyVERSE DFT).

### Glass Transition Temperature (Tg)

| Model | R² | MAE (K) | RMSE (K) |
|---|---|---|---|
| RF — Morgan FP (200 trees) | 0.864 | 28.3 | 41.3 |
| GCN (300 epochs) | 0.901 | 26.3 | 35.2 |
| GCN (1000 epochs) | **0.943** | **19.7** | **26.7** |
| E(3)-Equivariant (1000 epochs, 7k samples) | 0.937 | 20.9 | 28.0 |

### Electronic Bandgap

| Model | R² | MAE (eV) | RMSE (eV) |
|---|---|---|---|
| RF — Morgan FP (200 trees) | 0.832 | 0.391 | 0.611 |
| GCN (1000 epochs) | 0.950 | 0.250 | 0.326 |
| E(3)-Equivariant (1000 epochs, 4k samples) | **0.968** | **0.200** | **0.260** |

**Key findings:**
- RF saturates at 200 trees — the fingerprint representation is the performance ceiling, not forest size
- GCN surpasses RF at ~300 epochs and continues improving to 1000 epochs
- E(3)-equivariant models outperform GCN on bandgap, consistent with bandgap being sensitive to 3D orbital geometry
- Both deep models are data-hungry — equivariant R² scales from 0.48 (500 samples) to 0.94 (7k samples, 1000 epochs)

## Installation

```bash
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

## Quick start

```python
from polymatgen.core.monomer import Monomer
from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer

# Build a polystyrene sample
m = Monomer(name="styrene", smiles="C=Cc1ccccc1")
chains = [Chain([m], dp) for dp in [90, 100, 110]]
ps = Polymer(chains=chains, name="polystyrene")
print(ps.Mn)          # number-average molecular weight
print(ps.Mw)          # weight-average molecular weight
print(ps.dispersity)  # Mw / Mn

# Predict copolymer Tg using Fox equation
from polymatgen.properties.thermal import FoxEquation
fox = FoxEquation([(0.5, 373.0), (0.5, 273.0)])
print(fox.Tg)         # predicted Tg in Kelvin

# Look up a polymer in the reference database
from polymatgen.database.reference import get_polymer
ps_data = get_polymer("polystyrene")
print(ps_data["Tg"], ps_data["density"])

# Search experimental Tg data
from polymatgen.database.polyverse import search_tg
results = search_tg(min_tg=350.0, max_tg=450.0)
print(f"Found {len(results)} polymers with Tg between 350-450 K")

# Predict Tg with Random Forest
from polymatgen.ml.predictors import TgPredictor
rf = TgPredictor()
tg = rf.predict("[*]CC([*])c1ccccc1")
mean, std = rf.predict_with_uncertainty("[*]CC([*])c1ccccc1")
print(f"Tg: {mean:.1f} ± {std:.1f} K")

# Predict Tg with GCN
from polymatgen.ml.gcn_predictor import GCNTgPredictor
gcn = GCNTgPredictor(epochs=1000)
tg = gcn.predict("[*]CC([*])c1ccccc1")
print(f"GCN Tg: {tg:.1f} K")

# Predict bandgap with E(3)-equivariant GNN
from polymatgen.ml.equivariant_predictor import EquivariantBandgapPredictor
eq = EquivariantBandgapPredictor(epochs=1000, max_train_samples=4000)
bg = eq.predict("[*]CC([*])c1ccccc1")
print(f"Equivariant bandgap: {bg:.3f} eV")

# Inverse design — find polymers with Tg > 500 K and bandgap < 2 eV
from polymatgen.ml.predictors import BandgapPredictor
from polymatgen.ml.inverse_design import InverseDesigner
designer = InverseDesigner()
designer.add_constraint(TgPredictor(), min_val=500.0, name="Tg > 500 K")
designer.add_constraint(BandgapPredictor(), max_val=2.0, name="Bandgap < 2 eV")
results = designer.run(n_generations=20, population_size=50)
for smi, scores, fitness in results[:5]:
    print(smi, scores)

# Save and reload a polymer
from polymatgen.io.json_io import save_polymer, load_polymer
save_polymer(ps, "polystyrene.json")
loaded = load_polymer("polystyrene.json")
```

## Project structure

src/polymatgen/
├── core/          # Monomer, Chain, Polymer
├── properties/    # Thermal, solubility, mechanical calculators
├── io/            # JSON, LAMMPS, CSV
├── analysis/      # Chain stats, distribution, sequence
├── database/      # Reference data, PI1M, polyVERSE loaders
└── ml/            # RF, GCN, equivariant predictors, inverse design

## Data sources

| Dataset | Description | Size |
|---|---|---|
| Built-in reference | 10 common polymers with Tg, density, δ | 10 entries |
| PI1M | Polymer SMILES + SA scores | ~1M entries |
| polyVERSE chi | Flory-Huggins χ parameter | ~2,600 entries |
| polyVERSE bandgap | Electronic bandgap (DFT) | ~4,200 entries |
| polyVERSE gas | Gas permeability/diffusivity/solubility | ~400 entries |
| polyVERSE CED | Cohesive energy density | ~300 entries |
| PolyMetriX Tg | Curated experimental Tg | 7,367 entries |

## Requirements

- Python 3.11+
- RDKit
- NumPy
- pandas
- networkx
- scikit-learn
- torch
- torch-geometric
- e3nn

## License

MIT