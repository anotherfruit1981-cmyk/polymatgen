# 🧪 polymatgen

**A polymer-first materials informatics library for Python.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RDKit](https://img.shields.io/badge/powered%20by-RDKit-orange.svg)](https://www.rdkit.org/)

`polymatgen` is designed around polymer topology and chemistry rather than crystallographic symmetry. Built from scratch, it provides high-level tools for representing, analyzing, and querying complex polymer systems.

---

## ✨ Features

* **🏗️ Core Data Model** – Native classes for `Monomer`, `Chain`, and `Polymer` with seamless **RDKit** integration.
* **🌡️ Property Calculators** – Compute Fox equation ($T_g$), Flory-Huggins $\chi$, Hildebrand solubility, contour length, and radius of gyration.
* **📊 Analysis Suite** – Molecular weight distributions ($M_n, M_w$), dispersity, sequence analysis, and chain statistics.
* **💾 Robust IO** – Read and write support for **JSON**, **CSV**, and **LAMMPS** summary files.
* **📚 Database Access** – Built-in loaders for **PI1M** (1M SMILES) and **polyVERSE** (experimental data for 7,000+ polymers).
* **🤖 ML Predictors** – Pre-trained Random Forest models for $T_g$, Bandgap, and Cohesive Energy.

---

## 🚀 Installation

Install in editable mode using **pip**:
```bash
pip install -e .

Or using uv for faster dependency resolution:

Bash

uv pip install -e .

💡 Quick Start

1. Representing a Polymer

Python

from polymatgen.core.monomer import Monomer
from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer

# Define a monomer (Polystyrene)
m = Monomer(name="styrene", smiles="C=Cc1ccccc1")

# Create a sample with specific chain lengths
chains = [Chain([m], dp) for dp in [90, 100, 110]]
ps = Polymer(chains=chains, name="polystyrene")

print(f"Mn: {ps.Mn}")           # Number-average molecular weight
print(f"Mw: {ps.Mw}")           # Weight-average molecular weight
print(f"Dispersity: {ps.dispersity}") 

2. Machine Learning Predictions

Python

from polymatgen.ml.predictors import TgPredictor

predictor = TgPredictor()
# Predict Tg using a SMILES string
tg = predictor.predict("[*]CC([*])c1ccccc1") 
print(f"Predicted Tg: {tg:.1f} K")

📁 Project StructurePlaintextsrc/polymatgen/
├── core/         # Monomer, Chain, and Polymer classes
├── properties/   # Thermal, solubility, and mechanical calculators
├── io/           # JSON, LAMMPS, and CSV support
├── analysis/     # Chain stats and distribution tools
├── database/     # PI1M and polyVERSE reference loaders
└── ml/           # Morgan fingerprinting and RF predictors

# Random Forest (fast, good baseline)
from polymatgen.ml.predictors import TgPredictor
rf = TgPredictor()
print(rf.predict("[*]CC([*])c1ccccc1"))

# Graph Neural Network (learns from structure directly)
from polymatgen.ml.gcn_predictor import GCNTgPredictor
gcn = GCNTgPredictor(epochs=50)
print(gcn.predict("[*]CC([*])c1ccccc1"))

📊 Data Sources

POLYMATGEN DATA SOURCES SUMMARY
================================================================================
DATASET            SIZE           DESCRIPTION
--------------------------------------------------------------------------------
Built-in Reference 10 entries     Common polymers with Tg, density, and solubility.
PI1M               ~1,000,000     Polymer SMILES strings with SA (Synthetic 
                                  Accessibility) scores.
polyVERSE chi      ~2,600         Flory-Huggins chi (χ) interaction parameters.
polyVERSE bandgap  ~4,200         Electronic bandgap values calculated via DFT.
polyVERSE gas      ~400           Gas permeability, diffusivity, and solubility data.
polyVERSE CED      ~300           Cohesive Energy Density (CED) values.
PolyMetriX Tg      7,367          Curated experimental Glass Transition Temperature 
                                  (Tg) data points.
--------------------------------------------------------------------------------

🛠️ Requirements

Python 3.11+
RDKit
NumPy / Pandas
Scikit-learn
NetworkX

⚖️ License

Distributed under the MIT License. See LICENSE for more information.