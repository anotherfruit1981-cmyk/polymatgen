polymatgen
A polymer-first materials informatics library for Python — built from scratch. Designed around polymer topology and chemistry rather than crystallographic symmetry, polymatgen provides tools for representing, analysing, and querying polymer systems.

Features

Core data model — Monomer, Chain, Polymer classes with RDKit integration
Property calculators — Fox equation (Tg), Flory-Huggins χ, Hildebrand solubility parameter, contour length, radius of gyration
IO — JSON, LAMMPS summary, and CSV read/write
Analysis — molecular weight distribution, sequence analysis, chain statistics
Databases — built-in reference data, PI1M (1M polymer SMILES), polyVERSE (chi parameter, bandgap, gas permeability, cohesive energy density, experimental Tg for 7,367 polymers)
ML predictors — TgPredictor, BandgapPredictor, CohesiveEnergyPredictor (Morgan fingerprints + Random Forest, trained on polyVERSE/PolyMetriX data)

Installation
bashpip install -e .
Or with uv:
bashuv pip install -e .
Quick start
pythonfrom polymatgen.core.monomer import Monomer
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

# Predict Tg with ML
from polymatgen.ml.predictors import TgPredictor
predictor = TgPredictor()
tg = predictor.predict("[*]CC([*])c1ccccc1")  # polystyrene
print(f"Predicted Tg: {tg:.1f} K")

# Save and reload a polymer
from polymatgen.io.json_io import save_polymer, load_polymer
save_polymer(ps, "polystyrene.json")
loaded = load_polymer("polystyrene.json")
Project structure
src/polymatgen/
├── core/          # Monomer, Chain, Polymer
├── properties/    # Thermal, solubility, mechanical calculators
├── io/            # JSON, LAMMPS, CSV
├── analysis/      # Chain stats, distribution, sequence
├── database/      # Reference data, PI1M, polyVERSE loaders
└── ml/            # Morgan fingerprint features, RF predictors
Data sources
DatasetDescriptionSizeBuilt-in reference10 common polymers with Tg, density, δ10 entriesPI1MPolymer SMILES + SA scores~1M entriespolyVERSE chiFlory-Huggins χ parameter~2,600 entriespolyVERSE bandgapElectronic bandgap (DFT)~4,200 entriespolyVERSE gasGas permeability/diffusivity/solubility~400 entriespolyVERSE CEDCohesive energy density~300 entriesPolyMetriX TgCurated experimental Tg7,367 entries
Requirements

Python 3.11+
RDKit
NumPy
pandas
networkx
scikit-learn

License
MIT