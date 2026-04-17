"""
polymatgen — Polymer-first materials informatics library.
"""

from polymatgen.core.monomer import Monomer
from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer

from polymatgen.properties.thermal import FoxEquation
from polymatgen.properties.solubility import HildebrandSolubility, FloryHuggins
from polymatgen.properties.mechanical import ContourLength

from polymatgen.ml.features import psmiles_to_fingerprint, batch_fingerprints
from polymatgen.ml.predictors import TgPredictor, BandgapPredictor, CohesiveEnergyPredictor
from polymatgen.ml.gcn_predictor import (
    GCNTgPredictor, GCNBandgapPredictor, GCNCohesiveEnergyPredictor,
    smiles_to_graph,
)
from polymatgen.ml.equivariant_predictor import (
    EquivariantTgPredictor, EquivariantBandgapPredictor,
    EquivariantCohesiveEnergyPredictor,
    smiles_to_equivariant_graph, smiles_to_3d,
)
from polymatgen.ml.inverse_design import InverseDesigner

__version__ = "0.1.0"

__all__ = [
    # Core
    "Monomer", "Chain", "Polymer",
    # Properties
    "FoxEquation", "HildebrandSolubility", "FloryHuggins", "ContourLength",
    # ML — fingerprint-based
    "psmiles_to_fingerprint", "batch_fingerprints",
    "TgPredictor", "BandgapPredictor", "CohesiveEnergyPredictor",
    # ML — graph-based
    "GCNTgPredictor", "GCNBandgapPredictor", "GCNCohesiveEnergyPredictor",
    "smiles_to_graph",
    # ML — equivariant 3D
    "EquivariantTgPredictor", "EquivariantBandgapPredictor",
    "EquivariantCohesiveEnergyPredictor",
    "smiles_to_equivariant_graph", "smiles_to_3d",
    # ML — inverse design
    "InverseDesigner",
]