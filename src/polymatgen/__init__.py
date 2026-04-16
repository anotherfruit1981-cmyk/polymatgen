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

__version__ = "0.1.0"
__all__ = [
    "Monomer", "Chain", "Polymer",
    "FoxEquation", "HildebrandSolubility", "FloryHuggins", "ContourLength",
    "psmiles_to_fingerprint", "batch_fingerprints",
    "TgPredictor", "BandgapPredictor", "CohesiveEnergyPredictor",
]