from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt


class Monomer:
    def __init__(self, name: str, smiles: str):
        self.name = name
        self.smiles = smiles
        self._mol = Chem.MolFromSmiles(smiles)
        if self._mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

    @property
    def molecular_weight(self) -> float:
        return MolWt(self._mol)

    @property
    def atom_count(self) -> int:
        return self._mol.GetNumAtoms()

    def __repr__(self):
        return f"Monomer(name={self.name!r}, smiles={self.smiles!r})"
