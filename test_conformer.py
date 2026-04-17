from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCc1ccccc1")
mol = Chem.AddHs(mol)
result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
print("Conformer generation:", "OK" if result == 0 else "FAILED")
conf = mol.GetConformer()
pos = conf.GetPositions()
print("Atom positions shape:", pos.shape)
print("First 3 atoms xyz:")
for i in range(min(3, len(pos))):
    print(f"  atom {i}: {pos[i]}")