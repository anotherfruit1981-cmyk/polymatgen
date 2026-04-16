import pytest
from polymatgen.core.monomer import Monomer
from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer

def test_monomer_from_smiles():
    m = Monomer(name="styrene", smiles="C=Cc1ccccc1")
    assert m.atom_count == 8
    assert m.molecular_weight > 0

def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        Monomer(name="bad", smiles="not_a_smiles")

def test_chain_molecular_weight():
    m = Monomer(name="styrene", smiles="C=Cc1ccccc1")
    c = Chain(monomers=[m], degree_of_polymerization=100)
    assert c.molecular_weight > 0

def test_polymer_dispersity():
    m = Monomer(name="styrene", smiles="C=Cc1ccccc1")
    chains = [Chain([m], dp) for dp in [90, 100, 110]]
    p = Polymer(chains=chains, name="polystyrene")
    assert p.Mn > 0
    assert p.Mw >= p.Mn
    assert p.dispersity >= 1.0