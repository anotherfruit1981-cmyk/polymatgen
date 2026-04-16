import os
import pytest
from polymatgen.core.monomer import Monomer
from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer
from polymatgen.io.json_io import save_polymer, load_polymer, polymer_to_dict, polymer_from_dict
from polymatgen.io.lammps_io import write_lammps_summary, read_lammps_summary
from polymatgen.io.csv_io import export_chain_distribution, import_chain_distribution


@pytest.fixture
def sample_polymer():
    m = Monomer(name="styrene", smiles="C=Cc1ccccc1")
    chains = [Chain([m], dp) for dp in [90, 100, 110]]
    return Polymer(chains=chains, name="polystyrene")


# --- JSON ---
def test_polymer_to_dict(sample_polymer):
    d = polymer_to_dict(sample_polymer)
    assert d["name"] == "polystyrene"
    assert len(d["chains"]) == 3

def test_polymer_roundtrip_dict(sample_polymer):
    d = polymer_to_dict(sample_polymer)
    p2 = polymer_from_dict(d)
    assert p2.name == sample_polymer.name
    assert len(p2.chains) == len(sample_polymer.chains)

def test_save_and_load_json(sample_polymer, tmp_path):
    filepath = str(tmp_path / "polymer.json")
    save_polymer(sample_polymer, filepath)
    assert os.path.exists(filepath)
    loaded = load_polymer(filepath)
    assert loaded.name == sample_polymer.name
    assert abs(loaded.Mn - sample_polymer.Mn) < 0.01


# --- LAMMPS ---
def test_write_lammps_summary(sample_polymer, tmp_path):
    filepath = str(tmp_path / "polymer.lammps")
    write_lammps_summary(sample_polymer, filepath)
    assert os.path.exists(filepath)

def test_read_lammps_summary(sample_polymer, tmp_path):
    filepath = str(tmp_path / "polymer.lammps")
    write_lammps_summary(sample_polymer, filepath)
    stats = read_lammps_summary(filepath)
    assert stats["n_chains"] == 3
    assert abs(stats["Mn"] - sample_polymer.Mn) < 0.01


# --- CSV ---
def test_export_csv(sample_polymer, tmp_path):
    filepath = str(tmp_path / "chains.csv")
    export_chain_distribution(sample_polymer, filepath)
    assert os.path.exists(filepath)

def test_import_csv_roundtrip(sample_polymer, tmp_path):
    filepath = str(tmp_path / "chains.csv")
    export_chain_distribution(sample_polymer, filepath)
    rows = import_chain_distribution(filepath)
    assert len(rows) == len(sample_polymer.chains)
    assert rows[0]["tacticity"] == "atactic"