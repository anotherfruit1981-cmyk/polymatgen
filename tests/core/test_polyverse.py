import pytest
import os
from polymatgen.database.polyverse import (
    load_chi, search_chi_by_polymer, search_chi_by_smiles,
    load_bandgap, search_bandgap,
    load_gas_permeability, search_gas_permeability,
    load_cohesive_energy_density, search_ced, ced_to_hildebrand,
    CHI_PATH, BANDGAP_PATH, GAS_PATH, CED_PATH
)

# Skip all tests if data files are not present
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in [CHI_PATH, BANDGAP_PATH, GAS_PATH, CED_PATH]),
    reason="polyVERSE data files not found in src/polymatgen/data/"
)


# --- Chi parameter ---
def test_load_chi_returns_list():
    data = load_chi()
    assert isinstance(data, list)
    assert len(data) > 0

def test_load_chi_has_expected_columns():
    data = load_chi()
    assert "Polymer_SMILES" in data[0]
    assert "chi" in data[0]
    assert "temperature" in data[0]

def test_search_chi_by_polymer():
    results = search_chi_by_polymer("polyethylene")
    assert len(results) > 0

def test_search_chi_by_smiles_no_match():
    results = search_chi_by_smiles("NOTASMILES")
    assert results == []


# --- Bandgap ---
def test_load_bandgap_returns_list():
    data = load_bandgap()
    assert isinstance(data, list)
    assert len(data) > 0

def test_load_bandgap_has_expected_columns():
    data = load_bandgap()
    assert "smiles" in data[0]
    assert "bandgap_chain" in data[0]

def test_search_bandgap_range():
    results = search_bandgap(min_val=1.0, max_val=3.0)
    assert all(1.0 <= r["bandgap_chain"] <= 3.0 for r in results)

def test_search_bandgap_all():
    results = search_bandgap()
    assert len(results) > 0


# --- Gas permeability ---
def test_load_gas_permeability_returns_list():
    data = load_gas_permeability()
    assert isinstance(data, list)
    assert len(data) > 0

def test_load_gas_has_smiles():
    data = load_gas_permeability()
    assert "smiles_string" in data[0]

def test_search_gas_co2():
    results = search_gas_permeability(gas="CO2", prop="p_exp")
    assert all("smiles_string" in r for r in results)

def test_search_gas_invalid_column():
    with pytest.raises(ValueError):
        search_gas_permeability(gas="XENON", prop="p_exp")


# --- Cohesive energy density ---
def test_load_ced_returns_list():
    data = load_cohesive_energy_density()
    assert isinstance(data, list)
    assert len(data) > 0

def test_load_ced_has_expected_columns():
    data = load_cohesive_energy_density()
    assert "smiles1" in data[0]
    assert "value_COE" in data[0]

def test_search_ced_range():
    results = search_ced(min_val=10.0, max_val=50.0)
    assert all(10.0 <= r["value_COE"] <= 50.0 for r in results)

def test_ced_to_hildebrand():
    delta = ced_to_hildebrand(324.0)
    assert abs(delta - 18.0) < 0.01

def test_ced_to_hildebrand_negative():
    with pytest.raises(ValueError):
        ced_to_hildebrand(-1.0)