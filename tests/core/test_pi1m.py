import os
import pytest
import tempfile
import csv
from polymatgen.database.pi1m import (
    load_pi1m, search_by_sa_score, pi1m_stats,
    smiles_to_monomer, sample_pi1m
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small synthetic PI1M-format CSV for testing."""
    filepath = str(tmp_path / "PI1M_test.csv")
    rows = [
        {"SMILES": "*CC(*)c1ccccc1", "SA Score": 2.1},
        {"SMILES": "*CC(*)(C)C",     "SA Score": 1.5},
        {"SMILES": "*CC(*)C(=O)OC",  "SA Score": 2.8},
        {"SMILES": "*CC(*)Cl",       "SA Score": 4.2},
        {"SMILES": "*C(*)F",         "SA Score": 1.8},
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SMILES", "SA Score"])
        writer.writeheader()
        writer.writerows(rows)
    return filepath


# --- load_pi1m ---
def test_load_pi1m_returns_list(sample_csv):
    data = load_pi1m(sample_csv)
    assert isinstance(data, list)
    assert len(data) == 5

def test_load_pi1m_has_smiles(sample_csv):
    data = load_pi1m(sample_csv)
    assert "SMILES" in data[0]

def test_load_pi1m_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pi1m("nonexistent_file.csv")


# --- search_by_sa_score ---
def test_search_by_sa_score(sample_csv):
    results = search_by_sa_score(max_sa=3.0, filepath=sample_csv)
    assert all(r["SA Score"] <= 3.0 for r in results)

def test_search_excludes_hard(sample_csv):
    results = search_by_sa_score(max_sa=3.0, filepath=sample_csv)
    smiles_list = [r["SMILES"] for r in results]
    assert "*CC(*)Cl" not in smiles_list

def test_search_limit(sample_csv):
    results = search_by_sa_score(max_sa=10.0, limit=2, filepath=sample_csv)
    assert len(results) <= 2


# --- pi1m_stats ---
def test_stats_total_entries(sample_csv):
    stats = pi1m_stats(sample_csv)
    assert stats["total_entries"] == 5

def test_stats_has_sa_score(sample_csv):
    stats = pi1m_stats(sample_csv)
    assert "SA Score" in stats
    assert stats["SA Score"]["count_non_null"] == 5

def test_stats_sa_range(sample_csv):
    stats = pi1m_stats(sample_csv)
    assert stats["SA Score"]["min"] < stats["SA Score"]["max"]


# --- smiles_to_monomer ---
def test_smiles_to_monomer_removes_stars():
    result = smiles_to_monomer("*CC(*)c1ccccc1")
    assert "*" not in result

def test_smiles_to_monomer_replaces_with_H():
    result = smiles_to_monomer("*CC*")
    assert result == "[H]CC[H]"


# --- sample_pi1m ---
def test_sample_returns_correct_count(sample_csv):
    results = sample_pi1m(n=3, filepath=sample_csv)
    assert len(results) == 3

def test_sample_does_not_exceed_dataset(sample_csv):
    results = sample_pi1m(n=100, filepath=sample_csv)
    assert len(results) <= 5