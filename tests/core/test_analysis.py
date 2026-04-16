import pytest
from polymatgen.core.monomer import Monomer
from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer
from polymatgen.analysis.chain_stats import (
    radius_of_gyration, end_to_end_distance,
    characteristic_ratio, chain_summary
)
from polymatgen.analysis.distribution import (
    molecular_weight_moments, histogram, cumulative_distribution
)
from polymatgen.analysis.sequence import (
    monomer_composition, is_homopolymer, is_copolymer,
    sequence_blocks, blockiness, polymer_composition
)


@pytest.fixture
def styrene():
    return Monomer(name="styrene", smiles="C=Cc1ccccc1")

@pytest.fixture
def mma():
    return Monomer(name="mma", smiles="COC(=O)C(C)=C")

@pytest.fixture
def homo_chain(styrene):
    return Chain([styrene], degree_of_polymerization=100)

@pytest.fixture
def copoly_chain(styrene, mma):
    return Chain([styrene, mma], degree_of_polymerization=50)

@pytest.fixture
def sample_polymer(styrene):
    chains = [Chain([styrene], dp) for dp in [80, 90, 100, 110, 120]]
    return Polymer(chains=chains, name="polystyrene")


# --- Chain stats ---
def test_radius_of_gyration(homo_chain):
    rg = radius_of_gyration(homo_chain)
    assert rg > 0

def test_end_to_end_distance(homo_chain):
    r = end_to_end_distance(homo_chain)
    assert r > radius_of_gyration(homo_chain)

def test_characteristic_ratio(homo_chain):
    c = characteristic_ratio(homo_chain)
    assert c > 1.0

def test_chain_summary_keys(homo_chain):
    s = chain_summary(homo_chain)
    assert "Rg_angstrom" in s
    assert "r_end_to_end_angstrom" in s
    assert "C_inf" in s


# --- Distribution ---
def test_molecular_weight_moments(sample_polymer):
    moments = molecular_weight_moments(sample_polymer)
    assert moments["Mw"] >= moments["Mn"]
    assert moments["Mz"] >= moments["Mw"]
    assert moments["dispersity"] >= 1.0

def test_histogram_bin_count(sample_polymer):
    bins = histogram(sample_polymer, n_bins=5)
    assert len(bins) == 5
    assert sum(b["count"] for b in bins) == len(sample_polymer.chains)

def test_cumulative_distribution(sample_polymer):
    cdf = cumulative_distribution(sample_polymer)
    assert len(cdf) == len(sample_polymer.chains)
    assert cdf[-1][1] == 1.0
    fractions = [f for _, f in cdf]
    assert fractions == sorted(fractions)


# --- Sequence ---
def test_homopolymer_detection(homo_chain):
    assert is_homopolymer(homo_chain) is True
    assert is_copolymer(homo_chain) is False

def test_copolymer_detection(copoly_chain):
    assert is_copolymer(copoly_chain) is True

def test_monomer_composition_sums_to_one(copoly_chain):
    comp = monomer_composition(copoly_chain)
    assert abs(sum(comp.values()) - 1.0) < 1e-6

def test_sequence_blocks():
    m1 = Monomer("A", "C")
    m2 = Monomer("B", "CC")
    chain = Chain([m1, m1, m2, m2, m2, m1], degree_of_polymerization=1)
    blocks = sequence_blocks(chain)
    assert blocks == [("A", 2), ("B", 3), ("A", 1)]

def test_blockiness_homopolymer(homo_chain):
    b = blockiness(homo_chain)
    assert b == len(homo_chain.monomers)

def test_polymer_composition(sample_polymer):
    comp = polymer_composition(sample_polymer)
    assert "styrene" in comp
    assert abs(comp["styrene"] - 1.0) < 1e-6