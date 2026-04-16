import pytest
from polymatgen.properties.thermal import FoxEquation
from polymatgen.properties.solubility import HildebrandSolubility, FloryHuggins
from polymatgen.properties.mechanical import ContourLength


# --- Thermal ---
def test_fox_equation_single_component():
    fox = FoxEquation([(1.0, 373.0)])
    assert abs(fox.Tg - 373.0) < 0.01

def test_fox_equation_two_components():
    fox = FoxEquation([(0.5, 400.0), (0.5, 300.0)])
    assert 300.0 < fox.Tg < 400.0

def test_fox_equation_invalid_fractions():
    with pytest.raises(ValueError):
        FoxEquation([(0.4, 373.0), (0.4, 273.0)])

def test_fox_celsius():
    fox = FoxEquation([(1.0, 373.15)])
    assert abs(fox.Tg_celsius - 100.0) < 0.01


# --- Solubility ---
def test_hildebrand_delta():
    h = HildebrandSolubility([272, 269, 57], molar_volume=100.0)
    assert h.delta > 0

def test_miscibility_check():
    h = HildebrandSolubility([272, 269, 57], molar_volume=100.0)
    assert h.miscibility_check(h.delta) is True
    assert h.miscibility_check(h.delta + 100) is False

def test_flory_huggins_chi():
    fh = FloryHuggins(delta_polymer=18.0, delta_solvent=18.0,
                      molar_volume_solvent=100.0)
    assert abs(fh.chi) < 1e-6

def test_flory_huggins_miscibility():
    fh = FloryHuggins(delta_polymer=18.0, delta_solvent=18.5,
                      molar_volume_solvent=100.0)
    assert isinstance(fh.is_miscible, bool)


# --- Mechanical ---
def test_contour_length():
    cl = ContourLength(degree_of_polymerization=100)
    assert cl.length_nm > 0
    assert cl.length_angstrom == cl.length_nm * 10

def test_end_to_end_rms():
    cl = ContourLength(degree_of_polymerization=100)
    assert cl.end_to_end_rms < cl.length_angstrom

def test_contour_length_scales_with_dp():
    cl1 = ContourLength(degree_of_polymerization=100)
    cl2 = ContourLength(degree_of_polymerization=200)
    assert cl2.length_nm == 2 * cl1.length_nm