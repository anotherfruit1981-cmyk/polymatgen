import pytest
from polymatgen.database.reference import (
    get_polymer, list_polymers, search_by_property, POLYMER_DATABASE
)


def test_list_polymers():
    polymers = list_polymers()
    assert len(polymers) > 0
    assert "polystyrene" in polymers

def test_get_polymer_by_name():
    ps = get_polymer("polystyrene")
    assert ps["abbreviation"] == "PS"
    assert ps["Tg"] == 373.0

def test_get_polymer_by_abbreviation():
    ps = get_polymer("PS")
    assert ps["full_name"] == "Polystyrene"

def test_get_polymer_case_insensitive():
    ps = get_polymer("POLYSTYRENE")
    assert ps["abbreviation"] == "PS"

def test_get_polymer_not_found():
    with pytest.raises(KeyError):
        get_polymer("unobtainium")

def test_search_by_tg():
    results = search_by_property("Tg", min_val=350.0, max_val=400.0)
    assert len(results) > 0
    for name, tg in results:
        assert 350.0 <= tg <= 400.0

def test_search_by_density():
    results = search_by_property("density", max_val=1.0)
    assert len(results) > 0
    for name, d in results:
        assert d <= 1.0

def test_search_results_sorted():
    results = search_by_property("Tg")
    tgs = [tg for _, tg in results]
    assert tgs == sorted(tgs)

def test_all_entries_have_required_fields():
    required = ["full_name", "abbreviation", "Tg", "density", "delta"]
    for name, data in POLYMER_DATABASE.items():
        for field in required:
            assert field in data, f"{name} missing field {field}"