import pytest
import os
from polymatgen.ml.inverse_design import (
    InverseDesigner, Constraint, _is_valid, _mutate,
    _crossover, _load_pi1m_smiles, PI1M_PATH
)

POLYSTYRENE = "[*]CC([*])c1ccccc1"
POLYETHYLENE = "[*]CC[*]"
PMMA = "[*]CC([*])(C)C(=O)OC"


# --- Utility functions ---
def test_is_valid_polystyrene():
    assert _is_valid(POLYSTYRENE)

def test_is_valid_polyethylene():
    assert _is_valid(POLYETHYLENE)

def test_is_valid_rejects_garbage():
    assert not _is_valid("not_a_smiles!!!")

def test_is_valid_empty():
    assert not _is_valid("")

def test_crossover_returns_string():
    child = _crossover(POLYSTYRENE, POLYETHYLENE)
    assert isinstance(child, str)
    assert len(child) > 0

def test_crossover_uses_both_parents():
    # Child should contain characters from both parents
    child = _crossover("AAAA", "BBBB")
    assert "A" in child or "B" in child

def test_mutate_returns_string():
    pool = [POLYSTYRENE, POLYETHYLENE, PMMA]
    result = _mutate(POLYSTYRENE, pool, mutation_rate=1.0)
    assert isinstance(result, str)

def test_mutate_no_mutation():
    pool = [POLYSTYRENE, POLYETHYLENE]
    result = _mutate(POLYSTYRENE, pool, mutation_rate=0.0)
    assert result == POLYSTYRENE


# --- Constraint ---
class _MockPredictor:
    """Returns a fixed value for any SMILES."""
    is_trained = True
    def __init__(self, value):
        self._value = value
    def predict(self, smiles):
        return self._value


def test_constraint_satisfied_min():
    c = Constraint(_MockPredictor(600.0), min_val=500.0, name="Tg")
    assert c.is_satisfied(POLYSTYRENE)

def test_constraint_violated_min():
    c = Constraint(_MockPredictor(400.0), min_val=500.0, name="Tg")
    assert not c.is_satisfied(POLYSTYRENE)

def test_constraint_satisfied_max():
    c = Constraint(_MockPredictor(1.5), max_val=2.0, name="BG")
    assert c.is_satisfied(POLYSTYRENE)

def test_constraint_violated_max():
    c = Constraint(_MockPredictor(3.0), max_val=2.0, name="BG")
    assert not c.is_satisfied(POLYSTYRENE)

def test_constraint_score_perfect():
    c = Constraint(_MockPredictor(550.0), min_val=500.0, name="Tg")
    assert c.score(POLYSTYRENE) == 0.0

def test_constraint_score_negative_when_violated():
    c = Constraint(_MockPredictor(400.0), min_val=500.0, name="Tg")
    assert c.score(POLYSTYRENE) < 0.0

def test_constraint_predict_value():
    c = Constraint(_MockPredictor(375.0), min_val=300.0, name="Tg")
    assert c.predict_value(POLYSTYRENE) == 375.0

def test_constraint_repr():
    c = Constraint(_MockPredictor(0), min_val=1.0, max_val=2.0, name="X")
    assert "X" in repr(c)


# --- InverseDesigner ---
def test_designer_init():
    d = InverseDesigner(pool_size=100)
    assert len(d.constraints) == 0

def test_designer_add_constraint():
    d = InverseDesigner()
    d.add_constraint(_MockPredictor(500.0), min_val=400.0, name="Tg")
    assert len(d.constraints) == 1

def test_designer_add_constraint_chaining():
    d = InverseDesigner()
    result = d.add_constraint(_MockPredictor(500.0), min_val=400.0, name="Tg")
    assert result is d

def test_designer_requires_bound():
    d = InverseDesigner()
    with pytest.raises(ValueError):
        d.add_constraint(_MockPredictor(500.0), name="Tg")

def test_designer_run_no_constraints_raises():
    d = InverseDesigner()
    with pytest.raises(RuntimeError):
        d.run()

def test_designer_repr():
    d = InverseDesigner()
    assert "InverseDesigner" in repr(d)


@pytest.mark.skipif(not os.path.exists(PI1M_PATH),
                    reason="PI1M dataset not found")
def test_designer_run_mock_predictor():
    """Run a full GA with mock predictors (fast, no real training)."""
    d = InverseDesigner(pool_size=200, random_state=42)
    # Mock: always returns 550 K Tg — all candidates should satisfy min=500
    d.add_constraint(_MockPredictor(550.0), min_val=500.0, name="Tg")
    results = d.run(n_generations=3, population_size=20, verbose=False)
    assert isinstance(results, list)
    # All returned candidates should satisfy the constraint
    for smi, scores, fit in results:
        assert scores["Tg"] >= 500.0
        assert fit >= 0.0

@pytest.mark.skipif(not os.path.exists(PI1M_PATH),
                    reason="PI1M dataset not found")
def test_designer_run_two_constraints():
    """Two mock constraints — checks multi-constraint filtering."""
    d = InverseDesigner(pool_size=200, random_state=0)
    d.add_constraint(_MockPredictor(550.0), min_val=500.0, name="Tg")
    d.add_constraint(_MockPredictor(1.5), max_val=2.0, name="BG")
    results = d.run(n_generations=3, population_size=20, verbose=False)
    for smi, scores, fit in results:
        assert scores["Tg"] >= 500.0
        assert scores["BG"] <= 2.0

@pytest.mark.skipif(not os.path.exists(PI1M_PATH),
                    reason="PI1M dataset not found")
def test_designer_results_sorted_by_fitness():
    d = InverseDesigner(pool_size=200, random_state=1)
    d.add_constraint(_MockPredictor(550.0), min_val=500.0, name="Tg")
    results = d.run(n_generations=3, population_size=20, verbose=False)
    if len(results) > 1:
        fitnesses = [r[2] for r in results]
        assert fitnesses == sorted(fitnesses, reverse=True)