import pytest
import os
import numpy as np
from polymatgen.ml.equivariant_predictor import (
    smiles_to_3d, smiles_to_equivariant_graph,
    _atom_type_onehot, _build_radius_graph,
    EquivariantTgPredictor, EquivariantBandgapPredictor,
    EquivariantCohesiveEnergyPredictor,
    N_ATOM_TYPES, CUTOFF,
    TG_PATH, BANDGAP_PATH, CED_PATH
)

POLYSTYRENE = "[*]CC([*])c1ccccc1"
POLYETHYLENE = "[*]CC[*]"
PMMA = "[*]CC([*])(C)C(=O)OC"
TEST_SMILES = [POLYSTYRENE, POLYETHYLENE, PMMA]


# --- 3D conformer generation ---
def test_smiles_to_3d_returns_positions():
    pos, atomic_nums = smiles_to_3d(POLYSTYRENE)
    assert pos.shape[1] == 3
    assert len(atomic_nums) == pos.shape[0]

def test_smiles_to_3d_polyethylene():
    pos, atomic_nums = smiles_to_3d(POLYETHYLENE)
    assert pos.shape[0] > 0

def test_smiles_to_3d_positions_are_finite():
    pos, _ = smiles_to_3d(POLYSTYRENE)
    assert np.all(np.isfinite(pos))

def test_smiles_to_3d_invalid_raises():
    with pytest.raises(ValueError):
        smiles_to_3d("not_valid!!!")

def test_smiles_to_3d_star_handling():
    pos, atomic_nums = smiles_to_3d("[*]CC[*]")
    assert pos.shape[0] > 0


# --- Atom features ---
def test_atom_onehot_length():
    vec = _atom_type_onehot(6)  # Carbon
    assert len(vec) == N_ATOM_TYPES

def test_atom_onehot_carbon():
    vec = _atom_type_onehot(6)
    assert vec[1] == 1.0
    assert sum(vec) == 1.0

def test_atom_onehot_unknown():
    vec = _atom_type_onehot(999)
    assert vec[-1] == 1.0

def test_atom_onehot_hydrogen():
    vec = _atom_type_onehot(1)
    assert vec[0] == 1.0


# --- Radius graph ---
def test_radius_graph_shape():
    pos = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32)
    edges = _build_radius_graph(pos, cutoff=2.0)
    assert edges.shape[0] == 2

def test_radius_graph_nearby_connected():
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edges = _build_radius_graph(pos, cutoff=2.0)
    assert edges.shape[1] >= 2  # at least both directions

def test_radius_graph_far_not_connected():
    pos = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
    edges = _build_radius_graph(pos, cutoff=5.0)
    # Should still have edges (fallback to nearest neighbour)
    assert edges.shape[1] >= 2


# --- Full graph ---
def test_equivariant_graph_has_pos():
    graph = smiles_to_equivariant_graph(POLYSTYRENE)
    assert graph.pos is not None
    assert graph.pos.shape[1] == 3

def test_equivariant_graph_has_x():
    graph = smiles_to_equivariant_graph(POLYSTYRENE)
    assert graph.x.shape[1] == N_ATOM_TYPES

def test_equivariant_graph_has_edges():
    graph = smiles_to_equivariant_graph(POLYSTYRENE)
    assert graph.edge_index.shape[0] == 2

def test_equivariant_graph_nodes_match():
    graph = smiles_to_equivariant_graph(POLYSTYRENE)
    assert graph.x.shape[0] == graph.pos.shape[0]

def test_equivariant_graph_invalid_raises():
    with pytest.raises(ValueError):
        smiles_to_equivariant_graph("not_valid!!!")


# --- Equivariant Tg Predictor ---
@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_equivariant_tg_trains():
    p = EquivariantTgPredictor(epochs=2, max_train_samples=50)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_equivariant_tg_predict_returns_float():
    p = EquivariantTgPredictor(epochs=2, max_train_samples=50)
    tg = p.predict(POLYSTYRENE)
    assert isinstance(tg, float)

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_equivariant_tg_auto_trains():
    p = EquivariantTgPredictor(epochs=2, max_train_samples=50)
    assert not p.is_trained
    _ = p.predict(POLYSTYRENE)
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_equivariant_tg_batch():
    p = EquivariantTgPredictor(epochs=2, max_train_samples=50)
    results = p.predict_batch(TEST_SMILES)
    assert len(results) == 3
    for smi, val in results:
        assert isinstance(val, float)

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_equivariant_tg_repr():
    p = EquivariantTgPredictor(epochs=2, max_train_samples=50)
    assert "untrained" in repr(p)
    p.train()
    assert "trained" in repr(p)

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_equivariant_tg_save_load(tmp_path):
    p = EquivariantTgPredictor(epochs=2, max_train_samples=50)
    p.train()
    tg1 = p.predict(POLYSTYRENE)
    filepath = str(tmp_path / "eq_tg.pt")
    p.save(filepath)
    assert os.path.exists(filepath)
    p2 = EquivariantTgPredictor()
    p2.load(filepath)
    assert p2.is_trained
    tg2 = p2.predict(POLYSTYRENE)
    assert abs(tg1 - tg2) < 0.01


# --- Equivariant Bandgap Predictor ---
@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_equivariant_bandgap_trains():
    p = EquivariantBandgapPredictor(epochs=2, max_train_samples=50)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_equivariant_bandgap_predict():
    p = EquivariantBandgapPredictor(epochs=2, max_train_samples=50)
    bg = p.predict(POLYSTYRENE)
    assert isinstance(bg, float)


# --- Equivariant CED Predictor ---
@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_equivariant_ced_trains():
    p = EquivariantCohesiveEnergyPredictor(epochs=2, max_train_samples=50)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_equivariant_ced_predict():
    p = EquivariantCohesiveEnergyPredictor(epochs=2, max_train_samples=50)
    ced = p.predict(POLYSTYRENE)
    assert isinstance(ced, float)