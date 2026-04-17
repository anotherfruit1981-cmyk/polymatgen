import pytest
import os
import torch
from polymatgen.ml.gcn_predictor import (
    smiles_to_graph, GCNTgPredictor, GCNBandgapPredictor,
    GCNCohesiveEnergyPredictor, ATOM_FEATURE_DIM,
    TG_PATH, BANDGAP_PATH, CED_PATH
)

POLYSTYRENE = "[*]CC([*])c1ccccc1"
POLYETHYLENE = "[*]CC[*]"
PMMA = "[*]CC([*])(C)C(=O)OC"
TEST_SMILES = [POLYSTYRENE, POLYETHYLENE, PMMA]


# --- Graph construction ---
def test_smiles_to_graph_returns_data():
    graph = smiles_to_graph(POLYSTYRENE)
    assert graph.x is not None
    assert graph.edge_index is not None

def test_graph_node_feature_dim():
    graph = smiles_to_graph(POLYSTYRENE)
    assert graph.x.shape[1] == ATOM_FEATURE_DIM

def test_graph_has_edges():
    graph = smiles_to_graph(POLYSTYRENE)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] > 0

def test_graph_edges_undirected():
    graph = smiles_to_graph(POLYSTYRENE)
    # Number of edges should be even (both directions)
    assert graph.edge_index.shape[1] % 2 == 0

def test_graph_invalid_smiles_raises():
    with pytest.raises(ValueError):
        smiles_to_graph("not_valid!!!")

def test_graph_handles_star_smiles():
    # Should not raise
    graph = smiles_to_graph("[*]CC[*]")
    assert graph.x.shape[0] > 0

def test_graph_polyethylene():
    graph = smiles_to_graph(POLYETHYLENE)
    assert graph.x.shape[0] == 2  # two carbons after [H] substitution

def test_graph_features_in_range():
    graph = smiles_to_graph(POLYSTYRENE)
    # All normalised features should be between -1 and 2
    assert float(graph.x.min()) >= -1.0
    assert float(graph.x.max()) <= 2.0


# --- GCN Tg Predictor ---
@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_gcn_tg_trains():
    p = GCNTgPredictor(epochs=2, hidden_dim=16)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_gcn_tg_predict_returns_float():
    p = GCNTgPredictor(epochs=2, hidden_dim=16)
    tg = p.predict(POLYSTYRENE)
    assert isinstance(tg, float)
    assert tg > 0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_gcn_tg_auto_trains():
    p = GCNTgPredictor(epochs=2, hidden_dim=16)
    assert not p.is_trained
    tg = p.predict(POLYSTYRENE)
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_gcn_tg_batch():
    p = GCNTgPredictor(epochs=2, hidden_dim=16)
    results = p.predict_batch(TEST_SMILES)
    assert len(results) == 3
    for smi, val in results:
        assert isinstance(val, float)

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_gcn_tg_repr():
    p = GCNTgPredictor(epochs=2, hidden_dim=16)
    assert "untrained" in repr(p)
    p.train()
    assert "trained" in repr(p)

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_gcn_tg_save_load(tmp_path):
    p = GCNTgPredictor(epochs=2, hidden_dim=16)
    p.train()
    tg1 = p.predict(POLYSTYRENE)
    filepath = str(tmp_path / "gcn_tg.pt")
    p.save(filepath)
    assert os.path.exists(filepath)
    p2 = GCNTgPredictor()
    p2.load(filepath)
    assert p2.is_trained
    tg2 = p2.predict(POLYSTYRENE)
    assert abs(tg1 - tg2) < 0.01


# --- GCN Bandgap Predictor ---
@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_gcn_bandgap_trains():
    p = GCNBandgapPredictor(epochs=2, hidden_dim=16)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_gcn_bandgap_predict():
    p = GCNBandgapPredictor(epochs=2, hidden_dim=16)
    bg = p.predict(POLYSTYRENE)
    assert isinstance(bg, float)

@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_gcn_bandgap_evaluate():
    p = GCNBandgapPredictor(epochs=2, hidden_dim=16)
    p.train()
    metrics = p.evaluate_default()
    assert "r2" in metrics
    assert "mae" in metrics


# --- GCN CED Predictor ---
@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_gcn_ced_trains():
    p = GCNCohesiveEnergyPredictor(epochs=2, hidden_dim=16)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_gcn_ced_predict():
    p = GCNCohesiveEnergyPredictor(epochs=2, hidden_dim=16)
    ced = p.predict(POLYSTYRENE)
    assert isinstance(ced, float)
    assert ced > 0