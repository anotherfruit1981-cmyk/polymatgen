import pytest
import numpy as np
import os
from polymatgen.ml.features import (
    psmiles_to_fingerprint, batch_fingerprints, fingerprint_stats
)
from polymatgen.ml.predictors import (
    TgPredictor, BandgapPredictor, CohesiveEnergyPredictor,
    TG_PATH, BANDGAP_PATH, CED_PATH
)

POLYSTYRENE = "[*]CC([*])c1ccccc1"
POLYETHYLENE = "[*]CC[*]"
PMMA = "[*]CC([*])(C)C(=O)OC"
TEST_SMILES = [POLYSTYRENE, POLYETHYLENE, PMMA]


# --- Features ---
def test_fingerprint_shape():
    fp = psmiles_to_fingerprint(POLYSTYRENE)
    assert fp.shape == (2048,)

def test_fingerprint_binary():
    fp = psmiles_to_fingerprint(POLYSTYRENE)
    assert set(fp.tolist()).issubset({0, 1})

def test_fingerprint_custom_bits():
    fp = psmiles_to_fingerprint(POLYSTYRENE, n_bits=1024)
    assert fp.shape == (1024,)

def test_fingerprint_invalid_smiles():
    with pytest.raises(ValueError):
        psmiles_to_fingerprint("not_valid_smiles!!!")

def test_fingerprint_removes_stars():
    fp1 = psmiles_to_fingerprint("[*]CC[*]")
    fp2 = psmiles_to_fingerprint("[*]CC[*]", n_bits=2048)
    assert fp1.shape == fp2.shape

def test_batch_fingerprints_shape():
    X, valid_idx = batch_fingerprints(TEST_SMILES)
    assert X.shape == (3, 2048)
    assert valid_idx == [0, 1, 2]

def test_batch_fingerprints_skips_errors():
    smiles = [POLYSTYRENE, "INVALID!!!", POLYETHYLENE]
    X, valid_idx = batch_fingerprints(smiles, skip_errors=True)
    assert X.shape[0] == 2
    assert valid_idx == [0, 2]

def test_fingerprint_stats():
    X, _ = batch_fingerprints(TEST_SMILES)
    stats = fingerprint_stats(X)
    assert stats["n_samples"] == 3
    assert stats["n_bits"] == 2048
    assert 0 < stats["mean_bits_set"] < 2048
    assert 0 < stats["sparsity"] < 1


# --- Tg Predictor ---
@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_predictor_trains():
    p = TgPredictor(n_estimators=10)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_predictor_predict():
    p = TgPredictor(n_estimators=10)
    tg = p.predict(POLYSTYRENE)
    assert 200.0 < tg < 700.0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_predictor_auto_trains():
    p = TgPredictor(n_estimators=10)
    assert not p.is_trained
    tg = p.predict(POLYSTYRENE)
    assert p.is_trained
    assert tg > 0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_predictor_evaluate():
    p = TgPredictor(n_estimators=10)
    p.train()
    metrics = p.evaluate(test_fraction=0.2)
    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["r2"] > 0.0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_predictor_batch():
    p = TgPredictor(n_estimators=10)
    results = p.predict_batch(TEST_SMILES)
    assert len(results) == 3
    assert all(isinstance(r[1], float) for r in results)


# --- Bandgap Predictor ---
@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_bandgap_predictor_trains():
    p = BandgapPredictor(n_estimators=10)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_bandgap_predictor_predict():
    p = BandgapPredictor(n_estimators=10)
    bg = p.predict(POLYSTYRENE)
    assert bg >= 0.0

@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_bandgap_predictor_evaluate():
    p = BandgapPredictor(n_estimators=10)
    p.train()
    metrics = p.evaluate(test_fraction=0.2)
    assert metrics["r2"] > 0.0


# --- CED Predictor ---
@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_ced_predictor_trains():
    p = CohesiveEnergyPredictor(n_estimators=10)
    p.train()
    assert p.is_trained

@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_ced_predictor_predict():
    p = CohesiveEnergyPredictor(n_estimators=10)
    ced = p.predict(POLYSTYRENE)
    assert ced > 0.0

@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_ced_to_hildebrand():
    p = CohesiveEnergyPredictor(n_estimators=10)
    ced = p.predict(POLYSTYRENE)
    delta = ced ** 0.5
    assert delta > 0.0


# --- Save/load ---
@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_save_and_load_model(tmp_path):
    p = BandgapPredictor(n_estimators=10)
    p.train()
    filepath = str(tmp_path / "bandgap_model.joblib")
    p.save(filepath)
    assert os.path.exists(filepath)

    p2 = BandgapPredictor()
    p2.load(filepath)
    assert p2.is_trained
    bg1 = p.predict(POLYSTYRENE)
    bg2 = p2.predict(POLYSTYRENE)
    assert abs(bg1 - bg2) < 1e-6

# --- Uncertainty Quantification ---
@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_predict_with_uncertainty_returns_tuple():
    p = TgPredictor(n_estimators=10)
    mean, std = p.predict_with_uncertainty(POLYSTYRENE)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std >= 0.0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_uncertainty_mean_close_to_predict():
    p = TgPredictor(n_estimators=50)
    mean, std = p.predict_with_uncertainty(POLYSTYRENE)
    point = p.predict(POLYSTYRENE)
    # mean from trees should match point prediction closely
    assert abs(mean - point) < 1.0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_tg_batch_with_uncertainty():
    p = TgPredictor(n_estimators=10)
    results = p.predict_batch_with_uncertainty(TEST_SMILES)
    assert len(results) == 3
    for smi, mean, std in results:
        assert mean > 0
        assert std >= 0

@pytest.mark.skipif(not os.path.exists(TG_PATH),
                    reason="Tg dataset not found")
def test_uncertainty_threshold_splits_correctly():
    p = TgPredictor(n_estimators=10)
    out = p.uncertainty_threshold(TEST_SMILES)
    total = len(out["confident"]) + len(out["uncertain"])
    assert total == 3
    assert out["threshold"] is not None
    for r in out["confident"]:
        assert r[2] <= out["threshold"]
    for r in out["uncertain"]:
        assert r[2] > out["threshold"]

@pytest.mark.skipif(not os.path.exists(BANDGAP_PATH),
                    reason="Bandgap dataset not found")
def test_bandgap_predict_with_uncertainty():
    p = BandgapPredictor(n_estimators=10)
    mean, std = p.predict_with_uncertainty(POLYSTYRENE)
    assert mean >= 0.0
    assert std >= 0.0

@pytest.mark.skipif(not os.path.exists(CED_PATH),
                    reason="CED dataset not found")
def test_ced_predict_with_uncertainty():
    p = CohesiveEnergyPredictor(n_estimators=10)
    mean, std = p.predict_with_uncertainty(POLYSTYRENE)
    assert mean > 0.0
    assert std >= 0.0