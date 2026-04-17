"""
ML property predictors for polymatgen.

Each predictor:
- Loads its training data from the polyVERSE/PolyMetriX datasets
- Computes Morgan fingerprints from SMILES
- Trains a Random Forest regressor
- Exposes predict(), predict_with_uncertainty(), and evaluate() methods

Predictors included:
    TgPredictor              — glass transition temperature (K)
    BandgapPredictor         — electronic bandgap (eV)
    CohesiveEnergyPredictor  — cohesive energy density (MPa)
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TG_PATH = os.path.join(
    DATA_DIR,
    "LAMALAB_CURATED_Tg_structured_polymerclass_with_embeddings.csv"
)
BANDGAP_PATH = os.path.join(DATA_DIR, "bandgap_chain.csv")
CED_PATH = os.path.join(DATA_DIR, "Cohesive_energy_density_2025_06_23.csv")


class _BasePredictor:
    """
    Base class for all polymatgen ML predictors.
    Handles training, prediction, uncertainty quantification,
    evaluation and model persistence.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42,
                 n_bits: int = 2048, radius: int = 2):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_bits = n_bits
        self.radius = radius
        self.model = None
        self.is_trained = False
        self._train_smiles = []
        self._train_y = []

    def _make_model(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _get_fingerprints(self, smiles_list: list) -> tuple:
        from polymatgen.ml.features import batch_fingerprints
        return batch_fingerprints(
            smiles_list, radius=self.radius,
            n_bits=self.n_bits, skip_errors=True
        )

    def train(self):
        """Load data and train the model. Must be implemented by subclass."""
        raise NotImplementedError

    def predict(self, smiles: str) -> float:
        """
        Predict property value for a single polymer SMILES.

        Parameters
        ----------
        smiles : str — p-SMILES string

        Returns
        -------
        float — predicted property value
        """
        if not self.is_trained:
            self.train()
        from polymatgen.ml.features import psmiles_to_fingerprint
        fp = psmiles_to_fingerprint(smiles, radius=self.radius,
                                     n_bits=self.n_bits)
        return float(self.model.predict([fp])[0])

    def predict_with_uncertainty(self, smiles: str) -> tuple:
        """
        Predict property value with uncertainty estimate.

        Uses the variance across individual trees in the Random Forest
        to estimate prediction uncertainty.

        Parameters
        ----------
        smiles : str — p-SMILES string

        Returns
        -------
        tuple of (mean, std) where:
            mean : float — predicted property value
            std  : float — standard deviation across trees

        Notes
        -----
        A large std relative to mean indicates the model is uncertain
        about this prediction. High-uncertainty candidates are good
        targets for experimental synthesis to reduce model uncertainty.
        """
        if not self.is_trained:
            self.train()
        from polymatgen.ml.features import psmiles_to_fingerprint
        fp = psmiles_to_fingerprint(smiles, radius=self.radius,
                                     n_bits=self.n_bits)
        tree_preds = np.array([
            tree.predict([fp])[0]
            for tree in self.model.estimators_
        ])
        return float(tree_preds.mean()), float(tree_preds.std())

    def predict_batch(self, smiles_list: list) -> list:
        """
        Predict property values for a list of SMILES strings.

        Returns
        -------
        list of (smiles, predicted_value) tuples for valid SMILES
        """
        if not self.is_trained:
            self.train()
        X, valid_idx = self._get_fingerprints(smiles_list)
        if len(X) == 0:
            return []
        preds = self.model.predict(X)
        return [(smiles_list[i], float(p))
                for i, p in zip(valid_idx, preds)]

    def predict_batch_with_uncertainty(self, smiles_list: list) -> list:
        """
        Predict property values with uncertainty for a list of SMILES.

        Returns
        -------
        list of (smiles, mean, std) tuples for valid SMILES
        """
        if not self.is_trained:
            self.train()
        from polymatgen.ml.features import batch_fingerprints
        X, valid_idx = batch_fingerprints(
            smiles_list, radius=self.radius,
            n_bits=self.n_bits, skip_errors=True
        )
        if len(X) == 0:
            return []

        results = []
        for i, fp in zip(valid_idx, X):
            tree_preds = np.array([
                tree.predict([fp])[0]
                for tree in self.model.estimators_
            ])
            results.append((
                smiles_list[i],
                float(tree_preds.mean()),
                float(tree_preds.std()),
            ))
        return results

    def uncertainty_threshold(self, smiles_list: list,
                               max_std: float = None) -> dict:
        """
        Filter a list of SMILES by prediction uncertainty.

        Candidates below the threshold are confident predictions.
        Candidates above are flagged as high-priority for experimental
        synthesis to reduce model uncertainty.

        Parameters
        ----------
        smiles_list : list of str
        max_std     : float — uncertainty cutoff (default: median std)

        Returns
        -------
        dict with keys:
            'confident' : list of (smiles, mean, std) below threshold
            'uncertain' : list of (smiles, mean, std) above threshold
            'threshold' : float — the std cutoff used
        """
        results = self.predict_batch_with_uncertainty(smiles_list)
        if not results:
            return {"confident": [], "uncertain": [], "threshold": None}

        stds = np.array([r[2] for r in results])
        if max_std is None:
            max_std = float(np.median(stds))

        confident = [r for r in results if r[2] <= max_std]
        uncertain = [r for r in results if r[2] > max_std]

        return {
            "confident": confident,
            "uncertain": uncertain,
            "threshold": max_std,
        }

    def evaluate(self, test_fraction: float = 0.2) -> dict:
        """
        Evaluate model performance using a train/test split.

        Parameters
        ----------
        test_fraction : float — fraction of data to use as test set

        Returns
        -------
        dict with r2, mae, rmse on test set
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.ensemble import RandomForestRegressor

        X, valid_idx = self._get_fingerprints(self._train_smiles)
        y = np.array([self._train_y[i] for i in valid_idx])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_fraction,
            random_state=self.random_state
        )
        model = self._make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

        return {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }

    def save(self, filepath: str) -> None:
        """Save trained model to a file using joblib."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        import joblib
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load a previously saved model from a file."""
        import joblib
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return (f"{self.__class__.__name__}("
                f"n_estimators={self.n_estimators}, "
                f"n_bits={self.n_bits}, "
                f"status={status})")


class TgPredictor(_BasePredictor):
    """
    Predicts glass transition temperature (Tg) in Kelvin.

    Trained on the PolyMetriX curated experimental Tg dataset
    (7,367 polymers, literature-mined experimental values).
    Reliability filter: excludes 'red' reliability entries.

    Example
    -------
    predictor = TgPredictor()
    tg = predictor.predict("[*]CC([*])c1ccccc1")
    mean, std = predictor.predict_with_uncertainty("[*]CC([*])c1ccccc1")
    print(f"Tg: {mean:.1f} +/- {std:.1f} K")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(TG_PATH):
            raise FileNotFoundError(
                f"Tg dataset not found at {TG_PATH}"
            )
        df = pd.read_csv(TG_PATH)
        df = df[df["meta.reliability"] != "red"]
        df = df[df["labels.Exp_Tg(K)"].notna()]
        df = df[df["PSMILES"].notna()]

        self._train_smiles = df["PSMILES"].tolist()
        self._train_y = df["labels.Exp_Tg(K)"].tolist()

        X, valid_idx = self._get_fingerprints(self._train_smiles)
        y = np.array([self._train_y[i] for i in valid_idx])

        self.model = self._make_model()
        self.model.fit(X, y)
        self.is_trained = True
        print(f"TgPredictor trained on {len(y)} polymers.")


class BandgapPredictor(_BasePredictor):
    """
    Predicts electronic bandgap (eV) for polymer chains.

    Trained on the polyVERSE bandgap_chain dataset
    (4,209 DFT-computed bandgap values).

    Example
    -------
    predictor = BandgapPredictor()
    mean, std = predictor.predict_with_uncertainty("[*]CC([*])c1ccccc1")
    print(f"Bandgap: {mean:.3f} +/- {std:.3f} eV")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(BANDGAP_PATH):
            raise FileNotFoundError(
                f"Bandgap dataset not found at {BANDGAP_PATH}"
            )
        df = pd.read_csv(BANDGAP_PATH)
        df = df[df["bandgap_chain"].notna()]
        df = df[df["smiles"].notna()]

        self._train_smiles = df["smiles"].tolist()
        self._train_y = df["bandgap_chain"].tolist()

        X, valid_idx = self._get_fingerprints(self._train_smiles)
        y = np.array([self._train_y[i] for i in valid_idx])

        self.model = self._make_model()
        self.model.fit(X, y)
        self.is_trained = True
        print(f"BandgapPredictor trained on {len(y)} polymers.")


class CohesiveEnergyPredictor(_BasePredictor):
    """
    Predicts cohesive energy density (CED) in MPa.

    Trained on the polyVERSE cohesive energy density dataset
    (294 entries). The Hildebrand solubility parameter can be
    obtained as delta = sqrt(CED).

    Example
    -------
    predictor = CohesiveEnergyPredictor()
    mean, std = predictor.predict_with_uncertainty("[*]CC([*])c1ccccc1")
    print(f"CED: {mean:.2f} +/- {std:.2f} MPa")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(CED_PATH):
            raise FileNotFoundError(
                f"CED dataset not found at {CED_PATH}"
            )
        df = pd.read_csv(CED_PATH)
        df = df[df["value_COE"].notna()]
        df = df[df["smiles1"].notna()]

        self._train_smiles = df["smiles1"].tolist()
        self._train_y = df["value_COE"].tolist()

        X, valid_idx = self._get_fingerprints(self._train_smiles)
        y = np.array([self._train_y[i] for i in valid_idx])

        self.model = self._make_model()
        self.model.fit(X, y)
        self.is_trained = True
        print(f"CohesiveEnergyPredictor trained on {len(y)} polymers.")