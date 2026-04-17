"""
Graph Convolutional Network (GCN) predictors for polymatgen.

Instead of fixed-length Morgan fingerprints, these predictors learn
directly from the polymer graph structure using message-passing layers.
Each atom is a node; each bond is an edge. The GCN learns which
structural patterns matter for a given property.

Predictors included:
    GCNTgPredictor              — glass transition temperature (K)
    GCNBandgapPredictor         — electronic bandgap (eV)
    GCNCohesiveEnergyPredictor  — cohesive energy density (MPa)

Architecture:
    3 x GCNConv layers (64 hidden units)
    → Global mean pooling
    → 2 x Linear layers
    → Scalar output
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

# Atom feature dimension (see _atom_features below)
ATOM_FEATURE_DIM = 9


def _clean_psmiles(psmiles: str) -> str:
    """Replace polymerization point markers with hydrogen."""
    clean = psmiles.replace("[*]", "[H]")
    clean = clean.replace("*", "[H]")
    return clean


def _atom_features(atom) -> list:
    """
    Compute a fixed-length feature vector for a single RDKit atom.

    Features (9 total):
        - Atomic number (normalised by 118)
        - Degree (normalised by 6)
        - Formal charge (normalised by 4)
        - Number of Hs (normalised by 4)
        - Is in ring (0/1)
        - Is aromatic (0/1)
        - Hybridisation: SP, SP2, SP3 (one-hot, 3 values)
    """
    from rdkit.Chem import rdchem
    hyb = atom.GetHybridization()
    return [
        atom.GetAtomicNum() / 118.0,
        atom.GetDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        int(hyb == rdchem.HybridizationType.SP),
        int(hyb == rdchem.HybridizationType.SP2),
        int(hyb == rdchem.HybridizationType.SP3),
    ]


def smiles_to_graph(psmiles: str):
    """
    Convert a p-SMILES string to a torch_geometric Data object.

    Parameters
    ----------
    psmiles : str — polymer SMILES with * as polymerization points

    Returns
    -------
    torch_geometric.data.Data with:
        x         : node feature matrix (n_atoms, ATOM_FEATURE_DIM)
        edge_index: COO-format edge list (2, n_edges * 2)

    Raises
    ------
    ValueError if SMILES cannot be parsed or molecule has no atoms
    """
    import torch
    from torch_geometric.data import Data
    from rdkit import Chem

    clean = _clean_psmiles(psmiles)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {psmiles}")
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"Empty molecule from SMILES: {psmiles}")

    # Node features
    x = torch.tensor(
        [_atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float
    )

    # Edge index (undirected: add both directions)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        # Single-atom molecule — add self-loop
        edge_index = torch.zeros((2, 1), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class _GCNModel:
    """
    Lightweight 3-layer GCN with global mean pooling.
    Wraps torch_geometric so the import is deferred.
    """

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self._model = None

    def build(self):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv, global_mean_pool

        class _Net(nn.Module):
            def __init__(self, in_dim, hidden_dim):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, hidden_dim)
                self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.lin2 = nn.Linear(hidden_dim // 2, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

            def forward(self, x, edge_index, batch):
                x = self.relu(self.conv1(x, edge_index))
                x = self.relu(self.conv2(x, edge_index))
                x = self.relu(self.conv3(x, edge_index))
                x = global_mean_pool(x, batch)
                x = self.dropout(self.relu(self.lin1(x)))
                return self.lin2(x).squeeze(-1)

        self._model = _Net(ATOM_FEATURE_DIM, self.hidden_dim)
        return self._model


class _BaseGCNPredictor:
    """
    Base class for GCN-based property predictors.

    Uses a 3-layer GCN trained with Adam + MSE loss.
    Supports the same interface as _BasePredictor for drop-in use.
    """

    def __init__(self, hidden_dim: int = 64, epochs: int = 50,
                 lr: float = 1e-3, batch_size: int = 32,
                 random_state: int = 42):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self._y_mean = 0.0
        self._y_std = 1.0

    def _build_dataset(self, smiles_list: list, y_list: list) -> list:
        """Convert SMILES + labels to a list of torch_geometric Data objects."""
        import torch
        data_list = []
        for smi, y in zip(smiles_list, y_list):
            try:
                graph = smiles_to_graph(smi)
                graph.y = torch.tensor([y], dtype=torch.float)
                data_list.append(graph)
            except ValueError:
                continue
        return data_list

    def train(self):
        """Load data and train the GCN. Must be implemented by subclass."""
        raise NotImplementedError

    def _train_on(self, smiles_list: list, y_list: list):
        """Core training loop."""
        import torch
        from torch_geometric.loader import DataLoader

        torch.manual_seed(self.random_state)

        dataset = self._build_dataset(smiles_list, y_list)
        if len(dataset) == 0:
            raise RuntimeError("No valid molecules in training data.")

        # Normalise targets
        ys = np.array([d.y.item() for d in dataset])
        self._y_mean = float(ys.mean())
        self._y_std = float(ys.std()) if ys.std() > 0 else 1.0
        for d in dataset:
            d.y = (d.y - self._y_mean) / self._y_std

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        gcn = _GCNModel(hidden_dim=self.hidden_dim)
        net = gcn.build()
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                optimiser.zero_grad()
                pred = net(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}  "
                      f"loss={total_loss / len(loader):.4f}")

        self.model = net
        self.is_trained = True

    def predict(self, smiles: str) -> float:
        """
        Predict property value for a single polymer SMILES.

        Parameters
        ----------
        smiles : str — p-SMILES string

        Returns
        -------
        float — predicted property value (in original units)
        """
        if not self.is_trained:
            self.train()
        import torch
        self.model.eval()
        graph = smiles_to_graph(smiles)
        # Add batch vector (all zeros = single graph)
        batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        with torch.no_grad():
            pred = self.model(graph.x, graph.edge_index, batch)
        return float(pred.item()) * self._y_std + self._y_mean

    def predict_batch(self, smiles_list: list) -> list:
        """
        Predict property values for a list of SMILES strings.

        Returns
        -------
        list of (smiles, predicted_value) tuples for valid SMILES
        """
        if not self.is_trained:
            self.train()
        import torch
        self.model.eval()
        results = []
        for smi in smiles_list:
            try:
                graph = smiles_to_graph(smi)
                batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                with torch.no_grad():
                    pred = self.model(graph.x, graph.edge_index, batch)
                value = float(pred.item()) * self._y_std + self._y_mean
                results.append((smi, value))
            except ValueError:
                continue
        return results

    def evaluate(self, smiles_list: list, y_list: list,
                 test_fraction: float = 0.2) -> dict:
        """
        Evaluate on a held-out test split.

        Returns
        -------
        dict with r2, mae, rmse
        """
        import torch
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error

        pairs = [(s, y) for s, y in zip(smiles_list, y_list)]
        _, test_pairs = train_test_split(
            pairs, test_size=test_fraction,
            random_state=self.random_state
        )

        y_true, y_pred = [], []
        self.model.eval()
        for smi, y in test_pairs:
            try:
                graph = smiles_to_graph(smi)
                batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                with torch.no_grad():
                    pred = self.model(graph.x, graph.edge_index, batch)
                y_pred.append(float(pred.item()) * self._y_std + self._y_mean)
                y_true.append(y)
            except ValueError:
                continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        return {
            "n_test": len(y_true),
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }

    def save(self, filepath: str) -> None:
        """Save trained model weights."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        import torch
        torch.save({
            "state_dict": self.model.state_dict(),
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "hidden_dim": self.hidden_dim,
        }, filepath)
        print(f"GCN model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load saved model weights."""
        import torch
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location="cpu")
        self.hidden_dim = checkpoint["hidden_dim"]
        self._y_mean = checkpoint["y_mean"]
        self._y_std = checkpoint["y_std"]
        gcn = _GCNModel(hidden_dim=self.hidden_dim)
        net = gcn.build()
        net.load_state_dict(checkpoint["state_dict"])
        self.model = net
        self.is_trained = True
        print(f"GCN model loaded from {filepath}")

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return (f"{self.__class__.__name__}("
                f"hidden_dim={self.hidden_dim}, "
                f"epochs={self.epochs}, "
                f"status={status})")


class GCNTgPredictor(_BaseGCNPredictor):
    """
    GCN-based glass transition temperature predictor.

    Learns directly from the polymer graph structure rather than
    fixed-length fingerprints, capturing long-range backbone effects
    that Morgan fingerprints truncate at radius=2.

    Example
    -------
    predictor = GCNTgPredictor(epochs=50)
    predictor.train()
    tg = predictor.predict("[*]CC([*])c1ccccc1")
    print(f"GCN Tg: {tg:.1f} K")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(TG_PATH):
            raise FileNotFoundError(f"Tg dataset not found at {TG_PATH}")
        df = pd.read_csv(TG_PATH)
        df = df[df["meta.reliability"] != "red"]
        df = df[df["labels.Exp_Tg(K)"].notna()]
        df = df[df["PSMILES"].notna()]
        print(f"GCNTgPredictor: training on {len(df)} polymers "
              f"for {self.epochs} epochs...")
        self._train_smiles = df["PSMILES"].tolist()
        self._train_y = df["labels.Exp_Tg(K)"].tolist()
        self._train_on(self._train_smiles, self._train_y)
        print("GCNTgPredictor training complete.")

    def evaluate_default(self) -> dict:
        if not self.is_trained:
            self.train()
        return self.evaluate(self._train_smiles, self._train_y)


class GCNBandgapPredictor(_BaseGCNPredictor):
    """
    GCN-based electronic bandgap predictor.

    Example
    -------
    predictor = GCNBandgapPredictor(epochs=50)
    predictor.train()
    bg = predictor.predict("[*]CC([*])c1ccccc1")
    print(f"GCN Bandgap: {bg:.3f} eV")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(BANDGAP_PATH):
            raise FileNotFoundError(
                f"Bandgap dataset not found at {BANDGAP_PATH}")
        df = pd.read_csv(BANDGAP_PATH)
        df = df[df["bandgap_chain"].notna()]
        df = df[df["smiles"].notna()]
        print(f"GCNBandgapPredictor: training on {len(df)} polymers "
              f"for {self.epochs} epochs...")
        self._train_smiles = df["smiles"].tolist()
        self._train_y = df["bandgap_chain"].tolist()
        self._train_on(self._train_smiles, self._train_y)
        print("GCNBandgapPredictor training complete.")

    def evaluate_default(self) -> dict:
        if not self.is_trained:
            self.train()
        return self.evaluate(self._train_smiles, self._train_y)


class GCNCohesiveEnergyPredictor(_BaseGCNPredictor):
    """
    GCN-based cohesive energy density predictor.

    Example
    -------
    predictor = GCNCohesiveEnergyPredictor(epochs=50)
    predictor.train()
    ced = predictor.predict("[*]CC([*])c1ccccc1")
    print(f"GCN CED: {ced:.2f} MPa")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(CED_PATH):
            raise FileNotFoundError(
                f"CED dataset not found at {CED_PATH}")
        df = pd.read_csv(CED_PATH)
        df = df[df["value_COE"].notna()]
        df = df[df["smiles1"].notna()]
        print(f"GCNCohesiveEnergyPredictor: training on {len(df)} polymers "
              f"for {self.epochs} epochs...")
        self._train_smiles = df["smiles1"].tolist()
        self._train_y = df["value_COE"].tolist()
        self._train_on(self._train_smiles, self._train_y)
        print("GCNCohesiveEnergyPredictor training complete.")

    def evaluate_default(self) -> dict:
        if not self.is_trained:
            self.train()
        return self.evaluate(self._train_smiles, self._train_y)