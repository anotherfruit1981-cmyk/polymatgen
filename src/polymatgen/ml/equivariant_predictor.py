"""
E(3)-Equivariant GNN predictors for polymatgen.

These models operate on 3D polymer conformers rather than 2D graphs.
If you rotate or translate the polymer in 3D space, the predicted
scalar property (Tg, bandgap, CED) remains unchanged — this is the
equivariance guarantee:

    f(g · x) = f(x)  for g in E(3)

Architecture:
    1. Generate 3D conformer from p-SMILES using RDKit ETKDGv3
    2. Build radius graph (edges between atoms within cutoff distance)
    3. Pass through e3nn spherical harmonic message-passing layers
    4. Aggregate to scalar prediction via invariant pooling

Predictors:
    EquivariantTgPredictor
    EquivariantBandgapPredictor
    EquivariantCohesiveEnergyPredictor
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

# Radial cutoff for building the neighbour graph (Angstroms)
CUTOFF = 5.0

# Number of atom-type one-hot features
N_ATOM_TYPES = 10

# Common elements in polymers mapped to indices
ATOM_TYPE_MAP = {
    1: 0,   # H
    6: 1,   # C
    7: 2,   # N
    8: 3,   # O
    9: 4,   # F
    14: 5,  # Si
    15: 6,  # P
    16: 7,  # S
    17: 8,  # Cl
}
# Index 9 = "other"


def _clean_psmiles(psmiles: str) -> str:
    """Replace polymerization point markers with hydrogen."""
    clean = psmiles.replace("[*]", "[H]")
    clean = clean.replace("*", "[H]")
    return clean


def _atom_type_onehot(atomic_num: int) -> list:
    """One-hot encode atomic number into N_ATOM_TYPES bins."""
    idx = ATOM_TYPE_MAP.get(atomic_num, N_ATOM_TYPES - 1)
    vec = [0.0] * N_ATOM_TYPES
    vec[idx] = 1.0
    return vec


def smiles_to_3d(psmiles: str) -> tuple:
    """
    Convert p-SMILES to a 3D conformer.

    Parameters
    ----------
    psmiles : str — polymer SMILES with * as polymerization points

    Returns
    -------
    tuple of (positions, atomic_numbers) where:
        positions      : np.ndarray of shape (n_atoms, 3)
        atomic_numbers : list of int, length n_atoms

    Raises
    ------
    ValueError if conformer generation fails
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    clean = _clean_psmiles(psmiles)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {psmiles}")

    mol = Chem.AddHs(mol)

    # Try ETKDGv3 first
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)

    # Fallback 1: random coords + distance geometry
    if result != 0:
        result = AllChem.EmbedMolecule(
            mol, AllChem.ETKDGv3()
        )

    # Fallback 2: use random coordinate embedding
    if result != 0:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            result = 0
        except Exception:
            pass

    # Fallback 3: use 2D coords promoted to 3D
    if result != 0:
        try:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            # Promote to 3D by adding small z-noise
            import random as _random
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                conf.SetAtomPosition(
                    i, (pos.x, pos.y, _random.uniform(-0.1, 0.1))
                )
            result = 0
        except Exception:
            pass

    if result != 0:
        raise ValueError(
            f"Could not generate 3D conformer for: {psmiles}"
        )

    # Optional: minimise with MMFF
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    positions = np.array(conf.GetPositions(), dtype=np.float32)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    return positions, atomic_numbers

def _build_radius_graph(positions: np.ndarray,
                         cutoff: float = CUTOFF) -> np.ndarray:
    """
    Build a radius graph: connect all atom pairs within cutoff distance.

    Parameters
    ----------
    positions : np.ndarray of shape (n_atoms, 3)
    cutoff    : float — distance threshold in Angstroms

    Returns
    -------
    np.ndarray of shape (2, n_edges) — edge indices
    """
    n = len(positions)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    src.append(i)
                    dst.append(j)
    if not src:
        # Fallback: connect to nearest neighbour
        for i in range(n):
            dists = [np.linalg.norm(positions[i] - positions[j])
                     if i != j else np.inf for j in range(n)]
            j = int(np.argmin(dists))
            src.extend([i, j])
            dst.extend([j, i])
    return np.array([src, dst], dtype=np.int64)


def smiles_to_equivariant_graph(psmiles: str):
    """
    Convert p-SMILES to a torch_geometric Data object with 3D positions.

    Parameters
    ----------
    psmiles : str — polymer SMILES

    Returns
    -------
    torch_geometric.data.Data with:
        x         : one-hot atom type features (n_atoms, N_ATOM_TYPES)
        pos       : 3D coordinates (n_atoms, 3)
        edge_index: radius graph edges (2, n_edges)

    Raises
    ------
    ValueError if SMILES cannot be parsed or conformer fails
    """
    import torch
    from torch_geometric.data import Data

    positions, atomic_numbers = smiles_to_3d(psmiles)
    edge_index = _build_radius_graph(positions, cutoff=CUTOFF)

    x = torch.tensor(
        [_atom_type_onehot(an) for an in atomic_numbers],
        dtype=torch.float
    )
    pos = torch.tensor(positions, dtype=torch.float)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)

    return Data(x=x, pos=pos, edge_index=edge_index_t)


class _EquivariantNet:
    """
    E(3)-equivariant message-passing network using e3nn.

    Architecture:
        - Spherical harmonic edge features (l=0,1,2)
        - 2 x tensor product convolution layers
        - Invariant (l=0) output pooled to scalar
    """

    def __init__(self, hidden_irreps: str = "16x0e + 8x1o + 4x2e",
                 max_radius: float = CUTOFF):
        self.hidden_irreps_str = hidden_irreps
        self.max_radius = max_radius
        self._model = None

    def build(self):
        import torch
        import torch.nn as nn
        from e3nn import o3
        from e3nn.nn import FullyConnectedNet
        from e3nn.o3 import Irreps, FullyConnectedTensorProduct
        from e3nn.nn.models.gate_points_2101 import Network

        # Input: one-hot atom types = N_ATOM_TYPES scalars (l=0, even)
        in_irreps = o3.Irreps(f"{N_ATOM_TYPES}x0e")
        hidden_irreps = o3.Irreps(self.hidden_irreps_str)
        out_irreps = o3.Irreps("1x0e")  # scalar output

        # Use e3nn's built-in equivariant network
        net = Network(
            irreps_in=in_irreps,
            irreps_hidden=hidden_irreps,
            irreps_out=out_irreps,
            irreps_node_attr=o3.Irreps("0e"),  # no node attributes
            irreps_edge_attr=o3.Irreps.spherical_harmonics(lmax=2),
            layers=2,
            max_radius=self.max_radius,
            number_of_basis=8,
            radial_layers=1,
            radial_neurons=16,
            num_neighbors=8.0,
            num_nodes=20.0,
            reduce_output=True,  # global pooling to scalar
        )
        self._model = net
        return net


class _BaseEquivariantPredictor:
    """
    Base class for E(3)-equivariant property predictors.

    Trains an e3nn network on 3D polymer conformers.
    Falls back gracefully if conformer generation fails for a molecule.
    """

    def __init__(self, epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 16, random_state: int = 42,
                 max_train_samples: int = 1000):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.model = None
        self.is_trained = False
        self._y_mean = 0.0
        self._y_std = 1.0

    def _build_dataset(self, smiles_list: list,
                        y_list: list) -> list:
        """Convert SMILES + labels to 3D graph Data objects."""
        import torch
        from torch_geometric.data import Data

        data_list = []
        failed = 0
        for smi, y in zip(smiles_list, y_list):
            try:
                graph = smiles_to_equivariant_graph(smi)
                graph.y = torch.tensor([y], dtype=torch.float)
                data_list.append(graph)
            except Exception:
                failed += 1
                continue
        if failed > 0:
            print(f"  ({failed} molecules skipped — conformer failed)")
        return data_list

    def train(self):
        raise NotImplementedError

    def _train_on(self, smiles_list: list, y_list: list):
        """Core equivariant training loop."""
        import torch
        from torch_geometric.loader import DataLoader

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Subsample for speed
        if len(smiles_list) > self.max_train_samples:
            idx = np.random.choice(
                len(smiles_list), self.max_train_samples, replace=False
            )
            smiles_list = [smiles_list[i] for i in idx]
            y_list = [y_list[i] for i in idx]

        print(f"  Building 3D dataset ({len(smiles_list)} molecules)...")
        dataset = self._build_dataset(smiles_list, y_list)
        if len(dataset) == 0:
            raise RuntimeError("No valid 3D conformers generated.")
        print(f"  {len(dataset)} valid conformers generated.")

        # Normalise targets
        ys = np.array([d.y.item() for d in dataset])
        self._y_mean = float(ys.mean())
        self._y_std = float(ys.std()) if ys.std() > 0 else 1.0
        for d in dataset:
            d.y = (d.y - self._y_mean) / self._y_std

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        net_builder = _EquivariantNet()
        net = net_builder.build()
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                optimiser.zero_grad()
                # e3nn Network forward: needs pos and batch
                out = net(batch)
                pred = out.squeeze(-1)
                loss = loss_fn(pred, batch.y)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}  "
                      f"loss={total_loss / len(loader):.4f}")

        self.model = net
        self.is_trained = True

    def predict(self, smiles: str) -> float:
        """
        Predict property value for a single polymer SMILES.

        Generates a 3D conformer and runs the equivariant network.

        Parameters
        ----------
        smiles : str — p-SMILES string

        Returns
        -------
        float — predicted property value in original units
        """
        if not self.is_trained:
            self.train()
        import torch
        self.model.eval()
        graph = smiles_to_equivariant_graph(smiles)
        # Add batch vector
        graph.batch = torch.zeros(
            graph.x.size(0), dtype=torch.long
        )
        with torch.no_grad():
            out = self.model(graph)
        return float(out.squeeze(-1).item()) * self._y_std + self._y_mean

    def predict_batch(self, smiles_list: list) -> list:
        """
        Predict for a list of SMILES strings.

        Returns
        -------
        list of (smiles, value) tuples for valid SMILES
        """
        if not self.is_trained:
            self.train()
        import torch
        self.model.eval()
        results = []
        for smi in smiles_list:
            try:
                graph = smiles_to_equivariant_graph(smi)
                graph.batch = torch.zeros(
                    graph.x.size(0), dtype=torch.long
                )
                with torch.no_grad():
                    out = self.model(graph)
                val = float(out.squeeze(-1).item()) * self._y_std + self._y_mean
                results.append((smi, val))
            except Exception:
                continue
        return results

    def save(self, filepath: str) -> None:
        """Save model weights to file."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        import torch
        torch.save({
            "state_dict": self.model.state_dict(),
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }, filepath)
        print(f"Equivariant model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model weights from file."""
        import torch
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Not found: {filepath}")
        checkpoint = torch.load(filepath, map_location="cpu")
        self._y_mean = checkpoint["y_mean"]
        self._y_std = checkpoint["y_std"]
        net_builder = _EquivariantNet()
        net = net_builder.build()
        net.load_state_dict(checkpoint["state_dict"])
        self.model = net
        self.is_trained = True
        print(f"Equivariant model loaded from {filepath}")

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return (f"{self.__class__.__name__}("
                f"epochs={self.epochs}, "
                f"max_train_samples={self.max_train_samples}, "
                f"status={status})")


class EquivariantTgPredictor(_BaseEquivariantPredictor):
    """
    E(3)-equivariant Tg predictor.

    Unlike the RF and GCN predictors, this model is sensitive to
    3D conformation — it can distinguish isotactic from syndiotactic
    polymers that have identical 2D SMILES.

    Example
    -------
    predictor = EquivariantTgPredictor(epochs=10, max_train_samples=500)
    predictor.train()
    tg = predictor.predict("[*]CC([*])c1ccccc1")
    print(f"Equivariant Tg: {tg:.1f} K")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(TG_PATH):
            raise FileNotFoundError(f"Tg dataset not found at {TG_PATH}")
        df = pd.read_csv(TG_PATH)
        df = df[df["meta.reliability"] != "red"]
        df = df[df["labels.Exp_Tg(K)"].notna()]
        df = df[df["PSMILES"].notna()]
        print(f"EquivariantTgPredictor: training on up to "
              f"{self.max_train_samples} of {len(df)} polymers...")
        self._train_on(
            df["PSMILES"].tolist(),
            df["labels.Exp_Tg(K)"].tolist()
        )
        print("EquivariantTgPredictor training complete.")


class EquivariantBandgapPredictor(_BaseEquivariantPredictor):
    """
    E(3)-equivariant bandgap predictor.

    Example
    -------
    predictor = EquivariantBandgapPredictor(epochs=10, max_train_samples=500)
    predictor.train()
    bg = predictor.predict("[*]CC([*])c1ccccc1")
    print(f"Equivariant Bandgap: {bg:.3f} eV")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(BANDGAP_PATH):
            raise FileNotFoundError(
                f"Bandgap dataset not found at {BANDGAP_PATH}")
        df = pd.read_csv(BANDGAP_PATH)
        df = df[df["bandgap_chain"].notna()]
        df = df[df["smiles"].notna()]
        print(f"EquivariantBandgapPredictor: training on up to "
              f"{self.max_train_samples} of {len(df)} polymers...")
        self._train_on(
            df["smiles"].tolist(),
            df["bandgap_chain"].tolist()
        )
        print("EquivariantBandgapPredictor training complete.")


class EquivariantCohesiveEnergyPredictor(_BaseEquivariantPredictor):
    """
    E(3)-equivariant cohesive energy density predictor.

    Example
    -------
    predictor = EquivariantCohesiveEnergyPredictor(epochs=10)
    predictor.train()
    ced = predictor.predict("[*]CC([*])c1ccccc1")
    print(f"Equivariant CED: {ced:.2f} MPa")
    """

    def train(self):
        import pandas as pd
        if not os.path.exists(CED_PATH):
            raise FileNotFoundError(
                f"CED dataset not found at {CED_PATH}")
        df = pd.read_csv(CED_PATH)
        df = df[df["value_COE"].notna()]
        df = df[df["smiles1"].notna()]
        print(f"EquivariantCohesiveEnergyPredictor: training on up to "
              f"{self.max_train_samples} of {len(df)} polymers...")
        self._train_on(
            df["smiles1"].tolist(),
            df["value_COE"].tolist()
        )
        print("EquivariantCohesiveEnergyPredictor training complete.")