"""
polyVERSE dataset loaders for polymatgen.

polyVERSE is an open-source repository of informatics-ready polymer datasets
curated by the Ramprasad Group at Georgia Tech.

Source: https://github.com/Ramprasad-Group/polyVERSE
License: See repository LICENSE file.

Datasets included:
    chi_parameter.csv          - Flory-Huggins chi parameter (polymer/solvent pairs)
    bandgap_chain.csv          - Electronic bandgap (eV) from DFT
    Gas_permeability_...csv    - Gas permeability, diffusivity, solubility
    Cohesive_energy_density... - Cohesive energy density (related to Hildebrand delta)
"""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CHI_PATH = os.path.join(DATA_DIR, "chi_parameter.csv")
BANDGAP_PATH = os.path.join(DATA_DIR, "bandgap_chain.csv")
GAS_PATH = os.path.join(DATA_DIR, "Gas_permeability_solubility_diffusivity_wide.csv")
CED_PATH = os.path.join(DATA_DIR, "Cohesive_energy_density_2025_06_23.csv")
TG_PATH = os.path.join(DATA_DIR, "LAMALAB_CURATED_Tg_structured_polymerclass_with_embeddings.csv")

def _load_csv(filepath: str):
    """Internal helper to load a CSV with pandas."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Run: uv add pandas")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            f"Copy the file from polyVERSE into src/polymatgen/data/"
        )
    return pd.read_csv(filepath)


# --- Chi parameter ---

def load_chi(filepath: str = None) -> list:
    """
    Load the Flory-Huggins chi parameter dataset.

    Columns: Polymer, Polymer_SMILES, Solvent, Solvent_SMILES, chi, temperature, Reference

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or CHI_PATH)
    return df.to_dict(orient="records")


def search_chi_by_polymer(polymer_name: str, filepath: str = None) -> list:
    """
    Search chi parameter entries by polymer name (case-insensitive substring match).

    Parameters
    ----------
    polymer_name : str — e.g. 'polystyrene'

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or CHI_PATH)
    mask = df["Polymer"].str.lower().str.contains(polymer_name.lower(), na=False, regex=False)
    return df[mask].to_dict(orient="records")


def search_chi_by_smiles(smiles: str, filepath: str = None) -> list:
    """
    Search chi parameter entries by exact polymer SMILES.

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or CHI_PATH)
    mask = df["Polymer_SMILES"] == smiles
    return df[mask].to_dict(orient="records")


# --- Bandgap ---

def load_bandgap(filepath: str = None) -> list:
    """
    Load the electronic bandgap dataset.

    Columns: smiles, bandgap_chain (eV)

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or BANDGAP_PATH)
    return df.to_dict(orient="records")


def search_bandgap(min_val: float = None, max_val: float = None,
                   filepath: str = None) -> list:
    """
    Search bandgap dataset by value range (eV).

    Parameters
    ----------
    min_val : float — minimum bandgap in eV
    max_val : float — maximum bandgap in eV

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or BANDGAP_PATH)
    mask = df["bandgap_chain"].notna()
    if min_val is not None:
        mask &= df["bandgap_chain"] >= min_val
    if max_val is not None:
        mask &= df["bandgap_chain"] <= max_val
    return df[mask].to_dict(orient="records")


# --- Gas permeability ---

def load_gas_permeability(filepath: str = None) -> list:
    """
    Load the gas permeability/diffusivity/solubility dataset.

    Columns: smiles_string, d_exp/sim_{gas}, p_exp/sim_{gas}, s_exp/sim_{gas}
    where gas is CH4, CO2, H2, He, N2, O2.
    d = diffusivity, p = permeability, s = solubility
    exp = experimental, sim = simulated

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or GAS_PATH)
    return df.to_dict(orient="records")


def search_gas_permeability(gas: str, prop: str = "p_exp",
                             min_val: float = None, max_val: float = None,
                             filepath: str = None) -> list:
    """
    Search gas transport dataset by gas and property.

    Parameters
    ----------
    gas     : str   — gas name: 'CH4', 'CO2', 'H2', 'He', 'N2', 'O2'
    prop    : str   — property prefix: 'p_exp', 'p_sim', 'd_exp', 'd_sim',
                      's_exp', 's_sim' (default 'p_exp')
    min_val : float — minimum value
    max_val : float — maximum value

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or GAS_PATH)
    col = f"{prop}_{gas}"
    if col not in df.columns:
        available = [c for c in df.columns if c != "smiles_string"]
        raise ValueError(f"Column '{col}' not found. Available: {available}")
    mask = df[col].notna()
    if min_val is not None:
        mask &= df[col] >= min_val
    if max_val is not None:
        mask &= df[col] <= max_val
    return df[mask][["smiles_string", col]].to_dict(orient="records")


# --- Cohesive energy density ---

def load_cohesive_energy_density(filepath: str = None) -> list:
    """
    Load the cohesive energy density dataset.

    Columns: PID, smiles1, value_COE (cohesive energy density in MPa)

    The Hildebrand solubility parameter delta = sqrt(CED) in (MPa)^0.5.

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or CED_PATH)
    return df[["PID", "smiles1", "value_COE"]].to_dict(orient="records")


def search_ced(min_val: float = None, max_val: float = None,
               filepath: str = None) -> list:
    """
    Search cohesive energy density by value range (MPa).

    Parameters
    ----------
    min_val : float — minimum CED in MPa
    max_val : float — maximum CED in MPa

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or CED_PATH)
    mask = df["value_COE"].notna()
    if min_val is not None:
        mask &= df["value_COE"] >= min_val
    if max_val is not None:
        mask &= df["value_COE"] <= max_val
    return df[mask][["PID", "smiles1", "value_COE"]].to_dict(orient="records")


def ced_to_hildebrand(ced: float) -> float:
    """
    Convert cohesive energy density (MPa) to Hildebrand
    solubility parameter (MPa^0.5).

    delta = sqrt(CED)

    Parameters
    ----------
    ced : float — cohesive energy density in MPa

    Returns
    -------
    float — Hildebrand solubility parameter in (MPa)^0.5
    """
    import math
    if ced < 0:
        raise ValueError(f"CED must be non-negative, got {ced}")
    return math.sqrt(ced)

# --- Experimental Tg (PolyMetriX curated) ---

def load_tg(filepath: str = None) -> list:
    """
    Load the PolyMetriX curated experimental Tg dataset.

    Key columns:
        PSMILES            : polymer SMILES (p-SMILES format)
        labels.Exp_Tg(K)   : experimental glass transition temperature in Kelvin
        meta.polymer       : polymer name
        meta.polymer_class : polymer family classification
        meta.reliability   : data reliability score
        meta.source        : literature source

    Returns
    -------
    list of dicts (key columns only)
    """
    df = _load_csv(filepath or TG_PATH)
    key_cols = [
        "PSMILES",
        "labels.Exp_Tg(K)",
        "meta.polymer",
        "meta.polymer_class",
        "meta.reliability",
        "meta.source",
        "meta.num_of_points",
        "meta.std",
    ]
    available = [c for c in key_cols if c in df.columns]
    return df[available].to_dict(orient="records")


def search_tg(min_tg: float = None, max_tg: float = None,
              polymer_class: str = None,
              min_reliability: float = None,
              filepath: str = None) -> list:
    """
    Search the experimental Tg dataset by value range, polymer class,
    or reliability score.

    Parameters
    ----------
    min_tg          : float — minimum Tg in Kelvin
    max_tg          : float — maximum Tg in Kelvin
    polymer_class   : str   — polymer family e.g. 'polyacrylate', 'polyester'
    min_reliability : float — minimum reliability score (0-1)

    Returns
    -------
    list of dicts
    """
    df = _load_csv(filepath or TG_PATH)
    tg_col = "labels.Exp_Tg(K)"
    mask = df[tg_col].notna()

    if min_tg is not None:
        mask &= df[tg_col] >= min_tg
    if max_tg is not None:
        mask &= df[tg_col] <= max_tg
    if polymer_class is not None:
        mask &= df["meta.polymer_class"].str.lower().str.contains(
            polymer_class.lower(), na=False)
    if min_reliability is not None and "meta.reliability" in df.columns:
        mask &= df["meta.reliability"] >= min_reliability

    key_cols = [
        "PSMILES",
        "labels.Exp_Tg(K)",
        "meta.polymer",
        "meta.polymer_class",
        "meta.reliability",
        "meta.source",
    ]
    available = [c for c in key_cols if c in df.columns]
    return df[mask][available].to_dict(orient="records")


def tg_stats(filepath: str = None) -> dict:
    """
    Return summary statistics for the experimental Tg dataset.

    Returns
    -------
    dict with total entries, Tg min/max/mean, and polymer class counts
    """
    df = _load_csv(filepath or TG_PATH)
    tg = df["labels.Exp_Tg(K)"].dropna()
    classes = df["meta.polymer_class"].value_counts().to_dict() \
        if "meta.polymer_class" in df.columns else {}
    return {
        "total_entries": len(df),
        "Tg_K": {
            "count": len(tg),
            "min": round(float(tg.min()), 2),
            "max": round(float(tg.max()), 2),
            "mean": round(float(tg.mean()), 2),
        },
        "polymer_classes": classes,
    }
