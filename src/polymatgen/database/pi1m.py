"""
PI1M database loader for polymatgen.

PI1M is an open-source benchmark database of ~1 million polymer SMILES.
Paper: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c00726
Data:  https://github.com/RUIMINMA1996/PI1M

The CSV file (PI1M_v2.csv) has these columns:
    SMILES    : polymer SMILES string (p-SMILES, * = polymerization point)
    SA Score  : synthetic accessibility score 1-10 (lower = easier to synthesize)
"""

import os

DEFAULT_PI1M_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "PI1M_v2.csv"
)


def load_pi1m(filepath: str = None) -> list:
    """
    Load the PI1M dataset from a local CSV file.

    Parameters
    ----------
    filepath : str — path to PI1M_v2.csv (default: src/polymatgen/data/PI1M_v2.csv)

    Returns
    -------
    list of dicts with keys: SMILES, SA Score
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Run: uv add pandas")

    path = filepath or DEFAULT_PI1M_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PI1M file not found at: {path}\n"
            f"Download from: https://github.com/RUIMINMA1996/PI1M"
        )

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def search_by_sa_score(max_sa: float = 3.0, limit: int = 100,
                        filepath: str = None) -> list:
    """
    Return polymers that are easy to synthesize based on SA score.
    SA score ranges from 1 (easy) to 10 (hard).
    A cutoff of 3.0 is a common threshold for synthetically accessible molecules.

    Parameters
    ----------
    max_sa   : float — maximum SA score (default 3.0)
    limit    : int   — max results to return (default 100)
    filepath : str   — path to PI1M_v2.csv (optional)

    Returns
    -------
    list of dicts
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Run: uv add pandas")

    path = filepath or DEFAULT_PI1M_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PI1M file not found at: {path}\n"
            f"Download from: https://github.com/RUIMINMA1996/PI1M"
        )

    df = pd.read_csv(path)
    results = df[df["SA Score"] <= max_sa].head(limit)
    return results.to_dict(orient="records")


def pi1m_stats(filepath: str = None) -> dict:
    """
    Return summary statistics for the PI1M dataset.

    Parameters
    ----------
    filepath : str — path to PI1M_v2.csv (optional)

    Returns
    -------
    dict with total_entries, SA Score min/max/mean
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Run: uv add pandas")

    path = filepath or DEFAULT_PI1M_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PI1M file not found at: {path}\n"
            f"Download from: https://github.com/RUIMINMA1996/PI1M"
        )

    df = pd.read_csv(path)
    sa = df["SA Score"].dropna()

    return {
        "total_entries": len(df),
        "columns": list(df.columns),
        "SA Score": {
            "count_non_null": len(sa),
            "min": round(float(sa.min()), 4),
            "max": round(float(sa.max()), 4),
            "mean": round(float(sa.mean()), 4),
        }
    }


def smiles_to_monomer(psmiles: str) -> str:
    """
    Strip polymerization point markers (*) from a p-SMILES string
    to get a plain monomer SMILES compatible with RDKit.

    Parameters
    ----------
    psmiles : str — polymer SMILES e.g. '*CC(*)c1ccccc1'

    Returns
    -------
    str — monomer SMILES with * replaced by [H]
    """
    return psmiles.replace("*", "[H]")


def sample_pi1m(n: int = 10, filepath: str = None) -> list:
    """
    Return a random sample of n polymers from the dataset.

    Parameters
    ----------
    n        : int — number of random entries to return (default 10)
    filepath : str — path to PI1M_v2.csv (optional)

    Returns
    -------
    list of dicts
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Run: uv add pandas")

    path = filepath or DEFAULT_PI1M_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PI1M file not found at: {path}\n"
            f"Download from: https://github.com/RUIMINMA1996/PI1M"
        )

    df = pd.read_csv(path)
    return df.sample(n=min(n, len(df))).to_dict(orient="records")
