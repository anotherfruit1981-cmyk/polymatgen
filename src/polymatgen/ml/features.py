"""
Feature generation for polymer ML models.

Uses RDKit Morgan fingerprints computed from p-SMILES strings.
The polymerization point markers (*) are replaced with [H] before
fingerprinting so RDKit can parse the structure.
"""

import numpy as np


def _clean_psmiles(psmiles: str) -> str:
    """
    Convert polymer SMILES to a parseable SMILES by replacing
    polymerization point markers with hydrogen.

    Handles both bare * and bracketed [*] forms.
    """
    # Replace bracketed [*] first, then bare *
    clean = psmiles.replace("[*]", "[H]")
    clean = clean.replace("*", "[H]")
    return clean


def psmiles_to_fingerprint(psmiles: str, radius: int = 2,
                            n_bits: int = 2048) -> np.ndarray:
    """
    Convert a p-SMILES string to a Morgan fingerprint vector.

    Parameters
    ----------
    psmiles : str   — polymer SMILES with * as polymerization points
    radius  : int   — Morgan fingerprint radius (default 2 = ECFP4)
    n_bits  : int   — fingerprint length (default 2048)

    Returns
    -------
    np.ndarray of shape (n_bits,) with values 0 or 1

    Raises
    ------
    ValueError if SMILES cannot be parsed
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    clean = _clean_psmiles(psmiles)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {psmiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    return np.array(fp)


def batch_fingerprints(smiles_list: list, radius: int = 2,
                        n_bits: int = 2048,
                        skip_errors: bool = True) -> tuple:
    """
    Compute Morgan fingerprints for a list of SMILES strings.

    Parameters
    ----------
    smiles_list  : list of str — p-SMILES strings
    radius       : int         — Morgan radius (default 2)
    n_bits       : int         — fingerprint length (default 2048)
    skip_errors  : bool        — if True, skip unparseable SMILES
                                 if False, raise on first error

    Returns
    -------
    tuple of (X, valid_indices) where:
        X             : np.ndarray of shape (n_valid, n_bits)
        valid_indices : list of indices into smiles_list that succeeded
    """
    X = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        try:
            fp = psmiles_to_fingerprint(smi, radius=radius, n_bits=n_bits)
            X.append(fp)
            valid_indices.append(i)
        except ValueError:
            if not skip_errors:
                raise
            continue

    if len(X) == 0:
        return np.empty((0, n_bits), dtype=np.float64), valid_indices

    return np.array(X), valid_indices


def fingerprint_stats(X: np.ndarray) -> dict:
    """
    Return basic statistics about a fingerprint matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_bits)

    Returns
    -------
    dict with n_samples, n_bits, mean_bits_set, sparsity
    """
    n_samples, n_bits = X.shape
    bits_set = X.sum(axis=1)
    return {
        "n_samples": n_samples,
        "n_bits": n_bits,
        "mean_bits_set": round(float(bits_set.mean()), 2),
        "min_bits_set": int(bits_set.min()),
        "max_bits_set": int(bits_set.max()),
        "sparsity": round(float(1 - X.mean()), 4),
    }