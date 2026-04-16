import math

from polymatgen.core.chain import Chain


def radius_of_gyration(chain: Chain, bond_length: float = 1.54) -> float:
    """
    Estimate radius of gyration for a freely jointed chain.
    Rg = l * sqrt(n / 6)
    where l is bond length and n is number of backbone bonds.

    Parameters
    ----------
    chain : Chain
    bond_length : float — in Angstroms (default 1.54 for C-C)

    Returns
    -------
    Rg in Angstroms
    """
    n_bonds = chain.degree_of_polymerization * 2
    return bond_length * math.sqrt(n_bonds / 6.0)


def end_to_end_distance(chain: Chain, bond_length: float = 1.54) -> float:
    """
    RMS end-to-end distance for a freely jointed chain.
    r = l * sqrt(n)

    Returns
    -------
    r in Angstroms
    """
    n_bonds = chain.degree_of_polymerization * 2
    return bond_length * math.sqrt(n_bonds)


def characteristic_ratio(chain: Chain, bond_length: float = 1.54,
                          bond_angle: float = 109.5) -> float:
    """
    Flory characteristic ratio C_inf estimate using bond angle correction.
    C_inf = (1 + cos(theta)) / (1 - cos(theta))
    where theta is the supplement of the bond angle.

    Returns
    -------
    C_inf (dimensionless)
    """
    theta = math.radians(180.0 - bond_angle)
    cos_t = math.cos(theta)
    return (1.0 + cos_t) / (1.0 - cos_t)


def chain_summary(chain: Chain) -> dict:
    """
    Return a dictionary of key single-chain properties.
    """
    return {
        "DP": chain.degree_of_polymerization,
        "molecular_weight": round(chain.molecular_weight, 4),
        "tacticity": chain.tacticity,
        "Rg_angstrom": round(radius_of_gyration(chain), 4),
        "r_end_to_end_angstrom": round(end_to_end_distance(chain), 4),
        "C_inf": round(characteristic_ratio(chain), 4),
        "n_monomers": len(chain.monomers)
    }
