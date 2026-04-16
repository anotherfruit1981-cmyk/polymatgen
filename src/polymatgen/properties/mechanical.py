import math


class ContourLength:
    """
    Estimates the fully extended contour length of a polymer chain.

    Parameters
    ----------
    degree_of_polymerization : int
    bond_length : float — in Angstroms (default 1.54 A for C-C bond)
    bonds_per_monomer : int — number of backbone bonds per repeat unit (default 2)

    Example
    -------
    cl = ContourLength(degree_of_polymerization=100)
    print(cl.length_angstrom)
    """
    def __init__(self, degree_of_polymerization: int,
                 bond_length: float = 1.54,
                 bonds_per_monomer: int = 2):
        self.DP = degree_of_polymerization
        self.bond_length = bond_length
        self.bonds_per_monomer = bonds_per_monomer

    @property
    def n_bonds(self) -> int:
        return self.DP * self.bonds_per_monomer

    @property
    def length_angstrom(self) -> float:
        """Fully extended contour length in Angstroms."""
        return self.n_bonds * self.bond_length

    @property
    def length_nm(self) -> float:
        """Fully extended contour length in nanometres."""
        return self.length_angstrom / 10.0

    @property
    def end_to_end_rms(self) -> float:
        """
        Root mean square end-to-end distance for a freely jointed chain
        r = l * sqrt(n)
        """
        return self.bond_length * math.sqrt(self.n_bonds)

    def __repr__(self):
        return (f"ContourLength(DP={self.DP}, "
                f"L={self.length_nm:.2f} nm, "
                f"r_rms={self.end_to_end_rms:.2f} A)")
