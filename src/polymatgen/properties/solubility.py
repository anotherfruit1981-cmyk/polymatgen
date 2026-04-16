import math


class HildebrandSolubility:
    """
    Hildebrand solubility parameter estimation.
    delta = sqrt(sum(Fi) / V)
    where Fi are molar attraction constants and V is molar volume.

    Parameters
    ----------
    molar_attraction_constants : list of floats (Fi values in J^0.5 cm^1.5 mol^-1)
    molar_volume : float (cm^3 / mol)

    Example
    -------
    h = HildebrandSolubility([272, 269, 57], molar_volume=100.0)
    print(h.delta)
    """
    def __init__(self, molar_attraction_constants: list, molar_volume: float):
        self.Fi = molar_attraction_constants
        self.V = molar_volume

    @property
    def delta(self) -> float:
        """Hildebrand solubility parameter in (J/cm^3)^0.5."""
        return math.sqrt(sum(self.Fi) / self.V)

    def miscibility_check(self, solvent_delta: float, tolerance: float = 2.0) -> bool:
        """
        Simple miscibility check.
        If |delta_polymer - delta_solvent| < tolerance, likely miscible.
        Default tolerance of 2.0 (MPa)^0.5 is a common rule of thumb.
        """
        return abs(self.delta - solvent_delta) < tolerance

    def __repr__(self):
        return f"HildebrandSolubility(delta={self.delta:.2f} (J/cm^3)^0.5)"


class FloryHuggins:
    """
    Flory-Huggins interaction parameter chi.
    chi = V_solvent / (R * T) * (delta_polymer - delta_solvent)^2

    Parameters
    ----------
    delta_polymer : float  — solubility parameter of polymer (MPa^0.5)
    delta_solvent : float  — solubility parameter of solvent (MPa^0.5)
    molar_volume_solvent : float — molar volume of solvent (cm^3/mol)
    temperature : float — temperature in Kelvin (default 298 K)
    """
    R = 8.314  # J / (mol K)

    def __init__(self, delta_polymer: float, delta_solvent: float,
                 molar_volume_solvent: float, temperature: float = 298.0):
        self.delta_polymer = delta_polymer
        self.delta_solvent = delta_solvent
        self.V = molar_volume_solvent
        self.T = temperature

    @property
    def chi(self) -> float:
        """Flory-Huggins interaction parameter (dimensionless)."""
        delta_diff = (self.delta_polymer - self.delta_solvent) * 1000  # MPa^0.5 to Pa^0.5
        return (self.V * 1e-6 / (self.R * self.T)) * delta_diff ** 2

    @property
    def is_miscible(self) -> bool:
        """chi < 0.5 generally indicates miscibility."""
        return self.chi < 0.5

    def __repr__(self):
        return f"FloryHuggins(chi={self.chi:.4f}, miscible={self.is_miscible})"
