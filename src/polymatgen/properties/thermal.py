class FoxEquation:
    """
    Predicts glass transition temperature (Tg) of a copolymer
    using the Fox equation: 1/Tg = sum(wi / Tgi)

    Parameters
    ----------
    components : list of (weight_fraction, Tg_in_Kelvin) tuples

    Example
    -------
    fox = FoxEquation([(0.5, 373.0), (0.5, 273.0)])
    print(fox.Tg)  # predicted Tg in Kelvin
    """
    def __init__(self, components: list):
        self.components = components
        total = sum(w for w, _ in components)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weight fractions must sum to 1.0, got {total:.4f}")

    @property
    def Tg(self) -> float:
        """Predicted Tg in Kelvin."""
        return 1.0 / sum(w / tg for w, tg in self.components)

    @property
    def Tg_celsius(self) -> float:
        """Predicted Tg in Celsius."""
        return self.Tg - 273.15

    def __repr__(self):
        return f"FoxEquation(Tg={self.Tg:.2f} K, {self.Tg_celsius:.2f} C)"
