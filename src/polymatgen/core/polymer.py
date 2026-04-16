class Polymer:
    def __init__(self, chains: list, name: str = ""):
        self.chains = chains
        self.name = name

    @property
    def Mn(self) -> float:
        """Number-average molecular weight."""
        weights = [c.molecular_weight for c in self.chains]
        return sum(weights) / len(weights)

    @property
    def Mw(self) -> float:
        """Weight-average molecular weight."""
        weights = [c.molecular_weight for c in self.chains]
        total = sum(weights)
        return sum(w**2 for w in weights) / total

    @property
    def dispersity(self) -> float:
        """Dispersity = Mw / Mn"""
        return self.Mw / self.Mn

    def __repr__(self):
        return f"Polymer(name={self.name!r}, chains={len(self.chains)}, Mn={self.Mn:.1f}, Mw={self.Mw:.1f})"
