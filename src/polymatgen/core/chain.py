class Chain:
    def __init__(self, monomers: list, degree_of_polymerization: int, tacticity: str = "atactic"):
        self.monomers = monomers
        self.degree_of_polymerization = degree_of_polymerization
        self.tacticity = tacticity

    @property
    def molecular_weight(self) -> float:
        repeat_mw = sum(m.molecular_weight for m in self.monomers)
        return repeat_mw * self.degree_of_polymerization

    def __repr__(self):
        return f"Chain(monomers={self.monomers}, DP={self.degree_of_polymerization}, tacticity={self.tacticity!r})"
