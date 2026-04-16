"""
Built-in reference database of common polymer properties.
All values are approximate literature values at 298 K unless noted.

Units:
    Tg          : Kelvin
    density     : g/cm^3
    delta       : (MPa)^0.5  (Hildebrand solubility parameter)
    Cp          : J/(g·K)    (specific heat capacity)
"""

POLYMER_DATABASE = {
    "polystyrene": {
        "full_name": "Polystyrene",
        "abbreviation": "PS",
        "repeat_smiles": "C(Cc1ccccc1)",
        "Tg": 373.0,
        "density": 1.05,
        "delta": 18.5,
        "Cp": 1.3,
    },
    "polyethylene": {
        "full_name": "Polyethylene",
        "abbreviation": "PE",
        "repeat_smiles": "CC",
        "Tg": 148.0,
        "density": 0.95,
        "delta": 16.2,
        "Cp": 2.3,
    },
    "polypropylene": {
        "full_name": "Polypropylene",
        "abbreviation": "PP",
        "repeat_smiles": "CC(C)",
        "Tg": 253.0,
        "density": 0.91,
        "delta": 16.8,
        "Cp": 1.9,
    },
    "poly(methyl methacrylate)": {
        "full_name": "Poly(methyl methacrylate)",
        "abbreviation": "PMMA",
        "repeat_smiles": "COC(=O)C(C)(C)",
        "Tg": 378.0,
        "density": 1.18,
        "delta": 19.0,
        "Cp": 1.45,
    },
    "polyvinyl chloride": {
        "full_name": "Polyvinyl chloride",
        "abbreviation": "PVC",
        "repeat_smiles": "CCCl",
        "Tg": 354.0,
        "density": 1.40,
        "delta": 19.4,
        "Cp": 1.0,
    },
    "polytetrafluoroethylene": {
        "full_name": "Polytetrafluoroethylene",
        "abbreviation": "PTFE",
        "repeat_smiles": "C(F)(F)",
        "Tg": 200.0,
        "density": 2.20,
        "delta": 12.7,
        "Cp": 1.0,
    },
    "nylon 6": {
        "full_name": "Nylon 6 (Polycaprolactam)",
        "abbreviation": "PA6",
        "repeat_smiles": "NCCCCCC(=O)",
        "Tg": 323.0,
        "density": 1.14,
        "delta": 22.9,
        "Cp": 1.7,
    },
    "polyethylene terephthalate": {
        "full_name": "Polyethylene terephthalate",
        "abbreviation": "PET",
        "repeat_smiles": "OCCOC(=O)c1ccc(cc1)C(=O)",
        "Tg": 342.0,
        "density": 1.38,
        "delta": 21.9,
        "Cp": 1.0,
    },
    "polycarbonate": {
        "full_name": "Polycarbonate",
        "abbreviation": "PC",
        "repeat_smiles": "CC(C)(c1ccccc1)c1ccccc1OC(=O)O",
        "Tg": 423.0,
        "density": 1.20,
        "delta": 19.4,
        "Cp": 1.17,
    },
    "polyvinyl acetate": {
        "full_name": "Polyvinyl acetate",
        "abbreviation": "PVAc",
        "repeat_smiles": "CC(OC(C)=O)",
        "Tg": 304.0,
        "density": 1.19,
        "delta": 19.1,
        "Cp": 1.5,
    },
}


def get_polymer(name: str) -> dict:
    """
    Retrieve a polymer entry by name (case-insensitive).

    Parameters
    ----------
    name : str — common name or abbreviation

    Returns
    -------
    dict of polymer properties

    Raises
    ------
    KeyError if polymer not found
    """
    key = name.lower().strip()
    if key in POLYMER_DATABASE:
        return POLYMER_DATABASE[key]

    for k, v in POLYMER_DATABASE.items():
        if v.get("abbreviation", "").lower() == key:
            return v

    available = list_polymers()
    raise KeyError(
        f"Polymer '{name}' not found in reference database. "
        f"Available: {available}"
    )


def list_polymers() -> list:
    """Return a list of all polymer names in the reference database."""
    return sorted(POLYMER_DATABASE.keys())


def search_by_property(prop: str, min_val: float = None,
                        max_val: float = None) -> list:
    """
    Search the database for polymers within a property range.

    Parameters
    ----------
    prop    : str   — property key e.g. 'Tg', 'density', 'delta'
    min_val : float — minimum value (inclusive)
    max_val : float — maximum value (inclusive)

    Returns
    -------
    list of (name, value) tuples sorted by value
    """
    results = []
    for name, data in POLYMER_DATABASE.items():
        if prop not in data:
            continue
        val = data[prop]
        if min_val is not None and val < min_val:
            continue
        if max_val is not None and val > max_val:
            continue
        results.append((name, val))
    return sorted(results, key=lambda x: x[1])
