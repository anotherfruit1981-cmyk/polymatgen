import json

from polymatgen.core.chain import Chain
from polymatgen.core.monomer import Monomer
from polymatgen.core.polymer import Polymer


def polymer_to_dict(polymer: Polymer) -> dict:
    """Serialize a Polymer object to a dictionary."""
    return {
        "name": polymer.name,
        "chains": [
            {
                "degree_of_polymerization": chain.degree_of_polymerization,
                "tacticity": chain.tacticity,
                "monomers": [
                    {"name": m.name, "smiles": m.smiles}
                    for m in chain.monomers
                ]
            }
            for chain in polymer.chains
        ]
    }


def polymer_from_dict(data: dict) -> Polymer:
    """Deserialize a Polymer object from a dictionary."""
    chains = []
    for chain_data in data["chains"]:
        monomers = [
            Monomer(name=m["name"], smiles=m["smiles"])
            for m in chain_data["monomers"]
        ]
        chains.append(Chain(
            monomers=monomers,
            degree_of_polymerization=chain_data["degree_of_polymerization"],
            tacticity=chain_data["tacticity"]
        ))
    return Polymer(chains=chains, name=data["name"])


def save_polymer(polymer: Polymer, filepath: str) -> None:
    """Save a Polymer to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(polymer_to_dict(polymer), f, indent=2)
    print(f"Saved to {filepath}")


def load_polymer(filepath: str) -> Polymer:
    """Load a Polymer from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return polymer_from_dict(data)
