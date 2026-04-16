import csv

from polymatgen.core.polymer import Polymer


def export_chain_distribution(polymer: Polymer, filepath: str) -> None:
    """
    Export the chain distribution of a Polymer to a CSV file.
    Each row is one chain with its DP, molecular weight, and tacticity.
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chain_index", "degree_of_polymerization",
                         "molecular_weight", "tacticity", "monomers"])
        for i, chain in enumerate(polymer.chains):
            monomer_names = ",".join(m.name for m in chain.monomers)
            writer.writerow([
                i + 1,
                chain.degree_of_polymerization,
                round(chain.molecular_weight, 4),
                chain.tacticity,
                monomer_names
            ])
    print(f"Chain distribution exported to {filepath}")


def import_chain_distribution(filepath: str) -> list:
    """
    Read a chain distribution CSV back into a list of dicts.
    Each dict has keys: chain_index, degree_of_polymerization,
    molecular_weight, tacticity, monomers.
    """
    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "chain_index": int(row["chain_index"]),
                "degree_of_polymerization": int(row["degree_of_polymerization"]),
                "molecular_weight": float(row["molecular_weight"]),
                "tacticity": row["tacticity"],
                "monomers": row["monomers"]
            })
    return rows
