from polymatgen.core.chain import Chain
from polymatgen.core.polymer import Polymer


def monomer_composition(chain: Chain) -> dict:
    """
    Calculate the mole fraction of each monomer type in a chain.

    Returns
    -------
    dict mapping monomer name to mole fraction
    """
    counts = {}
    for m in chain.monomers:
        counts[m.name] = counts.get(m.name, 0) + 1
    total = sum(counts.values())
    return {name: round(count / total, 6) for name, count in counts.items()}


def is_homopolymer(chain: Chain) -> bool:
    """Return True if the chain contains only one monomer type."""
    return len(set(m.name for m in chain.monomers)) == 1


def is_copolymer(chain: Chain) -> bool:
    """Return True if the chain contains more than one monomer type."""
    return not is_homopolymer(chain)


def sequence_blocks(chain: Chain) -> list:
    """
    Identify contiguous blocks of the same monomer type.

    Returns
    -------
    list of (monomer_name, block_length) tuples

    Example
    -------
    [A, A, B, B, B, A] -> [('A', 2), ('B', 3), ('A', 1)]
    """
    if not chain.monomers:
        return []

    blocks = []
    current = chain.monomers[0].name
    count = 1

    for m in chain.monomers[1:]:
        if m.name == current:
            count += 1
        else:
            blocks.append((current, count))
            current = m.name
            count = 1
    blocks.append((current, count))
    return blocks


def blockiness(chain: Chain) -> float:
    """
    Blockiness score: average block length across all blocks.
    Higher = more blocky (closer to block copolymer).
    Lower = more alternating.

    Returns
    -------
    float — average block length
    """
    blocks = sequence_blocks(chain)
    if not blocks:
        return 0.0
    return sum(length for _, length in blocks) / len(blocks)


def polymer_composition(polymer: Polymer) -> dict:
    """
    Average monomer composition across all chains in the polymer.

    Returns
    -------
    dict mapping monomer name to average mole fraction
    """
    totals = {}
    for chain in polymer.chains:
        comp = monomer_composition(chain)
        for name, fraction in comp.items():
            totals[name] = totals.get(name, 0.0) + fraction

    n = len(polymer.chains)
    return {name: round(total / n, 6) for name, total in totals.items()}
