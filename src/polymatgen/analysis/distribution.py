import math

from polymatgen.core.polymer import Polymer


def molecular_weight_moments(polymer: Polymer) -> dict:
    """
    Compute statistical moments of the molecular weight distribution.

    Returns
    -------
    dict with Mn, Mw, Mz, dispersity, and standard deviation
    """
    weights = [c.molecular_weight for c in polymer.chains]
    n = len(weights)

    Mn = sum(weights) / n
    Mw = sum(w**2 for w in weights) / sum(weights)
    Mz = sum(w**3 for w in weights) / sum(w**2 for w in weights)
    dispersity = Mw / Mn
    std_dev = math.sqrt(sum((w - Mn)**2 for w in weights) / n)

    return {
        "Mn": round(Mn, 4),
        "Mw": round(Mw, 4),
        "Mz": round(Mz, 4),
        "dispersity": round(dispersity, 6),
        "std_dev": round(std_dev, 4),
        "n_chains": n
    }


def histogram(polymer: Polymer, n_bins: int = 10) -> list:
    """
    Bin the chain molecular weight distribution into a histogram.

    Returns
    -------
    list of dicts with keys: bin_min, bin_max, bin_mid, count, fraction
    """
    weights = [c.molecular_weight for c in polymer.chains]
    min_w = min(weights)
    max_w = max(weights)

    if min_w == max_w:
        return [{
            "bin_min": min_w,
            "bin_max": max_w,
            "bin_mid": min_w,
            "count": len(weights),
            "fraction": 1.0
        }]

    bin_width = (max_w - min_w) / n_bins
    bins = [0] * n_bins

    for w in weights:
        idx = min(int((w - min_w) / bin_width), n_bins - 1)
        bins[idx] += 1

    result = []
    for i, count in enumerate(bins):
        bin_min = min_w + i * bin_width
        bin_max = bin_min + bin_width
        result.append({
            "bin_min": round(bin_min, 2),
            "bin_max": round(bin_max, 2),
            "bin_mid": round((bin_min + bin_max) / 2, 2),
            "count": count,
            "fraction": round(count / len(weights), 4)
        })

    return result


def cumulative_distribution(polymer: Polymer) -> list:
    """
    Compute the cumulative molecular weight distribution.

    Returns
    -------
    list of (molecular_weight, cumulative_fraction) tuples, sorted by Mw
    """
    weights = sorted(c.molecular_weight for c in polymer.chains)
    n = len(weights)
    return [(round(w, 4), round((i + 1) / n, 6))
            for i, w in enumerate(weights)]
