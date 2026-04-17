"""
Inverse design module for polymatgen.

Instead of "structure -> property", this module does the reverse:
"target property -> propose structures".

Uses a genetic algorithm (GA) over the PI1M polymer SMILES space.
The GA iteratively selects, mutates, and crossbreeds candidates
scored by any polymatgen predictor.

Main class:
    InverseDesigner — multi-property constrained polymer search

Example
-------
from polymatgen.ml.predictors import TgPredictor, BandgapPredictor
from polymatgen.ml.inverse_design import InverseDesigner

designer = InverseDesigner()
designer.add_constraint(TgPredictor(), min_val=500.0)
designer.add_constraint(BandgapPredictor(), max_val=2.0)
results = designer.run(n_generations=20, population_size=50)
for smi, scores in results[:5]:
    print(smi, scores)
"""

import os
import random
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PI1M_PATH = os.path.join(DATA_DIR, "PI1M_v2.csv")


def _load_pi1m_smiles(n: int = 5000, seed: int = 42) -> list:
    """
    Load a random sample of SMILES from PI1M as the initial gene pool.

    Parameters
    ----------
    n    : int — number of SMILES to sample
    seed : int — random seed

    Returns
    -------
    list of SMILES strings
    """
    import pandas as pd
    if not os.path.exists(PI1M_PATH):
        raise FileNotFoundError(
            f"PI1M dataset not found at {PI1M_PATH}. "
            "The inverse designer uses PI1M as its gene pool."
        )
    df = pd.read_csv(PI1M_PATH, usecols=["SMILES"])
    df = df[df["SMILES"].notna()]
    sample = df["SMILES"].sample(
        n=min(n, len(df)), random_state=seed
    ).tolist()
    return sample


def _mutate(smiles: str, all_smiles: list,
            mutation_rate: float = 0.3) -> str:
    """
    Mutate a SMILES string.

    Three mutation operators (chosen randomly):
    1. Swap — replace with a random SMILES from the pool
    2. Fragment swap — replace a random atom/fragment token
    3. Return original (no mutation)

    Parameters
    ----------
    smiles       : str  — candidate to mutate
    all_smiles   : list — pool to draw replacement fragments from
    mutation_rate: float — probability of applying a mutation

    Returns
    -------
    str — mutated SMILES (may be the original if no mutation applied)
    """
    if random.random() > mutation_rate:
        return smiles

    op = random.choice(["swap", "fragment"])

    if op == "swap":
        # Replace entirely with a pool member
        return random.choice(all_smiles)

    if op == "fragment":
        # Tokenise SMILES and swap one token with one from a random pool member
        tokens = list(smiles)
        if len(tokens) < 2:
            return random.choice(all_smiles)
        donor = random.choice(all_smiles)
        donor_tokens = list(donor)
        if not donor_tokens:
            return smiles
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = random.choice(donor_tokens)
        return "".join(tokens)

    return smiles


def _crossover(smiles_a: str, smiles_b: str) -> str:
    """
    Single-point crossover between two SMILES strings.

    Splits both at a random point and joins the first half of A
    with the second half of B.

    Parameters
    ----------
    smiles_a : str
    smiles_b : str

    Returns
    -------
    str — child SMILES (may not be chemically valid;
          invalid ones are filtered during fitness evaluation)
    """
    if not smiles_a or not smiles_b:
        return smiles_a or smiles_b
    point_a = random.randint(1, max(1, len(smiles_a) - 1))
    point_b = random.randint(1, max(1, len(smiles_b) - 1))
    return smiles_a[:point_a] + smiles_b[point_b:]


def _is_valid(smiles: str) -> bool:
    """Check if a SMILES string is parseable by RDKit."""
    try:
        from rdkit import Chem
        from polymatgen.ml.gcn_predictor import _clean_psmiles
        clean = _clean_psmiles(smiles)
        mol = Chem.MolFromSmiles(clean)
        return mol is not None and mol.GetNumAtoms() > 0
    except Exception:
        return False


class Constraint:
    """
    A single property constraint for the inverse designer.

    Parameters
    ----------
    predictor : a polymatgen predictor with a .predict(smiles) method
    min_val   : float or None — minimum acceptable value
    max_val   : float or None — maximum acceptable value
    weight    : float — importance weight in the fitness function
    name      : str — human-readable label for this constraint
    """

    def __init__(self, predictor, min_val=None, max_val=None,
                 weight: float = 1.0, name: str = None):
        self.predictor = predictor
        self.min_val = min_val
        self.max_val = max_val
        self.weight = weight
        self.name = name or predictor.__class__.__name__

    def score(self, smiles: str) -> float:
        """
        Compute a fitness score for this constraint.

        Returns 0.0 if the prediction is within the target range,
        or a negative penalty proportional to how far outside
        the range the prediction falls.

        Returns
        -------
        float — 0.0 (perfect) to large negative (far from target)
        """
        try:
            val = self.predictor.predict(smiles)
        except Exception:
            return -1e6  # unpredictable = very bad

        penalty = 0.0
        if self.min_val is not None and val < self.min_val:
            penalty += (self.min_val - val)
        if self.max_val is not None and val > self.max_val:
            penalty += (val - self.max_val)
        return -penalty * self.weight

    def is_satisfied(self, smiles: str) -> bool:
        """Return True if the constraint is met for this SMILES."""
        try:
            val = self.predictor.predict(smiles)
            if self.min_val is not None and val < self.min_val:
                return False
            if self.max_val is not None and val > self.max_val:
                return False
            return True
        except Exception:
            return False

    def predict_value(self, smiles: str) -> float:
        """Return the raw predicted value for this SMILES."""
        try:
            return self.predictor.predict(smiles)
        except Exception:
            return float("nan")

    def __repr__(self):
        parts = [f"name={self.name}"]
        if self.min_val is not None:
            parts.append(f"min={self.min_val}")
        if self.max_val is not None:
            parts.append(f"max={self.max_val}")
        parts.append(f"weight={self.weight}")
        return f"Constraint({', '.join(parts)})"


class InverseDesigner:
    """
    Genetic algorithm-based inverse polymer designer.

    Searches the PI1M SMILES space for polymers satisfying
    one or more property constraints simultaneously.

    Parameters
    ----------
    pool_size    : int   — number of PI1M SMILES to use as gene pool
    random_state : int   — random seed for reproducibility
    mutation_rate: float — probability of mutating a candidate

    Example
    -------
    from polymatgen.ml.predictors import TgPredictor, BandgapPredictor
    from polymatgen.ml.inverse_design import InverseDesigner

    designer = InverseDesigner()
    designer.add_constraint(TgPredictor(), min_val=500.0,
                            name="Tg > 500 K")
    designer.add_constraint(BandgapPredictor(), max_val=2.0,
                            name="Bandgap < 2 eV")
    results = designer.run(n_generations=20, population_size=50)

    # Top candidates
    for smi, scores, fitness in results[:5]:
        print(smi, scores, f"fitness={fitness:.2f}")
    """

    def __init__(self, pool_size: int = 5000,
                 random_state: int = 42,
                 mutation_rate: float = 0.3):
        self.pool_size = pool_size
        self.random_state = random_state
        self.mutation_rate = mutation_rate
        self.constraints = []
        self._gene_pool = []
        random.seed(random_state)
        np.random.seed(random_state)

    def add_constraint(self, predictor, min_val=None, max_val=None,
                       weight: float = 1.0, name: str = None):
        """
        Add a property constraint to the search.

        Parameters
        ----------
        predictor : polymatgen predictor — any object with .predict(smiles)
        min_val   : float or None — lower bound on predicted property
        max_val   : float or None — upper bound on predicted property
        weight    : float — relative importance (default 1.0)
        name      : str — label for reporting

        Returns
        -------
        self (for chaining)
        """
        if min_val is None and max_val is None:
            raise ValueError(
                "At least one of min_val or max_val must be specified."
            )
        c = Constraint(predictor, min_val=min_val, max_val=max_val,
                       weight=weight,
                       name=name or predictor.__class__.__name__)
        self.constraints.append(c)
        print(f"Added constraint: {c}")
        return self

    def _fitness(self, smiles: str) -> float:
        """
        Compute total fitness score across all constraints.

        Higher is better. 0.0 = all constraints satisfied exactly.
        Negative = some constraints violated.
        """
        if not self.constraints:
            raise RuntimeError("No constraints added. Use add_constraint().")
        return sum(c.score(smiles) for c in self.constraints)

    def _initialise_population(self, population_size: int) -> list:
        """Sample valid SMILES from the gene pool."""
        if not self._gene_pool:
            print("Loading PI1M gene pool...")
            self._gene_pool = _load_pi1m_smiles(
                n=self.pool_size, seed=self.random_state
            )
            print(f"Gene pool: {len(self._gene_pool)} SMILES loaded.")

        candidates = []
        pool = list(self._gene_pool)
        random.shuffle(pool)
        for smi in pool:
            if _is_valid(smi):
                candidates.append(smi)
            if len(candidates) >= population_size:
                break

        if len(candidates) < population_size:
            print(f"Warning: only {len(candidates)} valid SMILES found "
                  f"(requested {population_size}).")
        return candidates

    def run(self, n_generations: int = 20,
            population_size: int = 50,
            elite_fraction: float = 0.2,
            crossover_fraction: float = 0.3,
            verbose: bool = True) -> list:
        """
        Run the genetic algorithm search.

        Parameters
        ----------
        n_generations    : int   — number of GA generations
        population_size  : int   — candidates per generation
        elite_fraction   : float — top fraction kept unchanged
        crossover_fraction: float — fraction created by crossover
        verbose          : bool  — print progress

        Returns
        -------
        list of (smiles, property_scores, fitness) tuples,
        sorted by fitness (best first). Only includes candidates
        where all constraints are satisfied.
        """
        if not self.constraints:
            raise RuntimeError("No constraints added. Use add_constraint().")

        # Initialise predictors
        if verbose:
            print("Training predictors...")
        for c in self.constraints:
            if not getattr(c.predictor, "is_trained", True):
                c.predictor.train()

        # Initialise population
        if verbose:
            print(f"Initialising population (size={population_size})...")
        population = self._initialise_population(population_size)

        n_elite = max(1, int(population_size * elite_fraction))
        n_crossover = int(population_size * crossover_fraction)
        n_mutate = population_size - n_elite - n_crossover

        best_fitness_history = []

        for gen in range(n_generations):
            # Score population
            scored = []
            for smi in population:
                fit = self._fitness(smi)
                scored.append((smi, fit))
            scored.sort(key=lambda x: x[1], reverse=True)

            best_fit = scored[0][1]
            best_fitness_history.append(best_fit)

            if verbose:
                n_satisfied = sum(
                    1 for smi, _ in scored
                    if all(c.is_satisfied(smi) for c in self.constraints)
                )
                print(f"  Gen {gen + 1:3d}/{n_generations} | "
                      f"best fitness={best_fit:+.2f} | "
                      f"constraints satisfied={n_satisfied}/{len(scored)}")

            # Elite selection
            elites = [smi for smi, _ in scored[:n_elite]]

            # Crossover
            children = []
            for _ in range(n_crossover):
                a, b = random.sample(elites, min(2, len(elites)))
                child = _crossover(a, b)
                if _is_valid(child):
                    children.append(child)
                else:
                    children.append(random.choice(elites))

            # Mutation
            mutants = []
            for _ in range(n_mutate):
                parent = random.choice(elites)
                mutant = _mutate(parent, self._gene_pool, self.mutation_rate)
                if _is_valid(mutant):
                    mutants.append(mutant)
                else:
                    mutants.append(parent)

            population = elites + children + mutants
            # Deduplicate
            seen = set()
            unique = []
            for smi in population:
                if smi not in seen:
                    seen.add(smi)
                    unique.append(smi)
            # Pad if needed
            while len(unique) < population_size:
                unique.append(random.choice(self._gene_pool))
            population = unique[:population_size]

        # Final scoring — return all candidates that satisfy constraints
        if verbose:
            print("\nFinal evaluation...")

        results = []
        for smi in population:
            if all(c.is_satisfied(smi) for c in self.constraints):
                scores = {c.name: c.predict_value(smi)
                          for c in self.constraints}
                fit = self._fitness(smi)
                results.append((smi, scores, fit))

        results.sort(key=lambda x: x[2], reverse=True)

        if verbose:
            print(f"Found {len(results)} candidates satisfying "
                  f"all {len(self.constraints)} constraint(s).")

        return results

    def __repr__(self):
        return (f"InverseDesigner("
                f"constraints={len(self.constraints)}, "
                f"pool_size={self.pool_size})")