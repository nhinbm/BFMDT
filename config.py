"""Default hyperparameters and CLI argument parsing for BFMDT."""

import argparse
from dataclasses import dataclass, field


@dataclass
class BFMDTConfig:
    """Centralized configuration for the BFMDT pipeline.

    Args:
        sigma (float | str): Fitting degree threshold, or 'auto'. Defaults to 'auto'.
        delta (float): RMI threshold for tree splitting. Defaults to 0.01.
        max_reducts (int): Maximum feature subsets per sigma. Defaults to 50.
        min_sigma_candidates (int): Minimum sigma candidates. Defaults to 5.
        max_sigma_candidates (int): Maximum sigma candidates. Defaults to 30.
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        test_size (float): Test split ratio for large datasets. Defaults to 0.2.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
        dataset_ids (list[int]): Dataset IDs to run benchmarks on. Defaults to 1-18.
    """

    # Classifier hyperparameters
    sigma: float | str = "auto"
    delta: float = 0.01
    max_reducts: int = 50

    # Sigma selection
    min_sigma_candidates: int = 5
    max_sigma_candidates: int = 30

    # Experiment settings
    n_folds: int = 5
    test_size: float = 0.2
    random_seed: int = 42
    dataset_ids: list[int] = field(default_factory=lambda: list(range(1, 19)))


def _parse_sigma(value):
    """Parse sigma as 'auto' or a float."""
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"sigma must be 'auto' or a float, got '{value}'"
        )


def parse_args(argv=None) -> BFMDTConfig:
    """Parse CLI arguments into a BFMDTConfig.

    Args:
        argv (list[str] | None): Arguments to parse. Defaults to sys.argv[1:].

    Returns:
        BFMDTConfig: Parsed configuration.
    """
    parser = argparse.ArgumentParser(
        description="BFMDT — Bi-directional Fusing Monotonic Decision Trees"
    )

    parser.add_argument(
        "--sigma", type=_parse_sigma, default="auto",
        help="Fitting degree threshold or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--delta", type=float, default=0.01,
        help="RMI threshold for tree splitting (default: 0.01)"
    )
    parser.add_argument(
        "--max-reducts", type=int, default=50,
        help="Max feature subsets per sigma (default: 50)"
    )
    parser.add_argument(
        "--min-sigma-candidates", type=int, default=5,
        help="Min sigma candidates (default: 5)"
    )
    parser.add_argument(
        "--max-sigma-candidates", type=int, default=30,
        help="Max sigma candidates (default: 30)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test split ratio for large datasets (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--datasets", type=int, nargs="+", default=list(range(1, 19)),
        metavar="ID",
        help="Dataset IDs to benchmark (default: 1-18)"
    )

    args = parser.parse_args(argv)

    return BFMDTConfig(
        sigma=args.sigma,
        delta=args.delta,
        max_reducts=args.max_reducts,
        min_sigma_candidates=args.min_sigma_candidates,
        max_sigma_candidates=args.max_sigma_candidates,
        n_folds=args.n_folds,
        test_size=args.test_size,
        random_seed=args.seed,
        dataset_ids=args.datasets,
    )


if __name__ == "__main__":
    config = parse_args()
    print(config)
