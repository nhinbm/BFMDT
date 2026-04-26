"""Centralized configuration: paths, URLs, dataset registry, and CLI parsing."""

import argparse
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "datasets/data"
YALE_DIR = os.path.join(DATA_DIR, "yale")
DRIVFACE_DIR = os.path.join(DATA_DIR, "DrivFace")
PEMS_SF_DIR = os.path.join(DATA_DIR, "pems-sf")
YALE_MAT_PATH = os.path.join(YALE_DIR, "Yale.mat")
OPENML_CACHE_DIR = os.path.join(DATA_DIR, "openml_cache")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ---------------------------------------------------------------------------
# Download sources
# ---------------------------------------------------------------------------
YALE_MAT_URL = "https://github.com/jundongl/scikit-feature/raw/master/skfeature/data/Yale.mat"
GDRIVE_FOLDER_ID = "1fnjGJ5VBu_m7XSfieKE3LU9prTb6hyjk"


# ---------------------------------------------------------------------------
# Dataset-shape sanity checks
# ---------------------------------------------------------------------------
PEMS_SF_EXPECTED_SHAPE = (440, 138672)
DRIVFACE_IMAGE_SIZE = (80, 80)


# ---------------------------------------------------------------------------
# Dataset registry — mirrors Table 4 of the BFMDT paper (IDs 1-18)
# ---------------------------------------------------------------------------
DATASET_NAMES_BY_ID = {
    1: "breast-wisconsin",
    2: "wine",
    3: "breast-cancer",
    4: "heart-disease",
    5: "hepatitis",
    6: "german-credit",
    7: "vehicle",
    8: "wdbc",
    9: "diabetes",
    10: "wine-quality",
    11: "divorce",
    12: "sonar",
    13: "turkiye-student",
    14: "Yale",
    15: "arcene",
    16: "SMK_CAN_187",
    17: "DrivFace",
    18: "PEMS-SF",
}
ALL_DATASET_NAMES = list(DATASET_NAMES_BY_ID.values())
NAME_TO_ID = {v: k for k, v in DATASET_NAMES_BY_ID.items()}


# ---------------------------------------------------------------------------
# Per-dataset metadata.
#   openml_name:          OpenML name; None for custom-loaded (DatasetRegistry).
#   label_mapping:        explicit nominal-label encoding; None -> alphabetical
#                         LabelEncoder.
#   allow_missing_values: contract for preprocessing. True  -> NaN is allowed
#                         and imputed by column mean. False -> NaN raises
#                         ValueError (assert clean).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetMeta:
    openml_name: str | None = None
    label_mapping: dict | None = None
    allow_missing_values: bool = False


DATASETS = {
    'breast-wisconsin': DatasetMeta(
        openml_name='breast-w',
        label_mapping={'benign': 0, 'malignant': 1},
        allow_missing_values=True,
    ),
    'arcene':          DatasetMeta(
        openml_name='arcene',
        label_mapping={'2': 0, '1': 1},
    ),
    'breast-cancer':   DatasetMeta(
        label_mapping={'no-recurrence-events': 0, 'recurrence-events': 1},
        allow_missing_values=True,
    ),
    'diabetes':        DatasetMeta(
        label_mapping={'tested_negative': 0, 'tested_positive': 1},
        allow_missing_values=True,
    ),
    'divorce':         DatasetMeta(
        openml_name='divorce_prediction',
        label_mapping={'0': 0, '1': 1},
    ),
    'german-credit':   DatasetMeta(
        label_mapping={'good': 0, 'bad': 1},
    ),
    'heart-disease':   DatasetMeta(
        label_mapping={'absent': 0, 'present': 1},
        allow_missing_values=True,
    ),
    'hepatitis':       DatasetMeta(
        label_mapping={'LIVE': 0, 'DIE': 1},
        allow_missing_values=True,
    ),
    'SMK_CAN_187':     DatasetMeta(
        openml_name='SMK',
        label_mapping={'1': 0, '2': 1},
    ),
    'sonar':           DatasetMeta(
        openml_name='sonar',
        label_mapping={'Rock': 0, 'Mine': 1},
    ),
    'turkiye-student': DatasetMeta(
        label_mapping={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4},
    ),
    'vehicle':         DatasetMeta(
        label_mapping={'opel': 0, 'saab': 1, 'van': 2, 'bus': 3},
    ),
    'wdbc':            DatasetMeta(
        openml_name='wdbc',
        label_mapping={'1': 0, '2': 1},
    ),
    'wine-quality':    DatasetMeta(
        openml_name='wine-quality-white',
        label_mapping={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6},
    ),
    'wine':            DatasetMeta(
        label_mapping={'1': 0, '2': 1, '3': 2},
        allow_missing_values=True,
    ),
    # Custom-loaded (handled by DatasetRegistry in datasets/loader.py)
    'Yale':            DatasetMeta(),
    'DrivFace':        DatasetMeta(),
    'PEMS-SF':         DatasetMeta(),
}


@dataclass
class BFMDTConfig:
    """Centralized configuration for the BFMDT pipeline.

    Args:
        sigma (float | str): Fitting degree threshold, or 'auto'. Defaults to 'auto'.
        delta (float): RMI threshold for tree splitting. Defaults to 0.01.
        max_reducts (int): Maximum feature subsets per sigma. Defaults to 50.
        min_sigma_candidates (int): Minimum sigma candidates. Defaults to 5.
        max_sigma_candidates (int): Maximum sigma candidates. Defaults to 30.
        max_sigma_iterations (int): Hard cap on sp adjustment iterations. Defaults to 1000.
        n_folds (int): Number of cross-validation folds. Defaults to 5.
        test_size (float): Test split ratio for large datasets. Defaults to 0.2.
        random_seed (int): Random seed for reproducibility. Defaults to 0.
        dataset_names (list[str]): Dataset names to benchmark. Defaults to all 18.
    """

    # Classifier hyperparameters
    sigma: float | str = "auto"
    delta: float = 0.01
    max_reducts: int = 50

    # Sigma selection
    min_sigma_candidates: int = 5
    max_sigma_candidates: int = 30
    max_sigma_iterations: int = 1000

    # Experiment settings
    n_folds: int = 5
    test_size: float = 0.2
    random_seed: int = 0
    mode: str = "cv"
    report: str = "all"
    dataset_names: list[str] = field(default_factory=lambda: list(ALL_DATASET_NAMES))

    # Algorithm version
    fitting_version: int = 1
    tree_version: int = 1


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


def _resolve_dataset_arg(values):
    """Resolve --datasets values (mix of IDs, names, or 'all') to a list of names.

    Each value can be:
      * 'all' (only valid as the sole argument) -> all 18 datasets
      * a numeric string '1'..'18' -> looked up via DATASET_NAMES_BY_ID
      * a dataset name from ALL_DATASET_NAMES
    """
    if values == ["all"]:
        return list(ALL_DATASET_NAMES)

    resolved = []
    for v in values:
        if v.isdigit():
            i = int(v)
            if i not in DATASET_NAMES_BY_ID:
                raise argparse.ArgumentTypeError(
                    f"Dataset ID must be 1-18, got {i}"
                )
            resolved.append(DATASET_NAMES_BY_ID[i])
        elif v in ALL_DATASET_NAMES:
            resolved.append(v)
        else:
            raise argparse.ArgumentTypeError(
                f"Unknown dataset '{v}'. Use a name from {ALL_DATASET_NAMES}, "
                f"an ID 1-18, or 'all'."
            )
    return resolved


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
        "--datasets", nargs="+", default=list(ALL_DATASET_NAMES),
        metavar="NAME_OR_ID",
        help="Dataset names (e.g. 'wine'), IDs (1-18 per paper Table 4), or 'all' (default: all)",
    )
    parser.add_argument(
        "--sigma", type=_parse_sigma, default="auto",
        help="Fitting degree threshold or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--delta", type=float, default=0.01,
        help="RMI threshold for tree splitting (default: 0.01)",
    )
    parser.add_argument(
        "--max-reducts", type=int, default=50,
        help="Max feature subsets per sigma (default: 50)",
    )
    parser.add_argument(
        "--min-sigma-candidates", type=int, default=5,
        help="Min sigma candidates (default: 5)",
    )
    parser.add_argument(
        "--max-sigma-candidates", type=int, default=30,
        help="Max sigma candidates (default: 30)",
    )
    parser.add_argument(
        "--max-sigma-iterations", type=int, default=1000,
        help="Hard cap on sp adjustment iterations in sigma selection (default: 1000)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Test split ratio for large datasets (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--mode", choices=["cv", "reporting"], default="cv",
        help="'cv' runs k-fold CV only; 'reporting' runs CV + comparison tables "
             "vs paper (select which via --report) (default: cv)",
    )
    parser.add_argument(
        "--report", choices=["sigma", "ca-mae", "reducts", "time", "all"], default="all",
        help="Which comparison table(s) to print in 'reporting' mode (default: all)",
    )
    parser.add_argument(
        "--fitting-version", type=int, choices=[1, 2], default=1,
        help="Fitting degree algorithm version: 1 = paper Algorithm 2 as published, "
             "2 = symmetric directional-inversion variant (default: 1)",
    )
    parser.add_argument(
        "--tree-version", type=int, choices=[1, 2], default=1,
        help="Monotonic decision tree version: 1 = matches paper Fig 3 / current "
             "tests (RMI dual + lower-bound splits), 2 = strict paper Eq 13/14/16 "
             "(ARMI=LEFT-sum, DRMI=RIGHT-sum, midpoint splits) (default: 1)",
    )

    args = parser.parse_args(argv)

    return BFMDTConfig(
        sigma=args.sigma,
        delta=args.delta,
        max_reducts=args.max_reducts,
        min_sigma_candidates=args.min_sigma_candidates,
        max_sigma_candidates=args.max_sigma_candidates,
        max_sigma_iterations=args.max_sigma_iterations,
        n_folds=args.n_folds,
        test_size=args.test_size,
        random_seed=args.seed,
        mode=args.mode,
        report=args.report,
        dataset_names=_resolve_dataset_arg(args.datasets),
        fitting_version=args.fitting_version,
        tree_version=args.tree_version,
    )
